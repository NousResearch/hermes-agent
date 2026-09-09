"""Plugin catalog — curated, Nous-approved Hermes plugins shipped with the repo.

Mirrors the ``optional-mcps/`` MCP-catalog pattern: one YAML file per entry under the in-tree
``plugin-catalog/`` directory, pinned to an exact 40-character commit SHA. Presence in the directory IS
the human-merged approval gate; SHA bumps are new, re-reviewed PRs; ``removed.yaml`` is the kill list
(installs of a removed name/repo are refused with the recorded reason). Full policy:
``plugin-catalog/README.md``.

Live refresh: the docs build publishes the same data as ONE JSON document
(``website/scripts/extract-plugins.py`` → ``/docs/api/plugin-catalog.json``, like the skills index), so
an installed Hermes sees new entries and removals without updating. Any fetch failure falls back to the
in-tree copy silently.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

CATALOG_TIERS = ("official", "community")
LIVE_CATALOG_URL = "https://hermes-agent.nousresearch.com/docs/api/plugin-catalog.json"
LIVE_CATALOG_TTL_SECONDS = 6 * 60 * 60
_REQUEST_TIMEOUT = 5.0
_MAX_LIVE_BYTES = 2 * 1024 * 1024

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")


@dataclass
class RemovedEntry:
    name: str
    repo: str = ""
    reason: str = ""
    date: str = ""


@dataclass
class CatalogCapabilities:
    provides_tools: List[str] = field(default_factory=list)
    provides_hooks: List[str] = field(default_factory=list)
    provides_middleware: List[str] = field(default_factory=list)
    requires_env: List[str] = field(default_factory=list)


@dataclass
class PluginCatalogEntry:
    name: str                 # catalog key, [a-z0-9_-]{1,64}
    repo: str                 # https:// git URL
    sha: str                  # 40-hex pinned commit — mandatory
    description: str
    maintainer: str
    tier: str = "community"
    requires_hermes: str = ""
    subdir: str = ""
    docs_url: str = ""
    platforms: List[str] = field(default_factory=list)  # empty = all OSes
    capabilities: CatalogCapabilities = field(default_factory=CatalogCapabilities)

    @property
    def install_identifier(self) -> str:
        """``_install_plugin_core`` identifier (``repo#subdir`` for monorepo entries)."""
        return f"{self.repo}#{self.subdir}" if self.subdir else self.repo

    def to_dict(self) -> Dict[str, Any]:
        caps = self.capabilities
        return {
            "name": self.name, "repo": self.repo, "sha": self.sha, "description": self.description,
            "maintainer": self.maintainer, "tier": self.tier, "requires_hermes": self.requires_hermes,
            "subdir": self.subdir, "docs_url": self.docs_url, "platforms": list(self.platforms),
            "capabilities": {
                "provides_tools": list(caps.provides_tools), "provides_hooks": list(caps.provides_hooks),
                "provides_middleware": list(caps.provides_middleware), "requires_env": list(caps.requires_env),
            },
        }


def get_catalog_dir() -> Path:
    """The ``plugin-catalog/`` directory shipped with this checkout."""
    return Path(__file__).resolve().parent.parent / "plugin-catalog"


# ── Parsing ──────────────────────────────────────────────────────────────────

def _str_list(raw: Any) -> List[str]:
    return [str(x) for x in raw if isinstance(x, (str, int, float))] if isinstance(raw, list) else []


def entry_from_mapping(data: Any, label: str) -> Optional[PluginCatalogEntry]:
    """Validate one entry mapping (YAML file or live-JSON element); ``None`` + warning on any failure."""
    if not isinstance(data, dict):
        logger.warning("Plugin catalog: %s: entry must be a mapping", label)
        return None
    name = str(data.get("name") or "")
    repo = str(data.get("repo") or "")
    sha = str(data.get("sha") or "").strip().lower()
    tier = str(data.get("tier") or "community")
    problem = (
        f"invalid name {name!r} (must match [a-z0-9_-]{{1,64}})" if not _NAME_RE.match(name)
        else f"repo must be an https:// URL (got {repo!r})" if not repo.startswith("https://")
        else f"sha must be a full 40-character hex commit SHA (got {data.get('sha')!r})" if not _SHA_RE.match(sha)
        else f"tier must be one of {'/'.join(CATALOG_TIERS)} (got {tier!r})" if tier not in CATALOG_TIERS
        else None)
    if problem:
        logger.warning("Plugin catalog: %s: %s", label, problem)
        return None
    caps_raw = data.get("capabilities")
    caps: Dict[str, Any] = caps_raw if isinstance(caps_raw, dict) else {}
    return PluginCatalogEntry(
        name=name, repo=repo, sha=sha,
        description=str(data.get("description") or "").strip(),
        maintainer=str(data.get("maintainer") or "").strip(), tier=tier,
        requires_hermes=str(data.get("requires_hermes") or "").strip(),
        subdir=str(data.get("subdir") or "").strip(), docs_url=str(data.get("docs_url") or "").strip(),
        platforms=_str_list(data.get("platforms")),
        capabilities=CatalogCapabilities(
            provides_tools=_str_list(caps.get("provides_tools")), provides_hooks=_str_list(caps.get("provides_hooks")),
            provides_middleware=_str_list(caps.get("provides_middleware")),
            requires_env=_str_list(caps.get("requires_env"))),
    )


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Plugin catalog: failed to read %s: %s", path, exc)
        return None


def _removed_from_list(raw_list: Any) -> List[RemovedEntry]:
    if not isinstance(raw_list, list):
        return []
    return [
        RemovedEntry(name=str(r["name"]), repo=str(r.get("repo") or ""), reason=str(r.get("reason") or ""),
                     date=str(r.get("date") or ""))
        for r in raw_list if isinstance(r, dict) and r.get("name")]


# ── In-tree catalog ──────────────────────────────────────────────────────────

def load_catalog(catalog_dir: Optional[Path] = None) -> List[PluginCatalogEntry]:
    """Every valid ``*.yaml`` entry in the catalog dir (``removed.yaml`` excluded), sorted by file name.
    Malformed entries are skipped with a warning — never raises."""
    root = catalog_dir or get_catalog_dir()
    if not root.is_dir():
        return []
    entries = []
    for path in sorted(root.glob("*.yaml")):
        if path.name == "removed.yaml":
            continue
        data = _read_yaml(path)
        entry = entry_from_mapping(data, str(path)) if data is not None else None
        if entry is not None:
            entries.append(entry)
    return entries


def load_removed_list(catalog_dir: Optional[Path] = None) -> List[RemovedEntry]:
    """``removed.yaml``'s ``removed:`` list; missing/malformed → empty."""
    path = (catalog_dir or get_catalog_dir()) / "removed.yaml"
    data = _read_yaml(path) if path.is_file() else None
    return _removed_from_list(data.get("removed")) if isinstance(data, dict) else []


def get_catalog_entry(name: str, catalog_dir: Optional[Path] = None) -> Optional[PluginCatalogEntry]:
    return next((e for e in load_catalog(catalog_dir) if e.name == name), None)


def filter_entries(entries: List[PluginCatalogEntry], query: str) -> List[PluginCatalogEntry]:
    """Case-insensitive substring match over name, description and declared tools; empty query = all."""
    q = (query or "").strip().lower()
    if not q:
        return entries
    return [e for e in entries
            if any(q in h.lower() for h in (e.name, e.description, *e.capabilities.provides_tools))]


def search_catalog(query: str) -> List[PluginCatalogEntry]:
    return filter_entries(load_catalog(), query)


# ── Removed / blocklist ──────────────────────────────────────────────────────

def _normalize_repo(url: str) -> str:
    return url.strip().rstrip("/").removesuffix(".git").lower()


def find_removed(name_or_repo: str, catalog_dir: Optional[Path] = None) -> Optional[RemovedEntry]:
    """Match a catalog name or repo URL (``.git``/trailing-slash insensitive) against the kill list.

    The in-tree list and the live-fetched list are UNIONED: a removal published after this checkout
    shipped must still block, and a stale live cache must not un-block an in-tree removal.
    """
    if not name_or_repo:
        return None
    candidate = name_or_repo.strip()
    candidate_repo = _normalize_repo(candidate)
    for entry in load_removed_list(catalog_dir) + (live_removed_list() if catalog_dir is None else []):
        if candidate == entry.name or (entry.repo and candidate_repo == _normalize_repo(entry.repo)):
            return entry
    return None


# ── Live catalog ─────────────────────────────────────────────────────────────

def _live_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "plugin-catalog.json"


def fetch_live_catalog(*, force: bool = False) -> Optional[Dict[str, Any]]:
    """The published ``plugin-catalog.json`` (``{"entries": [...], "removed": [...]}``), cached under
    ``HERMES_HOME/cache`` for :data:`LIVE_CATALOG_TTL_SECONDS`. ``None`` on ANY failure — callers fall
    back to the in-tree catalog."""
    cache = _live_cache_path()
    try:
        if not force and cache.is_file() and time.time() - cache.stat().st_mtime < LIVE_CATALOG_TTL_SECONDS:
            return json.loads(cache.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Plugin catalog: unreadable live cache %s: %s", cache, exc)
    try:
        import httpx
        resp = httpx.get(LIVE_CATALOG_URL, timeout=_REQUEST_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        if len(resp.content) > _MAX_LIVE_BYTES:
            raise ValueError("live catalog payload too large")
        data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            raise ValueError("unexpected live catalog payload")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception as exc:
        logger.debug("Plugin catalog: live fetch failed: %s", exc)
        try:  # stale cache still beats the in-tree copy when the network is down
            return json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else None
        except Exception:
            return None


def load_catalog_live() -> List[PluginCatalogEntry]:
    """Entries from the live (or cached) catalog, else the in-tree catalog."""
    data = fetch_live_catalog()
    if data is not None:
        entries = [e for i, raw in enumerate(data["entries"])
                   if (e := entry_from_mapping(raw, f"{LIVE_CATALOG_URL}#{i}")) is not None]
        if entries:
            return entries
    return load_catalog()


def live_removed_list() -> List[RemovedEntry]:
    data = fetch_live_catalog()
    return _removed_from_list(data.get("removed")) if data else []


def get_live_catalog_entry(name: str) -> Optional[PluginCatalogEntry]:
    return next((e for e in load_catalog_live() if e.name == name), None)


# ── Human summaries ──────────────────────────────────────────────────────────

def entry_capability_summary(entry: PluginCatalogEntry) -> str:
    """One paragraph shown at install/enable prompts: what the user is granting."""
    caps = entry.capabilities
    parts = [f"{label} {', '.join(items)}" for label, items in (
        ("registers tool(s):", caps.provides_tools), ("hook(s):", caps.provides_hooks),
        ("middleware:", caps.provides_middleware), ("requires env var(s):", caps.requires_env)) if items]
    bits = [f"{entry.name} ({entry.tier}, maintained by {entry.maintainer})"]
    if entry.description:
        bits.append(entry.description)
    bits.append(f"This plugin {'; '.join(parts) if parts else 'declares no tools, hooks, middleware, or env vars'}.")
    if entry.platforms:
        bits.append(f"Platforms: {', '.join(entry.platforms)}.")
    if entry.requires_hermes:
        bits.append(f"Requires Hermes {entry.requires_hermes}.")
    return " ".join(bits)
