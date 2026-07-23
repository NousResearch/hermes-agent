"""Plugin catalog — curated, Nous-approved Hermes plugins shipped with the repo.

Mirrors the ``optional-mcps/`` MCP-catalog pattern (see
:mod:`hermes_cli.mcp_catalog`): each catalog entry is a single YAML file under
the in-tree ``plugin-catalog/`` directory, pinned to an exact 40-character
commit SHA. Users discover entries via ``hermes plugins catalog`` /
``hermes plugins search`` and install them with
``hermes plugins install <name>``, which clones the pinned commit.

Catalog policy (see plugin-catalog/README.md for the full admission policy):
- Entries are added only by merging a PR into hermes-agent — presence in the
  ``plugin-catalog/`` directory is the human-merged approval gate.
- Every entry pins an exact 40-hex commit SHA. SHA bumps are new PRs,
  re-reviewed as diffs. The pinned release should be at least 2 weeks old at
  pin time, mirroring the optional-mcps supply-chain rules.
- ``plugin-catalog/removed.yaml`` is the blocklist: entries pulled from the
  catalog for security or policy reasons are recorded there so installs of
  the same name/repo are refused with the recorded reason.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml

logger = logging.getLogger(__name__)

CATALOG_TIERS = ("official", "community")

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class RemovedEntry:
    name: str
    repo: str = ""
    reason: str = ""
    date: str = ""            # ISO date string


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
    sha: str                  # 40-hex pinned commit — MANDATORY, validated
    description: str
    maintainer: str
    tier: str = "community"   # one of CATALOG_TIERS
    requires_hermes: str = "" # e.g. ">=0.19" (optional)
    subdir: str = ""          # optional path within the repo
    docs_url: str = ""
    platforms: List[str] = field(default_factory=list)  # empty = all OSes
    capabilities: CatalogCapabilities = field(default_factory=CatalogCapabilities)


# ─── Directory resolution ────────────────────────────────────────────────────


def get_catalog_dir() -> Path:
    """Return the ``plugin-catalog/`` directory shipped with this checkout.

    ``HERMES_PLUGIN_CATALOG_DIR`` overrides the location for tests only —
    read via ``os.getenv`` at call time so monkeypatched values take effect.
    """
    override = os.getenv("HERMES_PLUGIN_CATALOG_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "plugin-catalog"


# ─── Loading / validation ────────────────────────────────────────────────────


def _str_list(raw: Any) -> List[str]:
    """Coerce a YAML value into a list of strings (drop non-strings)."""
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, (str, int, float))]


def _parse_entry(path: Path) -> Optional[PluginCatalogEntry]:
    """Parse and validate one catalog YAML file.

    Returns ``None`` (after logging a warning) on any validation failure —
    the loader never raises for a bad entry.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Plugin catalog: failed to read %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Plugin catalog: %s: entry must be a mapping", path)
        return None

    name = str(data.get("name") or "")
    if not _NAME_RE.match(name):
        logger.warning(
            "Plugin catalog: %s: invalid name %r (must match [a-z0-9_-]{1,64})",
            path, name,
        )
        return None

    repo = str(data.get("repo") or "")
    if not repo.startswith("https://"):
        logger.warning(
            "Plugin catalog: %s: repo must be an https:// URL (got %r)",
            path, repo,
        )
        return None

    sha = str(data.get("sha") or "").strip().lower()
    if not _SHA_RE.match(sha):
        logger.warning(
            "Plugin catalog: %s: sha must be a full 40-character hex commit "
            "SHA (got %r)", path, data.get("sha"),
        )
        return None

    tier = str(data.get("tier") or "community")
    if tier not in CATALOG_TIERS:
        logger.warning(
            "Plugin catalog: %s: tier must be one of %s (got %r)",
            path, "/".join(CATALOG_TIERS), tier,
        )
        return None

    caps_raw = data.get("capabilities") or {}
    if not isinstance(caps_raw, dict):
        caps_raw = {}
    capabilities = CatalogCapabilities(
        provides_tools=_str_list(caps_raw.get("provides_tools")),
        provides_hooks=_str_list(caps_raw.get("provides_hooks")),
        provides_middleware=_str_list(caps_raw.get("provides_middleware")),
        requires_env=_str_list(caps_raw.get("requires_env")),
    )

    return PluginCatalogEntry(
        name=name,
        repo=repo,
        sha=sha,
        description=str(data.get("description") or "").strip(),
        maintainer=str(data.get("maintainer") or "").strip(),
        tier=tier,
        requires_hermes=str(data.get("requires_hermes") or "").strip(),
        subdir=str(data.get("subdir") or "").strip(),
        docs_url=str(data.get("docs_url") or "").strip(),
        platforms=_str_list(data.get("platforms")),
        capabilities=capabilities,
    )


def load_catalog() -> List[PluginCatalogEntry]:
    """Return all valid catalog entries, sorted by name.

    Parses every ``*.yaml`` in the catalog dir except ``removed.yaml``.
    Invalid entries are skipped with a logged warning; this function never
    raises for a malformed entry.
    """
    return _load_entries_from_dir(get_catalog_dir())


def _load_entries_from_dir(root: Path) -> List[PluginCatalogEntry]:
    """Parse all catalog entry files in *root* (skipping ``removed.yaml``)."""
    if not root.is_dir():
        return []
    entries: List[PluginCatalogEntry] = []
    for path in sorted(root.glob("*.yaml")):
        if path.name == "removed.yaml":
            continue
        entry = _parse_entry(path)
        if entry is not None:
            entries.append(entry)
    return entries


def get_catalog_entry(name: str) -> Optional[PluginCatalogEntry]:
    """Look up a single catalog entry by name."""
    for entry in load_catalog():
        if entry.name == name:
            return entry
    return None


def search_catalog(query: str) -> List[PluginCatalogEntry]:
    """Case-insensitive substring search over name, description, and
    declared tools. An empty query returns the whole catalog."""
    return filter_entries(load_catalog(), query)


def filter_entries(
    entries: List[PluginCatalogEntry], query: str
) -> List[PluginCatalogEntry]:
    """Filter *entries* with :func:`search_catalog` semantics.

    Lets callers that already hold a (possibly live-fetched) entry list
    apply the same matching rules without re-loading the catalog.
    """
    q = (query or "").strip().lower()
    if not q:
        return entries
    results: List[PluginCatalogEntry] = []
    for entry in entries:
        haystacks = [entry.name, entry.description]
        haystacks.extend(entry.capabilities.provides_tools)
        if any(q in h.lower() for h in haystacks):
            results.append(entry)
    return results


# ─── Removed / blocklist ─────────────────────────────────────────────────────


def _normalize_repo(url: str) -> str:
    """Normalize a repo URL for comparison (.git suffix and trailing slash
    stripped, lowercased)."""
    return url.strip().rstrip("/").removesuffix(".git").lower()


def load_removed_list() -> List[RemovedEntry]:
    """Load ``plugin-catalog/removed.yaml`` (the ``removed:`` list).

    Missing or malformed files yield an empty list — never raises.
    """
    path = get_catalog_dir() / "removed.yaml"
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Plugin catalog: failed to read %s: %s", path, exc)
        return []
    raw_list = data.get("removed") if isinstance(data, dict) else None
    if not isinstance(raw_list, list):
        return []
    removed: List[RemovedEntry] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "")
        if not name:
            continue
        removed.append(
            RemovedEntry(
                name=name,
                repo=str(raw.get("repo") or ""),
                reason=str(raw.get("reason") or ""),
                date=str(raw.get("date") or ""),
            )
        )
    return removed


def find_removed(name_or_repo: str) -> Optional[RemovedEntry]:
    """Match *name_or_repo* against the removed blocklist.

    Matches by exact catalog name OR by repo URL (normalized — ``.git``
    suffix and trailing slashes are ignored).
    """
    if not name_or_repo:
        return None
    candidate = name_or_repo.strip()
    candidate_repo = _normalize_repo(candidate)
    for entry in load_removed_list():
        if candidate == entry.name:
            return entry
        if entry.repo and candidate_repo == _normalize_repo(entry.repo):
            return entry
    return None


# ─── Live index ──────────────────────────────────────────────────────────────

# GitHub contents API for the in-repo catalog dir. Unauthenticated (60 req/hr
# rate limit) — fine for interactive use, and any failure falls back to the
# in-tree catalog silently.
_LIVE_INDEX_URL = (
    "https://api.github.com/repos/NousResearch/hermes-agent/contents/"
    "plugin-catalog?ref=main"
)
_LIVE_TTL_SECONDS = 6 * 60 * 60  # 6h
_REQUEST_TIMEOUT = 5.0


def _live_cache_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "plugin-catalog"


def fetch_live_catalog(*, force: bool = False) -> Optional[Path]:
    """Refresh the catalog cache from the GitHub repo; return the cache dir.

    Lists ``plugin-catalog/*.yaml`` via the GitHub contents API, raw-fetches
    each file, and stores them under ``<hermes_home>/cache/plugin-catalog/``
    with a 6-hour TTL (repeat searches don't re-hit the API). Returns the
    cache directory on success (or fresh cache), or ``None`` on ANY network
    or parse failure — callers then fall back to the in-tree catalog.
    """
    cache = _live_cache_dir()
    marker = cache / ".fetched"
    if not force and marker.is_file():
        try:
            age = time.time() - marker.stat().st_mtime
        except OSError:
            age = _LIVE_TTL_SECONDS + 1
        if age < _LIVE_TTL_SECONDS:
            return cache

    try:
        import httpx

        resp = httpx.get(
            _LIVE_INDEX_URL,
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        listing = resp.json()
        if not isinstance(listing, list):
            raise ValueError("unexpected contents-API payload")

        fetched: dict[str, str] = {}
        for item in listing:
            if not isinstance(item, dict):
                continue
            fname = str(item.get("name") or "")
            url = str(item.get("download_url") or "")
            if not fname.endswith(".yaml") or not url:
                continue
            file_resp = httpx.get(
                url, timeout=_REQUEST_TIMEOUT, follow_redirects=True
            )
            file_resp.raise_for_status()
            fetched[fname] = file_resp.text

        cache.mkdir(parents=True, exist_ok=True)
        # Replace stale cached entries wholesale so removed files disappear.
        for old in cache.glob("*.yaml"):
            if old.name not in fetched:
                old.unlink(missing_ok=True)
        for fname, text in fetched.items():
            (cache / fname).write_text(text, encoding="utf-8")
        marker.touch()
        return cache
    except Exception as exc:
        logger.debug("Plugin catalog: live index fetch failed: %s", exc)
        return None


def load_catalog_live() -> List[PluginCatalogEntry]:
    """Return catalog entries, preferring a live-fetched (or cached) index.

    Falls back silently to the in-tree catalog when the network is
    unavailable or the fetch fails.
    """
    cache = fetch_live_catalog()
    if cache is not None and any(
        p.name != "removed.yaml" for p in cache.glob("*.yaml")
    ):
        return _load_entries_from_dir(cache)
    return load_catalog()


# ─── Human summaries ─────────────────────────────────────────────────────────


def entry_capability_summary(entry: PluginCatalogEntry) -> str:
    """One-paragraph human summary of what an entry declares, shown at
    install prompts so the user knows what they're granting."""
    caps = entry.capabilities
    parts: List[str] = []
    if caps.provides_tools:
        parts.append(f"registers tool(s): {', '.join(caps.provides_tools)}")
    if caps.provides_hooks:
        parts.append(f"hook(s): {', '.join(caps.provides_hooks)}")
    if caps.provides_middleware:
        parts.append(f"middleware: {', '.join(caps.provides_middleware)}")
    if caps.requires_env:
        parts.append(f"requires env var(s): {', '.join(caps.requires_env)}")
    if not parts:
        capability_text = "declares no tools, hooks, middleware, or env vars"
    else:
        capability_text = "; ".join(parts)
    bits = [
        f"{entry.name} ({entry.tier}, maintained by {entry.maintainer})",
    ]
    if entry.description:
        bits.append(entry.description)
    bits.append(f"This plugin {capability_text}.")
    if entry.platforms:
        bits.append(f"Platforms: {', '.join(entry.platforms)}.")
    if entry.requires_hermes:
        bits.append(f"Requires Hermes {entry.requires_hermes}.")
    return " ".join(bits)
