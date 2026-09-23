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

import dataclasses
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
# Browse taxonomy for the catalog page / picker. Entries without one land on the Desktop shelf
# (the common case for community submissions); "general" is for plugins that fit no shelf.
CATALOG_CATEGORIES = ("desktop", "memory", "platform", "web", "tools", "voice", "automation", "models", "general")
LIVE_CATALOG_URL = "https://hermes-agent.nousresearch.com/docs/api/plugin-catalog.json"
LIVE_CATALOG_TTL_SECONDS = 6 * 60 * 60
# Past this age an offline cache no longer supplies PINS (the in-tree catalog does); its removals
# still count — a kill-list entry never expires.
LIVE_CATALOG_MAX_STALE_SECONDS = 24 * 60 * 60
LIVE_CATALOG_FAILURE_TTL_SECONDS = 60.0
_REQUEST_TIMEOUT = 5.0
_MAX_LIVE_BYTES = 2 * 1024 * 1024

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$")
# Catalog images may only come from GitHub: the Desktop catalog browser never fans out to
# third-party hosts, and a raw URL pinned to the entry's commit is as immutable as the sha.
IMAGE_HOSTS = ("raw.githubusercontent.com", "github.com")
IMAGE_HOST_SUFFIX = ".githubusercontent.com"


def is_allowed_image_url(url: str) -> bool:
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and bool(host) and (host in IMAGE_HOSTS or host.endswith(IMAGE_HOST_SUFFIX))
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
    category: str = "desktop"
    requires_hermes: str = ""
    subdir: str = ""
    docs_url: str = ""
    version: str = ""            # human label for the pinned sha ("1.4.0"); cosmetic, never parsed
    image: str = ""              # https image URL on a GitHub host; shown on catalog cards
    screenshots: List[str] = field(default_factory=list)  # GitHub-hosted https URLs; gallery on /docs/plugins/<name>
    readme: bool = False         # docs site renders the README from the pinned commit on the entry's page
    platforms: List[str] = field(default_factory=list)  # empty = all OSes
    title: str = ""              # human name ("NVIDIA App"); empty = derived from ``name``
    onboarding: bool = False     # curated: offered on the desktop onboarding card
    capabilities: CatalogCapabilities = field(default_factory=CatalogCapabilities)

    @property
    def install_identifier(self) -> str:
        """``_install_plugin_core`` identifier (``repo#subdir`` for monorepo entries)."""
        return f"{self.repo}#{self.subdir}" if self.subdir else self.repo

    def to_dict(self) -> Dict[str, Any]:
        caps = self.capabilities
        return {
            "name": self.name, "repo": self.repo, "sha": self.sha, "description": self.description,
            "maintainer": self.maintainer, "tier": self.tier, "category": self.category,
            "requires_hermes": self.requires_hermes,
            "subdir": self.subdir, "docs_url": self.docs_url, "version": self.version, "image": self.image,
            "screenshots": list(self.screenshots), "readme": self.readme,
            "platforms": list(self.platforms), "title": self.title, "onboarding": self.onboarding,
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
    category = str(data.get("category") or "desktop")
    problem = (
        f"invalid name {name!r} (must match [a-z0-9_-]{{1,64}})" if not _NAME_RE.match(name)
        else f"repo must be an https:// URL (got {repo!r})" if not repo.startswith("https://")
        else f"sha must be a full 40-character hex commit SHA (got {data.get('sha')!r})" if not _SHA_RE.match(sha)
        else f"tier must be one of {'/'.join(CATALOG_TIERS)} (got {tier!r})" if tier not in CATALOG_TIERS
        else f"category must be one of {'/'.join(CATALOG_CATEGORIES)} (got {category!r})"
        if category not in CATALOG_CATEGORIES
        else None)
    if problem:
        logger.warning("Plugin catalog: %s: %s", label, problem)
        return None
    caps_raw = data.get("capabilities")
    caps: Dict[str, Any] = caps_raw if isinstance(caps_raw, dict) else {}
    version = str(data.get("version") or "").strip()
    if version and not _VERSION_RE.match(version):
        logger.warning("Plugin catalog: %s: ignoring version %r (max 32 chars of [A-Za-z0-9._+-])", label, version)
        version = ""
    image = str(data.get("image") or "").strip()
    if image and not is_allowed_image_url(image):
        logger.warning("Plugin catalog: %s: ignoring image %r (must be https on a GitHub host)", label, image)
        image = ""
    screenshots = [s for s in _str_list(data.get("screenshots")) if is_allowed_image_url(s)]
    if len(screenshots) != len(_str_list(data.get("screenshots"))):
        logger.warning("Plugin catalog: %s: ignoring screenshots off GitHub hosts", label)
    return PluginCatalogEntry(
        name=name, repo=repo, sha=sha,
        description=str(data.get("description") or "").strip(),
        maintainer=str(data.get("maintainer") or "").strip(), tier=tier, category=category,
        requires_hermes=str(data.get("requires_hermes") or "").strip(),
        subdir=str(data.get("subdir") or "").strip(), docs_url=str(data.get("docs_url") or "").strip(),
        version=version, image=image, screenshots=screenshots, readme=data.get("readme") is not False,
        platforms=_str_list(data.get("platforms")),
        title=str(data.get("title") or "").strip(), onboarding=data.get("onboarding") is True,
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

_SCP_URL_RE = re.compile(r"^(?:[^@/\s]+@)?([^:/\s]+):(?!//)(.+)$")  # git@host:owner/repo


def _normalize_repo(url: str) -> str:
    """Canonical ``host/path`` for a repo URL: scheme, user, ``www.``, ``.git`` and trailing slashes are
    spelling, not identity — the kill list must match ``git@github.com:Evil/Bad.git`` when it names
    ``https://github.com/evil/bad``."""
    from urllib.parse import urlsplit
    text = url.strip()
    scp = _SCP_URL_RE.match(text)
    if scp:
        host, path = scp.group(1), scp.group(2)
    elif "://" in text:
        parts = urlsplit(text)
        host, path = parts.hostname or "", parts.path
    else:
        host, path = "", text
    host = host.lower().removeprefix("www.")
    path = path.strip("/").removesuffix(".git").rstrip("/").lower()
    return f"{host}/{path}" if host else path


def find_removed(name_or_repo: str, catalog_dir: Optional[Path] = None) -> Optional[RemovedEntry]:
    """Match a catalog name or repo URL (``.git``/trailing-slash insensitive) against the kill list.

    The in-tree list and the live-fetched list are UNIONED: a removal published after this checkout
    shipped must still block, and a stale live cache must not un-block an in-tree removal.
    """
    if not name_or_repo:
        return None
    entries = load_removed_list(catalog_dir)
    if catalog_dir is None:
        entries = entries + live_removed_list()
    return match_removed(name_or_repo, entries)


def resolved_removed_entries() -> List[RemovedEntry]:
    """The full kill list (in-tree UNION live) in one resolution. Callers that match many candidates
    — e.g. a plugins-hub rebuild annotating every installed plugin — resolve the list once instead
    of paying a live-catalog fetch per candidate."""
    return load_removed_list() + live_removed_list()


def cached_removed_entries() -> List[RemovedEntry]:
    """In-tree list UNION the last fetched live copy, with NO network round-trip — for the load-time and
    ``enable`` checks that run in every process and must never block on a dead catalog host."""
    cached = _stale_live_cache(_live_cache_path()) or {}
    return load_removed_list() + _removed_from_list(cached.get("removed"))


def match_removed(
    candidate: str, entries: List[RemovedEntry]
) -> Optional[RemovedEntry]:
    """One candidate against a pre-resolved kill list: exact name or normalized repo URL match."""
    if not candidate:
        return None
    text = candidate.strip()
    text_repo = _normalize_repo(text)
    for entry in entries:
        if text == entry.name or (
            entry.repo and text_repo == _normalize_repo(entry.repo)
        ):
            return entry
    return None


# ── Live catalog ─────────────────────────────────────────────────────────────

def _live_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "plugin-catalog.json"


# Wall-clock deadline of the last failed live fetch. Without it a dead catalog host costs one
# full request timeout PER CALL (the plugins hub and ``plugins list`` used to ask once per
# installed plugin), so the dashboard event loop stalled for minutes.
_live_fetch_failed_until = 0.0


def _stale_live_cache(cache: Path) -> Optional[Dict[str, Any]]:
    """A previously fetched copy still beats the in-tree one when the network is down — for
    :data:`LIVE_CATALOG_MAX_STALE_SECONDS`. Past that its pins may trail the checkout's own catalog
    (a 90-day-old cache outranked a freshly updated in-tree pin), so the entries are dropped and the
    caller falls back to in-tree; the removals are kept."""
    try:
        if not cache.is_file():
            return None
        data = json.loads(cache.read_text(encoding="utf-8"))
        if time.time() - cache.stat().st_mtime > LIVE_CATALOG_MAX_STALE_SECONDS:
            data = {**data, "entries": []}
        return data
    except Exception:
        return None


def fetch_live_catalog(*, force: bool = False) -> Optional[Dict[str, Any]]:
    """The published ``plugin-catalog.json`` (``{"entries": [...], "removed": [...]}``), cached under
    ``HERMES_HOME/cache`` for :data:`LIVE_CATALOG_TTL_SECONDS`. ``None`` on ANY failure — callers fall
    back to the in-tree catalog. A failed network attempt is remembered for
    :data:`LIVE_CATALOG_FAILURE_TTL_SECONDS` so a dead host costs one timeout per TTL window, not one
    per call (``force`` bypasses both caches)."""
    global _live_fetch_failed_until
    cache = _live_cache_path()
    try:
        if not force and cache.is_file() and time.time() - cache.stat().st_mtime < LIVE_CATALOG_TTL_SECONDS:
            return json.loads(cache.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Plugin catalog: unreadable live cache %s: %s", cache, exc)
    if not force and time.time() < _live_fetch_failed_until:
        return _stale_live_cache(cache)
    try:
        import httpx
        from hermes_constants import mkdir_under_hermes_home
        from utils import atomic_write_text

        resp = httpx.get(LIVE_CATALOG_URL, timeout=_REQUEST_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        if len(resp.content) > _MAX_LIVE_BYTES:
            raise ValueError("live catalog payload too large")
        data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            raise ValueError("unexpected live catalog payload")
        mkdir_under_hermes_home(cache.parent)
        # Atomic: a concurrent reader (gateway, TUI, a second CLI) must never see a half-written
        # document, which would read as a fetch failure and start its own 60 s failure window.
        atomic_write_text(cache, json.dumps(data), tmp_prefix=f"{cache.name}.tmp-")
        return data
    except Exception as exc:
        logger.debug("Plugin catalog: live fetch failed: %s", exc)
        _live_fetch_failed_until = time.time() + LIVE_CATALOG_FAILURE_TTL_SECONDS
        return _stale_live_cache(cache)


_in_tree_catalog_time: Optional[float] = -1.0  # -1 = not resolved yet; None = no git checkout


def in_tree_catalog_time() -> Optional[float]:
    """Commit time (epoch) of the last change to this checkout's ``plugin-catalog/``, or ``None`` when
    the install is not a git checkout (a release/pip install cannot be newer than the published doc).
    Resolved once per process."""
    global _in_tree_catalog_time
    if _in_tree_catalog_time != -1.0:
        return _in_tree_catalog_time
    root = get_catalog_dir().parent
    resolved: Optional[float] = None
    if (root / ".git").exists():
        try:
            import subprocess
            out = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%ct", "--", "plugin-catalog"],
                                 capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
            resolved = float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
        except Exception as exc:
            logger.debug("Plugin catalog: could not date the in-tree catalog: %s", exc)
    _in_tree_catalog_time = resolved
    return resolved


def _live_generated_time(data: Dict[str, Any]) -> Optional[float]:
    raw = data.get("generated_at")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        from datetime import datetime, timezone
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
    except ValueError:
        return None


def _prefer_in_tree_entry(tree: PluginCatalogEntry, live: PluginCatalogEntry, tree_is_newer: Optional[bool]) -> bool:
    """For one entry present in both sources with a different pin: the newer catalog wins. Newer is
    decided by the checkout's catalog commit time vs the doc's ``generated_at`` when both resolve;
    otherwise by the entries' ``version`` labels when both parse; otherwise the live doc wins (a release
    install's in-tree copy is frozen at release time)."""
    if tree.sha == live.sha:
        return False
    if tree_is_newer is not None:
        return tree_is_newer
    if tree.version and live.version:
        try:
            from packaging.version import Version
            return Version(tree.version) > Version(live.version)
        except Exception:
            return False
    return False


def load_catalog_live() -> List[PluginCatalogEntry]:
    """Entries from the live (or cached) catalog, else the in-tree catalog. When both name an entry at
    different pins the NEWER source supplies it — right after ``hermes update`` bumps an in-tree pin,
    a cache fetched before the bump must not re-install the old one (see :func:`_prefer_in_tree_entry`)."""
    data = fetch_live_catalog()
    if data is None:
        return load_catalog()
    entries = [e for i, raw in enumerate(data["entries"])
               if (e := entry_from_mapping(raw, f"{LIVE_CATALOG_URL}#{i}")) is not None]
    if not entries:
        return load_catalog()
    in_tree = {e.name: e for e in load_catalog()}
    live_t, tree_t = _live_generated_time(data), in_tree_catalog_time()
    tree_is_newer = (tree_t > live_t) if (live_t is not None and tree_t is not None) else None
    raw_by_name = {str(raw.get("name")): raw for raw in data["entries"] if isinstance(raw, dict)}
    return [in_tree[e.name] if e.name in in_tree and _prefer_in_tree_entry(in_tree[e.name], e, tree_is_newer)
            else _with_curated_fields(e, in_tree.get(e.name), raw_by_name.get(e.name) or {})
            for e in entries]


# Curated display fields a published doc older than the field does not carry. ``generated_at`` is the
# docs build time, not the content time, so a rebuild of an older catalog outranks a checkout that added
# the field; a doc that has the key (even ``false``) decides.
_CURATED_FIELDS = ("onboarding", "title")


def _with_curated_fields(live: PluginCatalogEntry, tree: Optional[PluginCatalogEntry], raw: Dict[str, Any]
                         ) -> PluginCatalogEntry:
    if tree is None or tree.sha != live.sha:
        return live
    missing = {key: getattr(tree, key) for key in _CURATED_FIELDS if key not in raw}
    return dataclasses.replace(live, **missing) if missing else live


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
