"""Community plugin index — fetch, cache, search, and name resolution.

Mirrors the Skills Hub catalog (``tools/skills_hub.py``): a static JSON index at a canonical URL,
cached under ``HERMES_HOME/cache/`` with a TTL, bundled seed as offline fallback / format reference.
Fallback chain: fresh cache → remote → stale cache → bundled seed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Canonical index location. Override via config key ``plugins.index_url``.
DEFAULT_INDEX_URL = "https://raw.githubusercontent.com/NousResearch/hermes-plugin-index/main/index.json"
INDEX_CACHE_TTL = 24 * 3600  # a stale cache still beats the bundled seed when remote is unreachable
SEED_INDEX_PATH = Path(__file__).parent / "data" / "plugin_index.json"
_FETCH_TIMEOUT = 10.0
_MAX_INDEX_BYTES = 5 * 1024 * 1024  # refuse absurdly large index payloads

SECURITY_FOOTER = (
    "Indexed \u2260 audited: inclusion in the index is a metadata review only, "
    "not a code audit. Review a plugin before enabling it.")


@dataclass
class PluginIndexEntry:
    """One community plugin index entry."""

    name: str
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    repo: str = ""                      # "owner/name"
    ref: str = ""                       # pinned tag or commit SHA
    subdir: Optional[str] = None        # path within the repo (monorepos)
    homepage: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    api_version: Optional[int] = None
    added_at: Optional[str] = None

    @property
    def install_identifier(self) -> str:
        """Identifier accepted by the existing install path (owner/repo[/subdir])."""
        return f"{self.repo}/{self.subdir}" if self.subdir else self.repo

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name, "description": self.description, "author": self.author,
            "tags": list(self.tags), "repo": self.repo, "ref": self.ref,
        }
        # Optional keys are emitted only when set (api_version: only when not None).
        for key, value in (
            ("subdir", self.subdir), ("homepage", self.homepage),
            ("capabilities", list(self.capabilities)),
            ("api_version", self.api_version), ("added_at", self.added_at),
        ):
            if value or (key == "api_version" and value is not None):
                d[key] = value
        return d


def _cache_path() -> Path:
    return get_hermes_home() / "cache" / "plugin_index.json"


def get_index_url() -> str:
    """Resolve the index URL: config override ``plugins.index_url`` or default."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        override = cfg_get(load_config_readonly(), "plugins", "index_url", default=None)
        if isinstance(override, str) and override.strip():
            return override.strip()
    except Exception:  # pragma: no cover - config loading must never break search
        logger.debug("plugin index: config override lookup failed", exc_info=True)
    return DEFAULT_INDEX_URL


def _parse_entries(raw: Any) -> List[PluginIndexEntry]:
    """Parse a decoded index document (object with ``plugins`` or bare list), skipping malformed items."""
    if isinstance(raw, dict):
        raw = raw.get("plugins", [])
        if not isinstance(raw, list):
            raise ValueError("Plugin index 'plugins' field must be a list.")
    elif not isinstance(raw, list):
        raise ValueError("Plugin index must be a JSON object or list.")

    entries: List[PluginIndexEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        repo = item.get("repo")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(repo, str) or repo.count("/") != 1 or not all(repo.split("/")):
            logger.debug("plugin index: skipping entry %r with invalid repo %r", name, repo)
            continue
        subdir = item.get("subdir")
        api_version = item.get("api_version")
        entries.append(PluginIndexEntry(
            name=name.strip(),
            description=str(item.get("description") or ""),
            author=str(item.get("author") or ""),
            tags=[str(t) for t in item.get("tags") or [] if isinstance(t, (str, int))],
            repo=repo.strip(),
            ref=str(item.get("ref") or ""),
            subdir=subdir.strip("/") if isinstance(subdir, str) and subdir.strip("/") else None,
            homepage=str(item["homepage"]) if item.get("homepage") else None,
            capabilities=[str(c) for c in item.get("capabilities") or []],
            api_version=int(api_version) if isinstance(api_version, (int, str)) and str(api_version).isdigit() else None,
            added_at=str(item["added_at"]) if item.get("added_at") else None))
    return entries


def _load_seed_entries() -> List[PluginIndexEntry]:
    try:
        return _parse_entries(json.loads(SEED_INDEX_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:  # pragma: no cover - bundled file
        logger.warning("plugin index: bundled seed unreadable: %s", exc)
        return []


def _read_cache(*, max_age: Optional[float]) -> Optional[List[PluginIndexEntry]]:
    """Return cached entries if the cache exists (and is younger than *max_age*)."""
    cache = _cache_path()
    try:
        if not cache.is_file():
            return None
        if max_age is not None and time.time() - cache.stat().st_mtime > max_age:
            return None
        return _parse_entries(json.loads(cache.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        logger.debug("plugin index: cache read failed: %s", exc)
        return None


def _write_cache(text: str) -> None:
    try:
        cache = _cache_path()
        cache.parent.mkdir(parents=True, exist_ok=True)
        from utils import atomic_write_text
        atomic_write_text(cache, text)
    except OSError as exc:  # pragma: no cover - best effort
        logger.debug("plugin index: cache write failed: %s", exc)


def _fetch_remote() -> Optional[List[PluginIndexEntry]]:
    """Fetch and parse the remote index; cache the raw payload on success."""
    url = get_index_url()
    try:
        import httpx

        resp = httpx.get(url, timeout=_FETCH_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        if len(text.encode("utf-8", errors="ignore")) > _MAX_INDEX_BYTES:
            raise ValueError("Plugin index payload exceeds size limit.")
        entries = _parse_entries(json.loads(text))
        _write_cache(text)
        return entries
    except Exception as exc:
        logger.debug("plugin index: remote fetch failed (%s): %s", url, exc)
        return None


def load_index(*, refresh: bool = False, offline: bool = False) -> tuple[List[PluginIndexEntry], str]:
    """Load the plugin index as ``(entries, source)``; source is ``"remote"``/``"cache"``/``"seed"``.
    Order: fresh cache (unless *refresh*) → remote (unless *offline*) → stale cache → seed."""
    if not refresh:
        cached = _read_cache(max_age=INDEX_CACHE_TTL)
        if cached is not None:
            return cached, "cache"
    if not offline:
        remote = _fetch_remote()
        if remote is not None:
            return remote, "remote"
    stale = _read_cache(max_age=None)
    if stale is not None:
        return stale, "cache"
    return _load_seed_entries(), "seed"


def _score_entry(entry: PluginIndexEntry, term: str) -> float:
    """Fuzzy relevance score for *entry* against lowercase *term* (0 = no match)."""
    import difflib
    name = entry.name.lower()
    tags = [t.lower() for t in entry.tags]
    if term == name:
        return 100.0
    ratio = difflib.SequenceMatcher(None, term, name).ratio()  # typo tolerance on the name
    signals = (
        (term in name, 80.0), (term in tags, 70.0), (any(term in t for t in tags), 55.0),
        (term in entry.description.lower(), 50.0), (term in entry.author.lower(), 40.0),
        (ratio >= 0.6, ratio * 60.0))
    return max((points for hit, points in signals if hit), default=0.0)


def search_index(
    entries: List[PluginIndexEntry], term: str, *, capability: Optional[str] = None
) -> List[PluginIndexEntry]:
    """Rank *entries* against *term* (fuzzy on name/description/tags/author)."""
    pool = entries
    if capability:
        cap = capability.lower()
        pool = [e for e in entries if any(cap == c.lower() for c in e.capabilities)]
    term = (term or "").strip().lower()
    if not term:
        return sorted(pool, key=lambda e: e.name)
    matched = [(e, s) for e in pool if (s := _score_entry(e, term)) > 0]
    matched.sort(key=lambda pair: (-pair[1], pair[0].name))
    return [e for e, _s in matched]


def resolve_name(
    entries: List[PluginIndexEntry], name: str
) -> tuple[Optional[PluginIndexEntry], List[PluginIndexEntry]]:
    """Resolve a bare *name*: ``(entry, candidates)`` — a unique case-insensitive match in
    ``entry``, else ``None`` with the partial matches (empty = nothing similar, >1 = ambiguous)."""
    lowered = name.strip().lower()
    exact = [e for e in entries if e.name.lower() == lowered]
    matches = exact or [e for e in entries if lowered in e.name.lower()]
    return (matches[0] if len(matches) == 1 else None), matches
