"""Community plugin index — fetch, cache, search, and name resolution.

Mirrors the Skills Hub catalog pattern (``tools/skills_hub.py``): a static
machine-readable JSON index hosted at a canonical URL, cached locally under
``HERMES_HOME/cache/`` with a TTL, with a bundled seed file as the offline
fallback and format reference.

Fallback chain: remote index → cached copy (fresh or stale) → bundled seed.

The index is discovery metadata ONLY.  **Indexed ≠ audited** — inclusion in
the index means the entry's metadata was reviewed, not that the plugin's code
was audited.  Install keeps its existing consent/review flow, and index
entries pin an immutable ref (tag or commit SHA).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Canonical index location. Override via config key ``plugins.index_url``.
# The index is maintained in this repository — ``hermes_cli/data/plugin_index.json``
# is the single source of truth: the raw GitHub URL below is the canonical fetch
# endpoint and the bundled copy is the offline fallback, so the two stay in sync
# by construction. Plugin authors submit entries via a PR to hermes-agent
# (see website/docs/user-guide/features/plugins.md).
DEFAULT_INDEX_URL = (
    "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/hermes_cli/data/plugin_index.json"
)

# Cache the fetched index for 24 hours; a stale cache is still preferred over
# the bundled seed when the remote is unreachable.
INDEX_CACHE_TTL = 24 * 3600

# Bundled seed — offline fallback and the machine-readable format reference.
SEED_INDEX_PATH = Path(__file__).parent / "data" / "plugin_index.json"

_FETCH_TIMEOUT = 10.0
_MAX_INDEX_BYTES = 5 * 1024 * 1024  # refuse absurdly large index payloads

SECURITY_FOOTER = (
    "Indexed \u2260 audited: inclusion in the index is a metadata review only, "
    "not a code audit. Review a plugin before enabling it."
)


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
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "tags": list(self.tags),
            "repo": self.repo,
            "ref": self.ref,
        }
        if self.subdir:
            d["subdir"] = self.subdir
        if self.homepage:
            d["homepage"] = self.homepage
        if self.capabilities:
            d["capabilities"] = list(self.capabilities)
        if self.api_version is not None:
            d["api_version"] = self.api_version
        if self.added_at:
            d["added_at"] = self.added_at
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
    """Parse a decoded index document into entries, skipping malformed items."""
    if isinstance(raw, dict):
        items = raw.get("plugins", [])
    elif isinstance(raw, list):  # bare-list form also accepted
        items = raw
    else:
        raise ValueError("Plugin index must be a JSON object or list.")
    if not isinstance(items, list):
        raise ValueError("Plugin index 'plugins' field must be a list.")

    entries: List[PluginIndexEntry] = []
    for item in items:
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
        entries.append(
            PluginIndexEntry(
                name=name.strip(),
                description=str(item.get("description") or ""),
                author=str(item.get("author") or ""),
                tags=[str(t) for t in item.get("tags") or [] if isinstance(t, (str, int))],
                repo=repo.strip(),
                ref=str(item.get("ref") or ""),
                subdir=str(subdir).strip("/") if isinstance(subdir, str) and subdir.strip("/") else None,
                homepage=str(item["homepage"]) if item.get("homepage") else None,
                capabilities=[str(c) for c in item.get("capabilities") or []],
                api_version=int(api_version) if isinstance(api_version, (int, str)) and str(api_version).isdigit() else None,
                added_at=str(item["added_at"]) if item.get("added_at") else None,
            )
        )
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
        if max_age is not None:
            age = time.time() - cache.stat().st_mtime
            if age > max_age:
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
        print(
            f"Warning: could not reach the plugin index at {url} "
            f"({exc.__class__.__name__}: {exc}); showing the cached or bundled index",
            file=sys.stderr,
        )
        return None


def load_index(*, refresh: bool = False, offline: bool = False) -> tuple[List[PluginIndexEntry], str]:
    """Load the plugin index.

    Returns ``(entries, source)`` where *source* is one of ``"remote"``,
    ``"cache"``, or ``"seed"``.

    Order: fresh cache (unless *refresh*) → remote → stale cache → bundled seed.
    ``offline=True`` skips the network entirely.
    """
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


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _score_entry(entry: PluginIndexEntry, term: str) -> float:
    """Fuzzy relevance score for *entry* against lowercase *term* (0 = no match)."""
    import difflib

    name = entry.name.lower()
    desc = entry.description.lower()
    tags = [t.lower() for t in entry.tags]

    if term == name:
        return 100.0
    score = 0.0
    if term in name:
        score = max(score, 80.0)
    if any(term == t for t in tags):
        score = max(score, 70.0)
    if any(term in t for t in tags):
        score = max(score, 55.0)
    if term in desc:
        score = max(score, 50.0)
    if term in entry.author.lower():
        score = max(score, 40.0)
    # Fuzzy close-match on the name for typo tolerance.
    ratio = difflib.SequenceMatcher(None, term, name).ratio()
    if ratio >= 0.6:
        score = max(score, ratio * 60.0)
    return score


def search_index(
    entries: List[PluginIndexEntry], term: str, *, capability: Optional[str] = None
) -> List[PluginIndexEntry]:
    """Rank *entries* against *term* (fuzzy on name/description/tags/author).

    An empty *term* matches everything (browse mode). ``capability`` filters
    entries by declared capability.
    """
    pool = entries
    if capability:
        cap = capability.lower()
        pool = [e for e in pool if any(cap == c.lower() for c in e.capabilities)]

    term = (term or "").strip().lower()
    if not term:
        return sorted(pool, key=lambda e: e.name)

    scored = [(e, _score_entry(e, term)) for e in pool]
    matched = [(e, s) for e, s in scored if s > 0]
    matched.sort(key=lambda pair: (-pair[1], pair[0].name))
    return [e for e, _s in matched]


def resolve_name(
    entries: List[PluginIndexEntry], name: str
) -> tuple[Optional[PluginIndexEntry], List[PluginIndexEntry]]:
    """Resolve a bare plugin *name* against the index.

    Returns ``(entry, candidates)``: an exact (case-insensitive) unique match
    in ``entry``, otherwise ``entry is None`` and ``candidates`` holds any
    partial matches (empty = nothing similar, >1 on exact = ambiguous).
    """
    lowered = name.strip().lower()
    exact = [e for e in entries if e.name.lower() == lowered]
    if len(exact) == 1:
        return exact[0], exact
    if len(exact) > 1:
        return None, exact
    partial = [e for e in entries if lowered in e.name.lower()]
    if len(partial) == 1:
        return partial[0], partial
    return None, partial
