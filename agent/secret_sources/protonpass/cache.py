"""Two-layer cache for resolved Proton Pass secrets.

* In-process (:data:`_CACHE`): saves repeated fetches WITHIN one process
  (CLI startup, gateway hot-reload, test suites).
* Disk (``<hermes_home>/cache/protonpass_cache.json``, mode 0600 in a 0700
  dir): saves repeated fetches ACROSS processes (scripts, cron, the gateway
  forking new agents).

Both layers store ONLY the secret values plus a token FINGERPRINT (a SHA-256
prefix embedded in the cache key); the token itself is NEVER persisted.  A
``cache_ttl_seconds <= 0`` disables BOTH layers (read and write).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

from agent.secret_sources._cache import CachedFetch as _CachedFetch
from agent.secret_sources._cache import DiskCache

from .session import _token_fingerprint

# Cache key: (token_fingerprint, vault, refs_signature, home).  The token is
# represented only by its fingerprint, never stored.  ``home`` scopes the
# IN-PROCESS L1 cache so a single long-lived process serving multiple Hermes
# profiles can't return a stale L1 entry across profiles (the disk L2 is
# already scoped by living under each profile's home dir; only its serialized
# string is profile-agnostic, see :func:`_cache_key_str`).
_CacheKey = Tuple[str, str, str, str]
_CACHE: Dict[_CacheKey, _CachedFetch] = {}

_DISK_CACHE_BASENAME = "protonpass_cache.json"


def _refs_signature(env_refs: Dict[str, str]) -> str:
    """Stable, value-free signature of the MODE B ref map for the cache key.

    Hashes the (sorted) env-var-name → ``pass://`` URI pairs.  The URIs are
    references (share/item/field ids), not secret values, but we hash them
    anyway so the cache key stays compact and uniform.
    """
    if not env_refs:
        return ""
    items = "\n".join(f"{k}={env_refs[k]}" for k in sorted(env_refs))
    return hashlib.sha256(items.encode("utf-8")).hexdigest()[:16]


def build_cache_key(
    service_token: str,
    vault: str,
    env_refs: Dict[str, str],
    home_path: Optional[Path] = None,
) -> _CacheKey:
    """Build the cache key for a fetch.  The token is reduced to a fingerprint.

    ``home_path`` is folded into the (in-process) key so a long-lived process
    that fetches for more than one Hermes profile keeps their L1 entries
    distinct.  When omitted we resolve the same home the disk layer uses so a
    direct caller and a ``home_path``-passing caller land on the same key.
    """
    return (
        _token_fingerprint(service_token),
        vault or "",
        _refs_signature(env_refs),
        str(_resolve_home(home_path)),
    )


def _resolve_home(home_path: Optional[Path] = None) -> Path:
    """Resolve the Hermes home dir for cache scoping.

    ``home_path`` is what ``load_hermes_dotenv()`` already resolved.  When it is
    omitted, use the same fallback as the shared secret-source cache substrate.
    """
    if home_path is not None:
        return home_path
    from agent.secret_sources._cache import resolve_cache_home

    return resolve_cache_home()


def _disk_cache_path(home_path: Optional[Path] = None) -> Path:
    """Return the disk cache path under hermes_home/cache/."""
    return _resolve_home(home_path) / "cache" / _DISK_CACHE_BASENAME


def _cache_key_str(cache_key: _CacheKey) -> str:
    """Serialize a cache key to a stable string for JSON storage.

    The leading element is the token *fingerprint*, never the token, so it is
    safe to persist.  The trailing ``home`` element is DELIBERATELY omitted: the
    disk file already lives under that home dir, so folding the path into the
    persisted string would be redundant (and would also churn the on-disk format
    for the N1 in-process-only fix).
    """
    token_fp, vault, refs_sig, _home = cache_key
    return f"{token_fp}|{vault}|{refs_sig}"


_DISK_CACHE: DiskCache[_CacheKey] = DiskCache(
    _DISK_CACHE_BASENAME,
    key_serializer=_cache_key_str,
)


def _read_disk_cache(cache_key: _CacheKey, ttl_seconds: float,
                     home_path: Optional[Path] = None) -> Optional[_CachedFetch]:
    """Return a cached entry from disk if fresh and the key matches, else None.

    The persisted ``key`` embeds the token fingerprint, so a changed token
    produces a different key and the stale entry is ignored (cache invalidation
    on token change).  Best-effort: any I/O or parse error returns None and we
    re-fetch.
    """
    return _DISK_CACHE.read(cache_key, ttl_seconds, home_path)


def _write_disk_cache(cache_key: _CacheKey, entry: _CachedFetch, ttl_seconds: float,
                      home_path: Optional[Path] = None) -> None:
    """Persist a cache entry to disk atomically with mode 0600.

    Stores only the values and the cache key (which embeds the token
    *fingerprint*, never the token).  Best-effort: any I/O error is swallowed
    (the next invocation will just re-fetch). We never want disk cache failures
    to break startup.  ``ttl_seconds <= 0`` is a no-op, matching reads and the
    shared cache substrate.
    """
    _DISK_CACHE.write(cache_key, entry, ttl_seconds, home_path)


def _reset_cache_for_tests(home_path: Optional[Path] = None) -> None:
    """Clear in-process AND disk caches.

    Tests can pass ``home_path`` to scope the disk cleanup to a tmpdir.
    Without it we fall back to the same default resolution as the cache writer
    itself.
    """
    _CACHE.clear()
    _DISK_CACHE.clear(home_path)
