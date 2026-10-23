"""``context_length_cache.yaml`` writes and their cross-process serialization.

Every agent (CLI, gateway sessions, cron jobs, subagents) persists probed context windows with a
load -> mutate -> write cycle; unserialized, one writer rewrites a stale snapshot and drops the
other's entry (or resurrects an invalidated, too-large length).
"""

import contextlib
import functools
import logging
import threading

from utils import atomic_yaml_write

logger = logging.getLogger(__name__)

_CONTEXT_CACHE_LOCK_HOLDER = threading.local()
_CONTEXT_CACHE_LOCK_TIMEOUT_SECONDS = 5.0


def _write_context_cache(cache: dict[str, int], bedrock_confirmed: dict | None = None) -> None:
    """Atomic write (a truncating write killed mid-dump leaves a partial file that
    _load_context_cache() swallows as {}, wiping EVERY cached length). Raises on failure."""
    from agent.model_metadata import _get_context_cache_path  # facade seam: tests patch the path there
    document = {"context_lengths": cache}
    if bedrock_confirmed:
        document["bedrock_confirmed_v1"] = bedrock_confirmed
    atomic_yaml_write(_get_context_cache_path(), document)


def locked_context_cache(fn):
    """Run ``fn``'s load -> mutate -> write cycle under the cache file's cross-process lock.

    The cache is best-effort: when the lock cannot be taken in time the update is skipped, never raised.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from agent.model_metadata import _get_context_cache_path  # facade seam: tests patch the path there
        from hermes_cli.auth import _file_lock
        path = _get_context_cache_path()
        lock_path = path.with_name(path.name + ".lock")
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(_file_lock(
                    lock_path, _CONTEXT_CACHE_LOCK_HOLDER, _CONTEXT_CACHE_LOCK_TIMEOUT_SECONDS,
                    f"Timed out waiting for the context length cache lock ({lock_path})"))
            except (OSError, TimeoutError):
                logger.debug("Skipping context length cache update", exc_info=True)
                return None
            return fn(*args, **kwargs)
    return wrapper


@locked_context_cache
def drop_context_cache_rows(stale_keys: set[str], label: str) -> None:
    """Remove every key shape of one cached entry (and its Bedrock provenance) under the cache lock.

    Only the file update is locked: callers evict their in-process probe caches unconditionally."""
    from agent.model_metadata import _load_context_cache_document  # facade seam: tests patch it there
    document = _load_context_cache_document()
    cache = document.get("context_lengths") or {}
    confirmed = document.get("bedrock_confirmed_v1") or {}
    confirmed = confirmed if isinstance(confirmed, dict) else {}
    if not any(k in cache or k in confirmed for k in stale_keys):
        return
    for k in stale_keys:
        cache.pop(k, None)
        confirmed.pop(k, None)
    try:
        _write_context_cache(cache, confirmed)
    except Exception:
        logger.debug("Failed to invalidate context length cache entry %s", label, exc_info=True)
