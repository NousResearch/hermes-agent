"""Cross-tool result cache with TTL and budget caps.

Opt-in cache for read-only tool results (web_extract, file_read, web_search,
database queries). Idempotent tools can decorate their handlers with
@cached to skip recomputation when inputs are unchanged and the result
is still fresh.

Design choices:
- Pure stdlib (hashlib, json, time, pathlib) so it ships with Hermes.
- Content-addressed key: sha256(canonical_json(args)).
- TTL is per-tool-set; default 5 minutes; overridden by tool metadata.
- Cache entries are pruned on every write to keep disk bounded.
- Failures are NEVER cached. Only successful returns with serializable
  payloads are eligible.

Usage::

    from agent.tool_cache import cached, cache_stats, clear_cache

    @cached(ttl_seconds=60)
    def my_expensive_query(url: str) -> dict:
        ...

    # Outside-inspect:
    cache_stats()        # {"entries": 12, "size_bytes": 8421, "hits": 4, "misses": 18}
    clear_cache("tool:my_expensive_query")  # scoped
    clear_cache()                            # whole cache
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional


_DEFAULT_TTL = 300  # 5 minutes
_MAX_ENTRIES = 1024
_MAX_BYTES = 64 * 1024 * 1024  # 64 MiB


def _cache_dir() -> Path:
    """Return the cache root directory, creating it if missing.

    Honours ``HERMES_TOOL_CACHE_DIR`` for tests and admin overrides.
    """
    override = os.environ.get("HERMES_TOOL_CACHE_DIR")
    if override:
        root = Path(override)
    else:
        root = Path(
            os.environ.get("LOCALAPPDATA")
            or (Path.home() / ".local" / "share")
        ) / "hermes" / "tool_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _canonical_key(args: tuple, kwargs: dict) -> str:
    """Build a deterministic cache key from positional and keyword args."""
    payload = {"args": args, "kwargs": kwargs}
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _entry_path(tool_name: str, key: str) -> Path:
    # Shard by first 2 chars to keep directories small.
    return _cache_dir() / tool_name / key[:2] / f"{key}.json"


def _read_entry(path: Path) -> Optional[Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return None
    expires_at = envelope.get("expires_at", 0)
    if expires_at and expires_at < time.time():
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return envelope.get("value")


def _write_entry(path: Path, value: Any, ttl_seconds: int = _DEFAULT_TTL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "value": value,
        "expires_at": int(time.time()) + ttl_seconds,
        "stored_at": int(time.time()),
    }
    # Atomic write: temp + rename, so concurrent readers never see half-written JSON.
    fd, tmp_path = tempfile.mkstemp(prefix=".cache-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(envelope, f, default=str, separators=(",", ":"))
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _enforce_budget() -> None:
    """Walk the cache and prune oldest entries until under budget."""
    root = _cache_dir()
    if not root.exists():
        return
    entries: list[tuple[float, Path, int]] = []
    total_bytes = 0
    for path in root.rglob("*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, path, stat.st_size))
        total_bytes += stat.st_size
    if len(entries) <= _MAX_ENTRIES and total_bytes <= _MAX_BYTES:
        return
    # Oldest first.
    entries.sort(key=lambda e: e[0])
    while (len(entries) > _MAX_ENTRIES or total_bytes > _MAX_BYTES) and entries:
        _, path, size = entries.pop(0)
        try:
            path.unlink()
            total_bytes -= size
        except OSError:
            continue


# In-memory hit/miss counters for observability. These reset on process start;
# the disk cache persists across restarts.
_stats: dict[str, int] = OrderedDict([("hits", 0), ("misses", 0), ("stores", 0)])


def reset_stats() -> None:
    """Reset hit/miss/store counters. Used by tests; not for production code."""
    _stats.clear()
    _stats.update({"hits": 0, "misses": 0, "stores": 0})


def cache_stats() -> dict[str, Any]:
    """Return current cache stats. Cheap; safe to call from any thread."""
    root = _cache_dir()
    count = 0
    total_bytes = 0
    if root.exists():
        for p in root.rglob("*.json"):
            try:
                count += 1
                total_bytes += p.stat().st_size
            except OSError:
                continue
    return {
        "entries": count,
        "size_bytes": total_bytes,
        "max_entries": _MAX_ENTRIES,
        "max_bytes": _MAX_BYTES,
        **{k: _stats.get(k, 0) for k in ("hits", "misses", "stores")},
    }


def clear_cache(tool_name: Optional[str] = None) -> int:
    """Delete cache entries. Returns the number of files removed.

    ``tool_name=None`` clears the whole cache.
    """
    root = _cache_dir()
    if not root.exists():
        return 0
    target = root if tool_name is None else root / tool_name
    if not target.exists():
        return 0
    removed = 0
    # Always use rglob: the layout is <tool>/<key[:2]>/<key>.json (3 levels).
    for p in target.rglob("*.json"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            continue
    # Prune empty parent dirs so the tree stays tidy.
    for d in sorted(target.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass
    return removed


def cached(
    *,
    ttl_seconds: int = _DEFAULT_TTL,
    key_from: Optional[Callable[..., tuple]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator. Caches successful returns keyed by deterministic args hash.

    ``key_from`` lets the caller build a custom cache key (e.g. to drop a
    timestamp arg). Default is the canonical JSON of all positional+keyword args.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = f"tool:{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                if key_from is not None:
                    key_args = key_from(*args, **kwargs)
                else:
                    key_args = (args, kwargs)
                key = _canonical_key(key_args, {})
                path = _entry_path(tool_name, key)
                hit = _read_entry(path)
                if hit is not None:
                    _stats["hits"] = _stats.get("hits", 0) + 1
                    return hit
                _stats["misses"] = _stats.get("misses", 0) + 1
                value = func(*args, **kwargs)
            except Exception:
                # Never cache the wrapper machinery; let exceptions propagate.
                raise
            # value is now defined — check serialisability before storing.
            # We don't use default=str because it would mask unserialisable types
            # (sets, custom objects) into strings; the wrapper must skip those.
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                return value
            try:
                _write_entry(path, value, ttl_seconds=ttl_seconds)
                _stats["stores"] = _stats.get("stores", 0) + 1
                _enforce_budget()
            except (OSError, TypeError, ValueError, RuntimeError):
                # Disk-full, permission, or serialisation edge cases: degrade silently.
                pass
            return value


        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        wrapper.tool_cache_name = tool_name  # type: ignore[attr-defined]
        return wrapper

    return decorator
