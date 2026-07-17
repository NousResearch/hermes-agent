"""Deduplicated, cached execution of expensive read-only health checks.

Gate-5 root cause: the same expensive auxiliary checks
(``control_plane_drift`` subprocess, ``admission_policy_mirror`` sha256,
``provenance_memory`` ``PRAGMA integrity_check`` over ~60k rows, ``nova_ssh``,
route-state) were re-run on *every* invocation of ``hermes-health-check.py``
(watchdog runs ``--local`` every 120s; the acceptance observer pings
``--gateway-only`` every 55s while vault-sync does heavy git I/O every 900s).
Each run costs 2-3s of CPU/disk and, because the gateway is a single asyncio
process on the same machine, starved the ``/health`` handler past the 2-second
bound.

This module fixes the *contention*, not the checks:
* checks are NEVER disabled or weakened;
* the 2-second session threshold is untouched;
* the expensive compute is moved OFF the request-critical path: a low-priority
  ``--refresh`` background job computes and caches the results; interactive and
  observer runs read the cache (``read_cached``) and never stall;
* each result is computed at most once per ``TTL_SECONDS`` window across all
  processes; concurrent callers share the cached result via an ``fcntl`` flock.

The cache is a single small JSON file guarded by an ``fcntl`` flock. A stale or
corrupt cache is ignored (recompute). Failures fall back to a live compute so
the check is never silently dropped.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

__all__ = ["TTL_SECONDS", "cached_check", "compute_with_cache"]

# One recompute per 60s is more than enough for health cadences (watchdog 120s,
# observer 55s) while still catching real state changes quickly.
TTL_SECONDS = 60

_DEFAULT_CACHE_DIR = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "state"


def _cache_path(name: str, cache_dir: Optional[str] = None) -> Path:
    root = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / f"health-check-{name}.cache.json"


def _lock_path(name: str, cache_dir: Optional[str] = None) -> Path:
    return _cache_path(name, cache_dir).with_suffix(".lock")


def compute_with_cache(
    name: str,
    compute: Callable[[], dict],
    *,
    ttl: int = TTL_SECONDS,
    cache_dir: Optional[str] = None,
) -> Tuple[dict, bool]:
    """Return ``(result, served_from_cache)``.

    ``compute`` is a zero-arg callable returning a JSON-serializable dict (the
    check result). It is invoked at most once per ``ttl`` seconds across all
    processes; concurrent callers within the window get the cached value.
    """
    cp = _cache_path(name, cache_dir)
    lk = _lock_path(name, cache_dir)
    now = time.time()

    # Fast path: fresh cache, no lock needed.
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("_ts", 0) + ttl >= now:
            return data.get("result", data), True
    except (OSError, ValueError, AttributeError):
        pass

    # Slow path: take the lock, re-check (another process may have computed),
    # else compute and write.
    try:
        lk.parent.mkdir(parents=True, exist_ok=True)
        with open(lk, "w") as lfh:
            fcntl.flock(lfh.fileno(), fcntl.LOCK_EX)
            try:
                data = json.loads(cp.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("_ts", 0) + ttl >= now:
                    return data.get("result", data), True
            except (OSError, ValueError, AttributeError):
                pass
            result = compute()
            try:
                cp.write_text(
                    json.dumps({"_ts": time.time(), "result": result}),
                    encoding="utf-8",
                )
            except OSError:
                pass
            return result, False
    except OSError:
        # Lock unavailable (e.g. platform without fcntl on the path): compute
        # live so the check is never dropped.
        return compute(), False


def read_cached(name: str, ttl: int = TTL_SECONDS, cache_dir: Optional[str] = None) -> Optional[dict]:
    """Return the cached result if fresh, else None (do NOT compute).

    Used by request-critical health checks so they never pay the cost of the
    expensive underlying check — a background refresher populates the cache.
    """
    cp = _cache_path(name, cache_dir)
    now = time.time()
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("_ts", 0) + ttl >= now:
            return data.get("result")
    except (OSError, ValueError, AttributeError):
        pass
    return None


def cached_check(
    name: str,
    compute: Callable[[], dict],
    *,
    ttl: int = TTL_SECONDS,
    cache_dir: Optional[str] = None,
) -> dict:
    """Convenience wrapper returning just the result dict."""
    result, _ = compute_with_cache(name, compute, ttl=ttl, cache_dir=cache_dir)
    return result


if __name__ == "__main__":
    import tempfile

    calls = {"n": 0}

    def work() -> dict:
        calls["n"] += 1
        return {"pass": True, "compute_count": calls["n"]}

    with tempfile.TemporaryDirectory() as td:
        r1, c1 = compute_with_cache("demo", work, ttl=60, cache_dir=td)
        r2, c2 = compute_with_cache("demo", work, ttl=60, cache_dir=td)
        assert c1 is False, "first call should compute"
        assert c2 is True, "second call should hit cache"
        assert r1 == r2, "cached result must match"
        assert calls["n"] == 1, f"compute ran {calls['n']} times, expected 1"
        print("OK: dedupe works, compute_count =", calls["n"])
