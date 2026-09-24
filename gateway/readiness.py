"""Bounded, non-destructive readiness probes for authenticated health surfaces."""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import load_yaml_file_readonly


_DISK_DEGRADED_PERCENT = 90.0
_CONNECTED_STATES = {"connected", "running", "ok"}


def _check(status: str, detail: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"status": status, **({"detail": detail} if detail else {}), **extra}


def _probe_state_db(home: Path) -> dict[str, Any]:
    """Read-only schema probe plus the process-wide corruption latch (``hermes_state_health``).

    The schema read only catches an unreadable header or schema; damage deeper in the file
    surfaces when a reader or writer touches it, and those publish into the latch. Reporting
    the latch here is what makes readiness and ``/api/status`` agree with the session list
    (#72046). ``detail="corrupt"`` is the one reason string consumers key off."""
    from hermes_state_health import STORAGE_CORRUPT, note_storage_error, storage_state

    path = home / "state.db"
    try:
        from hermes_state_backend import resolve_database_settings
        settings = resolve_database_settings(path)
        if settings.backend == "postgres":
            from hermes_state import SessionDB
            with SessionDB(path, read_only=True, database_settings=settings) as db, db._read_ctx() as conn:
                conn.execute("SELECT 1 FROM sessions LIMIT 1").fetchone()
            return _check("ok")
        if not path.exists():
            return _check("ok", "not initialized")
        if storage_state(path) == STORAGE_CORRUPT:
            return _check("degraded", STORAGE_CORRUPT)
        # Read-only schema query: catches unreadable/corrupt DBs without competing with
        # writers. ``closing`` is required — sqlite3's context manager only commits/rolls
        # back, never closes, so a bare ``with connect()`` leaks a connection per poll.
        with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=1.0)) as conn:
            # A readiness probe must never compete with normal state writers. See #69567, #69678.
            conn.execute("PRAGMA query_only = ON")
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        return _check("ok")
    except Exception as exc:
        if note_storage_error(path, exc):
            return _check("degraded", STORAGE_CORRUPT)
        return _check("degraded", type(exc).__name__)


def _probe_config(home: Path) -> dict[str, Any]:
    path = home / "config.yaml"
    if not path.exists():
        return _check("ok", "using defaults")
    try:
        raw = load_yaml_file_readonly(path)
    except Exception as exc:
        return _check("degraded", f"invalid config ({type(exc).__name__})")
    return _check("ok") if raw is None or isinstance(raw, dict) else _check("degraded", "top level is not a mapping")


def _probe_disk(home: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(home)
    except Exception as exc:
        return _check("degraded", type(exc).__name__)
    used_pct = round((usage.used / usage.total) * 100, 1) if usage.total else 0.0
    return _check("degraded" if used_pct >= _DISK_DEGRADED_PERCENT else "ok", used_percent=used_pct, free_bytes=usage.free)


def _probe_gateway(runtime_status: dict[str, Any]) -> dict[str, Any]:
    state = str(runtime_status.get("gateway_state") or "unknown")
    platforms = runtime_status.get("platforms")
    platforms = platforms if isinstance(platforms, dict) else {}
    connected = sum(
        isinstance(v, dict) and str(v.get("state") or v.get("status") or "").lower() in _CONNECTED_STATES
        for v in platforms.values()
    )
    return _check("ok" if state in {"running", "draining"} else "degraded", state=state,
                  connected_platforms=connected, platforms=len(platforms))


def _probe_session_store(runtime_status: dict[str, Any], state_db_probe: dict[str, Any]) -> dict[str, Any]:
    """Report the running gateway cache state, not an independent reopen.  A corrupt store is
    unavailable whatever the cache says: an open handle on a damaged file is not a working one."""
    if state_db_probe.get("detail") == "corrupt":
        return _check("unavailable", "corrupt")
    runtime_store = runtime_status.get("session_store")
    state = str(runtime_store.get("status") or "unknown") if isinstance(runtime_store, dict) else ""
    if state in {"ok", "unavailable", "retrying"}:
        return _check(state)
    # Older gateways publish no cache state: fall back to the state_db probe.
    return _check("ok" if state_db_probe.get("status") == "ok" else "unavailable")


def collect_runtime_readiness(
    *, configured_model: str, runtime_status: dict[str, Any] | None, active_api_runs: int = 0,
    process_completion_queue_depth: int = 0, active_delegations: int = 0,
) -> dict[str, Any]:
    """Bounded readiness diagnostics, no runtime mutation.  Even authenticated, probes
    expose status and counts only: never config values, credentials, paths, payloads."""
    home = get_hermes_home()
    runtime = runtime_status if isinstance(runtime_status, dict) else {}
    state_db_probe = _probe_state_db(home)
    checks = {
        "state_db": state_db_probe,
        "session_store": _probe_session_store(runtime, state_db_probe),
        "config": _probe_config(home),
        "model": _check("ok" if str(configured_model or "").strip() else "degraded"),
        "disk": _probe_disk(home),
        "gateway": _probe_gateway(runtime),
        "background_queues": _check(
            "ok", active_api_runs=max(0, int(active_api_runs)),
            process_completions=max(0, int(process_completion_queue_depth)),
            active_delegations=max(0, int(active_delegations)),
        ),
    }
    return {"status": "ok" if all(c.get("status") == "ok" for c in checks.values()) else "degraded", "checks": checks}


__all__ = ["collect_runtime_readiness"]
