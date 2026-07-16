"""Bounded, non-destructive readiness probes for authenticated health surfaces."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home


_DISK_DEGRADED_PERCENT = 90.0
_STATE_PROBE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="hermes-readiness-state",
)
_STATE_PROBE_LOCK = threading.Lock()
_STATE_PROBE_FUTURE: Future[dict[str, Any]] | None = None
_STATE_PROBE_HOME: Path | None = None
_STATE_PROBE_RESULT: dict[str, Any] | None = None


def _check(status: str, detail: str | None = None, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if detail:
        result["detail"] = detail
    result.update(extra)
    return result


def _probe_state_db(home: Path) -> dict[str, Any]:
    spec = None
    try:
        from hermes_cli.config import load_config_for_home
        from hermes_state import SessionDB
        from state_store import resolve_state_store

        config = load_config_for_home(home)
        spec = resolve_state_store(home, config, read_only=True)
        if spec.backend == "sqlite" and not spec.sqlite_path.exists():
            return _check("ok", "not initialized", backend="sqlite")

        if spec.backend == "sqlite":
            # Opening a SQLite database alone does not validate its contents.
            # Keep this query tiny so readiness never materializes session data.
            uri = f"file:{spec.sqlite_path.as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=1.0) as conn:
                conn.execute("PRAGMA query_only = ON")
                conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            return _check("ok", backend="sqlite")

        # The factory validates PostgreSQL availability and core-schema
        # capability during the read-only open without running broad queries.
        db = SessionDB.for_home(home, read_only=True, config=config)
        try:
            capabilities = {
                str(name): bool(enabled)
                for name, enabled in getattr(db, "capabilities", {}).items()
            }
            status = "ok" if capabilities.get("core_schema", False) else "degraded"
            return _check(
                status,
                backend="postgres",
                schema=spec.postgres_schema,
                capabilities=capabilities,
            )
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()
    except Exception as exc:
        extra = {"backend": spec.backend} if spec is not None else {}
        return _check("degraded", type(exc).__name__, **extra)


def _probe_state_db_without_blocking_event_loop(home: Path) -> dict[str, Any]:
    """Return a cached state probe while an aiohttp request loop is running."""
    global _STATE_PROBE_FUTURE, _STATE_PROBE_HOME, _STATE_PROBE_RESULT

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _probe_state_db(home)

    with _STATE_PROBE_LOCK:
        if _STATE_PROBE_FUTURE is not None and _STATE_PROBE_FUTURE.done():
            try:
                _STATE_PROBE_RESULT = _STATE_PROBE_FUTURE.result()
            except Exception as exc:
                _STATE_PROBE_RESULT = _check("degraded", type(exc).__name__)
            _STATE_PROBE_FUTURE = None

        if _STATE_PROBE_HOME != home:
            _STATE_PROBE_HOME = home
            _STATE_PROBE_RESULT = None
            _STATE_PROBE_FUTURE = None

        if _STATE_PROBE_FUTURE is None:
            _STATE_PROBE_FUTURE = _STATE_PROBE_EXECUTOR.submit(_probe_state_db, home)

        if _STATE_PROBE_RESULT is not None:
            return dict(_STATE_PROBE_RESULT)
    return _check("degraded", "probe pending")


def _probe_config(home: Path) -> dict[str, Any]:
    path = home / "config.yaml"
    if not path.exists():
        return _check("ok", "using defaults")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is not None and not isinstance(raw, dict):
            return _check("degraded", "top level is not a mapping")
        return _check("ok")
    except Exception as exc:
        return _check("degraded", f"invalid config ({type(exc).__name__})")


def _probe_disk(home: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(home)
        used_pct = round((usage.used / usage.total) * 100, 1) if usage.total else 0.0
        status = "degraded" if used_pct >= _DISK_DEGRADED_PERCENT else "ok"
        return _check(status, used_percent=used_pct, free_bytes=usage.free)
    except Exception as exc:
        return _check("degraded", type(exc).__name__)


def _probe_gateway(runtime_status: dict[str, Any]) -> dict[str, Any]:
    state = str(runtime_status.get("gateway_state") or "unknown")
    platforms = runtime_status.get("platforms")
    connected = 0
    configured = 0
    if isinstance(platforms, dict):
        configured = len(platforms)
        connected = sum(
            1
            for value in platforms.values()
            if isinstance(value, dict)
            and str(value.get("state") or value.get("status") or "").lower()
            in {"connected", "running", "ok"}
        )
    status = "ok" if state in {"running", "draining"} else "degraded"
    return _check(status, state=state, connected_platforms=connected, platforms=configured)


def collect_runtime_readiness(
    *,
    configured_model: str,
    runtime_status: dict[str, Any] | None,
    active_api_runs: int = 0,
    process_completion_queue_depth: int = 0,
    active_delegations: int = 0,
) -> dict[str, Any]:
    """Return bounded readiness diagnostics without mutating runtime state.

    The detailed health endpoint is authenticated. Even there, probes expose
    status and counts only: never config values, credentials, paths, commands,
    queue payloads, or exception messages.
    """
    home = get_hermes_home()
    runtime = runtime_status if isinstance(runtime_status, dict) else {}
    checks = {
        "state_db": _probe_state_db_without_blocking_event_loop(home),
        "config": _probe_config(home),
        "model": _check("ok" if str(configured_model or "").strip() else "degraded"),
        "disk": _probe_disk(home),
        "gateway": _probe_gateway(runtime),
        "background_queues": _check(
            "ok",
            active_api_runs=max(0, int(active_api_runs)),
            process_completions=max(0, int(process_completion_queue_depth)),
            active_delegations=max(0, int(active_delegations)),
        ),
    }
    overall = "ok" if all(item.get("status") == "ok" for item in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


__all__ = ["collect_runtime_readiness"]
