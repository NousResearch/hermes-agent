"""Bounded, non-destructive readiness probes for authenticated health surfaces."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home


_DISK_DEGRADED_PERCENT = 90.0


def _check(status: str, detail: str | None = None, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if detail:
        result["detail"] = detail
    result.update(extra)
    return result


def _probe_state_db(home: Path) -> dict[str, Any]:
    path = home / "state.db"
    if not path.exists():
        return _check("ok", "not initialized")
    try:
        # mode=rw prevents a health probe from creating or repairing state.
        uri = f"file:{path.as_posix()}?mode=rw"
        with sqlite3.connect(uri, uri=True, timeout=1.0) as conn:
            row = conn.execute("PRAGMA quick_check(1)").fetchone()
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
        if not row or str(row[0]).lower() != "ok":
            return _check("degraded", "integrity check failed")
        return _check("ok")
    except Exception as exc:
        return _check("degraded", type(exc).__name__)


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
        # Recommendation 15 may provide a durable snapshot on a sibling branch.
        # Recognize it without depending on its implementation or exposing paths.
        lkg_candidates = (
            home / "config.last-known-good.yaml",
            home / ".config.last-known-good.yaml",
        )
        if any(candidate.is_file() for candidate in lkg_candidates):
            return _check("degraded", "invalid config; last-known-good available")
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
    model_name: str,
    runtime_status: dict[str, Any] | None,
    api_run_queue_depth: int = 0,
    process_completion_queue_depth: int = 0,
    delegation_completion_queue_depth: int = 0,
) -> dict[str, Any]:
    """Return bounded readiness diagnostics without mutating runtime state.

    The detailed health endpoint is authenticated. Even there, probes expose
    status and counts only: never config values, credentials, paths, commands,
    queue payloads, or exception messages.
    """
    home = get_hermes_home()
    runtime = runtime_status if isinstance(runtime_status, dict) else {}
    checks = {
        "state_db": _probe_state_db(home),
        "config": _probe_config(home),
        "provider": _check("ok" if str(model_name or "").strip() else "degraded"),
        "disk": _probe_disk(home),
        "gateway": _probe_gateway(runtime),
        "scheduler": _check(
            "ok" if runtime.get("updated_at") and runtime.get("gateway_state") == "running" else "degraded",
            "gateway ticker status unavailable" if not runtime.get("updated_at") else None,
        ),
        "background_queues": _check(
            "ok",
            api_runs=max(0, int(api_run_queue_depth)),
            process_completions=max(0, int(process_completion_queue_depth)),
            delegation_completions=max(0, int(delegation_completion_queue_depth)),
        ),
    }
    overall = "ok" if all(item.get("status") == "ok" for item in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


__all__ = ["collect_runtime_readiness"]
