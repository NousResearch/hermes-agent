"""Bounded, non-destructive readiness probes for authenticated health surfaces."""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
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


def _unrepaired_corruption_marker(path: Path) -> bool:
    """True when the repair-attempt ledger matches the current file bytes.

    ``hermes_state`` writes ``state.db.repair-attempts.json`` beside the
    database when automatic schema surgery fails, keyed by a size+mtime_ns
    fingerprint of the exact bytes attempted, and deletes it on success
    (#86747). A ledger whose fingerprint still matches therefore means: this
    database is corrupt, automatic repair already failed on these bytes, and
    nothing has changed since — the strongest cheap corruption signal
    available to a bounded probe, and it works across processes (the repair
    may have been attempted by the CLI or a cron worker, not this gateway).

    Import-light on purpose: reads the sidecar JSON directly instead of
    importing ``hermes_state``. Any failed attempt on the current
    fingerprint is enough to report degraded — waiting for the attempt
    budget to exhaust would keep the probe green while repair retries churn.

    The ``size:mtime_ns`` fingerprint can false-match on filesystems with
    coarse mtime granularity if a repair rewrites the file to the same size
    within one timestamp tick — the probe then stays degraded until the next
    successful write bumps the mtime. Acceptable: the failure mode is a
    briefly pessimistic health signal, never a false green.
    """
    ledger_path = path.with_name(path.name + ".repair-attempts.json")
    try:
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        st = path.stat()
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    fingerprint = data.get("fingerprint")
    try:
        attempts = int(data.get("failed_attempts", 0))
    except (TypeError, ValueError):
        return False
    return attempts >= 1 and fingerprint == f"{st.st_size}:{st.st_mtime_ns}"


def _probe_state_db(home: Path) -> dict[str, Any]:
    path = home / "state.db"
    if not path.exists():
        return _check("ok", "not initialized")
    if _unrepaired_corruption_marker(path):
        # Report the corruption class without paths or messages (this feeds
        # public component rollups). "degraded" — not an error state that
        # would trip restart loops — but no longer a false green (OOF-106:
        # a page-corrupt state.db kept /api/status "ok" for 10+ days while
        # sessions silently failed to persist).
        return _check("degraded", "unrepaired corruption")
    try:
        # A readiness probe must never compete with normal state writers. A
        # read-only schema query still catches unreadable/corrupt databases
        # without taking a write reservation on every health poll.
        # ``closing(...)`` is required: sqlite3's connection context manager
        # only commits/rolls back — it never closes, so a bare ``with
        # sqlite3.connect(...)`` leaks one connection (and its fds) per
        # health poll in the long-running gateway (#69678/#69567 bug class).
        uri = f"file:{path.as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=1.0)) as conn:
            conn.execute("PRAGMA query_only = ON")
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
            # The schema read only touches page 1; page-level damage in the
            # canonical table b-trees sails past it (the OOF-106 false-green
            # gap). Walking one row of ``sessions`` descends its b-tree root
            # — still O(1) pages, still read-only, but it catches root-page
            # corruption of the table every session write depends on.
            # ``SELECT *`` on purpose: a narrower projection (e.g. ``id``)
            # can be satisfied from an index b-tree without ever touching
            # the table's pages. The row is fetched and discarded — probes
            # expose status only, never data. Guarded for pre-schema
            # databases where the table doesn't exist yet.
            has_sessions = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'"
            ).fetchone()
            if has_sessions:
                conn.execute("SELECT * FROM sessions LIMIT 1").fetchone()
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
        "state_db": _probe_state_db(home),
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
