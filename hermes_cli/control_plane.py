"""Value-free local control-plane diagnostics for Hermes runtime recovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable


_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_SAFE_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,120}$")
_STEER_BOUNDARY = "cannot_steer_until_current_tool_boundary"
_MODEL_POLICY_RECOMMENDED_ACTION = "interrupt_and_restore_fixed_model"
_CLI_QUEUE_DELIVERY = "queued_until_next_boundary"


def _safe_token(value: Any, *, default: str = "unknown") -> str:
    text = str(value or "").strip()
    if text and _SAFE_TOKEN_RE.match(text):
        return text
    return default


def _fingerprint_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def _safe_model_id(value: Any) -> str:
    text = str(value or "").strip()
    if text and _SAFE_MODEL_ID_RE.match(text):
        return text
    return ""


def _read_model_policy_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return {}
    return config if isinstance(config, dict) else {}


def _model_policy_status_fields(
    config: dict[str, Any],
    *candidate_models: Any,
) -> dict[str, Any]:
    try:
        from hermes_cli.model_policy import (
            check_fixed_model_policy,
            fixed_model_from_config,
        )
    except Exception:
        return {}

    required = _safe_model_id(fixed_model_from_config(config))
    if not required:
        return {}
    for candidate in candidate_models:
        safe_candidate = _safe_model_id(candidate)
        if not safe_candidate:
            continue
        check = check_fixed_model_policy(config, safe_candidate, action="reported")
        if not check.allowed:
            return {
                "model_policy_violation": True,
                "required_model": required,
                "model_policy_recommended_action": _MODEL_POLICY_RECOMMENDED_ACTION,
            }
    return {}


def _read_session_db_models(session_ids: list[str]) -> dict[str, str]:
    safe_ids = []
    for session_id in session_ids:
        text = str(session_id or "").strip()
        if text:
            safe_ids.append(text)
    if not safe_ids:
        return {}
    try:
        from hermes_state import SessionDB

        db = SessionDB()
    except Exception:
        return {}
    models: dict[str, str] = {}
    try:
        for session_id in sorted(set(safe_ids)):
            try:
                row = db.get_session(session_id)
            except Exception:
                row = None
            if isinstance(row, dict):
                model = _safe_model_id(row.get("model"))
                if model:
                    models[session_id] = model
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    return models


def _read_session_db_lifecycle(session_ids: list[str]) -> dict[str, dict[str, Any]]:
    safe_ids = []
    for session_id in session_ids:
        text = str(session_id or "").strip()
        if text:
            safe_ids.append(text)
    if not safe_ids:
        return {}
    try:
        from hermes_state import SessionDB

        db = SessionDB()
    except Exception:
        return {}
    lifecycle: dict[str, dict[str, Any]] = {}
    try:
        for session_id in sorted(set(safe_ids)):
            try:
                row = db.get_session(session_id)
            except Exception:
                row = None
            if not isinstance(row, dict):
                continue
            ended_at = _coerce_optional_float(row.get("ended_at"))
            latest_message_at = None
            message_rows = 0
            conn = getattr(db, "_conn", None)
            lock = getattr(db, "_lock", None)
            if conn is not None:
                try:
                    if lock is not None:
                        with lock:
                            latest = conn.execute(
                                "SELECT MAX(timestamp) AS latest_message_at, COUNT(*) AS message_rows "
                                "FROM messages WHERE session_id = ?",
                                (session_id,),
                            ).fetchone()
                    else:
                        latest = conn.execute(
                            "SELECT MAX(timestamp) AS latest_message_at, COUNT(*) AS message_rows "
                            "FROM messages WHERE session_id = ?",
                            (session_id,),
                        ).fetchone()
                    if latest is not None:
                        latest_message_at = _coerce_optional_float(latest["latest_message_at"])
                        message_rows = _coerce_nonnegative_int(latest["message_rows"])
                except Exception:
                    latest_message_at = None
                    message_rows = 0
            lifecycle[session_id] = {
                "ended": ended_at is not None,
                "end_reason": _safe_token(row.get("end_reason"), default=""),
                "messages_after_end": bool(
                    ended_at is not None
                    and latest_message_at is not None
                    and latest_message_at > ended_at
                ),
                "message_rows": message_rows,
            }
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    return lifecycle


def _runtime_dir() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "runtime"


def _control_plane_steer_queue_path() -> Path:
    return _runtime_dir() / "control_plane_steer_queue.json"


def _control_plane_steer_queue_lock_path() -> Path:
    return _runtime_dir() / "control_plane_steer_queue.lock"


def _read_control_plane_steer_queue_unlocked() -> list[dict[str, Any]]:
    try:
        with open(_control_plane_steer_queue_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except Exception:
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    safe: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        session_id = str(entry.get("session_id") or "").strip()
        text = str(entry.get("text") or "").strip()
        if not session_id or not text:
            continue
        safe.append(
            {
                "id": _safe_token(entry.get("id"), default=uuid.uuid4().hex),
                "session_id": session_id,
                "text": text,
                "created_at": _coerce_optional_float(entry.get("created_at")) or 0.0,
            }
        )
    return safe


def _write_control_plane_steer_queue_unlocked(entries: list[dict[str, Any]]) -> None:
    path = _control_plane_steer_queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh, sort_keys=True)
    os.replace(tmp, path)


def _control_plane_steer_queue_lock(*, owner_kind: str, session_id: str | None = None):
    from hermes_cli.active_sessions import _FileLock

    owner_metadata = {"owner_kind": owner_kind}
    if session_id:
        owner_metadata["session_id"] = session_id
    return _FileLock(
        _control_plane_steer_queue_lock_path(),
        owner_metadata=owner_metadata,
        timeout_seconds=1.0,
    )


def _live_cli_session_owner(session_id: str) -> dict[str, Any] | None:
    target = str(session_id or "").strip()
    if not target:
        return None
    try:
        from hermes_cli import active_sessions

        try:
            report = active_sessions.active_session_registry_status(no_lock=True)
        except TypeError:
            report = active_sessions.active_session_registry_status()
    except Exception:
        return None
    entries = report.get("entries") if isinstance(report, dict) else []
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("session_id") or "") != target:
            continue
        if entry.get("runtime_status") != "live":
            continue
        owner_kind = _safe_token(entry.get("owner_kind"), default="")
        surface = str(entry.get("surface") or "").split(":", 1)[0].strip().lower()
        if owner_kind == "cli" or surface == "cli":
            return entry
    return None


def _append_control_plane_cli_steer(session_id: str, text: str) -> int:
    target = str(session_id or "").strip()
    if not target:
        return 0
    with _control_plane_steer_queue_lock(
        owner_kind="control_steer_queue",
        session_id=target,
    ):
        entries = _read_control_plane_steer_queue_unlocked()
        entries.append(
            {
                "id": uuid.uuid4().hex,
                "session_id": target,
                "text": text,
                "created_at": time.time(),
            }
        )
        _write_control_plane_steer_queue_unlocked(entries)
        return sum(1 for entry in entries if entry.get("session_id") == target)


def _control_plane_steer_queue_counts(session_ids: list[str]) -> dict[str, int]:
    wanted = {str(session_id or "").strip() for session_id in session_ids}
    wanted.discard("")
    if not wanted:
        return {}
    try:
        with _control_plane_steer_queue_lock(owner_kind="control_steer_status"):
            entries = _read_control_plane_steer_queue_unlocked()
    except Exception:
        return {}
    counts: dict[str, int] = {session_id: 0 for session_id in wanted}
    for entry in entries:
        session_id = str(entry.get("session_id") or "")
        if session_id in counts:
            counts[session_id] += 1
    return {key: value for key, value in counts.items() if value > 0}


def consume_control_plane_steers(session_id: str, *, limit: int = 20) -> list[str]:
    """Return and remove queued cross-process CLI steers for *session_id*."""
    target = str(session_id or "").strip()
    if not target:
        return []
    try:
        max_items = max(1, int(limit))
    except (TypeError, ValueError):
        max_items = 20
    with _control_plane_steer_queue_lock(
        owner_kind="control_steer_consume",
        session_id=target,
    ):
        entries = _read_control_plane_steer_queue_unlocked()
        consumed: list[str] = []
        kept: list[dict[str, Any]] = []
        for entry in entries:
            if entry.get("session_id") == target and len(consumed) < max_items:
                consumed.append(str(entry.get("text") or ""))
            else:
                kept.append(entry)
        if len(kept) != len(entries):
            _write_control_plane_steer_queue_unlocked(kept)
        return [text for text in consumed if text.strip()]


def _session_identity_fields(value: Any) -> dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {"session_id": "unknown"}
    if text.startswith("agent:"):
        return {"session_id_fingerprint": _fingerprint_text(text)}
    if _SAFE_TOKEN_RE.match(text):
        return {"session_id": text}
    return {"session_id_fingerprint": _fingerprint_text(text)}


def _with_session_identity(payload: dict[str, Any], session_id: Any) -> dict[str, Any]:
    out = dict(payload)
    out.update(_session_identity_fields(session_id))
    return out


def _coerce_nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _coerce_optional_int(value: Any) -> int | None:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _metadata(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = entry.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _queued_steer_count(metadata: dict[str, Any]) -> int:
    count = max(
        _coerce_nonnegative_int(metadata.get("pending_steer_count")),
        _coerce_nonnegative_int(metadata.get("queued_steer_count")),
        _coerce_nonnegative_int(metadata.get("model_request_queued_steer_count")),
    )
    if count:
        return count
    for key in (
        "pending_steer_queued",
        "queued_steer",
        "has_queued_steer",
        "model_request_steer_queued",
    ):
        if _coerce_bool(metadata.get(key)):
            return 1
    return 0


def _activity_is_fresh(metadata: dict[str, Any], *, now: float, fresh_seconds: float) -> bool:
    for key in ("last_activity_age_seconds", "seconds_since_activity"):
        try:
            age = float(metadata.get(key))
        except (TypeError, ValueError):
            continue
        if 0 <= age <= fresh_seconds:
            return True
    for key in ("last_activity_ts", "runtime_last_activity_ts", "model_request_last_event_at"):
        try:
            ts = float(metadata.get(key))
        except (TypeError, ValueError):
            continue
        if ts > 0 and now - ts <= fresh_seconds:
            return True
    return False


def _worker_progress_evidence(
    entry: dict[str, Any],
    *,
    now: float,
    fresh_seconds: float,
) -> list[str]:
    metadata = _metadata(entry)
    evidence: list[str] = []
    current_tool = _safe_token(metadata.get("current_tool"), default="")
    if current_tool:
        evidence.append("current_tool")
    model_status = _safe_token(metadata.get("model_request_status"), default="")
    if model_status:
        evidence.append("model_request")
    if _activity_is_fresh(metadata, now=now, fresh_seconds=fresh_seconds):
        evidence.append("fresh_activity")
    if _queued_steer_count(metadata):
        evidence.append("queued_steer")
    return evidence


def _session_summary(
    entry: dict[str, Any],
    *,
    now: float,
    fresh_seconds: float,
    model_policy_config: dict[str, Any] | None = None,
    session_db_model: str | None = None,
    session_db_lifecycle: dict[str, Any] | None = None,
    control_queue_count: int = 0,
) -> dict[str, Any]:
    metadata = _metadata(entry)
    queued = max(_queued_steer_count(metadata), _coerce_nonnegative_int(control_queue_count))
    summary: dict[str, Any] = {
        "runtime_status": _safe_token(entry.get("runtime_status")),
        "queued_steer_count": queued,
        "worker_progress_evidence": _worker_progress_evidence(
            entry,
            now=now,
            fresh_seconds=fresh_seconds,
        ),
    }
    summary.update(_session_identity_fields(entry.get("session_id")))
    if queued:
        summary["steer_boundary"] = _STEER_BOUNDARY
    model_status = _safe_token(metadata.get("model_request_status"), default="")
    if model_status:
        summary["model_request_status"] = model_status
    if _coerce_bool(metadata.get("model_request_high_context")):
        summary["model_request_high_context"] = True
    policy_fields = _model_policy_status_fields(
        model_policy_config or {},
        metadata.get("model_request_model"),
        entry.get("model_request_model"),
        entry.get("model"),
        session_db_model,
    )
    if policy_fields:
        summary.update(policy_fields)
    lifecycle = session_db_lifecycle if isinstance(session_db_lifecycle, dict) else {}
    if lifecycle.get("ended") is True:
        evidence = []
        if summary.get("runtime_status") == "live":
            evidence.append("active_runtime_lease")
        if lifecycle.get("messages_after_end") is True:
            evidence.append("messages_after_end")
        if evidence:
            summary["db_lifecycle_status"] = "ended_with_live_runtime_evidence"
            summary["db_lifecycle_evidence"] = evidence
            summary["repair_recommendation"] = "inspect_before_reopen_or_close"
        else:
            summary["db_lifecycle_status"] = "ended"
        end_reason = _safe_token(lifecycle.get("end_reason"), default="")
        if end_reason:
            summary["db_end_reason"] = end_reason
    return summary


def _read_registry_status() -> dict[str, Any]:
    try:
        from hermes_cli import active_sessions

        return active_sessions.active_session_registry_status()
    except Exception:
        return {"checked": 0, "live": 0, "stale": 0, "entries": []}


def _read_runtime_status() -> dict[str, Any]:
    try:
        from gateway.status import read_runtime_status

        return read_runtime_status() or {}
    except Exception:
        return {}


def _runtime_active_agents(runtime_status: dict[str, Any]) -> int:
    try:
        from gateway.status import parse_active_agents

        return int(parse_active_agents(runtime_status.get("active_agents", 0)))
    except Exception:
        return _coerce_nonnegative_int(runtime_status.get("active_agents"))


def _listener_alive_from_runtime(runtime_status: dict[str, Any]) -> bool | None:
    try:
        from gateway.status import get_runtime_status_running_pid

        return get_runtime_status_running_pid(runtime_status) is not None
    except Exception:
        return None


def _normalize_ws_probe(ws_probe: Any) -> dict[str, Any]:
    if ws_probe is None:
        return {"probe_status": "not_run", "probe_ok": None}
    if callable(ws_probe):
        try:
            ws_probe = ws_probe()
        except Exception as exc:
            ws_probe = {"ok": False, "error_type": type(exc).__name__}
    if not isinstance(ws_probe, dict):
        return {"probe_status": "unknown", "probe_ok": None}
    ok = ws_probe.get("ok")
    if ok is None:
        ok = ws_probe.get("probe_ok")
    probe_ok = bool(ok) if ok is not None else None
    if probe_ok is True:
        status = "ok"
    elif probe_ok is False:
        status = "failed"
    else:
        status = _safe_token(ws_probe.get("status") or ws_probe.get("probe_status"), default="unknown")
    out: dict[str, Any] = {
        "probe_status": status,
        "probe_ok": probe_ok,
    }
    error_type = _safe_token(ws_probe.get("error_type"), default="")
    if error_type:
        out["error_type"] = error_type
    return out


def _normalize_ws_health(ws_health: Any) -> dict[str, Any]:
    if callable(ws_health):
        try:
            ws_health = ws_health()
        except Exception as exc:
            return {"health_status": "error", "error_type": _safe_token(type(exc).__name__)}
    if not isinstance(ws_health, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "active_clients",
        "closed_clients",
        "stale_closed_clients",
        "send_failures",
        "close_events",
    ):
        out[key] = _coerce_nonnegative_int(ws_health.get(key))
    failure_type = _safe_token(ws_health.get("last_send_failure_type"), default="")
    if failure_type:
        out["last_send_failure_type"] = failure_type
    return out


def count_close_wait_sockets(pid: int | None = None) -> int | None:
    """Return CLOSE_WAIT sockets for *pid* or this process when available."""
    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid or os.getpid())
        reader = getattr(proc, "net_connections", None) or getattr(proc, "connections", None)
        if not callable(reader):
            return None
        count = 0
        for conn in reader(kind="tcp"):
            status = getattr(conn, "status", "")
            if status == getattr(psutil, "CONN_CLOSE_WAIT", "CLOSE_WAIT") or status == "CLOSE_WAIT":
                count += 1
        return count
    except Exception:
        return None


def probe_websocket_url(url: str, *, timeout: float = 1.5) -> dict[str, Any]:
    """Connect to a WS URL and return a value-free probe result."""
    target = str(url or "").strip()
    if not target:
        return {"ok": None, "status": "not_configured"}
    try:
        from websockets.sync.client import connect

        with connect(target, open_timeout=timeout, close_timeout=timeout):
            return {"ok": True, "status": "ok"}
    except Exception as exc:
        return {"ok": False, "status": "failed", "error_type": type(exc).__name__}


def build_control_plane_status(
    *,
    session_id: str | None = None,
    registry_status: dict[str, Any] | None = None,
    runtime_status: dict[str, Any] | None = None,
    listener_alive: bool | None = None,
    ws_probe: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    ws_health: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    close_wait_count: int | None = None,
    now: float | None = None,
    fresh_activity_seconds: float = 120.0,
) -> dict[str, Any]:
    """Build a value-free local control-plane health report."""
    import time

    checked_at = float(now if now is not None else time.time())
    registry = registry_status if registry_status is not None else _read_registry_status()
    runtime = runtime_status if runtime_status is not None else _read_runtime_status()
    listener = listener_alive
    if listener is None:
        listener = _listener_alive_from_runtime(runtime)

    entries = registry.get("entries") if isinstance(registry, dict) else []
    if not isinstance(entries, list):
        entries = []
    target_session_id = str(session_id or "").strip()
    if target_session_id:
        entries = [entry for entry in entries if str(entry.get("session_id") or "") == target_session_id]

    model_policy_config = _read_model_policy_config()
    session_db_models = _read_session_db_models(
        [str(entry.get("session_id") or "") for entry in entries if isinstance(entry, dict)]
    )
    session_db_lifecycle = _read_session_db_lifecycle(
        [str(entry.get("session_id") or "") for entry in entries if isinstance(entry, dict)]
    )
    control_queue_counts = _control_plane_steer_queue_counts(
        [str(entry.get("session_id") or "") for entry in entries if isinstance(entry, dict)]
    )
    sessions = [
        _session_summary(
            entry,
            now=checked_at,
            fresh_seconds=fresh_activity_seconds,
            model_policy_config=model_policy_config,
            session_db_model=session_db_models.get(str(entry.get("session_id") or "")),
            session_db_lifecycle=session_db_lifecycle.get(str(entry.get("session_id") or "")),
            control_queue_count=control_queue_counts.get(str(entry.get("session_id") or ""), 0),
        )
        for entry in entries
        if isinstance(entry, dict)
    ]
    active_agents = _runtime_active_agents(runtime)
    active_worker_progress = bool(active_agents > 0 or any(s["worker_progress_evidence"] for s in sessions))

    websocket = _normalize_ws_probe(ws_probe)
    websocket.update(_normalize_ws_health(ws_health))

    if close_wait_count is None:
        close_wait_count = count_close_wait_sockets()
    normalized_close_wait = _coerce_optional_int(close_wait_count)

    ws_failed = websocket.get("probe_ok") is False
    stale_clients = _coerce_nonnegative_int(websocket.get("stale_closed_clients"))
    degraded = bool((listener is True and ws_failed) or stale_clients > 0)
    status = "degraded" if degraded else "ok"
    if degraded and active_worker_progress:
        restart_guidance = "defer_restart_active_worker_progress"
    elif degraded:
        restart_guidance = "restart_only_after_stuck_evidence"
    else:
        restart_guidance = "no_restart_indicated"

    violation_sessions = [s for s in sessions if s.get("model_policy_violation") is True]
    report = {
        "status": status,
        "listener_alive": listener,
        "websocket": websocket,
        "close_wait_count": normalized_close_wait,
        "active_worker_progress": active_worker_progress,
        "active_agents": active_agents,
        "restart_guidance": restart_guidance,
        "model_policy_violation": bool(violation_sessions),
        "sessions": sessions,
    }
    if violation_sessions:
        report["required_model"] = violation_sessions[0].get("required_model", "")
        report["model_policy_recommended_action"] = _MODEL_POLICY_RECOMMENDED_ACTION
    return report


def queue_control_plane_steer(
    *,
    agent: Any,
    text: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Queue a steer on an in-process agent without echoing the steer text."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return _with_session_identity({
            "status": "rejected",
            "reason": "empty_text",
        }, session_id)
    if agent is None or not hasattr(agent, "steer"):
        if session_id and _live_cli_session_owner(str(session_id)) is not None:
            try:
                count = _append_control_plane_cli_steer(str(session_id), cleaned)
            except Exception as exc:
                return _with_session_identity({
                    "status": "failed",
                    "reason": _safe_token(type(exc).__name__),
                }, session_id)
            return _with_session_identity({
                "status": "queued",
                "queued_steer_count": max(1, count),
                "delivery": _CLI_QUEUE_DELIVERY,
                "steer_boundary": _STEER_BOUNDARY,
                "control_channel": "cli_queue",
            }, session_id)
        return _with_session_identity({
            "status": "unsupported",
            "reason": "no_live_agent_control_channel",
            "steer_boundary": _STEER_BOUNDARY,
        }, session_id)
    try:
        accepted = bool(agent.steer(cleaned))
    except Exception as exc:
        return _with_session_identity({
            "status": "failed",
            "reason": _safe_token(type(exc).__name__),
        }, session_id)
    if not accepted:
        return _with_session_identity({
            "status": "rejected",
            "reason": "agent_rejected",
        }, session_id)
    count = 1
    summary = getattr(agent, "get_activity_summary", None)
    if callable(summary):
        try:
            activity = summary()
            if isinstance(activity, dict):
                count = max(1, _coerce_nonnegative_int(activity.get("pending_steer_count")))
        except Exception:
            count = 1
    else:
        count = max(1, _coerce_nonnegative_int(getattr(agent, "_pending_steer_count", 0)))
    return _with_session_identity({
        "status": "queued",
        "queued_steer_count": count,
        "delivery": "after_next_tool_boundary",
        "steer_boundary": _STEER_BOUNDARY,
    }, session_id)
