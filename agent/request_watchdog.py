"""Value-free watchdog state for long-running model requests."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

RECOVERY_RECOMMENDED_ACTION = "bounded_recovery_prompt"
COMPACT_FINALIZATION_RECOMMENDED_ACTION = "compact_finalization_prompt"


@dataclass
class RequestWatchdogThresholds:
    normal_alert_seconds: float = 180.0
    high_context_alert_seconds: float = 60.0
    terminal_recovery_seconds: float = 900.0
    poll_interval_seconds: float = 5.0
    high_context_tokens: int = 200_000


@dataclass
class RequestWatchdogRecord:
    session_id: str
    request_id: str
    request_id_fingerprint: str
    model: str
    provider: str
    started_at: float
    last_event_at: Optional[float]
    last_byte_at: Optional[float]
    estimated_context_tokens: int
    api_call_count: int
    queued_steer_count: int
    thresholds: RequestWatchdogThresholds
    queued_steer_count_fn: Optional[Callable[[], int]] = None
    done: threading.Event = field(default_factory=threading.Event)
    recovery_path: Optional[Path] = None
    _monitor_thread: Optional[threading.Thread] = None


def _safe_label(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 15)].rstrip() + " ...[truncated]"


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", "replace")).hexdigest()[:16]


def _queued_steer_count(agent: Any) -> int:
    fn = getattr(agent, "_pending_steer_status_count", None)
    if callable(fn):
        try:
            return max(0, int(fn()))
        except Exception:
            return 0
    try:
        return max(0, int(getattr(agent, "_pending_steer_count", 0) or 0))
    except Exception:
        return 0


def _high_context(record: RequestWatchdogRecord, thresholds: RequestWatchdogThresholds) -> bool:
    return int(record.estimated_context_tokens or 0) >= int(thresholds.high_context_tokens)


def _last_progress_at(record: RequestWatchdogRecord) -> float:
    return max(
        ts
        for ts in (
            record.started_at,
            record.last_event_at or 0.0,
            record.last_byte_at or 0.0,
        )
        if ts is not None
    )


def _status_message(record: RequestWatchdogRecord, status: str) -> str:
    if status == "terminal_recovery_needed":
        base = "active model request appears stalled; compact recovery state written"
    else:
        base = "active model request waiting"
    if record.queued_steer_count > 0:
        return base + "; steer will queue until tool boundary."
    return base + "."


def evaluate_request_watchdog(
    record: RequestWatchdogRecord,
    *,
    now: Optional[float] = None,
    thresholds: Optional[RequestWatchdogThresholds] = None,
) -> dict[str, Any]:
    thresholds = thresholds or record.thresholds
    now = time.time() if now is None else float(now)
    if record.queued_steer_count_fn is not None:
        try:
            record.queued_steer_count = max(0, int(record.queued_steer_count_fn()))
        except Exception:
            pass
    seconds_since_event = max(0.0, round(now - _last_progress_at(record), 1))
    high_context = _high_context(record, thresholds)
    alert_threshold = (
        thresholds.high_context_alert_seconds
        if high_context
        else thresholds.normal_alert_seconds
    )
    if seconds_since_event >= thresholds.terminal_recovery_seconds:
        status = "terminal_recovery_needed"
    elif seconds_since_event >= alert_threshold:
        status = "waiting"
    else:
        status = "healthy"
    return {
        "status": status,
        "message": _status_message(record, status),
        "high_context": high_context,
        "seconds_since_event": seconds_since_event,
        "alert_threshold_seconds": alert_threshold,
        "terminal_recovery_seconds": thresholds.terminal_recovery_seconds,
        "queued_steer_count": record.queued_steer_count,
        "request_id_fingerprint": record.request_id_fingerprint,
    }


def _active_session_metadata(record: RequestWatchdogRecord, status: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_request_status": status.get("status"),
        "model_request_status_message": status.get("message"),
        "model_request_id_fingerprint": record.request_id_fingerprint,
        "model_request_model": record.model,
        "model_request_provider": record.provider,
        "model_request_started_at": record.started_at,
        "model_request_last_event_at": record.last_event_at,
        "model_request_last_byte_at": record.last_byte_at,
        "model_request_estimated_context_tokens": record.estimated_context_tokens,
        "model_request_api_call_count": record.api_call_count,
        "model_request_seconds_since_event": status.get("seconds_since_event"),
        "model_request_queued_steer_count": record.queued_steer_count,
        "model_request_high_context": bool(status.get("high_context")),
        "model_request_steer_queued": record.queued_steer_count > 0,
    }


def _write_active_session_status(record: RequestWatchdogRecord, status: dict[str, Any]) -> None:
    if not record.session_id:
        return
    try:
        from hermes_cli.active_sessions import update_active_session_metadata

        update_active_session_metadata(
            session_id=record.session_id,
            metadata=_active_session_metadata(record, status),
        )
    except Exception:
        pass


def _clear_active_session_status(record: RequestWatchdogRecord) -> None:
    if not record.session_id:
        return
    try:
        from hermes_cli.active_sessions import update_active_session_metadata

        update_active_session_metadata(
            session_id=record.session_id,
            metadata={
                "model_request_status": None,
                "model_request_status_message": None,
                "model_request_id_fingerprint": None,
                "model_request_model": None,
                "model_request_provider": None,
                "model_request_started_at": None,
                "model_request_last_event_at": None,
                "model_request_last_byte_at": None,
                "model_request_estimated_context_tokens": None,
                "model_request_api_call_count": None,
                "model_request_seconds_since_event": None,
                "model_request_queued_steer_count": None,
                "model_request_high_context": None,
                "model_request_steer_queued": None,
            },
        )
    except Exception:
        pass


def poll_request_watchdog(
    record: RequestWatchdogRecord,
    *,
    now: Optional[float] = None,
    thresholds: Optional[RequestWatchdogThresholds] = None,
    write_status: bool = False,
) -> dict[str, Any]:
    status = evaluate_request_watchdog(record, now=now, thresholds=thresholds)
    if write_status:
        if status["status"] != "healthy":
            _write_active_session_status(record, status)
        elif record.last_event_at is not None or record.last_byte_at is not None:
            _clear_active_session_status(record)
    return status


def mark_request_watchdog_event(
    record: Optional[RequestWatchdogRecord],
    *,
    now: Optional[float] = None,
    byte_count: int = 0,
) -> None:
    if record is None or record.done.is_set():
        return
    ts = time.time() if now is None else float(now)
    record.last_event_at = ts
    if byte_count and byte_count > 0:
        record.last_byte_at = ts


def _default_recovery_directory() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "request_watchdog"
    except Exception:
        return Path.cwd() / "request_watchdog"


def _model_policy_recovery_line(packet: dict[str, Any]) -> str:
    model = _safe_label(packet.get("required_model") or packet.get("model") or "gpt-5.5")
    fixed = bool(packet.get("fixed_model_policy"))
    if fixed:
        return (
            f"Fixed model policy requires {model}; model switch requires explicit "
            "user approval."
        )
    return f"Keep {model}."


def build_bounded_recovery_prompt(packet: dict[str, Any]) -> str:
    session_id = _safe_label(packet.get("session_id"), limit=80)
    lines = [
        f"Resume session {session_id or 'unknown'} from current DB/log/repo state.",
        "First inspect current DB/log/repo state first.",
        "Do not replay broad history.",
        "If the task contract is known, use terminal-only write/test mode.",
        "If closeout or verification is the only remaining work, use compact final/blocked answer mode.",
        _model_policy_recovery_line(packet),
        (
            "queued steer cannot land until the active request exits or reaches "
            "a tool boundary; account for that before waiting or resubmitting."
        ),
        "Do not retry the same huge request unchanged after stale-call kills; emit/use a compact recovery or finalization packet first.",
        "Do not close unrelated sessions; act only on the exact session with current evidence.",
    ]
    if packet.get("closeout_only"):
        lines.append(
            "Closeout-only evidence is present; prefer compact finalization over another broad model request."
        )
    return "\n".join(lines)


def latest_recoverable_turn_state(
    *,
    session_id: str,
    directory: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    directory = Path(directory) if directory is not None else _default_recovery_directory()
    if not directory.exists():
        return None
    latest: Optional[tuple[float, dict[str, Any]]] = None
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("kind") != "high_context_request_watchdog_recovery":
            continue
        if str(payload.get("session_id") or "") != str(session_id or ""):
            continue
        payload["_path"] = str(path)
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        if latest is None or mtime > latest[0]:
            latest = (mtime, payload)
    return latest[1] if latest is not None else None


def write_recoverable_turn_state(
    record: RequestWatchdogRecord,
    *,
    status: dict[str, Any],
    directory: Optional[Path] = None,
) -> dict[str, Any]:
    directory = Path(directory) if directory is not None else _default_recovery_directory()
    directory.mkdir(parents=True, exist_ok=True)
    closeout_only = bool(
        status.get("closeout_only")
        or status.get("closeout_only_candidate")
        or status.get("compact_finalization_required")
    )
    repeated_stale_call_count = int(
        status.get("repeated_stale_call_count")
        or status.get("stale_call_kill_count")
        or status.get("same_request_stale_kill_count")
        or 0
    )
    recommended_action = (
        COMPACT_FINALIZATION_RECOMMENDED_ACTION
        if closeout_only
        else RECOVERY_RECOMMENDED_ACTION
    )
    prompt_packet = {
        "session_id": record.session_id,
        "required_model": record.model or "gpt-5.5",
        "model": record.model or "gpt-5.5",
        "fixed_model_policy": bool(status.get("fixed_model_policy")),
        "closeout_only": closeout_only,
    }
    payload = {
        "kind": "high_context_request_watchdog_recovery",
        "session_id": record.session_id,
        "request_id_fingerprint": record.request_id_fingerprint,
        "model": record.model,
        "provider": record.provider,
        "started_at": record.started_at,
        "last_event_at": record.last_event_at,
        "last_byte_at": record.last_byte_at,
        "estimated_context_tokens": record.estimated_context_tokens,
        "api_call_count": record.api_call_count,
        "queued_steer_count": record.queued_steer_count,
        "status": status,
        "repeated_stale_call_count": repeated_stale_call_count,
        "closeout_only": closeout_only,
        "recommended_action": recommended_action,
        "resume_prompt": build_bounded_recovery_prompt(prompt_packet),
    }
    path = directory / (
        f"{int(time.time() * 1000)}-{record.session_id or 'unknown'}-"
        f"{record.request_id_fingerprint}.json"
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    record.recovery_path = path
    return {
        "path": path,
        "session_id": record.session_id,
        "mass_close_sessions": False,
        "end_session_reason": "high_context_request_stalled",
        "recommended_action": recommended_action,
    }


def _monitor_loop(record: RequestWatchdogRecord) -> None:
    thresholds = record.thresholds
    while not record.done.wait(max(0.1, float(thresholds.poll_interval_seconds))):
        status = poll_request_watchdog(record, thresholds=thresholds, write_status=True)
        if status["status"] == "terminal_recovery_needed" and record.recovery_path is None:
            try:
                write_recoverable_turn_state(record, status=status)
                _write_active_session_status(record, status)
            except Exception:
                pass


def start_request_watchdog(
    agent: Any,
    *,
    request_id: str,
    api_call_count: int,
    estimated_context_tokens: int,
    now: Optional[float] = None,
    thresholds: Optional[RequestWatchdogThresholds] = None,
    start_monitor: bool = True,
) -> RequestWatchdogRecord:
    thresholds = thresholds or RequestWatchdogThresholds()
    started_at = time.time() if now is None else float(now)
    record = RequestWatchdogRecord(
        session_id=str(getattr(agent, "session_id", "") or ""),
        request_id=str(request_id or ""),
        request_id_fingerprint=_fingerprint(request_id),
        model=_safe_label(getattr(agent, "model", "")),
        provider=_safe_label(getattr(agent, "provider", "")),
        started_at=started_at,
        last_event_at=None,
        last_byte_at=None,
        estimated_context_tokens=max(0, int(estimated_context_tokens or 0)),
        api_call_count=max(0, int(api_call_count or 0)),
        queued_steer_count=_queued_steer_count(agent),
        thresholds=thresholds,
        queued_steer_count_fn=lambda: _queued_steer_count(agent),
    )
    try:
        agent._active_request_watchdog_record = record
    except Exception:
        pass
    if start_monitor:
        thread = threading.Thread(
            target=_monitor_loop,
            args=(record,),
            name=f"request-watchdog-{record.request_id_fingerprint}",
            daemon=True,
        )
        record._monitor_thread = thread
        thread.start()
    return record


def finish_request_watchdog(record: Optional[RequestWatchdogRecord], agent: Any = None) -> None:
    if record is None:
        return
    record.done.set()
    record.queued_steer_count_fn = None
    _clear_active_session_status(record)
    if agent is not None and getattr(agent, "_active_request_watchdog_record", None) is record:
        try:
            agent._active_request_watchdog_record = None
        except Exception:
            pass


def should_use_bounded_closeout_mode(
    *,
    estimated_context_tokens: int,
    remaining_work_kind: str,
    high_context_tokens: int = 200_000,
) -> bool:
    if int(estimated_context_tokens or 0) < int(high_context_tokens):
        return False
    text = str(remaining_work_kind or "").lower()
    return any(
        marker in text
        for marker in ("review", "report", "merge", "closeout", "verification")
    )
