"""Safe read-only session and event previews for the Telegram Mini App sidecar."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import time
from typing import Any

from hermes_constants import get_hermes_home

_ALLOWED_SESSION_STATES = {"observing", "waiting", "completed"}
_ALLOWED_TONES = {"ok", "warn", "muted"}
_ALLOWED_LOG_LEVELS = {"info", "warn", "error"}
_SOURCE_LABELS = {
    "cli": "Hermes CLI",
    "telegram": "Telegram",
    "discord": "Discord",
    "signal": "Signal",
    "cron": "Cron",
    "api": "API",
    "tui": "Hermes TUI",
    "desktop": "Desktop",
    "local": "Local",
}


def _preview_meta(source_label: str, *, source: str = "preview") -> dict[str, Any]:
    if source not in {"preview", "live-safe"}:
        source = "preview"
    return {
        "source": source,
        "source_label": source_label,
        "redaction": "safe-preview",
        "contains_live_actions": False,
    }


def _session_item(
    *,
    item_id: str,
    agent: str,
    state: str,
    meta: str,
    time_label: str,
    tone: str,
) -> dict[str, Any]:
    if state not in _ALLOWED_SESSION_STATES:
        state = "waiting"
    if tone not in _ALLOWED_TONES:
        tone = "muted"
    return {
        "id": item_id,
        "agent": agent,
        "state": state,
        "meta": meta,
        "time": time_label,
        "tone": tone,
    }


def _log_line(*, level: str, time_label: str, message: str) -> dict[str, str]:
    if level not in _ALLOWED_LOG_LEVELS:
        level = "info"
    return {"level": level, "time": time_label, "message": message}


def _safe_session_id(raw_id: Any) -> str:
    digest = hashlib.sha256(str(raw_id or "unknown").encode("utf-8", errors="ignore")).hexdigest()
    return f"session-{digest[:12]}"


def _source_label(source: Any) -> str:
    value = str(source or "").strip().lower()
    return _SOURCE_LABELS.get(value, "Hermes session")


def _coerce_timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def _relative_time_label(timestamp: Any, *, now: float) -> str:
    value = _coerce_timestamp(timestamp)
    if value is None:
        return "без времени"
    delta = max(0, int(now - value))
    if delta < 90:
        return "сейчас"
    minutes = delta // 60
    if minutes < 60:
        return f"{minutes} мин назад"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} ч назад"
    days = hours // 24
    if days < 7:
        return f"{days} дн назад"
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%d.%m")


def _message_count(row: dict[str, Any]) -> int:
    try:
        return max(0, int(row.get("message_count") or 0))
    except (TypeError, ValueError):
        return 0


def _sanitize_session_row(row: dict[str, Any], *, now: float) -> dict[str, Any]:
    ended_at = _coerce_timestamp(row.get("ended_at"))
    last_active = row.get("last_active") or row.get("started_at")
    live = ended_at is None
    count = _message_count(row)
    source_label = _source_label(row.get("source"))
    state = "observing" if live else "completed"
    tone = "warn" if live else "ok" if count > 0 else "muted"
    count_label = f"{count} сообщений" if count != 1 else "1 сообщение"
    meta = f"{count_label} · {source_label}"
    return _session_item(
        item_id=_safe_session_id(row.get("id")),
        agent=source_label,
        state=state,
        meta=meta,
        time_label=_relative_time_label(last_active, now=now),
        tone=tone,
    )


def _read_session_rows(session_db_factory: Callable[[], Any] | None) -> list[dict[str, Any]]:
    try:
        if session_db_factory is None:
            state_db = get_hermes_home() / "state.db"
            if not state_db.exists():
                return []
            from hermes_state import SessionDB

            session_db = SessionDB(db_path=state_db, read_only=True)
        else:
            session_db = session_db_factory()
        rows = session_db.list_sessions_rich(
            limit=5,
            include_children=False,
            order_by_last_active=True,
            include_archived=False,
        )
    except Exception:
        return []
    return [dict(row) for row in rows if isinstance(row, dict) or hasattr(row, "keys")]


def build_sessions_snapshot(
    *,
    session_db_factory: Callable[[], Any] | None = None,
    now: Callable[[], int | float] = time.time,
) -> dict[str, Any]:
    """Build a safe session index from read-only Hermes state.

    The response uses a strict projection and never returns raw state rows,
    message previews, filesystem locations, process ids, provider metadata,
    prompts, tokens, environment values, argv/cmdline values, or tool payloads.
    """
    current_time = float(now())
    rows = _read_session_rows(session_db_factory)
    return {
        "ok": True,
        "meta": _preview_meta("Safe session index", source="live-safe"),
        "items": [_sanitize_session_row(row, now=current_time) for row in rows],
    }


def build_logs_snapshot(
    *,
    sessions_provider: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build derived event lines from allowlisted facts only.

    M16 intentionally does not read raw log files. Event messages are fixed
    product-copy templates derived from safe snapshot counts and guardrail state.
    """
    try:
        sessions_snapshot = sessions_provider() if sessions_provider else build_sessions_snapshot()
        sessions = sessions_snapshot.get("items", []) if isinstance(sessions_snapshot, dict) else []
    except Exception:
        sessions = []

    session_count = len(sessions) if isinstance(sessions, list) else 0
    active_count = sum(1 for item in sessions if isinstance(item, dict) and item.get("state") == "observing")
    session_level = "warn" if active_count else "info"
    session_message = "Есть активные сессии наблюдения." if active_count else "Активных сессий наблюдения нет."

    return {
        "ok": True,
        "meta": _preview_meta("Derived safe event summary", source="live-safe"),
        "items": [
            _log_line(level="info", time_label="live", message="Read-only статус доступен."),
            _log_line(level=session_level, time_label="live", message=session_message),
            _log_line(level="info", time_label="safe", message=f"Сессий в безопасном индексе: {session_count}."),
            _log_line(level="warn", time_label="safe", message="Action routes заблокированы."),
        ],
    }
