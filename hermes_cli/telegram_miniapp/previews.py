"""Read-only session and log previews for the Telegram Mini App sidecar."""

from __future__ import annotations

from typing import Any

_ALLOWED_SESSION_STATES = {"observing", "waiting", "completed"}
_ALLOWED_TONES = {"ok", "warn", "muted"}
_ALLOWED_LOG_LEVELS = {"info", "warn", "error"}


def _preview_meta(source_label: str) -> dict[str, Any]:
    return {
        "source": "preview",
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


def build_sessions_snapshot() -> dict[str, Any]:
    """Build allowlisted session preview data.

    The response intentionally avoids raw runtime records, paths, PIDs, command
    lines, prompts, provider names, tokens, and user-generated content.
    """
    return {
        "ok": True,
        "meta": _preview_meta("Allowlisted session preview"),
        "items": [
            _session_item(
                item_id="codex-review-preview",
                agent="Codex review",
                state="completed",
                meta="проверка интерфейса и safety-copy завершена",
                time_label="12 мин назад",
                tone="ok",
            ),
            _session_item(
                item_id="miniapp-sidecar-preview",
                agent="Mini App sidecar",
                state="observing",
                meta="локальный статус доступен, публичный режим выключен",
                time_label="сейчас",
                tone="warn",
            ),
            _session_item(
                item_id="workspace-preview",
                agent="Workspace monitor",
                state="waiting",
                meta="нет активных команд из Mini App",
                time_label="фон",
                tone="muted",
            ),
        ],
    }


def build_logs_snapshot() -> dict[str, Any]:
    """Build allowlisted redacted event preview lines."""
    return {
        "ok": True,
        "meta": _preview_meta("Redacted event preview"),
        "items": [
            _log_line(level="info", time_label="M2", message="Status API работает только на чтение."),
            _log_line(level="info", time_label="M3", message="Telegram initData проверяется на сервере."),
            _log_line(level="warn", time_label="M5", message="Очередь одобрений подключена без исполнения действий."),
            _log_line(level="info", time_label="M6", message="Сессии и события отдаются как редактированный безопасный снимок."),
        ],
    }
