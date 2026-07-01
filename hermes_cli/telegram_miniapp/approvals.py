"""Read-only approval queue preview for the Telegram Mini App sidecar."""

from __future__ import annotations

from typing import Any

_ALLOWED_RISKS = {"read_only", "critical"}
_ALLOWED_STATUSES = {"waiting", "blocked"}


def _preview_meta(source_label: str) -> dict[str, Any]:
    return {
        "source": "preview",
        "source_label": source_label,
        "redaction": "safe-preview",
        "contains_live_actions": False,
    }


def _approval_item(
    *,
    item_id: str,
    title: str,
    source: str,
    risk: str,
    summary: str,
    requested_at: str,
    status: str,
    checks: list[str],
) -> dict[str, Any]:
    if risk not in _ALLOWED_RISKS:
        risk = "critical"
    if status not in _ALLOWED_STATUSES:
        status = "blocked"
    return {
        "id": item_id,
        "title": title,
        "source": source,
        "risk": risk,
        "summary": summary,
        "requested_at": requested_at,
        "status": status,
        "checks": checks,
    }


def build_approvals_snapshot() -> dict[str, Any]:
    """Build the allowlisted M5 approvals preview.

    The queue is intentionally synthetic until a durable approval engine exists.
    It must not include raw commands, paths, logs, tokens, PIDs, env values, or
    executable payloads.
    """
    return {
        "ok": True,
        "meta": _preview_meta("Allowlisted approval preview"),
        "items": [
            _approval_item(
                item_id="system-mode-change-preview",
                title="Изменить системный режим",
                source="будущий системный запрос",
                risk="critical",
                summary="Макет высокорискового действия перед ручным решением владельца.",
                requested_at="макет",
                status="blocked",
                checks=["нужен владелец", "нужна причина", "нужен rollback-план"],
            ),
            _approval_item(
                item_id="read-event-log-preview",
                title="Открыть журнал событий",
                source="безопасный маршрут",
                risk="read_only",
                summary="Действие только читает редактированную шкалу событий без секретов.",
                requested_at="сейчас",
                status="waiting",
                checks=["секреты скрыты", "команды не выполняются", "доступ только владельцу"],
            ),
        ],
    }
