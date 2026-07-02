"""Read-only approval queue preview for the Telegram Mini App sidecar."""

from __future__ import annotations

from typing import Any

_ALLOWED_RISKS = {"read_only", "critical"}
_ALLOWED_STATUSES = {"waiting", "blocked"}
_ALLOWED_DECISIONS = {"approve_once", "reject_once"}

# The bridge exposes four risk tiers; the preview UI carries two. Only a true
# read_only action shows the safe badge — anything state-changing (reversible,
# risky, critical) maps up to critical so the owner never under-reads the risk.
_RISK_TIER_TO_UI = {
    "read_only": "read_only",
    "reversible": "critical",
    "risky": "critical",
    "critical": "critical",
}


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


def build_live_approvals_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Map a redacted bridge snapshot into the sidecar approvals response.

    The bridge already projected away every sensitive field; this only shapes
    the safe items for the UI and carries the opaque ``snapshot_version`` plus
    per-item ``allowed_decisions`` the action gate needs. It never re-derives or
    exposes raw command/session data because it never sees any.
    """
    items: list[dict[str, Any]] = []
    for raw in snapshot.get("items", []) or []:
        if not isinstance(raw, dict):
            continue
        allowed = [d for d in (raw.get("allowed_decisions") or []) if d in _ALLOWED_DECISIONS]
        item = _approval_item(
            item_id=str(raw.get("approval_id", "")),
            title=str(raw.get("title", "Одобрение")),
            source=str(raw.get("source_label", "gateway approval")),
            risk=_RISK_TIER_TO_UI.get(str(raw.get("risk")), "critical"),
            summary=str(raw.get("summary", "Требуется решение владельца.")),
            requested_at=str(raw.get("requested_at", "")),
            status="waiting",
            checks=["решение владельца", "одноразовое approve/reject", "raw-команда не раскрывается"],
        )
        item["allowed_decisions"] = allowed
        items.append(item)
    return {
        "ok": True,
        "meta": {
            "source": "live-safe",
            "source_label": "Live gateway approvals",
            "redaction": "safe-preview",
            "contains_live_actions": True,
        },
        "snapshot_version": str(snapshot.get("snapshot_version", "")),
        "items": items,
    }
