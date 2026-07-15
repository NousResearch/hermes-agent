"""Read-only capability metadata for the Telegram Mini App."""

from __future__ import annotations

from typing import Any


def _preview_meta() -> dict[str, Any]:
    return {
        "source": "preview",
        "source_label": "Read-only capability matrix",
        "redaction": "safe-preview",
        "contains_live_actions": False,
    }


def _capability(*, item_id: str, label: str, enabled: bool, mode: str, reason: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "label": label,
        "enabled": enabled,
        "mode": mode,
        "reason": reason,
    }


def build_capabilities_snapshot(*, actions_enabled: bool = False) -> dict[str, Any]:
    """Build a static-safe capabilities matrix.

    The response intentionally contains only product capability labels. It does
    not inspect or expose runtime commands, paths, PIDs, env values, tokens,
    provider names, prompts, user payloads, raw logs, or config values.

    ``actions_enabled`` reflects whether the owner-confirmed Phase 1 decision
    endpoint (approve_once / reject_once) is live. Restart / process control
    stays blocked regardless.
    """
    if actions_enabled:
        approve = _capability(
            item_id="approve-action",
            label="Approve / reject",
            enabled=True,
            mode="owner-confirmed-action",
            reason="Approve/reject once доступны владельцу для pending-одобрений.",
        )
    else:
        approve = _capability(
            item_id="approve-action",
            label="Approve / reject",
            enabled=False,
            mode="blocked",
            reason="Требуется отдельный approved backend design.",
        )
    return {
        "ok": True,
        "meta": _preview_meta(),
        "items": [
            _capability(
                item_id="status-read",
                label="Статус системы",
                enabled=True,
                mode="read-only",
                reason="Доступен только allowlisted status snapshot.",
            ),
            _capability(
                item_id="approvals-read",
                label="Очередь одобрений",
                enabled=True,
                mode="read-only",
                reason="Доступен только preview без decision endpoint." if not actions_enabled else "Доступны pending-одобрения с owner-confirmed решением.",
            ),
            _capability(
                item_id="sessions-read",
                label="Сессии и события",
                enabled=True,
                mode="read-only",
                reason="Доступны только redacted preview items.",
            ),
            approve,
            _capability(
                item_id="restart-action",
                label="Restart / process control",
                enabled=False,
                mode="blocked",
                reason="Execution routes отсутствуют и должны проходить отдельный approval.",
            ),
        ],
    }
