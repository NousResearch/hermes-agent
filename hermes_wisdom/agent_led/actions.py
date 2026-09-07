"""Action dispatcher for agent-led recommendation buttons.

Resolves an opaque ``wa:<action>:<dedup>`` target through the delivery ledger
and routes it to the right consent-gated path. Publishing and installing
always go through the existing ``WisdomService`` gates; this module never
uploads or writes managed skills itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .history import SuggestionHistory, history_path
from .notify import STALE_ACTION_MESSAGE, DeliveryLedger
from .policy import load_policy


def handle_action(
    target: str,
    *,
    ledger: DeliveryLedger | None = None,
    history: SuggestionHistory | None = None,
    service: Any = None,
    mute_choice: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a structured outcome; never raises for stale/duplicate presses."""
    ledger_obj = ledger or DeliveryLedger(history_path().parent / "agent_led_delivery.json")
    hist = history or SuggestionHistory()
    current = now or datetime.now(timezone.utc)
    resolved = ledger_obj.resolve_action(target, at=current)
    if not resolved.get("ok"):
        return {"ok": False, "stale": True, "message": STALE_ACTION_MESSAGE}
    action = str(resolved["action"])
    event = resolved["event"]
    if action == "mute":
        # Legacy targets carry no authenticated revision or durable choice ID.
        # Even a duration from an old keyboard must open fresh native controls.
        return {
            "ok": True, "action": action, "open_mute_settings": True,
            "message": "Open /wisdom mute to choose your notification preference.",
        }
    policy = load_policy(client=getattr(service, "client", None))
    skill_id = str(event.get("skill_id") or "")
    content_hash = str((event.get("rendering_hints") or {}).get("content_hash") or "")
    dedup = str(event.get("dedup_key") or "")

    if action == "not_now":
        if not ledger_obj.mark_acted(dedup, action):
            return {"ok": False, "stale": True, "message": STALE_ACTION_MESSAGE}
        record = hist.record_dismissal(
            skill_id, content_hash, suppression_days=policy.dismiss_suppression_days, at=current,
            client=getattr(service, "client", None),
        )
        return {"ok": True, "action": action, "message": "Okay, I will not bring this up again for a while.", **record}

    if action in {"share", "review"}:
        # Share opens the resumable packaging flow; nothing is published here.
        from .share_flow import ShareFlow

        skill_path = _resolve_skill_path(skill_id)
        if skill_path is None:
            return {"ok": False, "stale": True, "message": "That skill is no longer on this device."}
        if not ledger_obj.mark_acted(dedup, action):
            return {"ok": False, "stale": True, "message": STALE_ACTION_MESSAGE}
        flow = ShareFlow()
        summary = flow.start(skill_path)
        return {
            "ok": True,
            "action": action,
            "published": False,
            "flow": summary,
            "message": (
                "Starting the share review. I will show you the package and portability "
                "notes before anything is uploaded."
            ),
        }

    if action in {"install", "update"}:
        if not ledger_obj.mark_acted(dedup, action):
            return {"ok": False, "stale": True, "message": STALE_ACTION_MESSAGE}
        reference = skill_id if not event.get("skill_version") else f"{skill_id}@v{event['skill_version']}"
        return {
            "ok": True,
            "action": action,
            "installed": False,
            "next": f"hermes wisdom {'install' if action == 'install' else 'update'} '{reference}' --plan --json",
            "message": "I will check prerequisites and walk you through setup before applying.",
        }

    if action in {"view", "view_changes", "view_portal"}:
        url = None
        if service is not None:
            try:
                url = service.portal_skill_url(skill_id)
            except Exception:
                url = None
        return {"ok": True, "action": action, "url": url}

    return {"ok": False, "stale": True, "message": STALE_ACTION_MESSAGE}


def _resolve_skill_path(skill_name: str) -> Path | None:
    try:
        from tools.skill_usage import _find_skill_dir

        path = _find_skill_dir(skill_name)
    except Exception:
        return None
    return path if path is not None and path.is_dir() else None
