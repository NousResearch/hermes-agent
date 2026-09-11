"""Read-only compatibility for pre-mediation recommendation controls.

The old JSON ledger did not bind a control to a native actor or destination.
Its targets cannot grant consent, consume a suggestion, or sync preferences.
Authenticated surfaces open current review/settings controls instead.
"""

from __future__ import annotations

import re
from typing import Any

from .notify import STALE_ACTION_MESSAGE

REPLACED_CONTROL_MESSAGE = (
    "This older control has been replaced. Review the current options below. "
    "Nothing was changed."
)


def handle_action(target: str, **_legacy_options: Any) -> dict[str, Any]:
    """Never read the old ledger or treat its data as authority, even in the CLI."""
    match = re.fullmatch(
        r"wa:(share|review|install|update|view|view_changes|view_portal|not_now|mute):"
        r"[A-Za-z0-9_-]{1,128}(?::(1d|1w|30d|forever|off))?",
        target,
    )
    if match is None or (match[2] and match[1] != "mute"):
        return {"ok": False, "stale": True, "message": STALE_ACTION_MESSAGE}
    action = match[1]
    command = (
        "mute"
        if action == "mute"
        else "browse"
        if action in {"view", "view_changes", "view_portal"}
        else "inbox"
    )
    return {
        "ok": True,
        "action": action,
        "command": command,
        "requires_fresh_consent": True,
        "open_mute_settings": command == "mute",
        "message": REPLACED_CONTROL_MESSAGE,
        "next": f"hermes wisdom {command}",
    }


def current_action_view(target, service, context):
    """Called after adapter authentication; no legacy-selected identity is used."""
    from gateway.wisdom_command import WisdomAction, WisdomCommandController, WisdomView

    result = handle_action(target)
    if not result["ok"]:
        return WisdomView("Collective Wisdom", result["message"])
    view = WisdomCommandController().execute(result["command"], service, context)
    if result["command"] == "inbox" and not context.is_group and not view.items:
        view.actions.extend([
            WisdomAction(
                "Browse team skills", "browse", local_command="/wisdom browse"
            ),
            WisdomAction(
                "Review local candidates",
                "candidates",
                local_command="/wisdom candidates",
            ),
        ])
    view.notice = "\n".join(filter(None, (view.notice, result["message"])))
    return view


def current_install_view(target, service, context):
    """An unversioned legacy click requests review, never approves latest bytes."""
    from gateway.wisdom_command import WisdomAction, WisdomCommandController, WisdomView

    match = re.fullmatch(
        r"wi:(plan|confirm):(install|update):[A-Za-z0-9_-]{1,128}", target,
    )
    if match is None:
        raise ValueError("Invalid Collective Wisdom action.")
    if match[1] == "confirm":
        return WisdomView(
            "Review required", REPLACED_CONTROL_MESSAGE,
            actions=[WisdomAction(
                "Browse team skills", "browse", local_command="/wisdom browse",
            )],
        )
    operation, reference = match[2], target.rsplit(":", 1)[1]
    controller = WisdomCommandController()
    if context.is_group:
        return controller.execute(f"{operation} {reference}", service, context)
    return controller.review_install(service, reference, kind=operation, context=context)
