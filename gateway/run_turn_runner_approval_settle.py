"""Tell the chat when an exec-approval prompt times out (messaging platforms).

``tools.approval_gateway_wait._await_gateway_decision`` calls ``entry.settle(reason)`` once the
wait ends. Only the TUI registered such a hook, so on Telegram / Slack / WhatsApp a card whose
timer ran out kept live buttons and the user never learned the command did NOT run. The turn
runner registers the hook here right after the prompt was delivered.

Best-effort by design: a failed notice is logged at debug — the approval already resolved as
"no", and nothing here may block the agent thread.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from agent.i18n import t
from gateway.platforms.base_exec_approval import approval_timeout_seconds, format_approval_timed_out_notice

logger = logging.getLogger(__name__)


def register_timeout_notice(
    runner, approval_data: dict, *, command: str, card_message_id: Optional[str]) -> None:
    """Arm a settle hook that posts the timed-out notice for ``approval_data['request_id']``.

    ``runner`` is the ``TurnRunner`` (for ``_ctx`` and ``_schedule``); ``card_message_id`` is the
    delivered BUTTON card's id when the adapter returned one, so the card itself is edited in place
    (which also drops its buttons). The plain-text prompt passes ``None``: it has no buttons to
    drop and rewriting it would erase the record of what was asked. ``command`` is the
    already-redacted command shown to the user. The notice is skipped when the run is no longer
    current (``ctx._run_still_current``).
    """
    from tools.approval import register_gateway_settle

    request_id = approval_data.get("request_id")
    session_key = runner._ctx.session_key or ""
    if not request_id or not session_key:
        return
    timeout_s = approval_timeout_seconds()

    def settle(reason: str) -> None:
        if reason != "timeout":
            return  # answered / interrupted / notify_failed already produced their own feedback
        # Same guard as every other late notice in TurnRunner: after /stop, /new or a restart the
        # turn is over and this chat belongs to a newer run — do not edit or post into it.
        still_current = getattr(runner._ctx, "_run_still_current", None)
        if callable(still_current) and not still_current():
            return
        runner._schedule(
            _post_timeout_notice(runner._ctx, command, card_message_id, timeout_s),
            "Approval timeout notice scheduling error")

    register_gateway_settle(session_key, request_id, settle)


async def _post_timeout_notice(ctx, command: str, card_message_id: Optional[str], timeout_s: int) -> None:
    from gateway.run import _interim_metadata

    adapter = ctx._status_adapter
    notice = format_approval_timed_out_notice(timeout_s)
    metadata = _interim_metadata(ctx._status_thread_metadata)
    try:
        # Send the notice as a NEW message, never an in-place edit of the button card.
        # The card was sent as platform-native markup (Telegram: HTML with <pre>); editing it
        # with the MarkdownV2 formatter escapes those tags into literal \<pre\> and the fallback
        # shows raw formatting glyphs (the "no formatting / literal backticks" complaint). A fresh
        # send goes through the same markdown pipeline as the final reply and renders cleanly.
        # The plain-text prompt path already passes card_message_id=None and posts a fresh message.
        await adapter.send(ctx._status_chat_id, notice, metadata=metadata)
    except Exception:
        logger.debug("Approval timeout notice failed", exc_info=True)
