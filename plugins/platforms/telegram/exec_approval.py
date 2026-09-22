"""Request-bound Telegram approval cards, retired by the existing core settlement hook.

The short callback token is a transport handle, never a session-level approval. All card
state lives on the Telegram loop; settlement may arrive from an agent worker while the send
is still in flight. Keep that outcome on the same record until the message id arrives.
"""

import asyncio
import itertools
import logging
from dataclasses import dataclass

from gateway.config import Platform
from gateway.platforms.base import ExecApprovalPrompt, SendResult, unauthorized_action_notice
from gateway.platforms.base_exec_approval import approval_timeout_seconds, format_approval_timed_out_notice

logger = logging.getLogger(__name__)

# The runner must not replace the adapter's pre-publication settlement hook with a timeout-only hook.
SETTLEMENT_MANAGED = {"exec_approval_settlement": True}


@dataclass
class ApprovalCard:
    session_key: str
    request_id: str
    chat_id: str
    choices: tuple[str, ...]
    message_id: str | None = None
    reason: str | None = None
    handled: bool = False


async def send_prompt(adapter, prompt: ExecApprovalPrompt) -> SendResult:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.constants import ParseMode
    from tools.approval import register_gateway_settle

    if not prompt.request_id:
        return SendResult(success=False, error="An interactive approval requires a request id")
    loop = asyncio.get_running_loop()
    timeout_s = approval_timeout_seconds()
    if not hasattr(adapter, "_approval_counter"):
        adapter._approval_counter = itertools.count(1)
    approval_id = next(adapter._approval_counter)
    card = ApprovalCard(prompt.session_key, prompt.request_id, str(prompt.chat_id), tuple(prompt.choices))
    adapter._approval_state[approval_id] = card

    async def edit_terminal_card():
        notices = {
            "timeout": format_approval_timed_out_notice(timeout_s),
            "interrupted": "Approval withdrawn: the operation was interrupted and did not run.",
            "session_closed": "Approval withdrawn: its owning operation ended without approval.",
            "notify_failed": "Approval withdrawn: the request could not be delivered reliably.",
            "resolved": "Approval answered elsewhere.",
        }
        notice = notices.get(card.reason or "", "This approval is no longer pending.")
        try:
            result = await adapter.edit_message(card.chat_id, card.message_id, notice, finalize=True)
            if result.success:
                return
        except Exception:
            logger.debug("Could not retire Telegram approval card", exc_info=True)
        # Preserve the existing timeout notice fallback, including the original topic and
        # interim marker: failed UI cleanup must not masquerade as a final agent response.
        from gateway.run import _interim_metadata
        try:
            await adapter.send(card.chat_id, notice, metadata=_interim_metadata(prompt.metadata))
        except Exception:
            logger.debug("Could not send Telegram approval settlement notice", exc_info=True)

    def settle_on_loop(reason: str):
        if card.handled:
            return
        card.reason = reason
        adapter._approval_state.pop(approval_id, None)
        if card.message_id is not None:
            loop.create_task(edit_terminal_card())

    def settle(reason: str):
        # Never do network I/O or touch the adapter map from a tool worker.
        loop.call_soon_threadsafe(settle_on_loop, reason)

    if not register_gateway_settle(card.session_key, card.request_id, settle):
        adapter._approval_state.pop(approval_id, None)
        # Already settled before publication: no ghost card and no text fallback for a dead request.
        return SendResult(success=True, raw_response=dict(SETTLEMENT_MANAGED))

    def on_sent(message):
        card.message_id = str(message.message_id)
        if card.reason is not None and not card.handled:
            loop.create_task(edit_terminal_card())

    def build():
        buttons = [InlineKeyboardButton(label, callback_data=f"ea:{choice}:{approval_id}")
                   for label, choice, _ in prompt.actions]
        return prompt.text, InlineKeyboardMarkup(adapter._rows_of_two(buttons)), on_sent

    result = await adapter._send_prompt(
        "send_exec_approval", prompt.chat_id, prompt.metadata, build, parse_mode=ParseMode.HTML,
        thread_id=adapter._metadata_thread_id(prompt.metadata), reply_to_mode=adapter._reply_to_mode,
        observe_late_sends=True)
    metadata = result.raw_response if isinstance(result.raw_response, dict) else {}
    if result.success or metadata.get("ambiguous"):
        result.raw_response = {**metadata, **SETTLEMENT_MANAGED}
    else:
        card.handled = True
        adapter._approval_state.pop(approval_id, None)
    return result


async def handle_callback(adapter, query, data: str, cb: dict) -> None:
    from tools.approval import resolve_gateway_approval

    parts = data.split(":", 2)
    if len(parts) != 3:
        return
    try:
        approval_id = int(parts[2])
    except ValueError:
        await query.answer(text="Invalid approval data.")
        return
    if not await adapter._callback_authorized(query, cb, unauthorized_action_notice(Platform.TELEGRAM)):
        return
    card = adapter._approval_state.get(approval_id)
    # Unbound/legacy controls never fall back to FIFO. Check the actual card as well as the user.
    if not isinstance(card, ApprovalCard) or not card.request_id:
        await query.answer(text="This approval is no longer pending.")
        return
    choice = parts[1]
    if (choice not in card.choices or str(cb["chat_id"]) != card.chat_id
            or str(getattr(query.message, "message_id", "")) != card.message_id):
        await query.answer(text="This button does not match the approval request.")
        return
    adapter._approval_state.pop(approval_id, None)
    card.handled = True
    # Resolve before rendering; a timeout/withdrawal racing the click must not look approved.
    count = resolve_gateway_approval(card.session_key, choice, request_id=card.request_id)
    logger.info("Telegram button resolved %d approval(s) for session %s (request_id=%s, choice=%s)",
                count, card.session_key, card.request_id, choice)
    labels = {"once": "✅ Approved once", "session": "✅ Approved for session",
              "always": "✅ Approved permanently", "deny": "❌ Denied"}
    label = labels[choice] if count else "This approval is no longer pending."
    text = f"{label} by {getattr(query.from_user, 'first_name', 'User')}" if count else label
    try:
        await query.answer(text=label)
    except Exception:
        # The decision has already committed. A failed/expired toast must not skip retirement.
        logger.debug("Could not acknowledge Telegram approval callback", exc_info=True)
    try:
        await adapter._edit_md_quiet(query, text)
    finally:
        if count and cb["chat_id"] is not None:
            adapter.resume_typing_for_chat(str(cb["chat_id"]))
