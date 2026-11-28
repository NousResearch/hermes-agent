"""Telegram approval card delivery and edits that keep resolved prompts visible.

Replacing an approval or slash-confirm prompt with only the decision ("✅ Approved once by Ana")
erases what was approved, so a shared chat loses its record. The edit keeps the prompt's own
rendering (``Message.text_html`` rebuilds it from the entities, code block included), drops the
"if you don't answer within N minutes" line that no longer applies, and appends the decision.
"""

from __future__ import annotations

import asyncio
import html
import logging
from typing import Any

from gateway.platforms.base import ExecApprovalPrompt, SendResult
from plugins.platforms.telegram.telegram_media_flood import flood_wait_seconds
from plugins.platforms.telegram.telegram_send_errors import FLOOD, may_have_delivered, send_error_kind

logger = logging.getLogger(__name__)


async def send_approval_card(
    adapter: Any, prompt: ExecApprovalPrompt, approval_id: int, keyboard: Any, *,
    parse_mode: Any, inline_wait_cap: float,
) -> SendResult:
    """Retry a short, explicit flood refusal once and keep a possibly delivered card usable."""
    from plugins.platforms.telegram.adapter import _redact_telegram_error_text

    async def send():
        return await adapter._send_control_message(
            prompt.chat_id, prompt.text, parse_mode=parse_mode, reply_markup=keyboard,
            thread_id=adapter._metadata_thread_id(prompt.metadata), metadata=prompt.metadata,
            reply_to_mode=adapter._reply_to_mode)

    try:
        try:
            msg = await send()
        except Exception as exc:  # health: allow BLE001 -- only an explicit flood refusal is retried
            delay = flood_wait_seconds(exc) if send_error_kind(exc) == FLOOD else None
            if delay is None or not 0 <= delay <= inline_wait_cap:
                raise
            await asyncio.sleep(delay)
            msg = await send()
    except Exception as exc:  # health: allow BLE001 -- classify the transport outcome and redact the error
        error = _redact_telegram_error_text(exc)
        logger.warning("[%s] send_exec_approval failed: %s", adapter.name, error)
        if not may_have_delivered(exc):
            return SendResult(success=False, error=error)
        result = SendResult(success=False, error=error, raw_response={"delivery_uncertain": True})
    else:
        result = SendResult(success=True, message_id=str(msg.message_id))
    adapter._approval_state[approval_id] = prompt.session_key
    if prompt.request_id:
        adapter._approval_request_ids[approval_id] = prompt.request_id
    return result


async def edit_keeping_prompt(adapter: Any, query: Any, decision: str) -> None:
    """Append *decision* to the prompt and drop its buttons; the adapter's decision-only edit runs when
    the prompt has no readable body or the appended line pushes it past Telegram's length cap."""
    from telegram.error import BadRequest, NetworkError

    body = getattr(query.message, "text_html", None) if query.message else None
    if isinstance(body, str) and body:
        body = body.replace(adapter._ea_deadline_line(), "")  # same escaping as text_html
        try:
            await query.edit_message_text(text=f"{body}\n\n— {html.escape(decision)}", parse_mode="HTML", reply_markup=None)
            return
        except BadRequest:  # too long / unparsable: the decision-only edit below still clears the buttons
            pass
        except NetworkError as exc:  # network/timeout: the edit may have landed, so keep the prompt as is
            logger.debug("Telegram resolved-prompt edit failed: %s", exc)
            return
    await adapter._edit_md_quiet(query, decision)
