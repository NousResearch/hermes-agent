from __future__ import annotations

"""Finite-choice picker support for the Telegram platform adapter."""

import time
from typing import Any, Callable, Optional

from gateway.platforms.base import SendResult


async def send_choice_picker(
    adapter: Any,
    chat_id: str,
    title: str,
    choices: list,
    session_key: str,
    on_choice_selected: Any,
    metadata: Optional[dict[str, Any]] = None,
    *,
    inline_keyboard_button: Any,
    inline_keyboard_markup: Any,
    parse_mode: Any,
    normalize_chat_id: Callable[[Any], Any],
    redact_error: Callable[[Any], str],
    logger: Any,
) -> SendResult:
    """Send a flat inline-keyboard choice picker (one tap to one value)."""
    if not adapter._bot:
        return SendResult(success=False, error="Not connected")

    try:
        buttons = []
        for i, choice in enumerate(choices):
            label = str(choice.get("label") or choice.get("value") or "")
            if choice.get("is_current"):
                label = f"✓ {label}"
            buttons.append(inline_keyboard_button(label, callback_data=f"cp:{i}"))
        if not buttons:
            return SendResult(success=False, error="No choices")
        row_size = 1 if any(choice.get("full_width") for choice in choices) else 2
        keyboard = inline_keyboard_markup([
            buttons[i : i + row_size] for i in range(0, len(buttons), row_size)
        ])

        thread_id = metadata.get("thread_id") if metadata else None
        reply_to_id = adapter._reply_to_message_id_for_send(
            None, metadata, reply_to_mode=adapter._reply_to_mode
        )
        msg = await adapter._send_message_with_thread_fallback(
            chat_id=normalize_chat_id(chat_id),
            text=adapter.format_message(title),
            parse_mode=parse_mode.MARKDOWN_V2,
            reply_markup=keyboard,
            reply_to_message_id=reply_to_id,
            **adapter._thread_kwargs_for_send(
                chat_id,
                thread_id,
                metadata,
                reply_to_message_id=reply_to_id,
                reply_to_mode=adapter._reply_to_mode,
            ),
            **adapter._link_preview_kwargs(),
        )

        adapter._choice_picker_state[str(chat_id)] = {
            "expires_at": time.monotonic() + 120,
            "msg_id": msg.message_id,
            "choices": choices,
            "requester_user_id": str((metadata or {}).get("requester_user_id") or ""),
            "session_key": session_key,
            "on_choice_selected": on_choice_selected,
        }
        return SendResult(success=True, message_id=str(msg.message_id))
    except Exception as exc:
        logger.warning(
            "[%s] send_choice_picker failed: %s",
            adapter.name,
            redact_error(exc),
        )
        return SendResult(success=False, error=redact_error(exc))


async def handle_choice_picker_callback(
    adapter: Any,
    query: Any,
    data: str,
    chat_id: str,
    *,
    parse_mode: Any,
    logger: Any,
) -> None:
    """Handle choice picker button taps (cp:<index>)."""
    state = adapter._choice_picker_state.get(chat_id)
    if not state:
        await query.answer(text="Picker expired — run the command again.")
        return

    query_message = getattr(query, "message", None)
    query_chat = getattr(query_message, "chat", None)
    query_message_id = getattr(query_message, "message_id", None)
    if query_message_id != state.get("msg_id"):
        await query.answer(text="This menu has expired. Run the command again.")
        return
    if time.monotonic() > float(state.get("expires_at") or 0):
        adapter._choice_picker_state.pop(chat_id, None)
        await query.answer(text="This menu has expired. Run the command again.")
        return
    requester_user_id = str(state.get("requester_user_id") or "")
    if requester_user_id and requester_user_id != str(
        getattr(query.from_user, "id", "")
    ):
        await query.answer(text="⛔ This menu belongs to another user.")
        return
    if not adapter._is_callback_user_authorized(
        str(getattr(query.from_user, "id", "")),
        chat_id=getattr(query_message, "chat_id", None),
        chat_type=(
            str(getattr(query_chat, "type", None))
            if getattr(query_chat, "type", None) is not None
            else None
        ),
        thread_id=(
            str(getattr(query_message, "message_thread_id", None))
            if getattr(query_message, "message_thread_id", None) is not None
            else None
        ),
        user_name=getattr(query.from_user, "first_name", None),
    ):
        await query.answer(text="⛔ You are not authorized to change this setting.")
        return

    try:
        idx = int(data[3:])
        choice = state["choices"][idx]
    except (ValueError, IndexError):
        await query.answer(text="Invalid selection.")
        return

    callback = state.get("on_choice_selected")
    if not callback:
        await query.answer(text="Picker expired.")
        return

    try:
        result_text = await callback(chat_id, str(choice.get("value") or ""))
    except Exception as exc:
        logger.error("Choice picker selection failed: %s", exc)
        result_text = f"Error applying selection: {exc}"

    try:
        await query.edit_message_text(
            text=adapter.format_message(result_text),
            parse_mode=parse_mode.MARKDOWN_V2,
            reply_markup=None,
        )
    except Exception:
        try:
            await query.edit_message_text(
                text=result_text,
                parse_mode=None,
                reply_markup=None,
            )
        except Exception:
            pass
    await query.answer()
    adapter._choice_picker_state.pop(chat_id, None)
