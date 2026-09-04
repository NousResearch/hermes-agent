from __future__ import annotations

"""Finite-choice picker support for the Telegram platform adapter."""

import asyncio
import secrets
import time
from typing import Any, Callable, Optional

from gateway.platforms.base import SendResult
from gateway.choice_picker import (
    ChoicePage,
    ChoiceProgress,
    PAGE_TIMEOUT_SECONDS,
    choice_action,
    choice_index,
    choice_label,
)


def _keyboard(choices, token, revision, button, markup):
    buttons = []
    for i, choice in enumerate(choices):
        label = (
            choice_label(choice, 62 if choice.get("is_current") else 64)
            if token
            else str(choice.get("label") or choice.get("value") or "")
        )
        if choice.get("is_current"):
            label = f"✓ {label}"
        action = choice_action(token, revision, i) if token else f"cp:{i}"
        buttons.append(button(label, callback_data=action))
    row_size = 1 if any(choice.get("full_width") for choice in choices) else 2
    return markup([buttons[i : i + row_size] for i in range(0, len(buttons), row_size)])


def _remove(adapter, chat_id, state):
    if adapter._choice_picker_state.get(chat_id) is state:
        adapter._choice_picker_state.pop(chat_id, None)
    handle = state.pop("expiry_handle", None)
    if handle:
        handle.cancel()


async def _expire(adapter, chat_id, state):
    if adapter._choice_picker_state.get(chat_id) is not state:
        return
    _remove(adapter, chat_id, state)
    try:
        await adapter._bot.edit_message_text(
            chat_id=chat_id,
            message_id=state["msg_id"],
            reply_markup=None,
            text="Menu expired. Run the command again.",
        )
    except Exception:
        try:
            await adapter._bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=state["msg_id"],
                reply_markup=None,
            )
        except Exception:
            pass


def cancel_choice_pages(adapter):
    for chat_id, state in list(getattr(adapter, "_choice_picker_state", {}).items()):
        if state.get("token"):
            _remove(adapter, chat_id, state)


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
        reusable = (metadata or {}).get("choice_pages") is True
        token = secrets.token_hex(4) if reusable else ""
        if reusable:
            if not (metadata or {}).get("requester_user_id"):
                return SendResult(success=False, error="Requester required")
            choices = ChoicePage(title, choices).choices
        if not choices:
            return SendResult(success=False, error="No choices")
        keyboard = _keyboard(
            choices, token, 0, inline_keyboard_button, inline_keyboard_markup
        )

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

        key = str(chat_id)
        old = adapter._choice_picker_state.get(key)
        if old:
            _remove(adapter, key, old)
        state = {
            "expires_at": time.monotonic() + PAGE_TIMEOUT_SECONDS,
            "msg_id": msg.message_id,
            "choices": choices,
            "requester_user_id": str((metadata or {}).get("requester_user_id") or ""),
            "session_key": session_key,
            "on_choice_selected": on_choice_selected,
            "token": token,
            "revision": 0,
            "thread_id": str(thread_id or ""),
            "busy": False,
        }
        adapter._choice_picker_state[key] = state
        if reusable:
            state["expiry_handle"] = asyncio.get_running_loop().call_later(
                PAGE_TIMEOUT_SECONDS,
                lambda: asyncio.create_task(_expire(adapter, key, state)),
            )
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
    inline_keyboard_button: Any = None,
    inline_keyboard_markup: Any = None,
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
        if state.get("token"):
            await _expire(adapter, chat_id, state)
        else:
            _remove(adapter, chat_id, state)
        await query.answer(text="This menu has expired. Run the command again.")
        return
    requester_user_id = str(state.get("requester_user_id") or "")
    if requester_user_id and requester_user_id != str(
        getattr(query.from_user, "id", "")
    ):
        await query.answer(text="⛔ This menu belongs to another user.")
        return
    if state.get("token") and (
        str(getattr(query_message, "chat_id", "")) != chat_id
        or str(getattr(query_message, "message_thread_id", None) or "")
        != state["thread_id"]
    ):
        await query.answer(text="This menu belongs to another conversation.")
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
        idx = (
            choice_index(data, state["token"], state["revision"], len(state["choices"]))
            if state.get("token")
            else int(data[3:])
        )
        if idx is None or idx < 0:
            raise ValueError("stale selection")
        choice = state["choices"][idx]
    except (ValueError, IndexError):
        await query.answer(text="Invalid selection.")
        return

    callback = state.get("on_choice_selected")
    if not callback:
        await query.answer(text="Picker expired.")
        return
    if state.get("busy"):
        await query.answer(text="Please wait.")
        return
    state["busy"] = True

    try:
        await query.answer()
        if adapter._choice_picker_state.get(chat_id) is not state:
            return
        if time.monotonic() > float(state.get("expires_at") or 0):
            if state.get("token"):
                await _expire(adapter, chat_id, state)
            else:
                _remove(adapter, chat_id, state)
            return
        result_text = await callback(chat_id, str(choice.get("value") or ""))
        if isinstance(result_text, ChoiceProgress):
            if not state.get("token"):
                raise TypeError("Reusable choice pages were not enabled")
            if adapter._choice_picker_state.get(chat_id) is not state:
                return
            await query.edit_message_text(
                text=adapter.format_message(result_text.text),
                parse_mode=parse_mode.MARKDOWN_V2,
                reply_markup=None,
            )
            if adapter._choice_picker_state.get(chat_id) is not state:
                return
            if time.monotonic() > state["expires_at"]:
                await _expire(adapter, chat_id, state)
                return
            result_text = await result_text.complete()
        if adapter._choice_picker_state.get(chat_id) is not state:
            return
        if state.get("token") and time.monotonic() > state["expires_at"]:
            await _expire(adapter, chat_id, state)
            return
        if isinstance(result_text, ChoicePage):
            if not state.get("token"):
                raise TypeError("Reusable choice pages were not enabled")
            revision = state["revision"] + 1
            keyboard = _keyboard(
                result_text.choices,
                state["token"],
                revision,
                inline_keyboard_button,
                inline_keyboard_markup,
            )
            await query.edit_message_text(
                text=adapter.format_message(result_text.title),
                parse_mode=parse_mode.MARKDOWN_V2,
                reply_markup=keyboard,
            )
            if adapter._choice_picker_state.get(chat_id) is state:
                state.update(choices=result_text.choices, revision=revision, busy=False)
            else:
                await adapter._bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=state["msg_id"],
                    reply_markup=None,
                )
            return
        if not isinstance(result_text, str):
            raise TypeError("Choice callback must return text or ChoicePage")
    except asyncio.CancelledError:
        _remove(adapter, chat_id, state)
        if state.get("token"):
            try:
                await adapter._bot.edit_message_reply_markup(
                    chat_id=chat_id, message_id=state["msg_id"], reply_markup=None
                )
            except Exception:
                pass
        raise
    except Exception as exc:
        logger.error("Choice picker selection failed: %s", exc)
        result_text = (
            "Unable to update menu."
            if state.get("token")
            else f"Error applying selection: {exc}"
        )

    if adapter._choice_picker_state.get(chat_id) is not state:
        return
    _remove(adapter, chat_id, state)

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
