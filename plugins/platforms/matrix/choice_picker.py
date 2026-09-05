"""Matrix finite choices and optional revision-bound reusable reaction pages."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from gateway.choice_picker import (
    ChoicePage,
    ChoiceProgress,
    PAGE_TIMEOUT_SECONDS,
    choice_label,
)
from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)


@dataclass
class MatrixChoicePickerPrompt:
    chat_id: str
    message_id: str
    session_key: str
    choices: dict[str, str]
    on_choice_selected: Any
    requester_user_id: str | None = None
    expires_at: float | None = None
    resolved: bool = False
    bot_reaction_events: dict[str, str] = field(default_factory=dict)
    reusable: bool = False
    busy: bool = False
    expiry_handle: Any = None
    metadata: dict = field(default_factory=dict)


def _page(title, choices, reactions, revision=None):
    values = {}
    lines = [title, ""]
    for emoji, choice in zip(reactions, choices):
        # Each reusable page has a new event ID, so old reaction targets cannot
        # select a row from its replacement even when the keycaps repeat.
        key = emoji
        value = str(choice.get("value") or "")
        label = (
            choice_label(choice)
            if revision is not None
            else str(choice.get("label") or value)
        )
        if revision is not None:
            label = re.sub(r"([\\`*_{}\[\]()<>#+.!|])", r"\\\1", label).replace(
                "@", "＠"
            )
            if label.startswith("/"):
                label = "／" + label[1:]
        if choice.get("is_current"):
            label = f"{label} ← current"
        values[key] = value
        lines.append(f"{key} {label}")
    lines.extend(["", "React to choose."])
    return "\n".join(lines), values


def _remove(adapter, prompt):
    prompt.resolved = True
    if adapter._choice_picker_prompts_by_event.get(prompt.message_id) is prompt:
        adapter._choice_picker_prompts_by_event.pop(prompt.message_id, None)
    if prompt.expiry_handle:
        prompt.expiry_handle.cancel()
        prompt.expiry_handle = None


async def _clear_reactions(adapter, prompt):
    events = list(prompt.bot_reaction_events.values())
    prompt.bot_reaction_events.clear()
    for event_id in events:
        try:
            await adapter.redact_message(prompt.chat_id, event_id, "choice page closed")
        except Exception:
            logger.debug("Could not remove choice reaction", exc_info=True)


async def _seed(adapter, prompt):
    for key in prompt.choices:
        if prompt.resolved:
            return
        try:
            event_id = await adapter._send_reaction(
                prompt.chat_id, prompt.message_id, key
            )
            if event_id:
                if prompt.resolved:
                    await adapter.redact_message(
                        prompt.chat_id, str(event_id), "choice page closed"
                    )
                    return
                prompt.bot_reaction_events[key] = str(event_id)
        except Exception:
            logger.debug("Could not add choice reaction", exc_info=True)


async def expire_choice_page(adapter, prompt):
    if prompt.resolved:
        return
    _remove(adapter, prompt)
    await _clear_reactions(adapter, prompt)
    try:
        await adapter.send(
            prompt.chat_id,
            "Menu expired. Run the command again.",
            reply_to=prompt.message_id,
            metadata=prompt.metadata,
        )
    except Exception:
        logger.debug("Could not mark choice page expired", exc_info=True)


def cancel_choice_pages(adapter):
    for prompt in list(
        getattr(adapter, "_choice_picker_prompts_by_event", {}).values()
    ):
        if prompt.reusable:
            _remove(adapter, prompt)


async def send_choice_picker(
    adapter,
    chat_id,
    title,
    choices,
    session_key,
    on_choice_selected,
    metadata,
    reactions,
    *,
    expires_at=None,
):
    if not adapter._client:
        return SendResult(success=False, error="Not connected")
    reusable = (metadata or {}).get("choice_pages") is True
    requester = str((metadata or {}).get("requester_user_id") or "") or None
    if reusable:
        if not requester:
            return SendResult(success=False, error="Requester required")
        try:
            choices = ChoicePage(title, choices).choices
        except ValueError as exc:
            return SendResult(success=False, error=str(exc))
    text, values = _page(title, choices, reactions, 0 if reusable else None)
    if not values:
        return SendResult(success=False, error="No choices")
    if reusable and len(text) > adapter.max_message_length:
        return SendResult(success=False, error="Choice page exceeds the message limit")
    result = await adapter.send(chat_id, text, metadata=metadata)
    if not result.success or not result.message_id:
        return result
    timeout = (
        PAGE_TIMEOUT_SECONDS if reusable else max(adapter._approval_timeout_seconds, 0)
    )
    deadline = expires_at if expires_at is not None else time.monotonic() + timeout
    prompt = MatrixChoicePickerPrompt(
        chat_id=chat_id,
        message_id=result.message_id,
        session_key=session_key,
        choices=values,
        on_choice_selected=on_choice_selected,
        requester_user_id=requester,
        expires_at=deadline,
        reusable=reusable,
        metadata=dict(metadata or {}),
        busy=True,
    )
    adapter._choice_picker_prompts_by_event[result.message_id] = prompt
    if reusable:
        prompt.expiry_handle = asyncio.get_running_loop().call_later(
            max(0, deadline - time.monotonic()),
            lambda: asyncio.create_task(expire_choice_page(adapter, prompt)),
        )
    try:
        await _seed(adapter, prompt)
    except asyncio.CancelledError:
        _remove(adapter, prompt)
        await _clear_reactions(adapter, prompt)
        raise
    prompt.busy = False
    return result


async def handle_choice_reaction(
    adapter, room_id, target, sender, key, event_id, reactions
):
    prompt = adapter._choice_picker_prompts_by_event.get(target)
    if not prompt or prompt.resolved or prompt.busy or room_id != prompt.chat_id:
        return
    if adapter._matrix_prompt_expired(prompt):
        if prompt.reusable:
            await expire_choice_page(adapter, prompt)
        else:
            _remove(adapter, prompt)
        return
    if prompt.reusable and sender != prompt.requester_user_id:
        return
    value = prompt.choices.get(key)
    if value is None and prompt.reusable:
        return
    # Claim before the async authorization check; another tap cannot enter
    # the callback while that check, page rendering or delivery is in flight.
    prompt.busy = True
    try:
        if not await adapter._validate_matrix_prompt_reactor(
            room_id, target, sender, prompt, "choice picker"
        ):
            return
        if value is None:
            await adapter._send_invalid_reaction_feedback(
                room_id,
                target,
                "That reaction is not one of the available choices.",
            )
            return
        if prompt.resolved or adapter._matrix_prompt_expired(prompt):
            await expire_choice_page(adapter, prompt)
            return
        if prompt.reusable:
            await adapter.send_read_receipt(room_id, event_id)
            if (
                adapter._choice_picker_prompts_by_event.get(target) is not prompt
                or prompt.resolved
            ):
                return
            if adapter._matrix_prompt_expired(prompt):
                await expire_choice_page(adapter, prompt)
                return
        else:
            _remove(adapter, prompt)
        result = await prompt.on_choice_selected(room_id, value)
        if isinstance(result, ChoiceProgress):
            if not prompt.reusable:
                raise TypeError("Reusable choice pages were not enabled")
            if prompt.resolved:
                return
            sent = await adapter.send(
                room_id, result.text, reply_to=target, metadata=prompt.metadata
            )
            if not sent.success:
                raise RuntimeError("Choice progress send failed")
            if prompt.resolved or adapter._matrix_prompt_expired(prompt):
                await expire_choice_page(adapter, prompt)
                return
            result = await result.complete()
        if prompt.reusable:
            if prompt.resolved:
                return
            if adapter._matrix_prompt_expired(prompt):
                await expire_choice_page(adapter, prompt)
                return
            if isinstance(result, ChoicePage):
                sent = await send_choice_picker(
                    adapter,
                    room_id,
                    result.title,
                    result.choices,
                    prompt.session_key,
                    prompt.on_choice_selected,
                    prompt.metadata,
                    reactions,
                    expires_at=prompt.expires_at,
                )
                if not sent.success:
                    raise RuntimeError("Choice page send failed")
                _remove(adapter, prompt)
                await _clear_reactions(adapter, prompt)
                return
            if not isinstance(result, str):
                raise TypeError("Choice callback must return text or ChoicePage")
            _remove(adapter, prompt)
            await _clear_reactions(adapter, prompt)
            await adapter.send(
                room_id,
                result or "Menu closed.",
                reply_to=target,
                metadata=prompt.metadata,
            )
        elif result:
            if not isinstance(result, str):
                raise TypeError("Reusable choice pages were not enabled")
            await adapter.send(room_id, result, reply_to=target)
    except asyncio.CancelledError:
        _remove(adapter, prompt)
        if prompt.reusable:
            await _clear_reactions(adapter, prompt)
        raise
    except Exception as exc:
        if prompt.reusable and prompt.resolved:
            return
        _remove(adapter, prompt)
        logger.error("Failed to apply choice from Matrix reaction: %s", exc)
        if prompt.reusable:
            await _clear_reactions(adapter, prompt)
            await adapter.send(
                room_id,
                "Unable to update menu.",
                reply_to=target,
                metadata=prompt.metadata,
            )
        else:
            await adapter.send(
                room_id, f"Failed to apply selection: {exc}", reply_to=target
            )
    finally:
        prompt.busy = False
