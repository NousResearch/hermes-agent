"""Plugin Adaptive Cards: the contract behind ``ctx.register_card_action_handler`` and :class:`CardReply`.

A plugin shows a card two ways: its slash command returns ``CardReply(card, text)``, or a click
handler answers with a card that replaces the clicked one. Adapters that render Adaptive Cards
implement ``send_card_message(chat_id, card, fallback_text, metadata) -> SendResult`` (Teams; like
``send`` it reports a failure in the result instead of raising); every other platform, and the
CLI/TUI/Desktop, shows ``CardReply.text``.

Routing: a button is an ``Action.Execute`` whose ``data`` carries ``{"hermes_plugin_action":
"<namespace>", ...}``. The client merges the card's input values into that same ``data`` on a click,
so when Hermes ships a plugin card it moves each such button's own keys under ``hermes_plugin_data``;
the click then splits back into :attr:`CardActionEvent.action_data` (the button's keys) and
:attr:`CardActionEvent.inputs` (what the user typed or picked). A card shipped another way has no
split: every key arrives in ``action_data`` and ``inputs`` is empty.

Security: the adapter's allowlist gate runs before any handler, but it only says the clicker may use
this bot. The handler must authorize :attr:`CardActionEvent.user_id` itself and treat
``action_data``/``inputs`` as client-controlled (re-validate ids and state server-side). Logs name
the plugin, namespace and sizes, never card content, button data or inputs.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROUTING_KEY = "hermes_plugin_action"
BUTTON_DATA_KEY = "hermes_plugin_data"
# A larger card is answered with text: Teams rejects activities over ~28 KB, the same budget as its
# text messages (``TeamsAdapter.MAX_MESSAGE_LENGTH``).
MAX_CARD_BYTES = 28_000


@dataclass(frozen=True)
class CardActionEvent:
    """One click, as the owning plugin's handler receives it."""

    platform: str
    namespace: str
    action_data: dict
    inputs: dict
    user_id: str  # the platform's verified id of the clicker (Teams: AAD object id)
    user_name: str
    chat_id: str
    message_id: str  # the clicked card's message id; "" when the platform does not say
    session_key: str  # the chat's session key, derived as its messages derive it; "" off the gateway


@dataclass(frozen=True)
class CardActionResult:
    """A click's answer: exactly one of ``card`` (replaces the clicked card) or ``text`` (shown briefly)."""

    card: Optional[dict] = None
    text: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.card is None) == (self.text is None):
            raise ValueError("CardActionResult takes exactly one of card or text")
        if self.card is not None:
            _require_card_json(self.card, "CardActionResult.card")
        if self.text is not None and not (isinstance(self.text, str) and self.text.strip()):
            raise ValueError("CardActionResult.text must be a non-empty string")


@dataclass(frozen=True)
class CardReply:
    """A plugin slash command's card answer. ``text`` is required: surfaces without cards show it."""

    card: dict
    text: str

    def __post_init__(self) -> None:
        _require_card_json(self.card, "CardReply.card")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("CardReply.text is required")


def _require_card_json(card: Any, what: str) -> None:
    """Refuse a card that is not a JSON object here, in the plugin's own call, not at delivery."""
    if not isinstance(card, dict):
        raise TypeError(f"{what} must be a dict (Adaptive Card JSON)")
    card_size(card)


def action_failed_text() -> str:
    from agent.i18n import t
    return t("gateway.plugin_card.action_failed")


def card_size(card: dict) -> int:
    """UTF-8 bytes of *card* as JSON (``TypeError``/``ValueError`` when it is not JSON)."""
    return len(json.dumps(card, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def prepare_card(card: dict) -> dict:
    """A copy of *card* in which every plugin button keeps its own keys under :data:`BUTTON_DATA_KEY`."""
    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        out = {key: walk(value) for key, value in node.items()}
        data = out.get("data")
        if (out.get("type") == "Action.Execute" and isinstance(data, dict)
                and isinstance(data.get(ROUTING_KEY), str) and BUTTON_DATA_KEY not in data):
            out["data"] = {ROUTING_KEY: data[ROUTING_KEY],
                           BUTTON_DATA_KEY: {k: v for k, v in data.items() if k != ROUTING_KEY}}
        return out

    return walk(card)


def split_action_data(data: dict) -> tuple[dict, dict]:
    """``(action_data, inputs)`` of a clicked button's ``data``, routing keys removed."""
    rest = {k: v for k, v in data.items() if k not in (ROUTING_KEY, BUTTON_DATA_KEY)}
    button = data.get(BUTTON_DATA_KEY)
    return (dict(button), rest) if isinstance(button, dict) else (rest, {})


def oversized_card(card: dict, owner: str, what: str) -> Optional[str]:
    """The text a card over :data:`MAX_CARD_BYTES` is answered with, else None (logs the size only)."""
    size = card_size(card)
    if size <= MAX_CARD_BYTES:
        return None
    logger.warning("%s from %s is %d bytes (limit %d); sending text instead", what, owner, size, MAX_CARD_BYTES)
    from agent.i18n import t
    return t("gateway.plugin_card.card_too_large", size=size, limit=MAX_CARD_BYTES)


def as_card_action_result(value: Any) -> CardActionResult:
    """A click handler's return value as a :class:`CardActionResult`: a ``dict`` is a card, a ``str``
    a text; anything else raises ``TypeError`` (``ValueError`` for an empty text)."""
    if isinstance(value, CardActionResult):
        return value
    if isinstance(value, dict):
        return CardActionResult(card=value)
    if isinstance(value, str):
        return CardActionResult(text=value)
    raise TypeError(f"expected CardActionResult, dict or str, got {type(value).__name__}")


def plugin_command_reply(result: Any) -> str | CardReply | None:
    """A plugin slash command's return value as delivered: a :class:`CardReply` as is (one that crossed
    the plugin-host wire arrives as a field record and is rebuilt), anything else truthy as ``str``."""
    if isinstance(result, dict):
        from hermes_cli.plugin_host_wire import Record
        if isinstance(result, Record) and set(result) == {"card", "text"}:
            return CardReply(card=result["card"], text=result["text"])
    if isinstance(result, CardReply):
        return result
    return str(result) if result else None


def plugin_reply_text(result: Any) -> Optional[str]:
    """What a surface without cards shows for a plugin command's return value."""
    reply = plugin_command_reply(result)
    return reply.text if isinstance(reply, CardReply) else reply


async def deliver_card_reply(runner: Any, source: Any, reply: str | CardReply | None, *,
                             command: str) -> Optional[str]:
    """Send a :class:`CardReply` as a card through the adapter answering *source* and return None;
    return the text to reply with instead when that adapter has no ``send_card_message``, the card is
    too large, or the send failed. A non-card reply passes through."""
    if not isinstance(reply, CardReply):
        return reply
    adapter = runner._delivery_adapter_for(source)
    send_card = getattr(adapter, "send_card_message", None)
    if send_card is None:
        return reply.text
    card = prepare_card(reply.card)
    if oversized_card(card, f"/{command}", "Plugin command card") is not None:
        return reply.text
    result = await send_card(source.chat_id, card, reply.text, metadata=runner._thread_metadata_for_source(source))
    if getattr(result, "success", False):
        return None
    logger.warning("Plugin command /%s: card send failed; replying with its text", command)
    return reply.text


async def _call_handler(runner: Any, source: Any, callback: Any, event: CardActionEvent) -> Any:
    """Run *callback* the way a plugin slash command runs: the chat's ``HERMES_SESSION_*`` bound,
    ``ctx.post_message`` aimed at the chat, sync handlers on the gateway pool (contextvars carried),
    never the loop thread (#105279)."""
    if runner is None:
        is_async = inspect.iscoroutinefunction(callback)
        result = await (callback(event) if is_async else asyncio.to_thread(callback, event))
        return await result if inspect.isawaitable(result) else result
    from gateway.plugin_messages import bind_live_source
    context = await runner._hm_plugin_session_context(source)
    with runner._session_env_scope(context), bind_live_source(source):
        if inspect.iscoroutinefunction(callback):
            return await callback(event)
        result = await runner._run_in_executor_with_context(callback, event)
        return await result if inspect.isawaitable(result) else result


async def run_card_action(runner: Any, source: Any, namespace: Any, data: dict, *, message_id: str,
                          timeout: float) -> Optional[CardActionResult]:
    """Answer one click with the handler of the plugin that registered *namespace* in the chat's
    profile, under that profile's scope; None when no plugin did. A handler that raises, outlives
    *timeout* or answers with something else gets the failure text (a sync one keeps its pool thread
    until it returns). A card is sized here; the adapter converts it."""
    scope = runner._async_profile_scope_for_source(source) if runner is not None else contextlib.nullcontext()
    async with scope:
        from hermes_cli.plugins_actions import card_action_handler
        entry = card_action_handler(namespace) if isinstance(namespace, str) else None
        if entry is None:
            return None
        action_data, inputs = split_action_data(data)
        event = CardActionEvent(
            platform=source.platform.value, namespace=namespace, action_data=action_data, inputs=inputs,
            user_id=source.user_id or "", user_name=source.user_name or "", chat_id=source.chat_id,
            message_id=message_id or "",
            session_key=runner._session_key_for_source(source) if runner is not None else "")
        owner = f"plugin {entry['plugin']} ({namespace})"
        try:
            result = as_card_action_result(
                await asyncio.wait_for(_call_handler(runner, source, entry["callback"], event), timeout))
            if result.card is None:
                return result
            card = prepare_card(result.card)
            too_large = oversized_card(card, owner, "Replacement card")
            return CardActionResult(text=too_large) if too_large else CardActionResult(card=card)
        except Exception as exc:  # health: allow BLE001 -- a plugin click handler's failure is answered, never raised into the invoke
            logger.warning("Card action handler of %s failed: %s", owner, type(exc).__name__)
            logger.debug("Card action handler of %s failed", owner, exc_info=True)
            return CardActionResult(text=action_failed_text())
