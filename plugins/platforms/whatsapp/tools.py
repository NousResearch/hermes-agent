"""Agent-callable native poll delivery for the WhatsApp platform plugin."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from tools.registry import tool_error
from tools.send_message_senders import _live_adapter
from tools.send_message_tool import _authorize_relay_target, _dispatch_on_gateway_loop

_WHATSAPP_CHAT_ID_RE = re.compile(
    r"^(?:[0-9]+(?:-[0-9]+)?@g\.us|[0-9]+@s\.whatsapp\.net)$"
)


def _clean_poll(args: dict[str, Any]) -> tuple[str, str, list[str], int] | str:
    chat_id = str(args.get("chat_id") or "").strip()
    if not _WHATSAPP_CHAT_ID_RE.fullmatch(chat_id):
        return "chat_id must be a valid WhatsApp chat ID ending in @g.us or @s.whatsapp.net"

    raw_question = args.get("question")
    if not isinstance(raw_question, str):
        return "question must be a string"
    question = raw_question.strip()
    if not question:
        return "question is required"

    raw_options = args.get("options")
    if not isinstance(raw_options, list):
        return "options must be an array"
    if any(not isinstance(option, str) for option in raw_options):
        return "every poll option must be a string"
    options = [option.strip() for option in raw_options]
    if any(not option for option in options):
        return "poll options must not be blank"
    if len(options) < 2:
        return "at least two poll options are required"
    if len(options) > 12:
        return "at most 12 poll options are supported"
    if len({option.casefold() for option in options}) != len(options):
        return "poll options must be unique"

    allow_multiple = args.get("allow_multiple", False)
    if not isinstance(allow_multiple, bool):
        return "allow_multiple must be a boolean"
    selectable_count = len(options) if allow_multiple else 1
    return chat_id, question, options, selectable_count


def _live_poll_adapter():
    from gateway.config import Platform

    return _live_adapter(Platform.WHATSAPP)


def _poll_tool_available() -> bool:
    """Expose the tool only while this profile has a poll-capable live adapter."""
    try:
        _, adapter = _live_poll_adapter()
        return callable(getattr(adapter, "send_poll", None))
    except Exception:  # noqa: BLE001 - capability probes must be non-fatal
        return False


async def _send_poll(
    runner: Any,
    send_poll: Callable[..., Any],
    chat_id: str,
    question: str,
    options: list[str],
    *,
    selectable_count: int,
) -> dict[str, Any]:
    result = await _dispatch_on_gateway_loop(
        runner,
        lambda: send_poll(
            chat_id,
            question,
            options,
            selectable_count=selectable_count,
        ),
        "whatsapp_send_poll: failed to schedule send on gateway loop",
    )
    if isinstance(result, dict):
        return result
    if not result.success:
        return {
            "error": f"WhatsApp poll send failed: {result.error or 'unknown error'}"
        }

    message_id = str(result.message_id or "").strip()
    if not message_id:
        return {
            "error": (
                "WhatsApp reported poll delivery without a message ID; delivery is "
                "unverified. Do not retry automatically."
            )
        }
    return {
        "success": True,
        "platform": "whatsapp",
        "chat_id": chat_id,
        "message_id": message_id,
    }


def whatsapp_send_poll_tool(args: dict[str, Any], **_: Any) -> str:
    cleaned = _clean_poll(args)
    if isinstance(cleaned, str):
        return tool_error(cleaned)
    chat_id, question, options, selectable_count = cleaned

    # Use the same active-profile adapter snapshot for classification and dispatch.
    # A concrete adapter here is native; the relay egress guard therefore does not
    # apply. If there is no native adapter, fail closed after preserving the relay
    # authorization boundary for any future fallback implementation.
    runner, adapter = _live_poll_adapter()
    if adapter is None:
        denial = _authorize_relay_target("whatsapp", chat_id, native_token=None)
        if denial:
            return tool_error(denial)
        return tool_error(
            "WhatsApp polls require a live WhatsApp adapter in the running gateway."
        )

    send_poll = getattr(adapter, "send_poll", None)
    if not callable(send_poll):
        return tool_error("The live WhatsApp adapter does not support native polls.")

    from model_tools import _run_async

    return json.dumps(
        _run_async(
            _send_poll(
                runner,
                send_poll,
                chat_id,
                question,
                options,
                selectable_count=selectable_count,
            )
        )
    )


def register_tools(ctx) -> None:
    ctx.register_tool(
        name="whatsapp_send_poll",
        toolset="whatsapp",
        description="Send a native WhatsApp poll to an exact WhatsApp chat ID.",
        schema={
            "name": "whatsapp_send_poll",
            "description": (
                "Send a native WhatsApp poll through this profile's live adapter. "
                "The destination must be an exact WhatsApp chat ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": (
                            "Exact WhatsApp chat ID, such as "
                            "120363000000000000@g.us for a group."
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": "Poll question.",
                    },
                    "options": {
                        "type": "array",
                        "description": "Two to twelve unique answer options.",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 12,
                    },
                    "allow_multiple": {
                        "type": "boolean",
                        "description": (
                            "When true, voters may select multiple answers; otherwise "
                            "the poll is single-choice."
                        ),
                        "default": False,
                    },
                },
                "required": ["chat_id", "question", "options"],
            },
        },
        handler=whatsapp_send_poll_tool,
        emoji="📊",
        check_fn=_poll_tool_available,
    )
