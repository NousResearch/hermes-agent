"""Telegram-scoped tool for creating a topic and starting its first agent turn."""

from __future__ import annotations

import asyncio
import json
from typing import Any


TELEGRAM_TOPIC_START_SCHEMA = {
    "name": "telegram_topic_start",
    "description": (
        "Create a new topic in the current Telegram chat and start an agent turn "
        "inside it. Use only when the user explicitly asks for a new topic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic_name": {
                "type": "string",
                "description": "Name of the Telegram topic to create.",
            },
            "prompt": {
                "type": "string",
                "description": "Initial user-authorized prompt to run in the new topic.",
            },
        },
        "required": ["topic_name", "prompt"],
    },
}


def _error(message: str) -> str:
    return json.dumps({"success": False, "error": message})


def _run_on_gateway_loop(coro, loop):
    if loop is None or loop.is_closed() or not loop.is_running():
        coro.close()
        raise RuntimeError("The live gateway event loop is not available")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=30)
    except Exception:
        future.cancel()
        raise


async def _create_topic_and_start(
    adapter: Any,
    *,
    chat_id: str,
    topic_name: str,
    prompt: str,
    user_id: str | None,
    profile: str | None,
) -> dict[str, Any]:
    thread_id = await adapter.create_handoff_thread(chat_id, topic_name)
    if not thread_id:
        return {"success": False, "error": f"Failed to create Telegram topic {topic_name!r}"}

    from gateway.config import Platform
    from gateway.delivery import looks_like_telegram_private_chat_id
    from gateway.session import SessionSource
    from gateway.wake import deliver_wake

    is_private = looks_like_telegram_private_chat_id(chat_id)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=str(chat_id),
        chat_name=topic_name,
        chat_type="dm" if is_private else "forum",
        user_id=str(chat_id) if is_private else (user_id or "system:kickoff"),
        thread_id=str(thread_id),
        profile=profile,
    )
    await deliver_wake(adapter, text=prompt, source=source)
    return {
        "success": True,
        "platform": "telegram",
        "chat_id": str(chat_id),
        "thread_id": str(thread_id),
        "topic_name": topic_name,
        "started": True,
    }


def telegram_topic_start(args: dict[str, Any], **_kwargs: Any) -> str:
    topic_name = str(args.get("topic_name") or "").strip()
    prompt = str(args.get("prompt") or "").strip()
    if not topic_name or not prompt:
        return _error("Both topic_name and prompt are required")

    from gateway.config import Platform
    from gateway.session_context import get_session_env

    if get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower() != "telegram":
        return _error("telegram_topic_start is available only from a Telegram session")
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
    if not chat_id:
        return _error("The current Telegram chat could not be resolved")

    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
    except Exception:
        runner = None
    adapter = runner.adapters.get(Platform.TELEGRAM) if runner is not None else None
    if adapter is None:
        return _error("The live Telegram gateway adapter is not available")

    try:
        result = _run_on_gateway_loop(
            _create_topic_and_start(
                adapter,
                chat_id=chat_id,
                topic_name=topic_name,
                prompt=prompt,
                user_id=get_session_env("HERMES_SESSION_USER_ID", "").strip() or None,
                profile=get_session_env("HERMES_SESSION_PROFILE", "").strip() or None,
            ),
            getattr(runner, "_gateway_loop", None),
        )
    except Exception as exc:
        return _error(f"Create topic and start failed: {exc}")
    return json.dumps(result)


def register_topic_tool(ctx: Any) -> None:
    ctx.register_tool(
        name="telegram_topic_start",
        toolset="telegram",
        schema=TELEGRAM_TOPIC_START_SCHEMA,
        handler=telegram_topic_start,
        emoji="🧵",
    )
