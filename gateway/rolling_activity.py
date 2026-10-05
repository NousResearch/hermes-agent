"""Bounded, single-bubble gateway activity rendering.

The queue transport remains owned by :mod:`gateway.run_turn_runner`; this module keeps the
rolling-only state and size policy separate from the already-large turn runner.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import suppress
from typing import Any, Optional


WORKING_HEADER = "⏳ Working…"
logger = logging.getLogger("gateway.run")


def terminal_header(
    result: Optional[dict[str, Any]], *, cancelled: bool = False
) -> str:
    """Choose the terminal state for a rolling activity bubble."""
    result = result or {}
    if cancelled or result.get("interrupted"):
        return "⏹️ Stopped"
    if (
        not result
        or result.get("failed")
        or result.get("partial")
        or result.get("completed") is False
        or bool(result.get("error"))
    ):
        return "⚠️ Needs attention"
    return "✅ Completed"


def configure_state(state: Any, grouping: str, initial_message_id: Optional[str]) -> None:
    """Attach rolling fields to the runner's shared editable-progress state."""
    state.progress_msg_id = initial_message_id
    state.rolling_header = WORKING_HEADER if grouping == "rolling" else None
    state.rolling_omitted_count = 0
    state.rolling_state_changed = False


def render_text(state: Any) -> str:
    """Render the current buffer, including the rolling header and omission count."""
    rendered = [str(line) for line in state.progress_lines]
    header = getattr(state, "rolling_header", None)
    omitted = getattr(state, "rolling_omitted_count", 0)
    if header:
        rendered.insert(0, header)
        if omitted:
            rendered.insert(1, f"… {omitted} earlier activities omitted")
    return "\n".join(rendered)


def fit_tail(state: Any) -> None:
    """Drop whole oldest entries until the rolling activity bubble fits."""
    while state.progress_lines and state._progress_len_fn(render_text(state)) > state._PROGRESS_TEXT_LIMIT:
        state.progress_lines.pop(0)
        state.rolling_omitted_count += 1


def absorb_marker(state: Any, raw: Any) -> tuple[bool, Any, bool]:
    """Apply a rolling control marker and return ``(handled, display_value, finish)``."""
    if not isinstance(raw, tuple) or not raw:
        return False, None, False
    marker = raw[0]
    if marker == "__activity_start__":
        return True, state.rolling_header or WORKING_HEADER, False
    if marker in {"__activity_state__", "__activity_finish__"} and len(raw) >= 2:
        state.rolling_header = str(raw[1])
        state.rolling_state_changed = True
        return True, state.rolling_header, marker == "__activity_finish__"
    return False, None, False


def has_renderable_state(state: Any) -> bool:
    return bool(
        state.progress_lines
        or getattr(state, "rolling_state_changed", False)
        or (getattr(state, "rolling_header", None) and getattr(state, "rolling_omitted_count", 0))
    )


async def start_hygiene_activity(runner: Any, event: Any, source: Any, user_config: Any) -> Optional[str]:
    """Seed rolling activity before a potentially long pre-turn compression."""
    from agent.secret_scope import get_secret
    from gateway.display_config import resolve_display_setting, resolve_tool_progress
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.run import _interim_metadata, _platform_config_key

    platform_key = _platform_config_key(source.platform)
    progress_mode, _ = resolve_tool_progress(
        user_config if isinstance(user_config, dict) else {},
        platform_key,
        get_secret("HERMES_TOOL_PROGRESS_MODE"),
    )
    grouping = resolve_display_setting(user_config or {}, platform_key, "tool_progress_grouping")
    adapter = runner._delivery_adapter_for(source)
    adapter_edit = getattr(type(adapter), "edit_message", None) if adapter is not None else None
    if (
        grouping != "rolling"
        or progress_mode in {"off", "log"}
        or runner._get_proxy_url()
        or adapter_edit in (None, BasePlatformAdapter.edit_message)
    ):
        return None
    reply_to = runner._reply_anchor_for_event(event)
    metadata, progress_reply_to, _ = runner._run_agent_progress_threading(source, reply_to, False)
    try:
        result = await adapter.send(
            source.chat_id,
            "⏳ Compressing conversation context…",
            reply_to=progress_reply_to,
            metadata=_interim_metadata(metadata),
        )
    except Exception:
        logger.debug("Session hygiene rolling activity send failed", exc_info=True)
        return None
    return str(result.message_id) if result.success and result.message_id else None


async def continue_hygiene_activity(
    runner: Any, source: Any, message_id: Optional[str], event_message_id: Optional[str]
) -> None:
    """Hand a pre-turn activity bubble to the normal rolling sender."""
    if not message_id:
        return
    adapter = runner._delivery_adapter_for(source)
    metadata, _, _ = runner._run_agent_progress_threading(source, event_message_id, False)
    kwargs = {"chat_id": source.chat_id, "message_id": message_id, "content": WORKING_HEADER}
    try:
        params = inspect.signature(adapter.edit_message).parameters
        if "metadata" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            kwargs["metadata"] = metadata
    except (TypeError, ValueError):
        pass
    try:
        await adapter.edit_message(**kwargs)
    except Exception:
        logger.debug("Session hygiene rolling activity edit failed", exc_info=True)


async def finish_turn_activity(turn_ctx: Any, progress_task: Any, logger: Any) -> None:
    """Flush the terminal header before final delivery or a queued rolling turn."""
    if not (
        progress_task
        and turn_ctx.progress_grouping == "rolling"
        and turn_ctx.tool_progress_enabled
        and turn_ctx.progress_queue is not None
    ):
        return
    current = asyncio.current_task()
    header = terminal_header(turn_ctx.activity_result, cancelled=bool(current and current.cancelling()))
    turn_ctx.progress_finish_header[0] = header
    turn_ctx.progress_queue.put(("__activity_finish__", header))
    try:
        await asyncio.wait_for(asyncio.shield(progress_task), timeout=5.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        progress_task.cancel()
        with suppress(asyncio.CancelledError):
            await progress_task
    except Exception:
        logger.warning("Rolling activity terminal flush failed", exc_info=True)
