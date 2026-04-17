"""Shared progress/status callback helpers for gateway foreground agent runs."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from gateway.config import Platform


@dataclass(slots=True)
class GatewayProgressRuntime:
    """Callback and progress-delivery wiring for one gateway foreground turn."""

    progress_mode: str
    tool_progress_enabled: bool
    progress_queue: queue.Queue[Any] | None
    thread_id: str | None
    thread_metadata: dict[str, Any] | None
    progress_callback: Callable[..., None] | None
    send_progress_messages: Callable[[], Awaitable[None]]
    step_callback: Callable[[int, list[Any]], None] | None
    status_callback: Callable[[str, str], None] | None
    status_adapter: Any
    status_chat_id: str | None


def _resolve_gateway_progress_mode(
    *,
    user_config: dict[str, Any],
) -> str:
    raw_tool_progress = user_config.get("display", {}).get("tool_progress")
    if raw_tool_progress is False:
        raw_tool_progress = "off"
    return raw_tool_progress or os.getenv("HERMES_TOOL_PROGRESS_MODE") or "all"


def _resolve_gateway_progress_threading(
    *,
    source: Any,
    event_message_id: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    if source.platform == Platform.SLACK:
        thread_id = source.thread_id or event_message_id
    else:
        thread_id = source.thread_id
    return thread_id, {"thread_id": thread_id} if thread_id else None


def _create_gateway_progress_callback(
    *,
    progress_mode: str,
    progress_queue: queue.Queue[Any] | None,
) -> Callable[..., None] | None:
    if progress_queue is None:
        return None

    last_tool = [None]
    last_progress_msg = [None]
    repeat_count = [0]

    def _progress_callback(
        event_type: str,
        tool_name: str | None = None,
        preview: str | None = None,
        args: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        if event_type not in ("tool.started",):
            return

        if progress_mode == "new" and tool_name == last_tool[0]:
            return
        last_tool[0] = tool_name

        from agent.display import get_tool_emoji, get_tool_preview_max_len

        emoji = get_tool_emoji(tool_name, default="⚙️")
        if progress_mode == "verbose":
            if args:
                preview_len = get_tool_preview_max_len()
                args_str = json.dumps(args, ensure_ascii=False, default=str)
                cap = preview_len if preview_len > 0 else 200
                if len(args_str) > cap:
                    args_str = args_str[: cap - 3] + "..."
                msg = f"{emoji} {tool_name}({list(args.keys())})\n{args_str}"
            elif preview:
                msg = f'{emoji} {tool_name}: "{preview}"'
            else:
                msg = f"{emoji} {tool_name}..."
            progress_queue.put(msg)
            return

        if preview:
            preview_len = get_tool_preview_max_len()
            cap = preview_len if preview_len > 0 else 40
            if len(preview) > cap:
                preview = preview[: cap - 3] + "..."
            msg = f'{emoji} {tool_name}: "{preview}"'
        else:
            msg = f"{emoji} {tool_name}..."

        if msg == last_progress_msg[0]:
            repeat_count[0] += 1
            progress_queue.put(("__dedup__", msg, repeat_count[0]))
            return

        last_progress_msg[0] = msg
        repeat_count[0] = 0
        progress_queue.put(msg)

    return _progress_callback


def _create_gateway_progress_sender(
    *,
    progress_queue: queue.Queue[Any] | None,
    adapter: Any,
    chat_id: str | None,
    metadata: dict[str, Any] | None,
    logger: Any,
) -> Callable[[], Awaitable[None]]:
    async def _send_progress_messages() -> None:
        if not progress_queue or not adapter:
            return

        progress_lines: list[str] = []
        progress_msg_id = None
        can_edit = True
        last_edit_ts = 0.0
        progress_edit_interval = 1.5

        while True:
            try:
                raw = progress_queue.get_nowait()
                if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                    _, base_msg, count = raw
                    if progress_lines:
                        progress_lines[-1] = f"{base_msg} (×{count + 1})"
                    msg = progress_lines[-1] if progress_lines else base_msg
                else:
                    msg = raw
                    progress_lines.append(msg)

                now = time.monotonic()
                remaining = progress_edit_interval - (now - last_edit_ts)
                if remaining > 0:
                    await asyncio.sleep(remaining)
                    continue

                if can_edit and progress_msg_id is not None:
                    full_text = "\n".join(progress_lines)
                    result = await adapter.edit_message(
                        chat_id=chat_id,
                        message_id=progress_msg_id,
                        content=full_text,
                    )
                    if not result.success:
                        err = (getattr(result, "error", "") or "").lower()
                        if "flood" in err or "retry after" in err:
                            logger.info(
                                "[%s] Progress edits disabled due to flood control",
                                adapter.name,
                            )
                        can_edit = False
                        await adapter.send(chat_id=chat_id, content=msg, metadata=metadata)
                else:
                    if can_edit:
                        full_text = "\n".join(progress_lines)
                        result = await adapter.send(
                            chat_id=chat_id,
                            content=full_text,
                            metadata=metadata,
                        )
                    else:
                        result = await adapter.send(
                            chat_id=chat_id,
                            content=msg,
                            metadata=metadata,
                        )
                    if result.success and result.message_id:
                        progress_msg_id = result.message_id

                last_edit_ts = time.monotonic()
                await asyncio.sleep(0.3)
                await adapter.send_typing(chat_id, metadata=metadata)

            except queue.Empty:
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                while not progress_queue.empty():
                    try:
                        raw = progress_queue.get_nowait()
                        if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                            _, base_msg, count = raw
                            if progress_lines:
                                progress_lines[-1] = f"{base_msg} (×{count + 1})"
                        else:
                            progress_lines.append(raw)
                    except Exception:
                        break
                if can_edit and progress_lines and progress_msg_id:
                    full_text = "\n".join(progress_lines)
                    try:
                        await adapter.edit_message(
                            chat_id=chat_id,
                            message_id=progress_msg_id,
                            content=full_text,
                        )
                    except Exception:
                        pass
                return
            except Exception as exc:
                logger.error("Progress message error: %s", exc)
                await asyncio.sleep(1)

    return _send_progress_messages


def _create_gateway_step_callback(
    *,
    hooks_ref: Any,
    loop_for_step: Any,
    source: Any,
    session_id: str,
    logger: Any,
) -> Callable[[int, list[Any]], None] | None:
    if not getattr(hooks_ref, "loaded_hooks", False):
        return None

    def _step_callback_sync(iteration: int, prev_tools: list[Any]) -> None:
        try:
            names: list[str] = []
            for tool in (prev_tools or []):
                if isinstance(tool, dict):
                    names.append(tool.get("name") or "")
                else:
                    names.append(str(tool))
            asyncio.run_coroutine_threadsafe(
                hooks_ref.emit(
                    "agent:step",
                    {
                        "platform": source.platform.value if source.platform else "",
                        "user_id": source.user_id,
                        "session_id": session_id,
                        "iteration": iteration,
                        "tool_names": names,
                        "tools": prev_tools,
                    },
                ),
                loop_for_step,
            )
        except Exception as exc:
            logger.debug("agent:step hook error: %s", exc)

    return _step_callback_sync


def _create_gateway_status_callback(
    *,
    adapter: Any,
    chat_id: str | None,
    metadata: dict[str, Any] | None,
    source: Any,
    loop_for_step: Any,
    logger: Any,
    should_forward_status: Callable[[Any, str, str], bool],
) -> Callable[[str, str], None]:
    def _status_callback_sync(event_type: str, message: str) -> None:
        if not adapter:
            return
        if not should_forward_status(source, event_type, message):
            return
        try:
            asyncio.run_coroutine_threadsafe(
                adapter.send(
                    chat_id,
                    message,
                    metadata=metadata,
                ),
                loop_for_step,
            )
        except Exception as exc:
            logger.debug("status_callback error (%s): %s", event_type, exc)

    return _status_callback_sync


def build_gateway_progress_runtime(
    *,
    user_config: dict[str, Any],
    source: Any,
    event_message_id: str | None,
    adapter: Any,
    hooks_ref: Any,
    loop_for_step: Any,
    session_id: str,
    logger: Any,
    should_forward_status: Callable[[Any, str, str], bool],
) -> GatewayProgressRuntime:
    """Build the per-turn callback and progress-delivery runtime for _run_agent()."""

    progress_mode = _resolve_gateway_progress_mode(user_config=user_config)
    tool_progress_enabled = (
        progress_mode != "off" and source.platform != Platform.WEBHOOK
    )
    progress_queue = queue.Queue() if tool_progress_enabled else None
    thread_id, thread_metadata = _resolve_gateway_progress_threading(
        source=source,
        event_message_id=event_message_id,
    )

    return GatewayProgressRuntime(
        progress_mode=progress_mode,
        tool_progress_enabled=tool_progress_enabled,
        progress_queue=progress_queue,
        thread_id=thread_id,
        thread_metadata=thread_metadata,
        progress_callback=_create_gateway_progress_callback(
            progress_mode=progress_mode,
            progress_queue=progress_queue,
        ),
        send_progress_messages=_create_gateway_progress_sender(
            progress_queue=progress_queue,
            adapter=adapter,
            chat_id=source.chat_id,
            metadata=thread_metadata,
            logger=logger,
        ),
        step_callback=_create_gateway_step_callback(
            hooks_ref=hooks_ref,
            loop_for_step=loop_for_step,
            source=source,
            session_id=session_id,
            logger=logger,
        ),
        status_callback=_create_gateway_status_callback(
            adapter=adapter,
            chat_id=source.chat_id,
            metadata=thread_metadata,
            source=source,
            loop_for_step=loop_for_step,
            logger=logger,
            should_forward_status=should_forward_status,
        ),
        status_adapter=adapter,
        status_chat_id=source.chat_id,
    )
