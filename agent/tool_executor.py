"""Tool call execution with frozen configuration.

Extracted from AIAgent._execute_tool_calls (run_agent.py L1397-1604).
Executes tool calls from assistant messages, routing agent-loop tools
directly and others through the tool registry.
"""

from dataclasses import dataclass
import json
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

from agent.display import (
    KawaiiSpinner,
    build_tool_preview as _build_tool_preview,
    get_cute_tool_message as _get_cute_tool_message_impl,
)

# NOTE: handle_function_call is imported lazily inside execute_tool_calls()
# to break a circular import: agent.tool_executor -> model_tools ->
# agent.tool_executor (for AGENT_LOOP_TOOLS).

# ---------------------------------------------------------------------------
# Canonical set of tools handled directly by the agent loop
# (not dispatched through the tool registry).
# Includes 'clarify' -- previously missing from model_tools._AGENT_LOOP_TOOLS
# (bug B1/SC-7).
# ---------------------------------------------------------------------------
AGENT_LOOP_TOOLS: frozenset = frozenset({
    "todo", "memory", "session_search", "clarify", "delegate_task",
})

MAX_TOOL_RESULT_CHARS = 100_000


@dataclass(frozen=True)
class ToolExecConfig:
    """Immutable configuration for tool execution.

    All fields correspond to AIAgent attributes that _execute_tool_calls
    previously read via ``self.X``.
    """

    todo_store: Any
    memory_store: Any
    session_db: Any
    clarify_callback: Optional[Callable]
    tool_delay: float
    tool_progress_callback: Optional[Callable]
    quiet_mode: bool
    verbose_logging: bool
    log_prefix: str
    log_prefix_chars: int


def execute_tool_calls(
    config: ToolExecConfig,
    assistant_message,
    messages: List[Dict],
    effective_task_id: str,
    *,
    is_interrupted: Callable[[], bool],
    log_msg_to_db: Callable[[Dict], None],
    on_tool_executed: Callable[[str], None],
    parent_agent: Optional[Any] = None,
) -> None:
    """Execute all tool calls from an assistant message.

    Appends tool result messages to *messages*.
    Checks *is_interrupted()* before each tool call.
    Calls *log_msg_to_db(msg)* after each appended message.
    Calls *on_tool_executed(tool_name)* after each successful tool execution.

    Parameters
    ----------
    config:
        Frozen configuration containing stores, callbacks, and display prefs.
    assistant_message:
        OpenAI-style assistant message whose ``.tool_calls`` are processed.
    messages:
        Conversation message list; tool results are appended in-place.
    effective_task_id:
        Task ID passed to ``handle_function_call`` for session isolation.
    is_interrupted:
        Callable returning ``True`` when the user has requested a stop.
    log_msg_to_db:
        Persistence callback invoked for every message appended to *messages*.
    on_tool_executed:
        Notification callback invoked with the tool name after each
        successful (non-skipped) tool execution.
    parent_agent:
        The parent ``AIAgent`` instance, passed to ``delegate_task``.
        If ``None``, ``delegate_task`` returns an error JSON.
    """
    # Lazy import to break circular dependency:
    # agent.tool_executor <-> model_tools
    from model_tools import handle_function_call

    for i, tool_call in enumerate(assistant_message.tool_calls, 1):
        # SAFETY: check interrupt BEFORE starting each tool.
        # If the user sent "stop" during a previous tool's execution,
        # do NOT start any more tools -- skip them all immediately.
        if is_interrupted():
            remaining_calls = assistant_message.tool_calls[i - 1:]
            if remaining_calls:
                print(
                    f"{config.log_prefix}⚡ Interrupt: skipping "
                    f"{len(remaining_calls)} tool call(s)"
                )
            for skipped_tc in remaining_calls:
                skip_msg = {
                    "role": "tool",
                    "content": "[Tool execution cancelled - user interrupted]",
                    "tool_call_id": skipped_tc.id,
                }
                messages.append(skip_msg)
                log_msg_to_db(skip_msg)
            break

        function_name = tool_call.function.name

        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logging.warning(f"Unexpected JSON error after validation: {e}")
            function_args = {}

        if not config.quiet_mode:
            args_str = json.dumps(function_args, ensure_ascii=False)
            args_preview = (
                args_str[: config.log_prefix_chars] + "..."
                if len(args_str) > config.log_prefix_chars
                else args_str
            )
            print(
                f"  📞 Tool {i}: {function_name}"
                f"({list(function_args.keys())}) - {args_preview}"
            )

        if config.tool_progress_callback:
            try:
                preview = _build_tool_preview(function_name, function_args)
                config.tool_progress_callback(function_name, preview)
            except Exception as cb_err:
                logging.debug(f"Tool progress callback error: {cb_err}")

        tool_start_time = time.time()

        # ----- agent-loop tools (routed directly) -----
        if function_name == "todo":
            from tools.todo_tool import todo_tool as _todo_tool

            function_result = _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=config.todo_store,
            )
            tool_duration = time.time() - tool_start_time
            if config.quiet_mode:
                print(
                    f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}"
                )

        elif function_name == "session_search" and config.session_db:
            from tools.session_search_tool import session_search as _session_search

            function_result = _session_search(
                query=function_args.get("query", ""),
                role_filter=function_args.get("role_filter"),
                limit=function_args.get("limit", 3),
                db=config.session_db,
            )
            tool_duration = time.time() - tool_start_time
            if config.quiet_mode:
                print(
                    f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}"
                )

        elif function_name == "memory":
            from tools.memory_tool import memory_tool as _memory_tool

            function_result = _memory_tool(
                action=function_args.get("action"),
                target=function_args.get("target", "memory"),
                content=function_args.get("content"),
                old_text=function_args.get("old_text"),
                store=config.memory_store,
            )
            tool_duration = time.time() - tool_start_time
            if config.quiet_mode:
                print(
                    f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}"
                )

        elif function_name == "clarify":
            from tools.clarify_tool import clarify_tool as _clarify_tool

            function_result = _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=config.clarify_callback,
            )
            tool_duration = time.time() - tool_start_time
            if config.quiet_mode:
                print(
                    f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}"
                )

        elif function_name == "delegate_task":
            # Guard: parent_agent=None means delegation is not available
            # (e.g. standalone tool_executor usage without a parent agent).
            if parent_agent is None:
                function_result = json.dumps({
                    "error": "delegate_task unavailable: no parent agent"
                })
                tool_duration = time.time() - tool_start_time
            else:
                from tools.delegate_tool import delegate_task as _delegate_task

                tasks_arg = function_args.get("tasks")
                if tasks_arg and isinstance(tasks_arg, list):
                    spinner_label = f"🔀 delegating {len(tasks_arg)} tasks"
                else:
                    goal_preview = (function_args.get("goal") or "")[:30]
                    spinner_label = (
                        f"🔀 {goal_preview}" if goal_preview else "🔀 delegating"
                    )
                spinner = None
                if config.quiet_mode:
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    spinner = KawaiiSpinner(
                        f"{face} {spinner_label}", spinner_type="dots"
                    )
                    spinner.start()
                _delegate_result = None
                try:
                    function_result = _delegate_task(
                        goal=function_args.get("goal"),
                        context=function_args.get("context"),
                        toolsets=function_args.get("toolsets"),
                        tasks=tasks_arg,
                        model=function_args.get("model"),
                        max_iterations=function_args.get("max_iterations"),
                        parent_agent=parent_agent,
                    )
                    _delegate_result = function_result
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(
                        "delegate_task",
                        function_args,
                        tool_duration,
                        result=_delegate_result,
                    )
                    if spinner:
                        spinner.stop(cute_msg)
                    elif config.quiet_mode:
                        print(f"  {cute_msg}")

        # ----- registry tools (quiet_mode with spinner) -----
        elif config.quiet_mode:
            face = random.choice(KawaiiSpinner.KAWAII_WAITING)
            tool_emoji_map = {
                "web_search": "🔍",
                "web_extract": "📄",
                "web_crawl": "🕸️",
                "terminal": "💻",
                "process": "⚙️",
                "read_file": "📖",
                "write_file": "✍️",
                "patch": "🔧",
                "search_files": "🔎",
                "browser_navigate": "🌐",
                "browser_snapshot": "📸",
                "browser_click": "👆",
                "browser_type": "⌨️",
                "browser_scroll": "📜",
                "browser_back": "◀️",
                "browser_press": "⌨️",
                "browser_close": "🚪",
                "browser_get_images": "🖼️",
                "browser_vision": "👁️",
                "image_generate": "🎨",
                "text_to_speech": "🔊",
                "vision_analyze": "👁️",
                "mixture_of_agents": "🧠",
                "skills_list": "📚",
                "skill_view": "📚",
                "schedule_cronjob": "⏰",
                "list_cronjobs": "⏰",
                "remove_cronjob": "⏰",
                "send_message": "📨",
                "todo": "📋",
                "memory": "🧠",
                "session_search": "🔍",
                "clarify": "❓",
                "execute_code": "🐍",
                "delegate_task": "🔀",
            }
            emoji = tool_emoji_map.get(function_name, "⚡")
            preview = _build_tool_preview(function_name, function_args) or function_name
            if len(preview) > 30:
                preview = preview[:27] + "..."
            spinner = KawaiiSpinner(
                f"{face} {emoji} {preview}", spinner_type="dots"
            )
            spinner.start()
            _spinner_result = None
            try:
                function_result = handle_function_call(
                    function_name, function_args, effective_task_id
                )
                _spinner_result = function_result
            finally:
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl(
                    function_name,
                    function_args,
                    tool_duration,
                    result=_spinner_result,
                )
                spinner.stop(cute_msg)

        # ----- registry tools (verbose / non-quiet mode) -----
        else:
            function_result = handle_function_call(
                function_name, function_args, effective_task_id
            )
            tool_duration = time.time() - tool_start_time

        result_preview = (
            function_result[:200] if len(function_result) > 200 else function_result
        )

        if config.verbose_logging:
            logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
            logging.debug(f"Tool result preview: {result_preview}...")

        # Guard against tools returning absurdly large content that would
        # blow up the context window. 100K chars ~ 25K tokens -- generous
        # enough for any reasonable tool output but prevents catastrophic
        # context explosions (e.g. accidental base64 image dumps).
        if len(function_result) > MAX_TOOL_RESULT_CHARS:
            original_len = len(function_result)
            function_result = (
                function_result[:MAX_TOOL_RESULT_CHARS]
                + f"\n\n[Truncated: tool response was {original_len:,} chars, "
                f"exceeding the {MAX_TOOL_RESULT_CHARS:,} char limit]"
            )

        tool_msg = {
            "role": "tool",
            "content": function_result,
            "tool_call_id": tool_call.id,
        }
        messages.append(tool_msg)
        log_msg_to_db(tool_msg)

        # Notify caller that this tool executed successfully.
        on_tool_executed(function_name)

        if not config.quiet_mode:
            response_preview = (
                function_result[: config.log_prefix_chars] + "..."
                if len(function_result) > config.log_prefix_chars
                else function_result
            )
            print(
                f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}"
            )

        # Post-tool interrupt check: if interrupted after this tool, skip rest.
        if is_interrupted() and i < len(assistant_message.tool_calls):
            remaining = len(assistant_message.tool_calls) - i
            print(
                f"{config.log_prefix}⚡ Interrupt: skipping "
                f"{remaining} remaining tool call(s)"
            )
            for skipped_tc in assistant_message.tool_calls[i:]:
                skip_msg = {
                    "role": "tool",
                    "content": "[Tool execution skipped - user sent a new message]",
                    "tool_call_id": skipped_tc.id,
                }
                messages.append(skip_msg)
                log_msg_to_db(skip_msg)
            break

        if config.tool_delay > 0 and i < len(assistant_message.tool_calls):
            time.sleep(config.tool_delay)
