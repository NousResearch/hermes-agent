"""Tool-call execution dispatch helpers (run_agent.py shard s5, c16).

Extracted verbatim from run_agent.py (wave 1, shard s5, cluster c16, 28
move-votes).  Method bodies are character-for-character copies; only this
header and the import block are new.  ``logger`` is bound to the same logger
name as run_agent's module logger so log records keep their origin.

Per-request helpers are imported lazily inside the methods
(``agent.tool_executor``, ``agent.tool_dispatch_helpers``,
``agent.chat_completion_helpers``, ``agent.agent_runtime_helpers``,
``tools.delegate_tool``); only ``get_active_env`` (used by
``_execute_tool_calls``) needs a module-level import.  Instance state
referenced via ``self.`` (``_executing_tools``, ``_delegate_depth``) stays on
``AIAgent`` and resolves through the MRO.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from tools.terminal_tool import get_active_env

logger = logging.getLogger("run_agent")


class ToolExecutionMixin:
    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls from the assistant message and append results to messages.

        The segment planner splits the batch into maximal contiguous runs of
        parallel-safe calls (read-only tools, non-overlapping file targets,
        opted-in MCP tools) separated by sequential barriers (interactive,
        unsafe, or unrecognized tools). Homogeneous batches keep their
        original single-path dispatch; mixed batches execute segment by
        segment in emission order so safe subsets still run concurrently
        while side-effect ordering is preserved.
        """
        tool_calls = assistant_message.tool_calls

        # Allow _vprint during tool execution even with stream consumers
        self._executing_tools = True
        try:
            if len(tool_calls) <= 1:
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            from agent.tool_dispatch_helpers import _plan_tool_batch_segments
            _active_env = get_active_env(effective_task_id)
            _exec_cwd = Path(_active_env.cwd) if _active_env is not None and _active_env.cwd else None
            segments = _plan_tool_batch_segments(tool_calls, execution_cwd=_exec_cwd)

            if len(segments) == 1:
                kind = segments[0][0]
                if kind == "parallel":
                    return self._execute_tool_calls_concurrent(
                        assistant_message, messages, effective_task_id, api_call_count
                    )
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            from agent.tool_executor import execute_tool_calls_segmented
            return execute_tool_calls_segmented(
                self, assistant_message, messages, effective_task_id, api_call_count,
                segments=segments,
            )
        finally:
            self._executing_tools = False

    def _dispatch_delegate_task(self, function_args: dict) -> str:
        """Single call site for delegate_task dispatch.

        New DELEGATE_TASK_SCHEMA fields only need to be added here to reach all
        invocation paths (concurrent, sequential, inline).
        """
        from tools.delegate_tool import (
            _strip_model_hidden_task_fields,
            delegate_task as _delegate_task,
        )
        # Delegations from the top-level MODEL always run in the background —
        # the model does not get to choose. delegate_task returns immediately
        # with a handle (one per task) and each subagent's result re-enters the
        # conversation as a new message when it finishes. This applies to BOTH
        # a single task and a fan-out batch (each task becomes its own
        # independent background subagent). The one exception:
        #   - A delegation from an ORCHESTRATOR SUBAGENT (depth > 0) stays
        #     synchronous: the orchestrator needs its workers' results within
        #     its own turn to compose a summary, and a subagent doesn't own the
        #     gateway session the async result would route back to.
        # The schema-level `background` param is intentionally ignored here.
        _is_subagent = getattr(self, "_delegate_depth", 0) > 0
        return _delegate_task(
            goal=function_args.get("goal"),
            context=function_args.get("context"),
            tasks=_strip_model_hidden_task_fields(function_args.get("tasks")),
            max_iterations=function_args.get("max_iterations"),
            role=function_args.get("role"),
            background=(not _is_subagent),
            parent_agent=self,
        )

    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str,
                     tool_call_id: Optional[str] = None, messages: list = None,
                     pre_tool_block_checked: bool = False,
                     skip_tool_request_middleware: bool = False,
                     tool_request_middleware_trace: Optional[list[dict[str, Any]]] = None,
                     skip_tool_execution_middleware: bool = False) -> str:
        """Forwarder — see ``agent.agent_runtime_helpers.invoke_tool``."""
        from agent.agent_runtime_helpers import invoke_tool
        return invoke_tool(
            self,
            function_name,
            function_args,
            effective_task_id,
            tool_call_id,
            messages,
            pre_tool_block_checked,
            skip_tool_request_middleware,
            tool_request_middleware_trace,
            skip_tool_execution_middleware,
        )

    @staticmethod
    def _wrap_verbose(label: str, text: str, indent: str = "     ") -> str:
        """Word-wrap verbose tool output to fit the terminal width.

        Splits *text* on existing newlines and wraps each line individually,
        preserving intentional line breaks (e.g. pretty-printed JSON).
        Returns a ready-to-print string with *label* on the first line and
        continuation lines indented.
        """
        import shutil as _shutil
        import textwrap as _tw
        cols = _shutil.get_terminal_size((120, 24)).columns
        wrap_width = max(40, cols - len(indent))
        out_lines: list[str] = []
        for raw_line in text.split("\n"):
            if len(raw_line) <= wrap_width:
                out_lines.append(raw_line)
            else:
                wrapped = _tw.wrap(raw_line, width=wrap_width,
                                   break_long_words=True,
                                   break_on_hyphens=False)
                out_lines.extend(wrapped or [raw_line])
        body = ("\n" + indent).join(out_lines)
        return f"{indent}{label}{body}"

    def _execute_tool_calls_concurrent(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Forwarder — see ``agent.tool_executor.execute_tool_calls_concurrent``."""
        from agent.tool_executor import execute_tool_calls_concurrent
        return execute_tool_calls_concurrent(self, assistant_message, messages, effective_task_id, api_call_count)

    def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Forwarder — see ``agent.tool_executor.execute_tool_calls_sequential``."""
        from agent.tool_executor import execute_tool_calls_sequential
        return execute_tool_calls_sequential(self, assistant_message, messages, effective_task_id, api_call_count)

    def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
        """Forwarder — see ``agent.chat_completion_helpers.handle_max_iterations``."""
        from agent.chat_completion_helpers import handle_max_iterations
        return handle_max_iterations(self, messages, api_call_count)
