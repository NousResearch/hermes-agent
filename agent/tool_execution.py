"""ToolExecutionMixin -- tool dispatch methods extracted from AIAgent.

Encapsulates concurrent and sequential tool call execution, tool invocation,
and steer-to-tool-result application.  Previously inline in AIAgent.
"""

from __future__ import annotations

from typing import Optional
import concurrent.futures
import contextvars
import json
import logging
import os
import re
import threading
import time

# Re-export utilities for mixin method access
from agent.utils import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


class ToolExecutionMixin:
    """Mixin providing tool execution methods for AIAgent.

    All methods use `self` to access agent state and are designed to be
    mixed into the AIAgent class.
    """

    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str,
                     tool_call_id: Optional[str] = None, messages: list = None,
                     pre_tool_block_checked: bool = False) -> str:
        """Invoke a single tool and return the result string. No display logic.

        Handles both agent-level tools (todo, memory, etc.) and registry-dispatched
        tools. Used by the concurrent execution path; the sequential path retains
        its own inline invocation for backward-compatible display handling.
        """
        # Check plugin hooks for a block directive before executing anything.
        block_message: Optional[str] = None
        if not pre_tool_block_checked:
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                block_message = get_pre_tool_call_block_message(
                    function_name, function_args, task_id=effective_task_id or "",
                )
            except Exception:
                pass
        if block_message is not None:
            return json.dumps({"error": block_message}, ensure_ascii=False)

        if function_name == "todo":
            from tools.todo_tool import todo_tool as _todo_tool
            return _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=self._todo_store,
            )
        elif function_name == "session_search":
            session_db = self._get_session_db_for_recall()
            if not session_db:
                from hermes_state import format_session_db_unavailable
                return json.dumps({"success": False, "error": format_session_db_unavailable()})
            from tools.session_search_tool import session_search as _session_search
            return _session_search(
                query=function_args.get("query", ""),
                role_filter=function_args.get("role_filter"),
                limit=function_args.get("limit", 3),
                db=session_db,
                current_session_id=self.session_id,
            )
        elif function_name == "memory":
            target = function_args.get("target", "memory")
            from tools.memory_tool import memory_tool as _memory_tool
            result = _memory_tool(
                action=function_args.get("action"),
                target=target,
                content=function_args.get("content"),
                old_text=function_args.get("old_text"),
                store=self._memory_store,
            )
            # Bridge: notify external memory provider of built-in memory writes
            if self._memory_manager and function_args.get("action") in ("add", "replace"):
                try:
                    self._memory_manager.on_memory_write(
                        function_args.get("action", ""),
                        target,
                        function_args.get("content", ""),
                        metadata=self._build_memory_write_metadata(
                            task_id=effective_task_id,
                            tool_call_id=tool_call_id,
                        ),
                    )
                except Exception:
                    pass
            return result
        elif self._memory_manager and self._memory_manager.has_tool(function_name):
            return self._memory_manager.handle_tool_call(function_name, function_args)
        elif function_name == "clarify":
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=self.clarify_callback,
            )
        elif function_name == "delegate_task":
            return self._dispatch_delegate_task(function_args)
        else:
            return handle_function_call(
                function_name, function_args, effective_task_id,
                tool_call_id=tool_call_id,
                session_id=self.session_id or "",
                enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                skip_pre_tool_call_hook=True,
            )

    def _apply_pending_steer_to_tool_results(self, messages: list, num_tool_msgs: int) -> None:
        """Append any pending /steer text to the last tool result in this turn.

        Called at the end of a tool-call batch, before the next API call.
        The steer is appended to the last ``role:"tool"`` message's content
        with a clear marker so the model understands it came from the user
        and NOT from the tool itself. Role alternation is preserved —
        nothing new is inserted, we only modify existing content.

        Args:
            messages: The running messages list.
            num_tool_msgs: Number of tool results appended in this batch;
                used to locate the tail slice safely.
        """
        if num_tool_msgs <= 0 or not messages:
            return
        steer_text = self._drain_pending_steer()
        if not steer_text:
            return
        # Find the last tool-role message in the recent tail. Skipping
        # non-tool messages defends against future code appending
        # something else at the boundary.
        target_idx = None
        for j in range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1):
            msg = messages[j]
            if isinstance(msg, dict) and msg.get("role") == "tool":
                target_idx = j
                break
        if target_idx is None:
            # No tool result in this batch (e.g. all skipped by interrupt);
            # put the steer back so the caller's fallback path can deliver
            # it as a normal next-turn user message.
            _lock = getattr(self, "_pending_steer_lock", None)
            if _lock is not None:
                with _lock:
                    if self._pending_steer:
                        self._pending_steer = self._pending_steer + "\n" + steer_text
                    else:
                        self._pending_steer = steer_text
            else:
                existing = getattr(self, "_pending_steer", None)
                self._pending_steer = (existing + "\n" + steer_text) if existing else steer_text
            return
        marker = f"\n\nUser guidance: {steer_text}"
        existing_content = messages[target_idx].get("content", "")
        if not isinstance(existing_content, str):
            # Anthropic multimodal content blocks — preserve them and append
            # a text block at the end.
            try:
                blocks = list(existing_content) if existing_content else []
                blocks.append({"type": "text", "text": marker.lstrip()})
                messages[target_idx]["content"] = blocks
            except Exception:
                # Fall back to string replacement if content shape is unexpected.
                messages[target_idx]["content"] = f"{existing_content}{marker}"
        else:
            messages[target_idx]["content"] = existing_content + marker
        logger.info(
            "Delivered /steer to agent after tool batch (%d chars): %s",
            len(steer_text),
            steer_text[:120] + ("..." if len(steer_text) > 120 else ""),
        )


    def _execute_tool_calls_concurrent(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute multiple tool calls concurrently using a thread pool.

        Results are collected in the original tool-call order and appended to
        messages so the API sees them in the expected sequence.
        """
        tool_calls = assistant_message.tool_calls
        num_tools = len(tool_calls)

        # ── Pre-flight: interrupt check ──────────────────────────────────
        if self._interrupt_requested:
            print(f"{self.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
            for tc in tool_calls:
                messages.append({
                    "role": "tool",
                    "name": tc.function.name,
                    "content": f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                    "tool_call_id": tc.id,
                })
            return

        # ── Parse args + pre-execution bookkeeping ───────────────────────
        parsed_calls = []  # list of (tool_call, function_name, function_args)
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            # Reset nudge counters
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # Checkpoint for file-mutating tools
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")
                except Exception:
                    pass

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass

            block_result = None
            blocked_by_guardrail = False
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                block_message = get_pre_tool_call_block_message(
                    function_name, function_args, task_id=effective_task_id or "",
                )
            except Exception:
                block_message = None

            if block_message is not None:
                block_result = json.dumps({"error": block_message}, ensure_ascii=False)
            else:
                guardrail_decision = self._tool_guardrails.before_call(function_name, function_args)
                if not guardrail_decision.allows_execution:
                    block_result = self._guardrail_block_result(guardrail_decision)
                    blocked_by_guardrail = True

            parsed_calls.append((tool_call, function_name, function_args, block_result, blocked_by_guardrail))

        # ── Logging / callbacks ──────────────────────────────────────────
        tool_names_str = ", ".join(name for _, name, _, _, _ in parsed_calls)
        if not self.quiet_mode:
            print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")
            for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls, 1):
                args_str = json.dumps(args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                    print(self._wrap_verbose("Args: ", json.dumps(args, indent=2, ensure_ascii=False)))
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}")

        for tc, name, args, block_result, blocked_by_guardrail in parsed_calls:
            if block_result is not None:
                continue
            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(name, args)
                    self.tool_progress_callback("tool.started", name, preview, args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

        for tc, name, args, block_result, blocked_by_guardrail in parsed_calls:
            if block_result is not None:
                continue
            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tc.id, name, args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

        # ── Concurrent execution ─────────────────────────────────────────
        # Each slot holds (function_name, function_args, function_result, duration, error_flag, blocked_flag)
        results = [None] * num_tools
        for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls):
            if block_result is not None:
                results[i] = (name, args, block_result, 0.0, True, True)

        # Touch activity before launching workers so the gateway knows
        # we're executing tools (not stuck).
        self._current_tool = tool_names_str
        self._touch_activity(f"executing {num_tools} tools concurrently: {tool_names_str}")

        # Capture CLI callbacks from the agent thread so worker threads can
        # register them locally.  Without this, _get_approval_callback() in
        # terminal_tool returns None in ThreadPoolExecutor workers, causing
        # the dangerous-command prompt to fall back to input() — which
        # deadlocks against prompt_toolkit's raw terminal mode (#13617).
        _parent_approval_cb = _get_approval_callback()
        _parent_sudo_cb = _get_sudo_password_callback()

        def _run_tool(index, tool_call, function_name, function_args):
            """Worker function executed in a thread."""
            # Register this worker tid so the agent can fan out an interrupt
            # to it — see AIAgent.interrupt().  Must happen first thing, and
            # must be paired with discard + clear in the finally block.
            _worker_tid = threading.current_thread().ident
            with self._tool_worker_threads_lock:
                self._tool_worker_threads.add(_worker_tid)
            # Race: if the agent was interrupted between fan-out (which
            # snapshotted an empty/earlier set) and our registration, apply
            # the interrupt to our own tid now so is_interrupted() inside
            # the tool returns True on the next poll.
            if self._interrupt_requested:
                try:
                    _set_interrupt(True, _worker_tid)
                except Exception:
                    pass
            # Set the activity callback on THIS worker thread so
            # _wait_for_process (terminal commands) can fire heartbeats.
            # The callback is thread-local; the main thread's callback
            # is invisible to worker threads.
            try:
                from tools.environments.base import set_activity_callback
                set_activity_callback(self._touch_activity)
            except Exception:
                pass
            # Propagate approval/sudo callbacks to this worker thread.
            # Mirrors cli.py run_agent() pattern (GHSA-qg5c-hvr5-hjgr).
            if _parent_approval_cb is not None:
                try:
                    _set_approval_callback(_parent_approval_cb)
                except Exception:
                    pass
            if _parent_sudo_cb is not None:
                try:
                    _set_sudo_password_callback(_parent_sudo_cb)
                except Exception:
                    pass
            start = time.time()
            try:
                result = self._invoke_tool(
                    function_name,
                    function_args,
                    effective_task_id,
                    tool_call.id,
                    messages=messages,
                    pre_tool_block_checked=True,
                )
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))
            results[index] = (function_name, function_args, result, duration, is_error, False)
            # Tear down worker-tid tracking.  Clear any interrupt bit we may
            # have set so the next task scheduled onto this recycled tid
            # starts with a clean slate.
            with self._tool_worker_threads_lock:
                self._tool_worker_threads.discard(_worker_tid)
            try:
                _set_interrupt(False, _worker_tid)
            except Exception:
                pass
            # Clear thread-local callbacks so a recycled worker thread
            # doesn't hold stale references to a disposed CLI instance.
            try:
                _set_approval_callback(None)
                _set_sudo_password_callback(None)
            except Exception:
                pass

        # Start spinner for CLI mode (skip when TUI handles tool progress)
        spinner = None
        if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
            face = random.choice(KawaiiSpinner.get_waiting_faces())
            spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=self._print_fn)
            spinner.start()

        try:
            runnable_calls = [
                (i, tc, name, args)
                for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls)
                if block_result is None
            ]
            futures = []
            if runnable_calls:
                max_workers = min(len(runnable_calls), _MAX_TOOL_WORKERS)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for i, tc, name, args in runnable_calls:
                        # Propagate ContextVars (e.g. _approval_session_key); mirrors asyncio.to_thread.
                        ctx = contextvars.copy_context()
                        f = executor.submit(ctx.run, _run_tool, i, tc, name, args)
                        futures.append(f)

                    # Wait for all to complete with periodic heartbeats so the
                    # gateway's inactivity monitor doesn't kill us during long
                    # concurrent tool batches. Also check for user interrupts
                    # so we don't block indefinitely when the user sends /stop
                    # or a new message during concurrent tool execution.
                    _conc_start = time.time()
                    _interrupt_logged = False
                    while True:
                        done, not_done = concurrent.futures.wait(
                            futures, timeout=5.0,
                        )
                        if not not_done:
                            break

                        # Check for interrupt — the per-thread interrupt signal
                        # already causes individual tools (terminal, execute_code)
                        # to abort, but tools without interrupt checks (web_search,
                        # read_file) will run to completion. Cancel any futures
                        # that haven't started yet so we don't block on them.
                        if self._interrupt_requested:
                            if not _interrupt_logged:
                                _interrupt_logged = True
                                self._vprint(
                                    f"{self.log_prefix}⚡ Interrupt: cancelling "
                                    f"{len(not_done)} pending concurrent tool(s)",
                                    force=True,
                                )
                            for f in not_done:
                                f.cancel()
                            # Give already-running tools a moment to notice the
                            # per-thread interrupt signal and exit gracefully.
                            concurrent.futures.wait(not_done, timeout=3.0)
                            break

                        _conc_elapsed = int(time.time() - _conc_start)
                        # Heartbeat every ~30s (6 × 5s poll intervals)
                        if _conc_elapsed > 0 and _conc_elapsed % 30 < 6:
                            _still_running = [
                                parsed_calls[futures.index(f)][1]
                                for f in not_done
                                if f in futures
                            ]
                            self._touch_activity(
                                f"concurrent tools running ({_conc_elapsed}s, "
                                f"{len(not_done)} remaining: {', '.join(_still_running[:3])})"
                            )
        finally:
            if spinner:
                # Build a summary message for the spinner stop
                completed = sum(1 for r in results if r is not None)
                total_dur = sum(r[3] for r in results if r is not None)
                spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

        # ── Post-execution: display per-tool results ─────────────────────
        for i, (tc, name, args, block_result, blocked_by_guardrail) in enumerate(parsed_calls):
            r = results[i]
            blocked = False
            if r is None:
                # Tool was cancelled (interrupt) or thread didn't return
                if self._interrupt_requested:
                    function_result = f"[Tool execution cancelled — {name} was skipped due to user interrupt]"
                else:
                    function_result = f"Error executing tool '{name}': thread did not return a result"
                tool_duration = 0.0
            else:
                function_name, function_args, function_result, tool_duration, is_error, blocked = r

                if not blocked:
                    function_result = self._append_guardrail_observation(
                        function_name,
                        function_args,
                        function_result,
                        failed=is_error,
                    )

                if is_error:
                    _err_text = _multimodal_text_summary(function_result)
                    result_preview = _err_text[:200] if len(_err_text) > 200 else _err_text
                    logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

                if not blocked and self.tool_progress_callback:
                    try:
                        self.tool_progress_callback(
                            "tool.completed", function_name, None, None,
                            duration=tool_duration, is_error=is_error,
                        )
                    except Exception as cb_err:
                        logging.debug(f"Tool progress callback error: {cb_err}")

                if self.verbose_logging:
                    logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                    logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            # Print cute message per tool
            if self._should_emit_quiet_tool_messages():
                cute_msg = _get_cute_tool_message_impl(name, args, tool_duration, result=function_result)
                self._safe_print(f"  {cute_msg}")
            elif not self.quiet_mode:
                _preview_str = _multimodal_text_summary(function_result)
                if self.verbose_logging:
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                    print(self._wrap_verbose("Result: ", _preview_str))
                else:
                    response_preview = _preview_str[:self.log_prefix_chars] + "..." if len(_preview_str) > self.log_prefix_chars else _preview_str
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s)")

            if not blocked and self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tc.id, name, args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=name,
                tool_use_id=tc.id,
                env=get_active_env(effective_task_id),
            ) if not _is_multimodal_tool_result(function_result) else function_result

            subdir_hints = self._subdirectory_hints.check_tool_call(name, args)
            if subdir_hints:
                if _is_multimodal_tool_result(function_result):
                    # Append the hint to the text summary part so the model
                    # still sees it; don't touch the image blocks.
                    _append_subdir_hint_to_multimodal(function_result, subdir_hints)
                else:
                    function_result += subdir_hints

            # Unwrap _multimodal dicts to an OpenAI-style content list so any
            # vision-capable provider receives [{type:text},{type:image_url}]
            # rather than a raw Python dict.  The Anthropic adapter already
            # accepts content lists; vision-capable OpenAI-compatible servers
            # (mlx-vlm, GPT-4o, …) accept image_url in tool messages natively.
            # Text-only servers that reject images are handled by the adaptive
            # _vision_supported recovery in the API retry loop.
            # String results pass through unchanged.
            _tool_content = (
                function_result["content"]
                if _is_multimodal_tool_result(function_result)
                else function_result
            )
            tool_msg = {
                "role": "tool",
                "name": name,
                "content": _tool_content,
                "tool_call_id": tc.id,
            }
            messages.append(tool_msg)

            # ── Per-tool /steer drain ───────────────────────────────────
            # Same as the sequential path: drain between each collected
            # result so the steer lands as early as possible.
            self._apply_pending_steer_to_tool_results(messages, 1)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools = len(parsed_calls)
        if num_tools > 0:
            turn_tool_msgs = messages[-num_tools:]
            enforce_turn_budget(turn_tool_msgs, env=get_active_env(effective_task_id))

        # ── /steer injection ──────────────────────────────────────────────
        # Append any pending user steer text to the last tool result so the
        # agent sees it on its next iteration. Runs AFTER budget enforcement
        # so the steer marker is never truncated. See steer() for details.
        if num_tools > 0:
            self._apply_pending_steer_to_tool_results(messages, num_tools)


    def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls sequentially (original behavior). Used for single calls or interactive tools."""
        for i, tool_call in enumerate(assistant_message.tool_calls, 1):
            # SAFETY: check interrupt BEFORE starting each tool.
            # If the user sent "stop" during a previous tool's execution,
            # do NOT start any more tools -- skip them all immediately.
            if self._interrupt_requested:
                remaining_calls = assistant_message.tool_calls[i-1:]
                if remaining_calls:
                    self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
                for skipped_tc in remaining_calls:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "name": skipped_name,
                        "content": f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                        "tool_call_id": skipped_tc.id,
                    }
                    messages.append(skip_msg)
                break

            function_name = tool_call.function.name

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logging.warning(f"Unexpected JSON error after validation: {e}")
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # Check plugin hooks for a block directive before executing.
            _block_msg: Optional[str] = None
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                _block_msg = get_pre_tool_call_block_message(
                    function_name, function_args, task_id=effective_task_id or "",
                )
            except Exception:
                pass

            _guardrail_block_decision: ToolGuardrailDecision | None = None
            if _block_msg is None:
                guardrail_decision = self._tool_guardrails.before_call(function_name, function_args)
                if not guardrail_decision.allows_execution:
                    _guardrail_block_decision = guardrail_decision

            _execution_blocked = _block_msg is not None or _guardrail_block_decision is not None

            if _execution_blocked:
                # Tool blocked by plugin or guardrail policy — skip counters,
                # callbacks, checkpointing, activity mutation, and real execution.
                pass
            else:
                # Reset nudge counters when the relevant tool is actually used
                if function_name == "memory":
                    self._turns_since_memory = 0
                elif function_name == "skill_manage":
                    self._iters_since_skill = 0

            if not self.quiet_mode:
                args_str = json.dumps(function_args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")
                    print(self._wrap_verbose("Args: ", json.dumps(function_args, indent=2, ensure_ascii=False)))
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

            if not _execution_blocked:
                self._current_tool = function_name
                self._touch_activity(f"executing tool: {function_name}")

            # Set activity callback for long-running tool execution (terminal
            # commands, etc.) so the gateway's inactivity monitor doesn't kill
            # the agent while a command is running.
            if not _execution_blocked:
                try:
                    from tools.environments.base import set_activity_callback
                    set_activity_callback(self._touch_activity)
                except Exception:
                    pass

            if not _execution_blocked and self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(function_name, function_args)
                    self.tool_progress_callback("tool.started", function_name, preview, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            if not _execution_blocked and self.tool_start_callback:
                try:
                    self.tool_start_callback(tool_call.id, function_name, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

            # Checkpoint: snapshot working dir before file-mutating tools
            if not _execution_blocked and function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(
                            work_dir, f"before {function_name}"
                        )
                except Exception:
                    pass  # never block tool execution

            # Checkpoint before destructive terminal commands
            if not _execution_blocked and function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass  # never block tool execution

            tool_start_time = time.time()

            if _block_msg is not None:
                # Tool blocked by plugin policy — return error without executing.
                function_result = json.dumps({"error": _block_msg}, ensure_ascii=False)
                tool_duration = 0.0
            elif _guardrail_block_decision is not None:
                # Tool blocked by tool-loop guardrail — synthesize exactly one
                # tool result for the original tool_call_id without executing.
                function_result = self._guardrail_block_result(_guardrail_block_decision)
                tool_duration = 0.0
            elif function_name == "todo":
                from tools.todo_tool import todo_tool as _todo_tool
                function_result = _todo_tool(
                    todos=function_args.get("todos"),
                    merge=function_args.get("merge", False),
                    store=self._todo_store,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
            elif function_name == "session_search":
                session_db = self._get_session_db_for_recall()
                if not session_db:
                    from hermes_state import format_session_db_unavailable
                    function_result = json.dumps({"success": False, "error": format_session_db_unavailable()})
                else:
                    from tools.session_search_tool import session_search as _session_search
                    function_result = _session_search(
                        query=function_args.get("query", ""),
                        role_filter=function_args.get("role_filter"),
                        limit=function_args.get("limit", 3),
                        db=session_db,
                        current_session_id=self.session_id,
                    )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
            elif function_name == "memory":
                target = function_args.get("target", "memory")
                from tools.memory_tool import memory_tool as _memory_tool
                function_result = _memory_tool(
                    action=function_args.get("action"),
                    target=target,
                    content=function_args.get("content"),
                    old_text=function_args.get("old_text"),
                    store=self._memory_store,
                )
                # Bridge: notify external memory provider of built-in memory writes
                if self._memory_manager and function_args.get("action") in ("add", "replace"):
                    try:
                        self._memory_manager.on_memory_write(
                            function_args.get("action", ""),
                            target,
                            function_args.get("content", ""),
                            metadata=self._build_memory_write_metadata(
                                task_id=effective_task_id,
                                tool_call_id=getattr(tool_call, "id", None),
                            ),
                        )
                    except Exception:
                        pass
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
            elif function_name == "clarify":
                from tools.clarify_tool import clarify_tool as _clarify_tool
                function_result = _clarify_tool(
                    question=function_args.get("question", ""),
                    choices=function_args.get("choices"),
                    callback=self.clarify_callback,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
            elif function_name == "delegate_task":
                tasks_arg = function_args.get("tasks")
                if tasks_arg and isinstance(tasks_arg, list):
                    spinner_label = f"🔀 delegating {len(tasks_arg)} tasks"
                else:
                    goal_preview = (function_args.get("goal") or "")[:30]
                    spinner_label = f"🔀 {goal_preview}" if goal_preview else "🔀 delegating"
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                self._delegate_spinner = spinner
                _delegate_result = None
                try:
                    function_result = self._dispatch_delegate_task(function_args)
                    _delegate_result = function_result
                finally:
                    self._delegate_spinner = None
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self._context_engine_tool_names and function_name in self._context_engine_tool_names:
                # Context engine tools (lcm_grep, lcm_describe, lcm_expand, etc.)
                spinner = None
                if self._should_emit_quiet_tool_messages():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _ce_result = None
                try:
                    function_result = self.context_compressor.handle_tool_call(function_name, function_args, messages=messages)
                    _ce_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Context engine tool '{function_name}' failed: {tool_error}"})
                    logger.error("context_engine.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_ce_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self._memory_manager and self._memory_manager.has_tool(function_name):
                # Memory provider tools (hindsight_retain, honcho_search, etc.)
                # These are not in the tool registry — route through MemoryManager.
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _mem_result = None
                try:
                    function_result = self._memory_manager.handle_tool_call(function_name, function_args)
                    _mem_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                    logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_mem_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self.quiet_mode:
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _spinner_result = None
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                        skip_pre_tool_call_hook=True,
                    )
                    _spinner_result = function_result
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            else:
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                        skip_pre_tool_call_hook=True,
                    )
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                tool_duration = time.time() - tool_start_time

            if isinstance(function_result, str):
                result_preview = function_result if self.verbose_logging else (
                    function_result[:200] if len(function_result) > 200 else function_result
                )
                _result_len = len(function_result)
            else:
                # Multimodal dict result (_multimodal=True) — not sliceable as string
                result_preview = function_result
                _result_len = len(str(function_result))

            # Log tool errors to the persistent error log so [error] tags
            # in the UI always have a corresponding detailed entry on disk.
            _is_error_result, _ = _detect_tool_failure(function_name, function_result)
            if not _execution_blocked:
                function_result = self._append_guardrail_observation(
                    function_name,
                    function_args,
                    function_result,
                    failed=_is_error_result,
                )
                result_preview = function_result if self.verbose_logging else (
                    function_result[:200] if len(function_result) > 200 else function_result
                )
            if _is_error_result:
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, _result_len)

            if not _execution_blocked and self.tool_progress_callback:
                try:
                    self.tool_progress_callback(
                        "tool.completed", function_name, None, None,
                        duration=tool_duration, is_error=_is_error_result,
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s)")

            if self.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                _log_result = _multimodal_text_summary(function_result)
                logging.debug(f"Tool result ({len(_log_result)} chars): {_log_result}")

            if not _execution_blocked and self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tool_call.id, function_name, function_args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=function_name,
                tool_use_id=tool_call.id,
                env=get_active_env(effective_task_id),
            ) if not _is_multimodal_tool_result(function_result) else function_result

            # Discover subdirectory context files from tool arguments
            subdir_hints = self._subdirectory_hints.check_tool_call(function_name, function_args)
            if subdir_hints:
                if _is_multimodal_tool_result(function_result):
                    _append_subdir_hint_to_multimodal(function_result, subdir_hints)
                else:
                    function_result += subdir_hints

            # Unwrap _multimodal dicts to an OpenAI-style content list
            # (see parallel path for rationale). String results pass through.
            _tool_content = (
                function_result["content"]
                if _is_multimodal_tool_result(function_result)
                else function_result
            )
            tool_msg = {
                "role": "tool",
                "name": function_name,
                "content": _tool_content,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_msg)

            # ── Per-tool /steer drain ───────────────────────────────────
            # Drain pending steer BETWEEN individual tool calls so the
            # injection lands as soon as a tool finishes — not after the
            # entire batch.  The model sees it on the next API iteration.
            self._apply_pending_steer_to_tool_results(messages, 1)

            if not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                    print(self._wrap_verbose("Result: ", function_result))
                else:
                    _fr_str = function_result if isinstance(function_result, str) else str(function_result)
                    response_preview = _fr_str[:self.log_prefix_chars] + "..." if len(_fr_str) > self.log_prefix_chars else _fr_str
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

            if self._interrupt_requested and i < len(assistant_message.tool_calls):
                remaining = len(assistant_message.tool_calls) - i
                self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
                for skipped_tc in assistant_message.tool_calls[i:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "name": skipped_name,
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id
                    }
                    messages.append(skip_msg)
                break

            if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                time.sleep(self.tool_delay)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools_seq = len(assistant_message.tool_calls)
        if num_tools_seq > 0:
            enforce_turn_budget(messages[-num_tools_seq:], env=get_active_env(effective_task_id))

        # ── /steer injection ──────────────────────────────────────────────
        # See _execute_tool_calls_parallel for the rationale. Same hook,
        # applied to sequential execution as well.
        if num_tools_seq > 0:
            self._apply_pending_steer_to_tool_results(messages, num_tools_seq)

