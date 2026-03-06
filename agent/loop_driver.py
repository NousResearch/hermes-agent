"""Python bridge between the Rust AgentLoopMachine and AIAgent.

The Rust side decides WHAT to do (state transitions, retry counters).
The Python side decides HOW to do it (API calls, tool execution, display).

Usage from run_agent.py:
    from agent.loop_driver import is_available, drive_loop
    if is_available():
        return drive_loop(self, messages, system_message, ...)
    else:
        # fallback to old while loop
"""

import json
import logging
import os
import random
import re
import time

logger = logging.getLogger(__name__)

try:
    from hermes_rs import (
        AgentLoopMachine,
        LoopState,
        Action,
        ResponseKind,
        strip_think_blocks as rs_strip_think,
        strip_tool_call_blocks as rs_strip_tool_calls,
        has_content_after_think as rs_has_content_after_think,
    )
    HAS_RUST_SM = True
except ImportError:
    HAS_RUST_SM = False


def is_available():
    """Check if the Rust state machine is available."""
    return HAS_RUST_SM


def classify_response(agent, assistant_message, finish_reason, content):
    """Classify a response into a ResponseKind for the Rust state machine."""
    from agent.trajectory import has_incomplete_scratchpad
    from agent.tool_call_parser import has_tool_call_start

    has_tool_calls = bool(getattr(assistant_message, "tool_calls", None))
    has_tag_start = has_tool_call_start(content)
    has_tag_close = "</tool_call>" in content

    tool_names_valid = True
    tool_json_valid = True
    if has_tool_calls:
        for tc in assistant_message.tool_calls:
            if tc.function.name not in agent.valid_tool_names:
                tool_names_valid = False
            try:
                args = tc.function.arguments
                if args and args.strip():
                    json.loads(args)
            except (json.JSONDecodeError, ValueError):
                tool_json_valid = False

    return AgentLoopMachine.classify_content(
        content,
        has_tool_calls,
        finish_reason or "stop",
        agent.api_mode == "codex_responses",
        has_tag_start,
        has_tag_close,
        has_incomplete_scratchpad(content),
        tool_names_valid,
        tool_json_valid,
    )


def drive_loop(
    agent,
    messages,
    active_system_prompt,
    system_message,
    conversation_history,
    user_message,
    effective_task_id,
):
    """Drive the main conversation loop using the Rust state machine.

    Returns the same dict as the old while-loop in run_conversation().
    """
    from agent.display import KawaiiSpinner
    from agent.prompt_caching import apply_anthropic_cache_control
    from agent.tool_prompt_injector import inject_tools_into_system_prompt, format_tool_response
    from agent.tool_response_adapter import adapt_response, should_adapt
    from agent.trajectory import has_incomplete_scratchpad

    sm = AgentLoopMachine(
        max_iterations=agent.max_iterations,
        needs_tool_adapter=agent._needs_tool_adapter,
        is_codex=(agent.api_mode == "codex_responses"),
        max_api_retries=6,
    )

    final_response = None
    interrupted = False
    codex_ack_continuations = 0
    agent._codex_auth_retry_attempted = False

    agent.clear_interrupt()

    while True:
        # ── Begin iteration ──
        overflow = sm.begin_iteration()
        if overflow is not None:
            # Max iterations exceeded
            final_response = agent._handle_max_iterations(messages, sm.iteration)
            break

        # ── CheckInterrupt ──
        if agent._interrupt_requested:
            sm.set_interrupted()
        t = sm.step(ResponseKind.Text)
        if t.action == Action.Break:
            interrupted = True
            if not agent.quiet_mode:
                agent._print(f"\n{agent.log_prefix}› Breaking out of tool loop due to interrupt...")
            break

        # ── Fire step_callback ──
        if agent.step_callback is not None:
            try:
                prev_tools = []
                for _m in reversed(messages):
                    if _m.get("role") == "assistant" and _m.get("tool_calls"):
                        prev_tools = [
                            tc["function"]["name"]
                            for tc in _m["tool_calls"]
                            if isinstance(tc, dict)
                        ]
                        break
                agent.step_callback(sm.iteration, prev_tools)
            except Exception as e:
                logger.debug("step_callback error (iter %s): %s", sm.iteration, e)

        if agent._skill_nudge_interval > 0 and "skill_manage" in agent.valid_tool_names:
            agent._iters_since_skill += 1

        # ── PrepareRequest ──
        t = sm.step(ResponseKind.Text)  # PrepareRequest → ApiCall

        api_messages = _build_api_messages(agent, messages, active_system_prompt)

        # ── ApiCall ──
        thinking_spinner = None
        if not agent.quiet_mode and agent._show_display:
            total_chars = sum(len(str(msg)) for msg in api_messages)
            approx_tokens = total_chars // 4
            agent._print(f"\n{agent.log_prefix}› Making API call #{sm.iteration}/{agent.max_iterations}...")
            agent._agent._print(f"{agent.log_prefix}   › Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
            agent._agent._print(f"{agent.log_prefix}   › Available tools: {len(agent.tools) if agent.tools else 0}")
        elif agent._show_display:
            verb = random.choice(KawaiiSpinner.THINKING_VERBS)
            spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
            thinking_spinner = KawaiiSpinner(f"{verb}...", spinner_type=spinner_type)
            thinking_spinner.start()

        api_start_time = time.time()
        response = None
        finish_reason = "stop"
        api_error_result = None

        # Retry loop for API call — driven by Rust retry counter
        while True:
            try:
                api_kwargs = agent._build_api_kwargs(api_messages)
                if agent.api_mode == "codex_responses":
                    api_kwargs = agent._preflight_codex_api_kwargs(api_kwargs, allow_stream=False)

                if os.getenv("HERMES_DUMP_REQUESTS", "").strip().lower() in {"1", "true", "yes", "on"}:
                    agent._dump_api_request_debug(api_kwargs, reason="preflight")

                response = agent._interruptible_api_call(api_kwargs)
                api_duration = time.time() - api_start_time

                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None

                if not agent.quiet_mode:
                    agent._print(f"{agent.log_prefix}›  API call completed in {api_duration:.2f}s")

                # Validate response shape
                response_invalid = _check_response_invalid(agent, response)
                if response_invalid:
                    t = sm.step(ResponseKind.Invalid)
                    if t.action == Action.Fail:
                        api_error_result = {
                            "messages": messages,
                            "completed": False,
                            "api_calls": sm.iteration,
                            "error": "Invalid API response after max retries",
                            "failed": True,
                        }
                        break
                    # Retry with backoff
                    wait_time = min(5 * (2 ** (sm.debug_counters()["api_retries"] - 1)), 120)
                    agent._print(f"{agent.log_prefix}⏳ Retrying in {wait_time}s...")
                    if _sleep_interruptible(agent, wait_time):
                        api_error_result = _interrupted_result(agent, messages, sm.iteration)
                        break
                    continue

                # Check finish_reason
                if agent.api_mode == "codex_responses":
                    status = getattr(response, "status", None)
                    incomplete_details = getattr(response, "incomplete_details", None)
                    incomplete_reason = None
                    if isinstance(incomplete_details, dict):
                        incomplete_reason = incomplete_details.get("reason")
                    else:
                        incomplete_reason = getattr(incomplete_details, "reason", None)
                    if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                        finish_reason = "length"
                    else:
                        finish_reason = "stop"
                else:
                    finish_reason = response.choices[0].finish_reason

                # Track token usage
                _track_usage(agent, response, api_messages)

                # Good response — exit retry loop
                t = sm.step(ResponseKind.Text)  # ApiCall → ValidateResponse
                break

            except InterruptedError:
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                interrupted = True
                final_response = "Operation interrupted."
                api_error_result = {"_break_outer": True}
                break

            except Exception as api_error:
                if thinking_spinner:
                    thinking_spinner.stop("error, retrying...")
                    thinking_spinner = None

                error_result = _handle_api_error(
                    agent, api_error, api_messages, messages,
                    active_system_prompt, system_message, sm, conversation_history,
                )
                if error_result is not None:
                    if error_result.get("_continue_retry"):
                        continue
                    api_error_result = error_result
                    break
                # If None, we somehow fell through — treat as fatal
                api_error_result = {
                    "messages": messages,
                    "completed": False,
                    "api_calls": sm.iteration,
                    "error": str(api_error),
                    "failed": True,
                }
                break

        # Handle results from API retry loop
        if api_error_result is not None:
            if api_error_result.get("_break_outer"):
                break
            agent._persist_session(messages, conversation_history)
            return api_error_result

        if interrupted:
            break

        # ── ValidateResponse ──
        if finish_reason == "length":
            t = sm.step(ResponseKind.Truncated)
            # Handle truncation
            if len(messages) > 1:
                rolled_back = agent._get_messages_up_to_last_assistant(messages)
                agent._cleanup_task_resources(effective_task_id)
                agent._persist_session(messages, conversation_history)
                return {
                    "final_response": None,
                    "messages": rolled_back,
                    "api_calls": sm.iteration,
                    "completed": False,
                    "partial": True,
                    "error": "Response truncated due to output length limit",
                }
            else:
                agent._persist_session(messages, conversation_history)
                return {
                    "final_response": None,
                    "messages": messages,
                    "api_calls": sm.iteration,
                    "completed": False,
                    "failed": True,
                    "error": "First response truncated",
                }

        t = sm.step(ResponseKind.Text)  # ValidateResponse → ParseResponse

        # ── ParseResponse ──
        try:
            if agent.api_mode == "codex_responses":
                assistant_message, finish_reason = agent._normalize_codex_response(response)
            else:
                assistant_message = response.choices[0].message

            if assistant_message.content and not agent.quiet_mode:
                agent._print(f"{agent.log_prefix}› Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

            # Delegate progress callback
            if (assistant_message.content and agent.tool_progress_callback
                    and getattr(agent, '_delegate_depth', 0) > 0):
                _think_text = re.sub(
                    r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '',
                    assistant_message.content.strip()
                ).strip()
                first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                if first_line:
                    try:
                        agent.tool_progress_callback("_thinking", first_line)
                    except Exception:
                        pass

            content = assistant_message.content or ""
            kind = classify_response(agent, assistant_message, finish_reason, content)

            # ── CheckScratchpad ──
            if kind == ResponseKind.IncompleteScratchpad:
                t = sm.step(kind)  # ParseResponse → CheckScratchpad
                t = sm.step(ResponseKind.Text)  # CheckScratchpad decision
                if t.action == Action.Retry:
                    agent._print(f"{agent.log_prefix}△  Incomplete scratchpad, retrying ({t.message})...")
                    continue
                elif t.action == Action.Fail:
                    rolled_back = agent._get_messages_up_to_last_assistant(messages)
                    agent._cleanup_task_resources(effective_task_id)
                    agent._persist_session(messages, conversation_history)
                    return {
                        "final_response": None,
                        "messages": rolled_back,
                        "api_calls": sm.iteration,
                        "completed": False,
                        "partial": True,
                        "error": t.message,
                    }
            else:
                # Reset scratchpad counter on clean response
                pass

            # ── AdaptToolCalls (for local models) ──
            if agent._needs_tool_adapter and should_adapt(response, agent.model):
                adapt_response(response, agent.tools)
                assistant_message = response.choices[0].message
                content = assistant_message.content or ""
                # Re-classify after adaptation
                kind = classify_response(agent, assistant_message, finish_reason, content)

                t = sm.step(kind)  # ParseResponse → AdaptToolCalls → next

                if kind == ResponseKind.TruncatedToolCall:
                    if t.action == Action.Nudge:
                        nudge = {
                            "role": "user",
                            "content": (
                                "Your tool call was truncated (the JSON was cut off). "
                                "Please retry with a simpler, shorter tool call. "
                                "If you need to delegate multiple tasks, call the tool "
                                "once per task instead of packing them all into one call."
                            ),
                        }
                        messages.append(nudge)
                        agent._log_msg_to_db(nudge)
                        agent._print(f"{agent.log_prefix}△  {t.message}")
                        continue
                    else:
                        # Strip broken tags and treat as text
                        raw = getattr(assistant_message, '_original_content', content)
                        from agent.tool_call_parser import strip_tool_calls as _strip_tc
                        cleaned = _strip_tc(raw)
                        cleaned = re.sub(r'<tool_call>.*', '', cleaned, flags=re.DOTALL)
                        assistant_message.content = cleaned.strip() or None
                        content = assistant_message.content or ""
                        kind = ResponseKind.Text
            elif not agent._needs_tool_adapter:
                t = sm.step(kind)  # ParseResponse → ValidateToolCalls

            # ── HandleCodexIncomplete ──
            if agent.api_mode == "codex_responses" and finish_reason == "incomplete" and kind != ResponseKind.ToolCalls:
                if sm.state == LoopState.HandleCodexIncomplete or kind == ResponseKind.CodexIncomplete:
                    t_codex = sm.step(ResponseKind.CodexIncomplete)
                    if t_codex.action == Action.Retry:
                        interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
                        if (interim_msg.get("content") or "").strip() or (interim_msg.get("reasoning") or "").strip():
                            messages.append(interim_msg)
                            agent._log_msg_to_db(interim_msg)
                        if not agent.quiet_mode:
                            agent._print(f"{agent.log_prefix}↻ {t_codex.message}")
                        agent._session_messages = messages
                        agent._save_session_log(messages)
                        continue
                    elif t_codex.action == Action.Fail:
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": sm.iteration,
                            "completed": False,
                            "partial": True,
                            "error": t_codex.message,
                        }

            # ── ValidateToolCalls ──
            has_tc = bool(getattr(assistant_message, "tool_calls", None))

            if has_tc:
                # Validate names
                invalid_names = [
                    tc.function.name for tc in assistant_message.tool_calls
                    if tc.function.name not in agent.valid_tool_names
                ]
                if invalid_names:
                    t = sm.step(ResponseKind.InvalidToolNames)
                    if t.action == Action.Retry:
                        agent._print(f"{agent.log_prefix}△  Invalid tool: '{invalid_names[0][:80]}' — {t.message}")
                        continue
                    elif t.action == Action.Fail:
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": sm.iteration,
                            "completed": False,
                            "partial": True,
                            "error": t.message,
                        }

                # Validate JSON args
                invalid_json = []
                for tc in assistant_message.tool_calls:
                    args = tc.function.arguments
                    if not args or not args.strip():
                        tc.function.arguments = "{}"
                        continue
                    try:
                        json.loads(args)
                    except json.JSONDecodeError as e:
                        invalid_json.append((tc.function.name, str(e)))

                if invalid_json:
                    t = sm.step(ResponseKind.InvalidToolJson)
                    if t.action == Action.Retry:
                        agent._print(f"{agent.log_prefix}△  Invalid JSON for '{invalid_json[0][0]}' — {t.message}")
                        continue
                    elif t.action == Action.Nudge:
                        tool_name, error_msg = invalid_json[0]
                        recovery = {
                            "role": "user",
                            "content": (
                                f"Your tool call to '{tool_name}' had invalid JSON arguments. "
                                f"Error: {error_msg}. "
                                f"For tools with no required parameters, use an empty object: {{}}. "
                                f"Please retry with valid JSON or respond without the tool."
                            ),
                        }
                        messages.append(recovery)
                        agent._log_msg_to_db(recovery)
                        continue

                # ── ExecuteTools ──
                t = sm.step(ResponseKind.ToolCalls)  # ValidateToolCalls → ExecuteTools

                assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)

                turn_content = assistant_message.content or ""
                if turn_content and agent._has_content_after_think_block(turn_content):
                    agent._last_content_with_tools = turn_content
                    if agent.quiet_mode and agent._show_display:
                        clean = agent._strip_think_blocks(turn_content).strip()
                        if clean:
                            agent._print(f"  ┊ › {clean}")

                messages.append(assistant_msg)
                agent._log_msg_to_db(assistant_msg)

                agent._execute_tool_calls(assistant_message, messages, effective_task_id)

                if agent.compression_enabled and agent.context_compressor.should_compress():
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message,
                        approx_tokens=agent.context_compressor.last_prompt_tokens
                    )

                agent._session_messages = messages
                agent._save_session_log(messages)

                t = sm.step(ResponseKind.Text)  # ExecuteTools → CheckInterrupt
                continue

            else:
                # ── HandleFinalResponse ──
                # Transition SM from ValidateToolCalls → HandleFinalResponse
                sm.step(ResponseKind.Text)

                final_response = content

                # Strip leftover <tool_call> tags
                if "<tool_call>" in final_response:
                    final_response = rs_strip_tool_calls(final_response)

                # Check empty after think
                if not rs_has_content_after_think(final_response) and final_response.strip():
                    t = sm.step(ResponseKind.EmptyAfterThink)
                    if t.action == Action.Nudge:
                        reasoning_text = agent._extract_reasoning(assistant_message)
                        agent._print(f"{agent.log_prefix}△  Empty after think block — {t.message}")
                        if reasoning_text:
                            agent._print(f"{agent.log_prefix}   Reasoning: {reasoning_text[:500]}")
                        nudge = {
                            "role": "user",
                            "content": (
                                "Your response was empty. Please respond directly to my "
                                "request without overthinking it. Just answer."
                            ),
                        }
                        messages.append(nudge)
                        agent._log_msg_to_db(nudge)
                        continue
                    elif t.action == Action.Fail:
                        # Try fallback from prior tool_calls turn
                        fallback = getattr(agent, '_last_content_with_tools', None)
                        if fallback:
                            agent._last_content_with_tools = None
                            for i in range(len(messages) - 1, -1, -1):
                                msg = messages[i]
                                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                                    tool_names = [
                                        tc.get("function", {}).get("name", "unknown")
                                        for tc in msg["tool_calls"]
                                    ]
                                    msg["content"] = f"Calling {', '.join(tool_names)}..."
                                    break
                            final_response = agent._strip_think_blocks(fallback).strip()
                            break

                        reasoning_text = agent._extract_reasoning(assistant_message)
                        empty_msg = {
                            "role": "assistant",
                            "content": final_response,
                            "reasoning": reasoning_text,
                            "finish_reason": finish_reason,
                        }
                        messages.append(empty_msg)
                        agent._log_msg_to_db(empty_msg)
                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": final_response or None,
                            "messages": messages,
                            "api_calls": sm.iteration,
                            "completed": False,
                            "partial": True,
                            "error": t.message,
                        }
                elif not final_response.strip():
                    # Truly empty response
                    t = sm.step(ResponseKind.EmptyAfterThink)
                    if t.action == Action.Nudge:
                        nudge = {
                            "role": "user",
                            "content": "Your response was empty. Please answer directly.",
                        }
                        messages.append(nudge)
                        agent._log_msg_to_db(nudge)
                        continue
                    elif t.action == Action.Fail:
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": sm.iteration,
                            "completed": False,
                            "partial": True,
                            "error": t.message,
                        }

                # Premature quit detection: model gave up mid-task
                if (
                    agent.api_mode != "codex_responses"
                    and agent.tools
                    and sm.iteration > 1
                    and len(final_response) < 300
                ):
                    _quit_phrases = [
                        "read-only", "permission denied", "cannot ",
                        "let me try", "try a different", "instead",
                        "i'll create", "unable to", "failed",
                    ]
                    _resp_lower = final_response.lower()
                    _looks_like_quit = any(p in _resp_lower for p in _quit_phrases)

                    if not hasattr(agent, '_premature_quit_nudges'):
                        agent._premature_quit_nudges = 0

                    if _looks_like_quit and agent._premature_quit_nudges < 2:
                        agent._premature_quit_nudges += 1
                        interim = agent._build_assistant_message(assistant_message, finish_reason)
                        messages.append(interim)
                        agent._log_msg_to_db(interim)
                        nudge = {
                            "role": "user",
                            "content": (
                                "Don't give up — keep going. Try the current working "
                                "directory or ~/. Use your tools to complete the task."
                            ),
                        }
                        messages.append(nudge)
                        agent._log_msg_to_db(nudge)
                        if not agent.quiet_mode:
                            agent._print(f"{agent.log_prefix}› Model appeared to give up, nudging to continue ({agent._premature_quit_nudges}/2)...")
                        continue
                    elif not _looks_like_quit:
                        agent._premature_quit_nudges = 0

                # Codex ack continuation check
                if (
                    agent.api_mode == "codex_responses"
                    and agent.valid_tool_names
                    and codex_ack_continuations < 2
                    and agent._looks_like_codex_intermediate_ack(
                        user_message=user_message,
                        assistant_content=final_response,
                        messages=messages,
                    )
                ):
                    codex_ack_continuations += 1
                    interim_msg = agent._build_assistant_message(assistant_message, "incomplete")
                    messages.append(interim_msg)
                    agent._log_msg_to_db(interim_msg)
                    cont_msg = {
                        "role": "user",
                        "content": (
                            "[System: Continue now. Execute the required tool calls and only "
                            "send your final answer after completing the task.]"
                        ),
                    }
                    messages.append(cont_msg)
                    agent._log_msg_to_db(cont_msg)
                    agent._session_messages = messages
                    agent._save_session_log(messages)
                    continue

                codex_ack_continuations = 0

                # Strip think blocks from final
                final_response = rs_strip_think(final_response).strip()

                final_msg = agent._build_assistant_message(assistant_message, finish_reason)
                messages.append(final_msg)
                agent._log_msg_to_db(final_msg)

                if not agent.quiet_mode:
                    agent._print(f"{agent.log_prefix}› Conversation completed after {sm.iteration} API call(s)")
                break

        except Exception as e:
            error_msg = f"Error during API call #{sm.iteration}: {e}"
            agent._print(f"✕ {error_msg}")
            if agent.verbose_logging:
                logging.exception("Detailed error:")

            # Fill in error results for pending tool calls
            _fill_pending_tool_errors(messages, error_msg, agent)

            if sm.iteration >= agent.max_iterations - 1:
                final_response = f"Encountered repeated errors: {error_msg}"
                break

    # ── Post-loop ──
    if sm.iteration >= agent.max_iterations and final_response is None:
        final_response = agent._handle_max_iterations(messages, sm.iteration)

    completed = final_response is not None and sm.iteration < agent.max_iterations

    agent._save_trajectory(messages, user_message, completed)
    agent._cleanup_task_resources(effective_task_id)
    agent._persist_session(messages, conversation_history)

    if final_response and not interrupted:
        agent._honcho_sync(user_message, final_response)

    result = {
        "final_response": final_response,
        "messages": messages,
        "api_calls": sm.iteration,
        "completed": completed,
        "partial": False,
        "interrupted": interrupted,
    }

    if interrupted and agent._interrupt_message:
        result["interrupt_message"] = agent._interrupt_message

    agent.clear_interrupt()

    # Log state machine debug info
    if agent.verbose_logging:
        logger.debug("State machine counters: %s", sm.debug_counters())

    return result


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_api_messages(agent, messages, active_system_prompt):
    """Build the API message list with system prompt, caching, and prefills."""
    from agent.prompt_caching import apply_anthropic_cache_control
    from agent.tool_prompt_injector import inject_tools_into_system_prompt

    api_messages = []
    for msg in messages:
        api_msg = msg.copy()
        if msg.get("role") == "assistant":
            reasoning_text = msg.get("reasoning")
            if reasoning_text:
                api_msg["reasoning_content"] = reasoning_text
        if "reasoning" in api_msg:
            api_msg.pop("reasoning")
        if "finish_reason" in api_msg:
            api_msg.pop("finish_reason")
        api_messages.append(api_msg)

    effective_system = active_system_prompt or ""
    if agent.ephemeral_system_prompt:
        effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
    if agent._honcho_context:
        effective_system = (effective_system + "\n\n" + agent._honcho_context).strip()
    if agent._needs_tool_adapter and agent.tools:
        effective_system = inject_tools_into_system_prompt(effective_system, agent.tools)
    if effective_system:
        api_messages = [{"role": "system", "content": effective_system}] + api_messages

    if agent.prefill_messages:
        sys_offset = 1 if effective_system else 0
        for idx, pfm in enumerate(agent.prefill_messages):
            api_messages.insert(sys_offset + idx, pfm.copy())

    if agent._use_prompt_caching:
        api_messages = apply_anthropic_cache_control(api_messages, cache_ttl=agent._cache_ttl)

    return api_messages


def _check_response_invalid(agent, response):
    """Return True if the response shape is invalid."""
    if agent.api_mode == "codex_responses":
        output_items = getattr(response, "output", None) if response is not None else None
        if response is None or not isinstance(output_items, list) or len(output_items) == 0:
            return True
    else:
        if (response is None or not hasattr(response, 'choices')
                or response.choices is None or len(response.choices) == 0):
            return True
    return False


def _track_usage(agent, response, api_messages):
    """Track token usage from response."""
    from agent.model_metadata import save_context_length

    if not hasattr(response, 'usage') or not response.usage:
        return

    if agent.api_mode == "codex_responses":
        prompt_tokens = getattr(response.usage, 'input_tokens', 0) or 0
        completion_tokens = getattr(response.usage, 'output_tokens', 0) or 0
        total_tokens = getattr(response.usage, 'total_tokens', None) or (prompt_tokens + completion_tokens)
    else:
        prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
        completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
        total_tokens = getattr(response.usage, 'total_tokens', 0) or 0

    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    agent.context_compressor.update_from_response(usage_dict)

    if agent.context_compressor._context_probed:
        ctx = agent.context_compressor.context_length
        save_context_length(agent.model, agent.base_url, ctx)
        agent._print(f"{agent.log_prefix}💾 Cached context length: {ctx:,} tokens for {agent.model}")
        agent.context_compressor._context_probed = False

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_api_calls += 1

    if agent._use_prompt_caching:
        details = getattr(response.usage, 'prompt_tokens_details', None)
        cached = getattr(details, 'cached_tokens', 0) or 0 if details else 0
        written = getattr(details, 'cache_write_tokens', 0) or 0 if details else 0
        prompt = usage_dict["prompt_tokens"]
        hit_pct = (cached / prompt * 100) if prompt > 0 else 0
        if not agent.quiet_mode:
            agent._print(f"{agent.log_prefix}   › Cache: {cached:,}/{prompt:,} tokens ({hit_pct:.0f}% hit, {written:,} written)")


def _sleep_interruptible(agent, seconds):
    """Sleep in small increments. Returns True if interrupted."""
    end = time.time() + seconds
    while time.time() < end:
        if agent._interrupt_requested:
            return True
        time.sleep(0.2)
    return False


def _interrupted_result(agent, messages, api_calls):
    """Build an interrupted result dict."""
    agent.clear_interrupt()
    return {
        "final_response": "Operation interrupted.",
        "messages": messages,
        "api_calls": api_calls,
        "completed": False,
        "interrupted": True,
    }


def _handle_api_error(agent, api_error, api_kwargs, messages,
                       active_system_prompt, system_message, sm, conversation_history):
    """Handle API errors with context-aware retry logic.

    Returns:
        dict with result to return, or dict with _continue_retry=True, or None.
    """
    from agent.model_metadata import parse_context_limit_from_error, get_next_probe_tier

    error_msg = str(api_error).lower()
    status_code = getattr(api_error, "status_code", None)

    # Check interrupt
    if agent._interrupt_requested:
        agent._persist_session(messages, conversation_history)
        return _interrupted_result(agent, messages, sm.iteration)

    # 413 payload too large
    is_payload_too_large = (
        status_code == 413
        or 'request entity too large' in error_msg
        or 'payload too large' in error_msg
        or 'error code: 413' in error_msg
    )
    if is_payload_too_large:
        original_len = len(messages)
        approx_tokens = sum(len(str(m)) for m in messages) // 4
        messages[:], active_system_prompt = agent._compress_context(
            messages, system_message, approx_tokens=approx_tokens
        )
        if len(messages) < original_len:
            return {"_continue_retry": True}
        return {
            "messages": messages,
            "completed": False,
            "api_calls": sm.iteration,
            "error": "Request payload too large (413). Cannot compress further.",
            "partial": True,
        }

    # Context length errors
    is_context_error = any(p in error_msg for p in [
        'context length', 'context size', 'maximum context',
        'token limit', 'too many tokens', 'reduce the length',
        'exceeds the limit', 'context window',
    ])
    if is_context_error:
        compressor = agent.context_compressor
        old_ctx = compressor.context_length
        parsed_limit = parse_context_limit_from_error(error_msg)
        new_ctx = parsed_limit if (parsed_limit and parsed_limit < old_ctx) else get_next_probe_tier(old_ctx)

        if new_ctx and new_ctx < old_ctx:
            compressor.context_length = new_ctx
            compressor.threshold_tokens = int(new_ctx * compressor.threshold_percent)
            compressor._context_probed = True

        original_len = len(messages)
        approx_tokens = sum(len(str(m)) for m in messages) // 4
        messages[:], active_system_prompt = agent._compress_context(
            messages, system_message, approx_tokens=approx_tokens
        )
        if len(messages) < original_len or (new_ctx and new_ctx < old_ctx):
            return {"_continue_retry": True}
        return {
            "messages": messages,
            "completed": False,
            "api_calls": sm.iteration,
            "error": "Context length exceeded, cannot compress further",
            "partial": True,
        }

    # Codex 401 refresh: try to refresh credentials before treating as fatal
    if (
        agent.api_mode == "codex_responses"
        and agent.provider == "openai-codex"
        and status_code == 401
        and not getattr(agent, '_codex_auth_retry_attempted', False)
    ):
        agent._codex_auth_retry_attempted = True
        if agent._try_refresh_codex_client_credentials(force=True):
            agent._print(f"{agent.log_prefix}› Codex auth refreshed after 401. Retrying request...")
            return {"_continue_retry": True}

    # Non-retryable client errors
    is_client_error = (
        isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 413
        and not is_context_error
    )
    if is_client_error:
        return {
            "final_response": None,
            "messages": messages,
            "api_calls": sm.iteration,
            "completed": False,
            "failed": True,
            "error": str(api_error),
        }

    # Generic retry with backoff
    t = sm.step(ResponseKind.Invalid)  # Use Invalid as generic error signal
    if t.action == Action.Fail:
        raise api_error  # Re-raise after max retries

    wait_time = min(2 ** sm.debug_counters()["api_retries"], 60)
    if _sleep_interruptible(agent, wait_time):
        return _interrupted_result(agent, messages, sm.iteration)

    return {"_continue_retry": True}


def _fill_pending_tool_errors(messages, error_msg, agent):
    """Fill in error results for any pending tool calls without responses."""
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, dict):
            break
        if msg.get("role") == "tool":
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            answered_ids = {
                m["tool_call_id"]
                for m in messages[idx + 1:]
                if isinstance(m, dict) and m.get("role") == "tool"
            }
            for tc in msg["tool_calls"]:
                if tc["id"] not in answered_ids:
                    err = {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"Error executing tool: {error_msg}",
                    }
                    messages.append(err)
                    agent._log_msg_to_db(err)
            break
        break
