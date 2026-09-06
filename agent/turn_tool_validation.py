"""Tool-call validation for the conversation turn loop: unknown tool names (with
auto-repair and the 3-strike partial exit) and malformed JSON arguments (retry, then
recovery tool results).

Role alternation is preserved on every path: an invalid batch is answered with tool-role
error results (never a user message), and the exits close any open tool-result tail
(#48879). Nothing here imports ``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.message_metadata import append_message
from agent.message_sanitization import close_interrupted_tool_sequence, coalesce_tool_call_id

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class ToolValidationVerdict:
    """Outcome of ``validate_tool_calls``.

    ``action``: ``"ok"`` (dispatch the calls), ``"continue"`` (re-issue the API call —
    error results / retry state were recorded) or ``"return"`` (terminal partial
    result in ``result``). ``mixed_invalid_batch`` is True when the batch contains BOTH
    valid and unknown tool names: only the invalid calls get error results, the valid
    ones run."""

    action: str
    result: Optional[Dict[str, Any]]
    mixed_invalid_batch: bool


def _preview_name(name: str) -> str:
    return name[:80] + "..." if len(name) > 80 else name


def _append_tool_error_results(messages, tool_calls, content_for) -> None:
    """One tool-role result per call so every tool_call keeps a matching result."""
    for tc in tool_calls:
        append_message(messages, {
            "role": "tool",
            "name": tc.function.name,
            "tool_call_id": coalesce_tool_call_id(tc),
            "content": content_for(tc),
        })


def _json_structure_incomplete(raw: str) -> bool:
    """Whether a malformed JSON value ended inside a string or container.

    This distinguishes an interrupted serialization from complete-but-invalid text without
    guessing from its final character (``not json`` does not become "truncated").
    """
    stack: List[str] = []
    in_string = escaped = False
    pairs = {"}": "{", "]": "["}
    for char in raw:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    return in_string or bool(stack)


def _python_preflight_supported() -> bool:
    """Compile locally only when it exactly matches execute_code's target interpreter."""
    try:
        from tools.code_execution_env import _resolve_child_python
        from tools.code_execution_tool import _get_execution_mode
        from tools.terminal_tool import _get_env_config

        if _get_env_config().get("env_type") != "local":
            return False
        child_python = _resolve_child_python(_get_execution_mode())
        return os.path.normcase(os.path.realpath(child_python)) == os.path.normcase(
            os.path.realpath(sys.executable)
        )
    except Exception:
        logger.debug("Could not resolve execute_code interpreter for source preflight", exc_info=True)
        return False


def _python_source_error(agent: Any, tool_name: str, args: Any) -> Optional[tuple[Exception, str]]:
    """Return a local compile error for direct or tool-search-bridged execute_code."""
    if not isinstance(args, dict):
        return None
    if tool_name == "tool_call":
        from agent.tool_executor import _unwrap_tool_search_call

        resolved_name, resolved_args, scope_block = _unwrap_tool_search_call(
            agent, tool_name, args
        )
        if resolved_name != "execute_code" or scope_block is not None:
            return None
        args = resolved_args
    elif tool_name != "execute_code":
        return None
    source = args.get("code")
    if not isinstance(source, str) or not source.strip():
        return None
    try:
        compile(source, "<execute_code>", "exec")
    except (SyntaxError, ValueError) as exc:
        return exc, source
    return None


def _partial_exit(agent, messages, conversation_history, api_call_count, final_response: str) -> Dict[str, Any]:
    """Terminal partial result. Prior retries or an earlier tool batch leave a tool-result
    tail; close it as interrupt aborts do so the next turn is not tool→user (#48879).
    This path never reaches finalize_turn, so persist here."""
    close_interrupted_tool_sequence(messages, final_response)
    agent._persist_session(messages, conversation_history)
    return {
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "partial": True,
        "error": final_response,
    }


def validate_tool_calls(
    agent: Any, assistant_message: Any, finish_reason: str, *, messages: List[Dict[str, Any]],
    conversation_history: Any, api_call_count: int, effective_task_id: Any,
) -> ToolValidationVerdict:
    """Validate ``assistant_message.tool_calls`` in place (ids uniquified, names
    repaired, dict/empty args normalized to JSON strings). Strikes for invalid names
    advance only when a turn has NO valid call, so a degenerate model still halts at
    3. Incomplete payloads are rejected batch-wide and retried with one shared bounded
    budget before any tool can run."""
    from agent.conversation_loop import _invalid_tool_name_error_content

    tool_calls = assistant_message.tool_calls
    valid_names = agent.valid_tool_names

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ToolValidationVerdict:
        return ToolValidationVerdict(action=action, result=result, mixed_invalid_batch=_mixed_invalid_batch)

    # Uniquify duplicate tool-call ids BEFORE any downstream consumer: the
    # pre-API sanitizer keeps only the first call/result per id.
    agent._uniquify_tool_call_ids(tool_calls)

    # Repair mismatched tool names before validating (model hallucinations).
    for tc in tool_calls:
        if tc.function.name not in valid_names:
            repaired = agent._repair_tool_call(tc.function.name)
            if repaired:
                print(f"{agent.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                tc.function.name = repaired
    invalid_tool_calls = [tc.function.name for tc in tool_calls if tc.function.name not in valid_names]
    # Mixed batch: error-result ONLY the invalid calls and run the valid
    # ones; voiding the turn discards real work. Strikes advance only when a
    # turn has NO valid call, so a degenerate model still halts at 3.
    _mixed_invalid_batch = bool(invalid_tool_calls) and any(
        tc.function.name in valid_names for tc in tool_calls
    )
    if _mixed_invalid_batch:
        agent._invalid_tool_retries = 0
        _n_valid = sum(1 for tc in tool_calls if tc.function.name in valid_names)
        agent._buffer_vprint(
            f"⚠️  Unknown tool '{_preview_name(invalid_tool_calls[0])}' in batch — erroring that call, "
            f"executing {_n_valid} valid call(s)"
        )
    elif invalid_tool_calls:
        agent._invalid_tool_retries += 1
        # Return helpful error to model — model can agent-correct next turn
        invalid_preview = _preview_name(invalid_tool_calls[0])
        agent._buffer_vprint(f"⚠️  Unknown tool '{invalid_preview}' — sending error to model for agent-correction ({agent._invalid_tool_retries}/3)")

        if agent._invalid_tool_retries >= 3:
            agent._flush_status_buffer()
            agent._vprint(f"{agent.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
            agent._invalid_tool_retries = 0
            return _verdict("return", _partial_exit(
                agent, messages, conversation_history, api_call_count,
                f"Model generated invalid tool call: {invalid_preview}",
            ))

        append_message(messages, agent._build_assistant_message(assistant_message, finish_reason))
        # See _invalid_tool_name_error_content for the blank-name anti-priming rationale (#47967).
        _append_tool_error_results(
            messages, tool_calls,
            lambda tc: (
                _invalid_tool_name_error_content(tc.function.name, valid_names)
                if tc.function.name not in valid_names
                else "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
            ),
        )
        return _verdict("continue")
    # Reset retry counter on successful tool call validation
    agent._invalid_tool_retries = 0

    # Validate tool call arguments are valid JSON; empty strings become empty
    # objects (common model quirk).
    invalid_json_args = []
    parsed_args = []
    for tc in tool_calls:
        args = tc.function.arguments
        if isinstance(args, (dict, list)):
            parsed_args.append((tc, args))
            tc.function.arguments = json.dumps(args)
            continue
        if args is not None and not isinstance(args, str):
            tc.function.arguments = args = str(args)
        if not args or not args.strip():
            tc.function.arguments = "{}"
            continue
        try:
            parsed_args.append((tc, json.loads(args)))
        except json.JSONDecodeError as e:
            # A mixed-batch invalid-name call never executes (error result later);
            # don't let its broken args trigger the whole-turn JSON retry.
            if not (_mixed_invalid_batch and tc.function.name not in valid_names):
                invalid_json_args.append((tc.function.name, str(e)))

    if invalid_json_args:
        invalid_names = {n for n, _ in invalid_json_args}
        # Routers may rewrite finish_reason "length" → "tool_calls". Inspect JSON
        # structure rather than the final byte so complete malformed text keeps the
        # established repair path.
        _incomplete = any(
            _json_structure_incomplete(tc.function.arguments or "")
            for tc in tool_calls if tc.function.name in invalid_names
        )
        if _incomplete:
            agent._invalid_tool_payload_retries += 1
            n = agent._invalid_tool_payload_retries
            agent._vprint(
                f"{agent.log_prefix}⚠️  Incomplete tool call arguments detected "
                f"(finish_reason={finish_reason!r}) — retrying without execution ({n}/3).",
                force=True,
            )
            if n < 3:
                return _verdict("continue")
            agent._invalid_tool_payload_retries = 0
            agent._cleanup_task_resources(effective_task_id)
            return _verdict("return", _partial_exit(
                agent, messages, conversation_history, api_call_count,
                "Model repeatedly generated invalid or incomplete tool-call payloads; "
                "no tools from the invalid batches were executed",
            ))

        agent._invalid_tool_payload_retries += 1
        tool_name, error_msg = invalid_json_args[0]
        agent._buffer_vprint(f"⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

        if agent._invalid_tool_payload_retries < 3:
            agent._buffer_vprint(f"🔄 Retrying API call ({agent._invalid_tool_payload_retries}/3)...")
            # Don't add anything to messages, just retry the API call
            return _verdict("continue")
        # Instead of returning partial, inject tool error results so the model can recover.
        # Using tool results (not user messages) preserves role alternation.
        agent._buffer_vprint("⚠️  Injecting recovery tool results for invalid JSON...")
        agent._invalid_tool_payload_retries = 0  # Reset for next attempt
        # Append the assistant message with its (broken) tool_calls, then one
        # error result per call.
        append_message(messages, agent._build_assistant_message(assistant_message, finish_reason))

        def _json_error_result(tc) -> str:
            if tc.function.name not in invalid_names:
                return "Skipped: other tool call in this response had invalid JSON."
            err = next(e for n, e in invalid_json_args if n == tc.function.name)
            return (
                f"Error: Invalid JSON arguments. {err}. "
                f"For tools with no required parameters, use an empty object: {{}}. "
                f"Please retry with valid JSON."
            )

        _append_tool_error_results(messages, tool_calls, _json_error_result)
        return _verdict("continue")

    python_errors = []
    if any(tc.function.name in {"execute_code", "tool_call"} for tc, _ in parsed_args):
        if _python_preflight_supported():
            for tc, args in parsed_args:
                if _mixed_invalid_batch and tc.function.name not in valid_names:
                    continue
                error = _python_source_error(agent, tc.function.name, args)
                if error is not None:
                    python_errors.append((tc, *error))

    if python_errors:
        agent._invalid_tool_payload_retries += 1
        n = agent._invalid_tool_payload_retries
        first_tc, first_error, _ = python_errors[0]
        first_message = first_error.msg if isinstance(first_error, SyntaxError) else str(first_error)
        agent._buffer_vprint(
            f"⚠️  Python source in '{first_tc.function.name}' does not compile: "
            f"{first_message} ({n}/3)"
        )
        if n >= 3:
            agent._invalid_tool_payload_retries = 0
            agent._cleanup_task_resources(effective_task_id)
            return _verdict("return", _partial_exit(
                agent, messages, conversation_history, api_call_count,
                "Model repeatedly generated invalid or incomplete tool-call payloads; "
                "no tools from the invalid batches were executed",
            ))

        append_message(messages, agent._build_assistant_message(assistant_message, finish_reason))
        errors_by_call = {id(tc): (exc, source) for tc, exc, source in python_errors}

        def _python_error_result(tc) -> str:
            error = errors_by_call.get(id(tc))
            if error is None:
                return "Skipped: another tool call in this response contained invalid Python source."
            exc, source = error
            if isinstance(exc, SyntaxError):
                detail = f"{exc.msg} at line {exc.lineno or '?'}, column {exc.offset or '?'}"
            else:
                detail = str(exc)
            source_bytes = len(source.encode("utf-8", errors="replace"))
            return (
                "Error: Python source did not compile, so no call in this batch was executed. "
                f"Received {len(source)} characters / {source_bytes} bytes; "
                f"{detail}. Resend the complete source, or split it into smaller "
                "independently complete calls; do not send only the missing closing syntax."
            )

        _append_tool_error_results(messages, tool_calls, _python_error_result)
        return _verdict("continue")

    # Reset retry counter on a complete, valid payload batch.
    agent._invalid_tool_payload_retries = 0
    return _verdict("ok")
