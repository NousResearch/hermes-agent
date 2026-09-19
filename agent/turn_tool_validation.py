"""Tool-call validation for the conversation turn loop: unknown tool names (with
auto-repair and the 3-strike partial exit) and malformed JSON arguments (retry, then
recovery tool results).

Truncated argument JSON is refused outright (never dispatched). When the provider flagged the
cut (``finish_reason="length"``) the finish_reason phase already ran the bounded recovery, so
this module keeps its terminal refusal; when a router HID the truncation behind
``"tool_calls"``/``"stop"`` this module returns the ``"truncate"`` verdict so the loop can run
the same bounded chunking recovery instead of ending the turn partial.

Role alternation is preserved on every path: an invalid batch is answered with tool-role
error results (never a user message), and the exits close any open tool-result tail
(#48879). Nothing here imports ``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.message_metadata import append_message
from agent.message_sanitization import close_interrupted_tool_sequence, coalesce_tool_call_id
from agent.turn_failure_copy import site_copy, stamp_failure

logger = logging.getLogger("agent.conversation_loop")

# ``finish_reason`` values that mean "the provider itself flagged an output-limit cut". The
# streaming assembler rewrites a truncated tool call to "length"; a provider/wire path that
# delivers "length" verbatim lands in the same bucket. Any OTHER reason on a batch whose
# arguments the provider left incomplete is a router-rewritten (hidden) truncation.
_PROVIDER_TRUNCATION_FINISH_REASONS = frozenset({"length"})


@dataclass
class ToolValidationVerdict:
    """Outcome of ``validate_tool_calls``.

    ``action``: ``"ok"`` (dispatch the calls), ``"continue"`` (re-issue the API call —
    error results / retry state were recorded), ``"return"`` (terminal partial result in
    ``result``) or ``"truncate"`` (a router-hidden truncation the loop must recover from via
    the bounded chunking path). ``mixed_invalid_batch`` is True when the batch contains BOTH
    valid and unknown tool names: only the invalid calls get error results, the valid
    ones run. ``hidden_truncation`` is set (with ``action="truncate"``) when a
    router-rewritten truncation needs the loop's bounded chunking recovery instead of a
    terminal partial exit."""

    action: str
    result: Optional[Dict[str, Any]]
    mixed_invalid_batch: bool
    hidden_truncation: Optional["HiddenTruncationRequest"] = None


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


def _partial_exit(agent, messages, conversation_history, api_call_count, final_response: str) -> Dict[str, Any]:
    """Terminal partial result. Prior retries or an earlier tool batch leave a tool-result
    tail; close it as interrupt aborts do so the next turn is not tool→user (#48879).
    This path never reaches finalize_turn, so persist here."""
    close_interrupted_tool_sequence(messages, final_response)
    agent._persist_session(messages, conversation_history)
    return stamp_failure({
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "partial": True,
        "error": final_response,
    }, "truncated", True)


@dataclass
class HiddenTruncationRequest:
    """A router-rewritten truncation discovered after the ``finish_reason`` phase.

    ``finish_reason`` was NOT ``"length"``, so ``recover_from_truncation`` never ran; the
    broken argument JSON is the only evidence. The turn must not end partial here — the loop
    owns every recovery budget (``_ChunkingProgress`` survives the per-iteration
    ``TurnRetryState`` rebuild) and runs the bounded chunking recovery on this request.

    ``assistant_message`` is the still-unstaged, unappended normalized message, so the
    truncation phase can measure the cut-off payload exactly as it does on the ``length``
    path (``_truncated_payload_size`` reads the raw argument strings).
    """

    assistant_message: Any
    finish_reason: Any
    broken_tools: List[str]


def _arguments_cut_off_mid_stream(args: Any) -> bool:
    """``args`` looks like the provider stopped mid-payload (output limit), not like a
    complete-but-malformed payload.

    Two independent signals, both required:

    - the argument JSON never closed its braces/brackets (a payload that HAS closed them but
      is still unparsable is ''complete then garbage'' — e.g. a stray trailing token — and
      must never be reclassified as an output-limit truncation), and
    - the stripped string does not end in ``}``/``]`` (the composer never emitted the final
      structural character).

    Together they keep a genuinely malformed complete payload on the historical path while
    still recognising every real mid-stream cut (unterminated string, dangling key/delimiter,
    unbalanced nesting).
    """
    if not isinstance(args, str):
        return False
    raw = args.rstrip()
    if raw.endswith(("}", "]")):
        return False
    return raw.count("{") > raw.count("}") or raw.count("[") > raw.count("]")


def _hidden_truncation_request(
    agent: Any, assistant_message: Any, finish_reason: Any, incomplete_names,
) -> Optional[HiddenTruncationRequest]:
    """The bounded-recovery request for a truncation the router hid behind a non-``length``
    ``finish_reason``, or ``None`` when this turn must stay ordinary.

    Only calls the provider left incomplete qualify (the same ``_incomplete_tool_call_names``
    verdict the ``length`` path uses: ``json.loads`` fails AND the repair fallback gives up)
    AND that additionally look cut off mid-stream (``_arguments_cut_off_mid_stream``). A
    provider-flagged truncation is therefore recovered in-turn; everything else keeps the
    existing behaviour untouched.
    """
    _incomplete = {
        tc.function.name for tc in (getattr(assistant_message, "tool_calls", None) or [])
        if tc.function.name in incomplete_names
        and _arguments_cut_off_mid_stream(tc.function.arguments)
    }
    if not _incomplete:
        return None
    return HiddenTruncationRequest(
        assistant_message=assistant_message, finish_reason=finish_reason,
        broken_tools=sorted(_incomplete),
    )


def validate_tool_calls(
    agent: Any, assistant_message: Any, finish_reason: str, *, messages: List[Dict[str, Any]],
    conversation_history: Any, api_call_count: int, effective_task_id: Any,
) -> ToolValidationVerdict:
    """Validate ``assistant_message.tool_calls`` in place (ids uniquified, names
    repaired, dict/empty args normalized to JSON strings). Strikes for invalid names
    advance only when a turn has NO valid call, so a degenerate model still halts at
    3; args cut off mid-stream (routers rewrite ``length`` → ``tool_calls``) are refused
    outright rather than retried."""
    from agent.conversation_loop import _invalid_tool_name_error_content

    tool_calls = assistant_message.tool_calls
    valid_names = agent.valid_tool_names

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None,
                 hidden_truncation: Optional[HiddenTruncationRequest] = None) -> ToolValidationVerdict:
        return ToolValidationVerdict(
            action=action, result=result, mixed_invalid_batch=_mixed_invalid_batch,
            hidden_truncation=hidden_truncation,
        )

    # Uniquify duplicate tool-call ids BEFORE any downstream consumer: the
    # pre-API sanitizer keeps only the first call/result per id.
    agent._uniquify_tool_call_ids(tool_calls)

    # Repair mismatched tool names before validating (model hallucinations).
    for tc in tool_calls:
        if tc.function.name not in valid_names:
            repaired = agent._repair_tool_call(tc.function.name)
            if repaired:
                agent._vprint(f"{agent.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'",
                              force=True, diagnostic=True)
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
            agent._vprint(f"{agent.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True, diagnostic=True)
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
    for tc in tool_calls:
        args = tc.function.arguments
        if isinstance(args, (dict, list)):
            tc.function.arguments = json.dumps(args)
            continue
        if args is not None and not isinstance(args, str):
            tc.function.arguments = args = str(args)
        if not args or not args.strip():
            tc.function.arguments = "{}"
            continue
        try:
            json.loads(args)
        except json.JSONDecodeError as e:
            # A mixed-batch invalid-name call never executes (error result later);
            # don't let its broken args trigger the whole-turn JSON retry.
            if not (_mixed_invalid_batch and tc.function.name not in valid_names):
                invalid_json_args.append((tc.function.name, str(e)))

    if invalid_json_args:
        invalid_names = {n for n, _ in invalid_json_args}
        # Routers may rewrite finish_reason "length" → "tool_calls", hiding
        # truncation; args not ending in } or ] (stripped) were cut off
        # mid-stream.
        _truncated = any(
            not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
            for tc in tool_calls if tc.function.name in invalid_names
        )
        if _truncated:
            agent._vprint(
                f"{agent.log_prefix}⚠️  Truncated tool call arguments detected "
                f"(finish_reason={finish_reason!r}) — refusing to execute.",
                force=True, diagnostic=True,
            )
            agent._invalid_json_retries = 0
            if finish_reason in _PROVIDER_TRUNCATION_FINISH_REASONS:
                # The provider SAID the output limit was hit (the streaming assembler rewrites
                # the reason; some providers/wire paths deliver it verbatim).
                # recover_from_truncation already consumed its budget for this attempt, so
                # ending here is correct and a second recovery would double-count the same
                # truncation.
                agent._cleanup_task_resources(effective_task_id)
                return _verdict("return", _partial_exit(
                    agent, messages, conversation_history, api_call_count,
                    "Response truncated due to output length limit",
                ))
            # Hidden truncation, and it was NOT handled at the finish_reason phase: the router
            # reported e.g. "tool_calls"/"stop" while the output limit cut the arguments off, so
            # recover_from_truncation never ran and the whole bounded chunking recovery was
            # skipped. Hand the request to the loop instead of ending the turn partial — the
            # loop owns the chunking budget (``_ChunkingProgress`` survives the per-iteration
            # ``TurnRetryState`` rebuild) and applies the exact same semantics as the
            # ``length`` path.
            from agent.turn_truncation import _incomplete_tool_call_names

            _hidden = _hidden_truncation_request(
                agent, assistant_message, finish_reason,
                _incomplete_tool_call_names(assistant_message),
            )
            if _hidden is not None:
                agent._vprint(
                    f"{agent.log_prefix}↻ Truncation hidden by finish_reason={finish_reason!r} — "
                    f"routing to bounded chunking recovery (tool call "
                    f"{', '.join(_hidden.broken_tools[:3])} was cut off).",
                    force=True,
                )
                return _verdict("truncate", hidden_truncation=_hidden)
            # Genuinely malformed args the provider did not leave incomplete (they merely do not
            # end in }/]): nothing proves an output-limit cut, so keep the historical refusal.
            agent._cleanup_task_resources(effective_task_id)
            return _verdict("return", _partial_exit(
                agent, messages, conversation_history, api_call_count, site_copy("truncated"),
            ))

        agent._invalid_json_retries += 1
        tool_name, error_msg = invalid_json_args[0]
        agent._buffer_vprint(f"⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

        if agent._invalid_json_retries < 3:
            agent._buffer_vprint(f"🔄 Retrying API call ({agent._invalid_json_retries}/3)...")
            # Don't add anything to messages, just retry the API call
            return _verdict("continue")
        # Instead of returning partial, inject tool error results so the model can recover.
        # Using tool results (not user messages) preserves role alternation.
        agent._buffer_vprint("⚠️  Injecting recovery tool results for invalid JSON...")
        agent._invalid_json_retries = 0  # Reset for next attempt
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

    # Reset retry counter on successful JSON validation
    agent._invalid_json_retries = 0
    return _verdict("ok")
