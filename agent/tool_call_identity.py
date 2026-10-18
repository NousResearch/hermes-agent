"""Reject ambiguous completed tool batches before IDs or approvals change."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from agent.message_metadata import append_message
from agent.message_sanitization import close_interrupted_tool_sequence
from agent.tool_dispatch_helpers import make_tool_result_message

logger = logging.getLogger(__name__)


def _call_id(call: Any) -> str:
    """Full provider identity of one tool call, for conflict detection.

    Every identity field the provider supplies is part of the identity, including the
    response-item half of a composite ``call_id|item`` id. Distinct response items are
    distinct calls: two provider items that share only the pairing half are reconciled by
    ``uniquify_tool_call_ids`` (provider-minted parallel ids, #130363), so collapsing them
    to the pairing half here would reject a batch that repair exists to fix.
    """
    fields = (getattr(call, "call_id", None), getattr(call, "id", None),
              getattr(call, "response_item_id", None))
    return "|".join(value.strip() for value in fields if isinstance(value, str) and value.strip())


def _arguments_key(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return arguments
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def _is_empty_arguments(arguments: Any) -> bool:
    if isinstance(arguments, str):
        if not arguments.strip():
            return True
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return False
    return arguments in (None, {}, [])


def _conflicting_calls(tool_calls: list | None) -> tuple[list, set[str]]:
    """Split a batch into first-seen calls and the IDs shared with an empty shadow.

    Reused IDs with different, non-empty arguments are legitimate parallel calls
    that ``_uniquify_tool_call_ids`` repairs; only an empty-argument call sharing
    an ID with a different call is treated as a conflict.
    """
    seen = {}
    conflicts = set()
    unique_calls = []
    for call in tool_calls or []:
        cid = _call_id(call)
        key = (call.function.name, _arguments_key(call.function.arguments))
        empty = _is_empty_arguments(call.function.arguments)
        if cid and cid in seen:
            seen_key, seen_empty = seen[cid]
            if seen_key != key and (empty or seen_empty):
                conflicts.add(cid)
        else:
            unique_calls.append(call)
            if cid:
                seen[cid] = (key, empty)
    return unique_calls, conflicts


def _append_rejection(agent, assistant_message, finish_reason, unique_calls, messages) -> None:
    rejected = copy.copy(assistant_message)
    rejected.tool_calls = unique_calls
    transcript = agent._build_assistant_message(rejected, finish_reason)
    append_message(messages, transcript)
    error = json.dumps({
        "error": "Conflicting tool call IDs",
        "message": (
            "The same tool call ID described different names or arguments. "
            "This entire batch was not executed. Resend the intended calls "
            "with unique IDs and complete arguments, without empty shadows."
        ),
    })
    for call in transcript["tool_calls"]:
        append_message(messages, make_tool_result_message(
            call["function"]["name"], error, call["id"],
            effect_disposition="none",
        ))


def reject_conflicting_tool_call_ids(
    agent: Any,
    assistant_message: Any,
    finish_reason: str,
    *,
    messages: list,
    conversation_history: Any,
    api_call_count: int,
    effective_task_id: Any,
) -> tuple[bool, dict | None]:
    """Reject the whole completed batch when one received ID names different actions.

    Never choose a payload or rename a shadow into an executable action. Each
    rejected identity gets one explicit result so the model can resend safely.
    """
    unique_calls, conflicts = _conflicting_calls(assistant_message.tool_calls)
    if not conflicts:
        agent._invalid_tool_identity_retries = 0
        return False, None

    attempts = getattr(agent, "_invalid_tool_identity_retries", 0) + 1
    agent._invalid_tool_identity_retries = attempts
    logger.warning(
        "Rejected tool batch with %d conflicting call identities; "
        "no tools executed (attempt %d/3)",
        len(conflicts), attempts,
    )
    _append_rejection(agent, assistant_message, finish_reason, unique_calls, messages)
    if attempts < 3:
        return True, None

    final_response = (
        "Model returned conflicting tool call IDs repeatedly. "
        "The conflicting batches were not executed."
    )
    close_interrupted_tool_sequence(messages, final_response)
    agent._cleanup_task_resources(effective_task_id)
    agent._persist_session(messages, conversation_history)
    return True, {
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "partial": True,
        "error": final_response,
    }
