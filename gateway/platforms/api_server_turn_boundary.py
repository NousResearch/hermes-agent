"""Current-turn boundary inside an agent transcript returned to the OpenAI-compat routes.

``AIAgent.run_conversation`` hands back the FULL transcript (client history + this turn),
and the routes need the index where this turn starts to build Responses ``output`` items,
``run.completed`` turn transcripts and the stored ``previous_response_id`` history.
"""
from typing import Any, Dict, List

_TRANSCRIPT_IDENTITY_KEYS = ("role", "content", "tool_calls", "tool_call_id")


def _same_transcript_prefix(agent_messages: List[Any], prefix: List[Any]) -> bool:
    """True when ``agent_messages`` starts with ``prefix`` by what each message *says*.

    The API layer builds bare ``{"role", "content"}`` dicts while the agent stamps its copies
    with ``timestamp`` / ``_db_persisted`` / ``reasoning`` / ``finish_reason``; whole-dict
    equality therefore never matched and every chained turn re-appended the full prior
    transcript (#95137, #101644, #82513)."""
    if len(agent_messages) < len(prefix):
        return False
    for got, want in zip(agent_messages, prefix):
        if not isinstance(got, dict) or not isinstance(want, dict):
            if got != want:
                return False
            continue
        if any(got.get(k) != want.get(k) for k in _TRANSCRIPT_IDENTITY_KEYS):
            return False
    return True


def response_turn_start_index(
    conversation_history: List[Dict[str, Any]], user_message: Any, result: Dict[str, Any],
) -> int:
    """Index in ``result["messages"]`` where this turn's assistant/tool rows begin (0 = all).

    Anchored on this turn's user row (the loop's canonical ``reanchor_current_turn_user_idx``),
    not on prefix equality with the client history: the loop repairs host-fed history before
    the first call (merging consecutive assistant/user rows, dropping stray tool results) and
    compaction rewrites it, so the returned transcript legitimately stops sharing a prefix with
    ``conversation_history`` and a prefix match then returned 0 — the whole transcript was
    treated as the current turn, replaying earlier ``function_call`` items and doubling the
    stored history on every chained turn (#89891).

    Mocked/legacy paths return only this turn's suffix (no user row): the prefix match stays as
    the fallback for them.
    """
    from agent.turn_context import reanchor_current_turn_user_idx

    agent_messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(agent_messages, list) or not agent_messages:
        return 0
    user_idx = reanchor_current_turn_user_idx(agent_messages, user_message)
    if user_idx >= 0:
        return user_idx + 1
    prior = list(conversation_history)
    expected_prefix = prior + [{"role": "user", "content": user_message}]
    if _same_transcript_prefix(agent_messages, expected_prefix):
        return len(expected_prefix)
    if prior and _same_transcript_prefix(agent_messages, prior):
        return len(prior)
    return 0
