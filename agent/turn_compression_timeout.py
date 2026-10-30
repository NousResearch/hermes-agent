"""Recovery envelopes for a summary that timed out before turn-start admission."""

import logging

logger = logging.getLogger("agent.conversation_loop")


def preflight_timeout_result(agent, exc, conversation_history):
    """Keep accepted input on a transient timeout; retain permanent no-progress semantics."""
    from agent.agent_runtime_helpers import note_turn_persisted
    from agent.conversation_compression import context_compression_timed_out
    from agent.conversation_loop import _partial_turn_result

    retryable = context_compression_timed_out(agent)
    messages = list(conversation_history or [])
    if retryable:
        messages = exc.pending_messages
        agent._persist_session(messages, exc.pending_history)
    else:
        note_turn_persisted(agent)
    logger.warning("Turn-start compression stopped without a provider call: %s", exc)
    return _partial_turn_result(
        str(exc), messages, 0, failed=True, compression_exhausted=True,
        compression_retryable=retryable, turn_exit_reason="context_compression_timeout",
        failure_reason="timeout" if retryable else "context_overflow", failure_retryable=retryable,
    )
