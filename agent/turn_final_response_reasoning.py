"""Trusted-route promotion of reasoning-only final responses."""
from agent.reasoning_promotion import answer_in_reasoning_capability


def promote_final_reasoning(agent, assistant_message, finish_reason, messages, api_call_count, logger):
    # Reasoning-only clean stop on a trusted route (elsewhere reasoning is private and takes the
    # ladder): some parsers (vLLM nemotron_v3 past ~500K prompt tokens) file the whole answer as
    # reasoning when the closing delimiter is missing; ``stop`` means generation is complete, so
    # promote it BEFORE the ladder. ``length`` (cut off mid-thought) stays on the continuation path.
    # The promoted text is RETURNED as the answer but never written into the assistant
    # row's ``content``: chain-of-thought stored as ordinary content is indistinguishable
    # from a real reply on every history surface (#111761). The row keeps ``content``
    # empty with the text in its reasoning fields and carries the promoted text as the
    # ``api_content`` sidecar, so the next turn still replays it byte-identically.
    # Anthropic thinking (signed ``thinking`` block, or a plugin's ``*.native_assistant`` carrier
    # of native Claude turns) is a summary written by a separate model, never the answer: it
    # takes the empty-response continuation below instead.
    _content = assistant_message.content
    _promoted = None
    if (
        finish_reason == "stop"
        and not assistant_message.tool_calls
        and (_content is None or (isinstance(_content, str) and not _content.strip()))
        and not any(
            isinstance(d, dict) and (
                (d.get("type") in ("thinking", "redacted_thinking") and (d.get("signature") or d.get("data")))
                or str(d.get("type") or "").endswith(".native_assistant")
            )
            for d in getattr(assistant_message, "reasoning_details", None) or ()
        )
        and answer_in_reasoning_capability(agent)
    ):
        _promoted = agent._extract_reasoning(assistant_message) or None
        if _promoted:
            # WARNING, not INFO: a model that keeps ending turns this way is stalled
            # (planning monologue, zero tool calls) while the turn reports "complete".
            logger.warning(
                "Reasoning-only clean stop (%d chars) — returning the reasoning as the final "
                "response (model=%s provider=%s api_calls=%d tool_turns=%d)",
                len(_promoted), agent.model, agent.provider, api_call_count,
                sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")),
            )
    return _promoted
