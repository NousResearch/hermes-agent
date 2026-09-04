"""Successful API-call cleanup for the conversation retry loop."""


def complete_successful_call(agent, _retry, api_request_id, api_call_count):
    """Perform successful-call cleanup; the caller owns retry-loop control flow."""
    _retry.has_retried_429 = False  # Reset on success
    # Note: don't clear the retry buffer here — an "API call
    # success" only means we got bytes back, not that we got
    # usable content. Empty responses still loop through the
    # empty-retry path below; the buffer is cleared when
    # genuinely successful content is detected later (~L4127).
    # Clear Nous rate limit state on successful request —
    # proves the limit has reset and other sessions can
    # resume hitting Nous.
    if agent.provider == "nous":
        try:
            from agent.nous_rate_guard import clear_nous_rate_limit
            clear_nous_rate_limit()
        except Exception:
            pass
    from agent import relay_llm

    relay_llm.complete_logical_call(
        api_request_id,
        outcome="success",
    )
    agent._touch_activity(f"API call #{api_call_count} completed")
