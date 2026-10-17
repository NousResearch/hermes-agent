"""Keep chat summary retries within a verified LM Studio context window."""

from agent.turn_retry_state import TurnRetryState


def _summary_fits(agent, kwargs) -> bool:
    from agent.model_metadata import estimate_request_tokens_rough
    # The SDK merges extra_body after its typed fields, with extra_body taking precedence.
    extra_body = kwargs.get("extra_body")
    payload = {**kwargs, **extra_body} if isinstance(extra_body, dict) else kwargs
    request_tokens = estimate_request_tokens_rough(payload.get("messages", []), tools=payload.get("tools"))
    reserve = max(agent._requested_output_cap_from_api_kwargs({key: payload.get(key)}) or 0
                  for key in ("max_tokens", "max_completion_tokens"))
    return request_tokens + reserve <= agent.context_compressor.context_length


def chat_summary_attempt(agent, api_messages: list, api_request_id: str):
    """Build the ordinary summary payload and recover a routed 404 at most once."""
    from agent import chat_completion_helpers as helpers
    from hermes_cli.models_lmstudio_instances import recover_stale_instance

    # Keep the ordinary builder and tools so the summary preserves the cached prefix.
    summary_kwargs = agent._build_api_kwargs(api_messages)
    helpers.sanitize_outbound_kwargs(agent, summary_kwargs)
    recovery = TurnRetryState()

    def _call(retry_count):
        return helpers._managed_summary_call(
            agent, api_request_id, summary_kwargs, agent._interruptible_api_call,
            retry_count=retry_count,
        )

    def _attempt(retry_count: int) -> str:
        nonlocal summary_kwargs
        try:
            response = _call(retry_count)
        except Exception as error:
            if not recover_stale_instance(agent, recovery, getattr(error, "status_code", None), summary_kwargs):
                raise
            summary_kwargs = agent._build_api_kwargs(api_messages)
            helpers.sanitize_outbound_kwargs(agent, summary_kwargs)
            if not _summary_fits(agent, summary_kwargs):
                raise
            response = _call(retry_count)
        return helpers._summary_text(agent, response)

    return _attempt
