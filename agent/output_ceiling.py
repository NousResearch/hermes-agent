"""Opt-in host limits checked after provider request transformations.

Reject incompatible assembly rather than changing configured reasoning semantics.
An omitted cap is not evidence of a bounded request (notably on Codex subscriptions).
"""


def validate_output_ceiling(agent, kwargs: dict, limit: int | None) -> None:
    if limit is None:
        return

    mode = agent.api_mode
    extra = kwargs.get("extra_body") or {}
    if not isinstance(extra, dict):
        raise ValueError("Host output limit cannot validate a non-object extra_body")
    # SDK extra_body fields override the typed request fields at serialization.
    body = {**kwargs, **extra}
    if mode == "bedrock_converse":
        values = [("inferenceConfig.maxTokens", (body.get("inferenceConfig") or {}).get("maxTokens"))]
    elif mode == "anthropic_messages":
        values = [("max_tokens", body.get("max_tokens"))]
    elif mode == "codex_responses":
        values = [("max_output_tokens", body.get("max_output_tokens"))]
    elif mode == "chat_completions":
        values = [(key, body[key]) for key in ("max_tokens", "max_completion_tokens") if key in body]
    else:
        raise ValueError(f"Host output limit is not supported by transport {mode!r}")

    if not values:
        raise ValueError(f"Host output limit {limit} requires an explicit wire cap; transport {mode!r} omitted it")
    for key, value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"Host output limit {limit} requires an explicit positive integer wire cap; "
                f"transport {mode!r} omitted or invalidated it"
            )
        if value > limit:
            raise ValueError(
                f"Host output limit {limit} is incompatible with provider thinking/output assembly: "
                f"{key} requires {value}. Increase the configured limit or choose compatible reasoning; "
                "the request was not sent and the reasoning budget was not reduced."
            )

    # GeminiNativeClient performs another conversion inside chat.completions.create.
    # Check that same output calculation before handing the request to the client.
    if mode == "chat_completions":
        from agent.gemini_native_adapter import is_native_gemini_base_url, _effective_gemini_max_output_tokens

        if is_native_gemini_base_url(getattr(agent, "base_url", None) or ""):
            thinking = extra.get("thinking_config") or extra.get("thinkingConfig")
            native_limit = _effective_gemini_max_output_tokens(kwargs.get("max_tokens"), thinking)
            if native_limit > limit:
                raise ValueError(
                    f"Host output limit {limit} is incompatible with native Gemini thinking/output assembly: "
                    f"maxOutputTokens requires {native_limit}. Increase the configured limit or choose "
                    "compatible reasoning; the request was not sent."
                )
