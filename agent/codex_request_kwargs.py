"""Build the Responses request with route-specific schema and replay controls."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_codex_kwargs(agent, api_messages, tools_for_api, reasoning_config, request_overrides, cache_scope_id):
    from agent.chat_completion_helpers import _consume_ephemeral_max_output
    from agent.codex_responses_adapter import classify_responses_route
    from agent.native_compaction import native_compaction_context_management
    is_codex_backend, is_xai_responses, is_github_responses = classify_responses_route(agent)
    # Replay window for proxy routes (``other:*``); None on every agent built before this key
    # existed, which leaves replay uncapped.
    from agent.codex_proxy_replay_window import proxy_replay_max_turns as _proxy_replay_max_turns
    # Native server-side compaction (gpt-5.6 on direct OpenAI / ChatGPT Codex routes
    # only) — None on every other route/model, leaving the request unchanged.
    context_management = native_compaction_context_management(agent, is_codex_backend=is_codex_backend,
        is_xai_responses=is_xai_responses, is_github_responses=is_github_responses)
    # xAI's /responses endpoint 400s on ``pattern``/``format`` schema keywords and on
    # ``enum`` values containing ``/`` — strip them (#27197). Deep-copy first: the
    # sanitizers mutate in place and tools_for_api aliases agent.tools (#27907).
    if is_xai_responses:
        try:
            import copy as _copy
            from tools.schema_sanitizer import strip_pattern_and_format, strip_slash_enum
            tools_for_api = _copy.deepcopy(tools_for_api)
            tools_for_api, _ = strip_pattern_and_format(tools_for_api)
            tools_for_api, _ = strip_slash_enum(tools_for_api)
        except Exception as exc:
            logger.warning("%s⚠️ Failed to sanitize tool schemas for xAI: %s", getattr(agent, "log_prefix", ""), exc, exc_info=True)
    ephemeral_out = _consume_ephemeral_max_output(agent)
    return agent._get_transport().build_kwargs(model=agent.model,
        messages=agent._prepare_messages_for_non_vision_model(api_messages), tools=tools_for_api,
        reasoning_config=reasoning_config, session_id=getattr(agent, "session_id", None),
        cache_scope_id=cache_scope_id, base_url=agent.base_url,
        max_tokens=ephemeral_out if ephemeral_out is not None else agent.max_tokens,
        timeout=agent._resolved_api_call_timeout(), request_overrides=request_overrides,
        provider=getattr(agent, "provider", None), is_github_responses=is_github_responses,
        is_codex_backend=is_codex_backend, is_xai_responses=is_xai_responses,
        github_reasoning_extra=agent._github_models_reasoning_extra_body() if is_github_responses else None,
        replay_encrypted_reasoning=bool(getattr(agent, "_codex_reasoning_replay_enabled", True)),
        proxy_replay_max_turns=_proxy_replay_max_turns(agent),
        context_management=context_management, text_verbosity=getattr(agent, "text_verbosity", None))
