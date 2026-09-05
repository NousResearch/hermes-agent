"""Per-attempt request assembly for the conversation turn's retry loop: re-apply the
reasoning echo pad and prompt-cache decoration for the CURRENT provider (a fallback may
differ from the primary), build ``api_kwargs``, run the surrogate/ASCII chokepoints, Codex
preflight, OpenRouter cache bypass, Copilot ``x-initiator``, the LLM request middleware,
the ``pre_api_request`` hook and the debug dump. Nothing here imports
``agent.conversation_loop`` at module level.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from agent.message_sanitization import (
    _sanitize_structure_non_ascii, _sanitize_structure_surrogates
)
from utils import env_var_enabled

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class ApiRequestBuild:
    """Always ``action == "fallthrough"``; the fields are the request-local values the caller
    rebinds for the attempt."""

    action: str
    api_messages: Any
    _moa_prepared_request: Any
    tools_for_api: Any
    api_kwargs: Any
    _original_api_kwargs: Any
    _llm_middleware_trace: Any


def _set_extra_header(api_kwargs: Any, key: str, value: str) -> None:
    """Copy-on-write header set (the dict may be shared with the transport's defaults)."""
    _xh = dict(api_kwargs.get("extra_headers") or {})
    _xh[key] = value
    api_kwargs["extra_headers"] = _xh


def _fire_pre_api_request_hook(
    agent: Any, api_kwargs: Any, api_messages: Any, _llm_middleware_trace: Any, *, messages: Any,
    original_user_message: Any, approx_tokens: Any, total_chars: Any, retry_count: Any,
    api_call_count: Any, api_request_id: Any, api_start_time: Any, effective_task_id: Any,
    turn_id: Any,
) -> None:
    from agent.conversation_loop import _system_prompt_for_hooks

    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook as _invoke_hook
        if has_hook("pre_api_request"):
            request_messages = api_kwargs.get("messages")
            if not isinstance(request_messages, list):
                request_messages = api_kwargs.get("input")
            if not isinstance(request_messages, list):
                request_messages = api_messages
            # Shallow copies: plugins may retain the lists; deepcopy is costly.
            # ``request_messages``/``conversation_history`` are raw langfuse passthroughs.
            # Anthropic (``system``) and Responses/Codex (``instructions``) move the system
            # prompt out of messages; pass it for observability.
            _invoke_hook(
                "pre_api_request",
                task_id=effective_task_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                session_id=agent.session_id or "",
                user_message=original_user_message,
                conversation_history=list(messages),
                platform=agent.platform or "",
                model=agent.model,
                provider=agent.provider,
                base_url=agent.base_url,
                api_mode=agent.api_mode,
                api_call_count=api_call_count,
                retry_count=retry_count,
                request_messages=list(request_messages) if isinstance(request_messages, list) else [],
                system_prompt=_system_prompt_for_hooks(api_kwargs, request_messages),
                message_count=len(api_messages),
                tool_count=len(agent.tools or []),
                approx_input_tokens=approx_tokens,
                request_char_count=total_chars,
                max_tokens=agent.max_tokens,
                started_at=api_start_time,
                middleware_trace=list(_llm_middleware_trace),
                request=agent._api_request_payload_for_hook(api_kwargs),
            )
    except Exception:
        pass


def build_api_request(
    agent: Any, *, api_messages: Any, _moa_prepared_request: Any, tools_for_api: Any,
    system_message: Any, messages: Any, original_user_message: Any, approx_tokens: Any,
    total_chars: Any, retry_count: Any, api_call_count: Any, api_request_id: Any,
    api_start_time: Any, effective_task_id: Any, turn_id: Any,
) -> ApiRequestBuild:
    """Assemble the attempt's request in the original order (every mutation happens BEFORE
    middleware/hooks/debug dumps observe the payload)."""
    from agent.conversation_loop import (
        _moa_client_consumes_prepared_request, _redecorate_prompt_cache_for_provider,
    )

    agent._reset_stream_delivery_tracking()
    # Per-attempt first-chunk timestamp so a stale value never leaks into post_api_request.
    agent._last_api_first_chunk_at = None
    # api_messages was built for the primary; a fallback (DeepSeek / Kimi / MiMo) may
    # require reasoning_content — re-apply the echo-back pad (idempotent) and re-render
    # the prompt-cache decoration for the current provider.
    agent._reapply_reasoning_echo_for_provider(api_messages)
    api_messages, _moa_prepared_request, tools_for_api = (
        _redecorate_prompt_cache_for_provider(
            agent, api_messages, system_message=system_message, moa_prepared=_moa_prepared_request,
            tools_for_api=tools_for_api,
        )
    )
    if tools_for_api == agent.tools:
        api_kwargs = agent._build_api_kwargs(api_messages)
    else:
        api_kwargs = agent._build_api_kwargs(api_messages, tools_for_api=tools_for_api)
    # Surrogate chokepoint: tool descriptions, extra_body and kwargs strings can carry
    # invalid code points (HTTP 400). One walk makes the payload json.dumps()-safe.
    # Outbound-request surrogate chokepoint (#50959): the messages were scrubbed above, but the rest of the
    # request body — tool/function descriptions (session_search's ±-heavy text is the recorded repro),
    # extra_body, system strings routed via kwargs — can still carry invalid code points that providers
    # reject with a non-retryable HTTP 400 ("invalid unicode code point"). One in-place walk here guarantees
    # the entire payload json.dumps()-safe regardless of which leaf produced the string. Fast no-op when the
    # payload is clean.
    _sanitize_structure_surrogates(api_kwargs)
    if agent._force_ascii_payload:
        _sanitize_structure_non_ascii(api_kwargs)
    if agent.api_mode == "codex_responses":
        api_kwargs = agent._get_transport().preflight_kwargs(
            api_kwargs, allow_stream=False, is_github_responses=agent._is_copilot_url(),
            sanitize_harmony_tokens=agent._is_codex_backend(),
        )
    # OpenRouter caching replays identical responses, even empty ones; an empty-response
    # retry must bypass the cache.
    if agent._empty_content_retries > 0 and agent._is_openrouter_url():
        _set_extra_header(api_kwargs, "X-OpenRouter-Cache", "false")
    # Copilot x-initiator: first call of a user turn is "user" (billed premium);
    # tool-loop follow-ups keep the default "agent".
    if getattr(agent, "_is_user_initiated_turn", False) and agent._is_copilot_url():
        _set_extra_header(api_kwargs, "x-initiator", "user")
        agent._is_user_initiated_turn = False
    try:
        from hermes_cli.middleware import apply_llm_request_middleware

        _llm_request_mw = apply_llm_request_middleware(
            api_kwargs, task_id=effective_task_id, turn_id=turn_id, api_request_id=api_request_id,
            session_id=agent.session_id or "", platform=agent.platform or "", model=agent.model,
            provider=agent.provider, base_url=agent.base_url, api_mode=agent.api_mode,
            api_call_count=api_call_count,
        )
        api_kwargs = _llm_request_mw.payload
        _original_api_kwargs = _llm_request_mw.original_payload
        _llm_middleware_trace = _llm_request_mw.trace
    except Exception:
        _original_api_kwargs = dict(api_kwargs)
        _llm_middleware_trace = []

    _fire_pre_api_request_hook(
        agent, api_kwargs, api_messages, _llm_middleware_trace, messages=messages,
        original_user_message=original_user_message, approx_tokens=approx_tokens,
        total_chars=total_chars, retry_count=retry_count, api_call_count=api_call_count,
        api_request_id=api_request_id, api_start_time=api_start_time,
        effective_task_id=effective_task_id, turn_id=turn_id,
    )

    if env_var_enabled("HERMES_DUMP_REQUESTS"):
        agent._dump_api_request_debug(api_kwargs, reason="preflight")

    # Private to the in-process MoA facade; added after middleware/hooks/debug dumps so
    # none serializes it into the provider payload. Re-read the live client:
    # rotation/fallback/cleanup rebuild agent.client between attempts; a native OpenAI
    # client rejects this key (TypeError).
    if _moa_prepared_request is not None and agent.provider == "moa":
        if _moa_client_consumes_prepared_request(agent.client):
            api_kwargs["_moa_prepared_request"] = _moa_prepared_request
        else:
            logger.warning(
                "MoA client replaced mid-turn (client=%s); sending the "
                "prepared prompt without the MoA handshake",
                type(agent.client).__name__,
            )
    return ApiRequestBuild(
        "fallthrough", api_messages, _moa_prepared_request, tools_for_api, api_kwargs,
        _original_api_kwargs, _llm_middleware_trace,
    )
