"""OpenAI Responses API (Codex) transport.

Delegates to the existing adapter functions in agent/codex_responses_adapter.py.
This transport owns format conversion and normalization — NOT client lifecycle,
streaming, or the _run_codex_stream() call path.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall


def _content_cache_key(instructions: str, tools: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Content-address the prompt cache key from the static request prefix.

    Returns ``pck_<sha256[:24]>`` of (instructions + sorted tool schemas), or
    None when there is nothing static to key on. The cache key is a routing
    hint only — never a correctness boundary — so two requests sharing a system
    prompt and tool set intentionally resolve to the same warm prefix bucket.

    The fix this exists for: recurring cron jobs build session_id as
    ``cron_<id>_<timestamp>``, so using session_id as the cache key made every
    fire cache-cold. The static prefix (identity + tools) is identical across
    fires, so hashing it gives a stable key that stays warm within the
    provider's cache TTL. Sorting tools by name keeps the hash insertion-order
    independent.
    """
    if not instructions and not tools:
        return None
    tools_part = ""
    if tools:
        sorted_tools = sorted(
            (t for t in tools if isinstance(t, dict)),
            key=lambda t: str(t.get("name") or t.get("type") or ""),
        )
        tools_part = json.dumps(
            sorted_tools, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    # \x00 separator so instructions ending in the tool JSON can't collide with
    # a request whose instructions contain that JSON and whose tools are empty.
    content = f"{instructions or ''}\x00{tools_part}"
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"pck_{digest}"


class ResponsesApiTransport(ProviderTransport):
    """Transport for api_mode='codex_responses'.

    Wraps the functions extracted into codex_responses_adapter.py (PR 1).
    """

    # Issuer kind of the most recent build_kwargs / convert_messages call.
    # Used as a fallback when normalize_response is invoked without an
    # explicit ``issuer_kind`` kwarg, so reasoning items captured from a
    # response are stamped with the endpoint that minted them. Plain class
    # attribute default; mutated on the instance, not the class.
    _last_issuer_kind: Optional[str] = None

    @property
    def api_mode(self) -> str:
        return "codex_responses"

    def _resolve_issuer_kind(self, params: Dict[str, Any]) -> str:
        """Classify the current Responses endpoint from transport params."""
        from agent.codex_responses_adapter import _classify_responses_issuer
        return _classify_responses_issuer(
            is_xai_responses=params.get("is_xai_responses") is True,
            is_github_responses=params.get("is_github_responses") is True,
            is_codex_backend=params.get("is_codex_backend") is True,
            base_url=params.get("base_url"),
        )

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI chat messages to Responses API input items."""
        from agent.codex_responses_adapter import _chat_messages_to_responses_input
        issuer = self._resolve_issuer_kind(kwargs)
        self._last_issuer_kind = issuer
        return _chat_messages_to_responses_input(
            messages,
            is_xai_responses=kwargs.get("is_xai_responses") is True,
            is_github_responses=kwargs.get("is_github_responses") is True,
            replay_encrypted_reasoning=bool(
                kwargs.get("replay_encrypted_reasoning", True)
            ),
            current_issuer_kind=issuer,
        )

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI tool schemas to Responses API function definitions."""
        from agent.codex_responses_adapter import _responses_tools
        return _responses_tools(tools)

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build Responses API kwargs.

        Calls convert_messages and convert_tools internally.

        params:
            instructions: str — system prompt (extracted from messages[0] if not given)
            reasoning_config: dict | None — {effort, enabled}
            session_id: str | None — transcript/session id; drives the xAI
                x-grok-conv-id header and the Codex cache-scope headers, and is
                the fallback prompt_cache_key when there is no static prefix to
                content-address
            max_tokens: int | None — max_output_tokens
            timeout: float | None — per-request timeout forwarded to the SDK
            request_overrides: dict | None — extra kwargs merged in
            provider: str | None — provider name for backend-specific logic
            base_url: str | None — endpoint URL
            base_url_hostname: str | None — hostname for backend detection
            is_github_responses: bool — Copilot/GitHub models backend
            is_codex_backend: bool — chatgpt.com/backend-api/codex
            is_xai_responses: bool — xAI/Grok backend
            github_reasoning_extra: dict | None — Copilot reasoning params
        """
        from agent.codex_responses_adapter import (
            _chat_messages_to_responses_input,
            _responses_tools,
        )

        from run_agent import DEFAULT_AGENT_IDENTITY

        instructions = params.get("instructions", "")
        payload_messages = messages
        if not instructions:
            if messages and messages[0].get("role") == "system":
                instructions = str(messages[0].get("content") or "").strip()
                payload_messages = messages[1:]
        if not instructions:
            instructions = DEFAULT_AGENT_IDENTITY

        is_github_responses = params.get("is_github_responses") is True
        is_codex_backend = params.get("is_codex_backend") is True
        is_xai_responses = params.get("is_xai_responses") is True
        replay_encrypted_reasoning = bool(
            params.get("replay_encrypted_reasoning", True)
        )

        # Resolve the issuing endpoint for this call. Stashed on the
        # transport so normalize_response can stamp it onto reasoning
        # items captured from the response, and passed to the input
        # converter so foreign-issuer reasoning blocks in history are
        # dropped before the API rejects them.
        issuer_kind = self._resolve_issuer_kind(params)
        self._last_issuer_kind = issuer_kind

        # Resolve reasoning effort
        reasoning_effort = "medium"
        reasoning_enabled = True
        reasoning_config = params.get("reasoning_config")
        if reasoning_config and isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                reasoning_enabled = False
            elif reasoning_config.get("effort"):
                reasoning_effort = reasoning_config["effort"]

        _effort_clamp = {"minimal": "low"}
        if "gpt-5.6" in (model or "").lower():
            # Ultra is the Codex product tier; the Responses API wire value is max.
            _effort_clamp["ultra"] = "max"
        if params.get("is_xai_responses", False):
            # xAI Responses tops out at high; keep generic stronger values usable.
            _effort_clamp.update({"xhigh": "high", "max": "high", "ultra": "high"})
        reasoning_effort = _effort_clamp.get(reasoning_effort, reasoning_effort)

        response_tools = _responses_tools(tools)

        # xAI server-side web search.
        #
        # grok models on xAI's /v1/responses surface (notably
        # grok-composer-2.5-fast on SuperGrok OAuth) have a *native*,
        # server-executed web search.  When the model is handed a
        # client-side function literally named ``web_search``, it routes
        # the intent to that native engine — but because the tool is
        # declared as a plain ``function`` rather than xAI's first-class
        # ``{"type": "web_search"}`` built-in, the server-side search is
        # dispatched but never reconciled: the response streams reasoning
        # + ``web_search_call`` progress items, the searches never reach
        # ``status="completed"`` in the assembled output, no final
        # message is emitted, and ``_normalize_codex_response`` correctly
        # sees reasoning-with-no-answer and reports ``incomplete``.  The
        # turn then burns 3 continuation retries and fails with "Codex
        # response remained incomplete after 3 continuation attempts".
        # Verified live against grok-composer-2.5-fast (2026-06).
        #
        # Fix: when the agent HAS a client-side ``web_search`` function (i.e.
        # the user enabled the web toolset), declare xAI's native
        # ``web_search`` built-in instead so the search actually runs to
        # completion server-side and the model streams a real answer.  The
        # Responses API rejects two tools sharing the name ``web_search``
        # (HTTP 400 "Duplicate tool names"), so we drop the client-side
        # ``web_search`` function for the xAI path and let the native tool
        # satisfy it.  All other client-side tools (read_file, terminal,
        # web_extract, MCP tools, …) are untouched and continue to dispatch
        # through Hermes's agent loop.
        #
        # Scope: we ONLY swap in the native built-in when the client
        # ``web_search`` was actually present.  We do NOT force-enable Grok
        # server-side search on turns where the user never had web enabled —
        # that would silently route around Hermes's web-provider config and
        # tool-trace/citation plumbing for every xai-oauth turn.  The swap is
        # a 1:1 replacement of an already-requested capability, not an
        # additive grant.
        #
        # NOTE: for the swapped case this routes ``web_search`` to Grok's
        # native search engine for xAI sessions instead of Hermes's
        # configured web provider (Tavily/etc.), and those results bypass
        # Hermes's tool-trace / citation plumbing (they arrive baked into the
        # model's answer rather than as a tool result the loop observes).
        # Scoped to ``is_xai_responses`` deliberately; narrow to specific
        # models if a future grok variant should keep the client-side
        # function.
        if is_xai_responses and response_tools:
            has_client_web_search = any(
                isinstance(t, dict) and t.get("name") == "web_search"
                for t in response_tools
            )
            if has_client_web_search:
                filtered = [
                    t for t in response_tools
                    if not (isinstance(t, dict) and t.get("name") == "web_search")
                ]
                filtered.append({"type": "web_search"})
                response_tools = filtered

        # ``tools`` MUST be omitted entirely when there are no functions to
        # expose: the openai SDK's ``responses.stream()`` / ``responses.parse()``
        # eagerly call ``_make_tools(tools)`` which does ``for tool in tools``
        # without a None guard, so passing ``tools=None`` raises
        # ``TypeError: 'NoneType' object is not iterable`` before any HTTP
        # request is issued (openai==2.24.0).  Reported for the
        # ``openai-codex`` / ``gpt-5.5`` combo on chatgpt.com/backend-api/codex
        # (#32892) when the agent runs without external tools registered.
        kwargs = {
            "model": model,
            "instructions": instructions,
            "input": _chat_messages_to_responses_input(
                payload_messages,
                is_xai_responses=is_xai_responses,
                is_github_responses=is_github_responses,
                replay_encrypted_reasoning=replay_encrypted_reasoning,
                current_issuer_kind=issuer_kind,
            ),
            "store": False,
        }
        if response_tools:
            kwargs["tools"] = response_tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = True

        session_id = params.get("session_id")
        # prompt_cache_key is content-addressed from the static prefix
        # (instructions + tools), NOT session_id — recurring cron jobs carry a
        # per-fire timestamp in session_id (cron_<id>_<ts>) that made every run
        # cache-cold. session_id is left untouched for transcript isolation and
        # the cache-scope routing headers below. Falls back to session_id when
        # there is no static content to hash.
        cache_key = _content_cache_key(instructions, response_tools) or session_id
        # xAI Responses takes prompt_cache_key in extra_body (set further
        # down); GitHub Models opts out of cache-key routing entirely.
        if not is_github_responses and not is_xai_responses and cache_key:
            kwargs["prompt_cache_key"] = cache_key

        if reasoning_enabled and is_xai_responses:
            from agent.model_metadata import grok_supports_reasoning_effort

            # Ask xAI to echo back encrypted reasoning items so we can
            # replay them on subsequent turns for cross-turn coherence.
            # See agent/codex_responses_adapter._chat_messages_to_responses_input
            # for the May 2026 reversal of the earlier suppression gate.
            kwargs["include"] = (
                ["reasoning.encrypted_content"] if replay_encrypted_reasoning else []
            )
            # xAI rejects `reasoning.effort` on grok-4 / grok-4-fast / grok-3
            # / grok-code-fast / grok-4.20-0309-* with HTTP 400 even though
            # those models reason natively. Only send the effort dial when
            # the target model is on the allowlist; otherwise send no
            # `reasoning` key at all and let the model reason on its own.
            if grok_supports_reasoning_effort(model):
                kwargs["reasoning"] = {"effort": reasoning_effort}
        elif reasoning_enabled:
            if is_github_responses:
                github_reasoning = params.get("github_reasoning_extra")
                if github_reasoning is not None:
                    kwargs["reasoning"] = github_reasoning
            else:
                kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
                kwargs["include"] = (
                    ["reasoning.encrypted_content"] if replay_encrypted_reasoning else []
                )
        elif not is_github_responses and not is_xai_responses:
            kwargs["include"] = []

        request_overrides = params.get("request_overrides")
        if request_overrides:
            kwargs.update(request_overrides)

        # xAI Responses API rejects ``service_tier`` (HTTP 400 "Argument not
        # supported: service_tier") — hit when ``/fast`` priority-processing
        # mode lingers from a prior model in the same session, or when a
        # user explicitly sets ``agent.service_tier`` in config.yaml.  The
        # main-loop guard (``resolve_fast_mode_overrides`` only returns
        # ``service_tier`` for OpenAI fast-eligible models) doesn't cover
        # those leak paths, so strip defensively when targeting xAI.  See
        # #28490 for the original report.
        if is_xai_responses:
            kwargs.pop("service_tier", None)

        # Forward per-request timeout to the SDK so OpenAI/Anthropic clients
        # honor it.  Without this, ``providers.<id>.request_timeout_seconds``
        # is silently dropped on the main agent Codex path while the
        # chat_completions path and auxiliary Codex adapter both forward it.
        timeout = kwargs.get("timeout", params.get("timeout"))
        if (
            isinstance(timeout, (int, float))
            and not isinstance(timeout, bool)
            and 0 < float(timeout) < float("inf")
        ):
            kwargs["timeout"] = float(timeout)
        else:
            kwargs.pop("timeout", None)

        if is_codex_backend:
            # The Codex backend rejects body-level ``extra_headers`` with
            # HTTP 400, but the OpenAI SDK's ``extra_headers`` kwarg maps
            # to actual HTTP request headers (not body fields).  We need
            # these headers for cache-scope routing so prompt cache hits
            # remain high.  Send session_id / x-client-request-id as HTTP
            # headers while keeping ``prompt_cache_key`` in the body for
            # standard OpenAI routing as a belt-and-braces fallback.
            cache_scope_id = str(session_id or "").strip()
            if cache_scope_id:
                existing_extra_headers = kwargs.get("extra_headers")
                merged_extra_headers: Dict[str, str] = {}
                if isinstance(existing_extra_headers, dict):
                    merged_extra_headers.update(
                        {
                            str(key): str(value)
                            for key, value in existing_extra_headers.items()
                            if key and value is not None
                        }
                    )
                merged_extra_headers["session_id"] = cache_scope_id
                merged_extra_headers["x-client-request-id"] = cache_scope_id
                kwargs["extra_headers"] = merged_extra_headers

        max_tokens = params.get("max_tokens")
        if max_tokens is not None and not is_codex_backend:
            kwargs["max_output_tokens"] = max_tokens

        if is_xai_responses and session_id:
            existing_extra_headers = kwargs.get("extra_headers")
            merged_extra_headers: Dict[str, str] = {}
            if isinstance(existing_extra_headers, dict):
                merged_extra_headers.update(
                    {
                        str(key): str(value)
                        for key, value in existing_extra_headers.items()
                        if key and value is not None
                    }
                )
            merged_extra_headers["x-grok-conv-id"] = session_id
            kwargs["extra_headers"] = merged_extra_headers

            # xAI Responses cache-routing — body-level field per
            # https://docs.x.ai/developers/advanced-api-usage/prompt-caching/maximizing-cache-hits.
            # Sent via extra_body (not the typed kwarg) so it survives openai
            # SDK builds whose Responses.stream() signature has dropped the field.
            existing_extra_body = kwargs.get("extra_body")
            merged_extra_body: Dict[str, Any] = {}
            if isinstance(existing_extra_body, dict):
                merged_extra_body.update(existing_extra_body)
            merged_extra_body.setdefault("prompt_cache_key", cache_key)
            kwargs["extra_body"] = merged_extra_body

        return kwargs

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize Codex Responses API response to NormalizedResponse."""
        from agent.codex_responses_adapter import (
            _normalize_codex_response,
        )

        # Issuer for this response = explicit kwarg if the caller knows it,
        # otherwise the stash from the matching build_kwargs/convert_messages
        # call. Either way it gets stamped onto reasoning items so future
        # turns can detect a model swap and drop foreign-issuer blobs.
        issuer_kind = kwargs.get("issuer_kind") or self._last_issuer_kind
        # _normalize_codex_response returns (SimpleNamespace, finish_reason_str)
        msg, finish_reason = _normalize_codex_response(response, issuer_kind=issuer_kind)

        tool_calls = None
        if msg and msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                provider_data = {}
                if hasattr(tc, "call_id") and tc.call_id:
                    provider_data["call_id"] = tc.call_id
                if hasattr(tc, "response_item_id") and tc.response_item_id:
                    provider_data["response_item_id"] = tc.response_item_id
                tool_calls.append(ToolCall(
                    id=tc.id if hasattr(tc, "id") else (tc.function.name if hasattr(tc, "function") else None),
                    name=tc.function.name if hasattr(tc, "function") else getattr(tc, "name", ""),
                    arguments=tc.function.arguments if hasattr(tc, "function") else getattr(tc, "arguments", "{}"),
                    provider_data=provider_data or None,
                ))

        # Extract reasoning items for provider_data
        provider_data = {}
        if msg and hasattr(msg, "codex_reasoning_items") and msg.codex_reasoning_items:
            provider_data["codex_reasoning_items"] = msg.codex_reasoning_items
        if msg and hasattr(msg, "codex_message_items") and msg.codex_message_items:
            provider_data["codex_message_items"] = msg.codex_message_items
        if msg and hasattr(msg, "reasoning_details") and msg.reasoning_details:
            provider_data["reasoning_details"] = msg.reasoning_details

        return NormalizedResponse(
            content=msg.content if msg else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            reasoning=msg.reasoning if msg and hasattr(msg, "reasoning") else None,
            usage=None,  # Codex usage is extracted separately in normalize_usage()
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Check Codex Responses API response has valid output structure.

        Returns True only if response.output is a non-empty list.
        Does NOT check output_text fallback — the caller handles that
        with diagnostic logging for stream backfill recovery.
        """
        if response is None:
            return False
        output = getattr(response, "output", None)
        if not isinstance(output, list) or not output:
            return False
        return True

    def preflight_kwargs(
        self,
        api_kwargs: Any,
        *,
        allow_stream: bool = False,
        is_github_responses: bool = False,
    ) -> dict:
        """Validate and sanitize Codex API kwargs before the call.

        Normalizes input items, strips unsupported fields, validates structure.
        """
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs
        return _preflight_codex_api_kwargs(
            api_kwargs,
            allow_stream=allow_stream,
            is_github_responses=is_github_responses,
        )

    def map_finish_reason(self, raw_reason: str) -> str:
        """Map Codex response.status to OpenAI finish_reason.

        Codex uses response.status ('completed', 'incomplete') +
        response.incomplete_details.reason for granular mapping.
        This method handles the simple status string; the caller
        should check incomplete_details separately for 'max_output_tokens'.
        """
        _MAP = {
            "completed": "stop",
            "incomplete": "length",
            "failed": "stop",
            "cancelled": "stop",
        }
        return _MAP.get(raw_reason, "stop")


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("codex_responses", ResponsesApiTransport)
