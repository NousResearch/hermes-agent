"""Tests for the ResponsesApiTransport (Codex)."""

import json
import pytest
from types import SimpleNamespace

from agent.transports import get_transport
from agent.transports.types import NormalizedResponse


@pytest.fixture
def transport():
    import agent.transports.codex  # noqa: F401
    return get_transport("codex_responses")


class TestCodexTransportBasic:

    def test_api_mode(self, transport):
        assert transport.api_mode == "codex_responses"

    def test_registered_on_import(self, transport):
        assert transport is not None

    def test_convert_tools(self, transport):
        tools = [{
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        }]
        result = transport.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "terminal"


class TestCodexBuildKwargs:







    def test_cache_key_is_content_addressed_not_session_id(self, transport):
        """prompt_cache_key is content-addressed from the static prefix
        (instructions + tools), not the session_id. This keeps recurring cron
        jobs — whose session_id carries a per-fire timestamp — on a stable warm
        cache key. The key is a 'pck_' hash and must NOT equal session_id."""
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=messages, tools=[],
            session_id="cron_job42_20260624_143000",
        )
        pck = kw.get("prompt_cache_key", "")
        assert pck.startswith("pck_")
        assert pck != "cron_job42_20260624_143000"

    def test_cache_key_stable_across_session_ids(self, transport):
        """Same static prefix + different session_id (e.g. two cron fires of the
        same job) must yield the same prompt_cache_key — the whole point of the
        fix: repeated fires reuse the warm prefix instead of going cold."""
        messages = [{"role": "user", "content": "Hi"}]
        kw1 = transport.build_kwargs(
            model="gpt-5.4", messages=messages, tools=[],
            session_id="cron_job42_20260624_143000",
        )
        kw2 = transport.build_kwargs(
            model="gpt-5.4", messages=messages, tools=[],
            session_id="cron_job42_20260624_143500",
        )
        assert kw1["prompt_cache_key"] == kw2["prompt_cache_key"]


    def test_github_responses_drops_message_item_id_end_to_end(self, transport):
        # #32716: Copilot binds codex_message_items ids to a backend
        # "connection" that doesn't survive credential rotation, a gateway
        # restart, or load-balancer churn — replaying a stale id gets HTTP
        # 401 "input item ID does not belong to this connection", even for
        # ids well under the #27038 64-char length cap. build_kwargs must
        # thread is_github_responses through to the input converter so the
        # id never reaches the request.
        messages = [
            {"role": "system", "content": "You are Hermes."},
            {
                "role": "assistant",
                "content": "pong",
                "codex_message_items": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [{"type": "output_text", "text": "pong"}],
                        "id": "msg_short_but_connection_scoped",
                        "phase": "final_answer",
                    }
                ],
            },
        ]
        kw = transport.build_kwargs(
            model="gpt-5.5", messages=messages, tools=[],
            is_github_responses=True,
        )
        message_item = next(item for item in kw["input"] if item.get("type") == "message")
        assert "id" not in message_item
        assert message_item["phase"] == "final_answer"
        assert message_item["status"] == "in_progress"
        assert message_item["content"] == [{"type": "output_text", "text": "pong"}]



    def test_non_github_responses_keeps_message_item_id_end_to_end(self, transport):
        messages = [
            {"role": "system", "content": "You are Hermes."},
            {
                "role": "assistant",
                "content": "pong",
                "codex_message_items": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "pong"}],
                        "id": "msg_short_id",
                    }
                ],
            },
        ]
        kw = transport.build_kwargs(
            model="gpt-5.5", messages=messages, tools=[],
            is_codex_backend=True,
        )
        message_item = next(item for item in kw["input"] if item.get("type") == "message")
        assert message_item["id"] == "msg_short_id"

    @pytest.mark.parametrize("model", [
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.2",
        "gpt-5.1-codex-max",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.1-chat-latest",
        "gpt-5",
        "gpt-5-codex",
        "gpt-4.1",
        "openai.gpt-5.5-pro",
        "openai/gpt-5.1-codex-2026-01-01",
    ])
    def test_extended_cache_models_set_24h_prompt_cache_retention(self, transport, model):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model=model, messages=messages, tools=[],
            session_id="test-session",
            base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
        )
        assert kw["prompt_cache_retention"] == "24h"

    @pytest.mark.parametrize("model", ["gpt-5.6", "gpt-4o", "o3"])
    def test_prompt_cache_retention_omitted_for_other_model_families(self, transport, model):
        kw = transport.build_kwargs(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            session_id="test-session",
            base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
        )
        assert "prompt_cache_retention" not in kw

    @pytest.mark.parametrize("base_url", [
        "https://api.openai.com/v1",
        "https://example.openai.azure.com/openai/v1",
        "https://api.x.ai/v1",
        "https://models.github.ai/inference",
        "https://api.githubcopilot.com",
        "https://chatgpt.com/backend-api/codex",
        "https://responses.example.com/v1",
        "https://bedrock-mantle.us-west-2.api.aws.example/v1",
        "https://example.com/bedrock-mantle.us-west-2.api.aws/v1",
    ])
    def test_prompt_cache_retention_omitted_for_non_mantle_endpoints(self, transport, base_url):
        kw = transport.build_kwargs(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            base_url=base_url,
        )
        assert "prompt_cache_retention" not in kw

    def test_xai_responses_sends_cache_key_via_extra_body(self, transport):
        """xAI's Responses API documents ``prompt_cache_key`` as the
        body-level cache-routing key (the ``x-grok-conv-id`` header is
        Chat-Completions-only). Passing it via ``extra_body`` is robust
        against openai SDK builds whose ``Responses.stream()`` kwarg
        signature ever drops the field — the body field still serializes
        and reaches xAI either way. The ``x-grok-conv-id`` header is kept
        as a belt-and-braces fallback so cache routing survives even
        when the body field would be stripped by an intermediate proxy.
        Ref: https://docs.x.ai/developers/advanced-api-usage/prompt-caching/maximizing-cache-hits
        """
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="grok-4.3", messages=messages, tools=[],
            session_id="conv-xai-1",
            is_xai_responses=True,
        )
        assert "prompt_cache_key" not in kw
        # Body-level prompt_cache_key is content-addressed (pck_ hash), not the
        # raw session_id, so recurring cron fires stay on a stable warm key.
        eb_pck = kw.get("extra_body", {}).get("prompt_cache_key", "")
        assert eb_pck.startswith("pck_")
        assert eb_pck != "conv-xai-1"
        # x-grok-conv-id stays the session/transcript id, not the cache key.
        assert kw.get("extra_headers", {}).get("x-grok-conv-id") == "conv-xai-1"

    def test_xai_responses_extra_body_preserves_caller_fields(self, transport):
        """When the caller already supplies ``extra_body`` (e.g. via
        request_overrides), the xAI cache-key injection must merge into
        the existing dict instead of overwriting it. Caller-supplied
        ``prompt_cache_key`` wins (setdefault semantics) so user overrides
        aren't silently clobbered by the transport."""
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="grok-4.3", messages=messages, tools=[],
            session_id="conv-xai-1",
            is_xai_responses=True,
            request_overrides={"extra_body": {"prompt_cache_key": "caller-override", "other_field": 42}},
        )
        eb = kw.get("extra_body", {})
        assert eb.get("prompt_cache_key") == "caller-override"
        assert eb.get("other_field") == 42





    @pytest.mark.parametrize("length", [64, 65])
    def test_codex_cache_scope_boundary(self, transport, length):
        session_id = "s" * length
        scope = transport.build_kwargs(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            session_id=session_id,
            is_codex_backend=True,
            request_overrides={"extra_headers": {"x-test": "1"}},
        )["extra_headers"]

        assert scope["x-test"] == "1"
        assert len(scope["session_id"]) <= 64
        assert scope["x-client-request-id"] == scope["session_id"]
        if length == 64:
            assert scope["session_id"] == session_id
        else:
            assert scope["session_id"].startswith("pck_")
            assert scope["session_id"] != session_id












    def test_xai_injects_native_web_search_when_client_web_search_present(self, transport, monkeypatch):
        """When the active/configured search backend is xAI, swap client
        ``web_search`` for Grok's native built-in so server-side search
        completes (otherwise the turn stalls as incomplete → 3 retries).
        Non-conflicting client tools are preserved.
        """
        import agent.transports.codex as codex_mod

        monkeypatch.setattr(codex_mod, "_xai_prefers_native_web_search", lambda: True)
        messages = [{"role": "user", "content": "Find current prices."}]
        kw = transport.build_kwargs(
            model="grok-composer-2.5-fast", messages=messages,
            tools=[
                {"type": "function", "function": {
                    "name": "read_file", "description": "Read a file.",
                    "parameters": {"type": "object",
                                   "properties": {"path": {"type": "string"}}}}},
                {"type": "function", "function": {
                    "name": "web_search", "description": "Search the web.",
                    "parameters": {"type": "object",
                                   "properties": {"query": {"type": "string"}}}}},
            ],
            is_xai_responses=True,
        )
        tool_types = [t.get("type") for t in kw.get("tools", [])]
        assert "web_search" in tool_types, kw.get("tools")
        # Non-conflicting client-side tools are preserved.
        names = [t.get("name") for t in kw.get("tools", []) if t.get("type") == "function"]
        assert "read_file" in names
        assert "web_search" not in names
        assert "hermes_web_search" not in names

    def test_xai_renames_client_web_search_when_firecrawl_configured(self, transport, monkeypatch):
        """Configured Firecrawl (or any non-xai backend) must keep Hermes
        dispatch — rename the wire tool so Grok cannot hijack ``web_search``.
        """
        import agent.transports.codex as codex_mod

        monkeypatch.setattr(codex_mod, "_xai_prefers_native_web_search", lambda: False)
        messages = [{"role": "user", "content": "Find current prices."}]
        kw = transport.build_kwargs(
            model="grok-4.5", messages=messages,
            tools=[
                {"type": "function", "function": {
                    "name": "read_file", "description": "Read a file.",
                    "parameters": {"type": "object",
                                   "properties": {"path": {"type": "string"}}}}},
                {"type": "function", "function": {
                    "name": "web_search", "description": "Search the web.",
                    "parameters": {"type": "object",
                                   "properties": {"query": {"type": "string"}}}}},
            ],
            is_xai_responses=True,
        )
        tools = kw.get("tools", [])
        assert not any(t.get("type") == "web_search" for t in tools), tools
        names = [t.get("name") for t in tools if t.get("type") == "function"]
        assert "read_file" in names
        assert "hermes_web_search" in names
        assert "web_search" not in names

    def test_xai_normalize_maps_client_web_search_alias_back(self, transport, monkeypatch):
        """Alias used on the wire must become ``web_search`` for Hermes dispatch."""
        import agent.transports.codex as codex_mod

        msg = SimpleNamespace(
            content=None,
            reasoning=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    call_id="call_1",
                    response_item_id="fc_1",
                    function=SimpleNamespace(
                        name=codex_mod._XAI_CLIENT_WEB_SEARCH_ALIAS,
                        arguments='{"query":"hermes"}',
                    ),
                )
            ],
            codex_reasoning_items=None,
            codex_message_items=None,
            reasoning_details=None,
        )
        response = SimpleNamespace(output=[], status="completed")

        monkeypatch.setattr(
            "agent.codex_responses_adapter._normalize_codex_response",
            lambda resp, issuer_kind=None: (msg, "tool_calls"),
        )
        normalized = transport.normalize_response(response)

        assert normalized.tool_calls is not None
        assert len(normalized.tool_calls) == 1
        assert normalized.tool_calls[0].name == "web_search"

    def test_xai_does_not_inject_native_web_search_without_client_web_search(self, transport):
        """The native ``web_search`` built-in is a 1:1 swap for an
        already-requested client ``web_search`` — NOT an additive grant.  A
        turn whose toolset has no ``web_search`` (user never enabled the web
        toolset) must not get Grok server-side search force-injected, which
        would silently bypass Hermes's web-provider config and tool-trace
        plumbing for every xai-oauth turn.
        """
        messages = [{"role": "user", "content": "Read this file."}]
        kw = transport.build_kwargs(
            model="grok-composer-2.5-fast", messages=messages,
            tools=[{"type": "function", "function": {
                "name": "read_file", "description": "Read a file.",
                "parameters": {"type": "object",
                               "properties": {"path": {"type": "string"}}}}}],
            is_xai_responses=True,
        )
        tools = kw.get("tools", [])
        assert not any(t.get("type") == "web_search" for t in tools), tools
        names = [t.get("name") for t in tools if t.get("type") == "function"]
        assert "read_file" in names


    def test_non_xai_path_does_not_inject_native_web_search(self, transport):
        """Native web_search injection is scoped to xAI — Codex/GitHub paths
        keep the client-side web_search function untouched."""
        messages = [{"role": "user", "content": "Search."}]
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=messages,
            tools=[{"type": "function", "function": {
                "name": "web_search", "description": "Search the web.",
                "parameters": {"type": "object",
                               "properties": {"query": {"type": "string"}}}}}],
            is_xai_responses=False,
        )
        tools = kw.get("tools", [])
        assert not any(t.get("type") == "web_search" for t in tools)
        assert any(
            t.get("type") == "function" and t.get("name") == "web_search"
            for t in tools
        )

    # --- Grok reasoning-effort capability allowlist ---
    # api.x.ai 400s with "Model X does not support parameter reasoningEffort"
    # on grok-4 / grok-4-fast / grok-3 / grok-code-fast / grok-4.20-0309-*.
    # Those models reason natively but don't expose the dial. The transport
    # must omit the `reasoning` key for them.  As of May 2026 we DO request
    # ``reasoning.encrypted_content`` back from xAI on every model —
    # see test_xai_reasoning_effort_passed for the rationale.

    def test_xai_grok_4_20_0309_variants_omit_reasoning_effort(self, transport):
        """grok-4.20-0309-(non-)reasoning reject the effort dial.

        Counterintuitively, only grok-4.20-multi-agent-0309 accepts it.
        """
        messages = [{"role": "user", "content": "Hi"}]
        for model in ("grok-4.20-0309-reasoning", "grok-4.20-0309-non-reasoning"):
            kw = transport.build_kwargs(
                model=model, messages=messages, tools=[],
                is_xai_responses=True,
                reasoning_config={"effort": "high"},
            )
            assert "reasoning" not in kw, f"{model} must not receive reasoning"


class TestXaiWebSearchBackendPreference:
    """``_xai_prefers_native_web_search`` must honor web backend config."""

    def test_explicit_firecrawl_prefers_client(self, monkeypatch):
        import agent.transports.codex as codex_mod

        monkeypatch.setattr(
            "agent.web_search_registry.get_active_search_provider",
            lambda: SimpleNamespace(name="firecrawl"),
        )
        assert codex_mod._xai_prefers_native_web_search() is False

    def test_explicit_search_backend_xai_prefers_native(self, monkeypatch):
        import agent.transports.codex as codex_mod

        monkeypatch.setattr(
            "agent.web_search_registry.get_active_search_provider",
            lambda: SimpleNamespace(name="xai"),
        )
        assert codex_mod._xai_prefers_native_web_search() is True

    def test_resolved_non_xai_provider_prefers_client(self, monkeypatch):
        import agent.transports.codex as codex_mod

        monkeypatch.setattr(
            "agent.web_search_registry.get_active_search_provider",
            lambda: SimpleNamespace(name="firecrawl"),
        )
        assert codex_mod._xai_prefers_native_web_search() is False

    def test_no_provider_legacy_fallback_xai(self, monkeypatch):
        """When no provider is registered, fall back to _get_search_backend."""
        import agent.transports.codex as codex_mod

        monkeypatch.setattr(
            "agent.web_search_registry.get_active_search_provider",
            lambda: None,
        )
        monkeypatch.setattr(
            "tools.web_tools._get_search_backend",
            lambda: "xai",
        )
        assert codex_mod._xai_prefers_native_web_search() is True

    def test_no_provider_legacy_fallback_non_xai(self, monkeypatch):
        """When no provider is registered and backend isn't xai, keep client."""
        import agent.transports.codex as codex_mod

        monkeypatch.setattr(
            "agent.web_search_registry.get_active_search_provider",
            lambda: None,
        )
        monkeypatch.setattr(
            "tools.web_tools._get_search_backend",
            lambda: "firecrawl",
        )
        assert codex_mod._xai_prefers_native_web_search() is False


class TestCodexValidateResponse:

    def test_none_response(self, transport):
        assert transport.validate_response(None) is False


    def test_valid_output(self, transport):
        r = SimpleNamespace(output=[{"type": "message", "content": []}])
        assert transport.validate_response(r) is True




    @pytest.mark.parametrize("reason", ["max_output_tokens", "length", "", None])
    def test_empty_output_other_incomplete_reasons_remain_invalid(self, transport, reason):
        r = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason=reason),
            output=[],
            output_text="",
        )
        assert transport.validate_response(r) is False


class TestCodexMapFinishReason:

    def test_completed(self, transport):
        assert transport.map_finish_reason("completed") == "stop"





class TestCodexNormalizeResponse:

    def test_text_response(self, transport):
        """Normalize a simple text Codex response."""
        r = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="Hello world")],
                    status="completed",
                ),
            ],
            status="completed",
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  input_tokens_details=None, output_tokens_details=None),
        )
        nr = transport.normalize_response(r)
        assert isinstance(nr, NormalizedResponse)
        assert nr.content == "Hello world"
        assert nr.finish_reason == "stop"

    def test_message_items_preserved_in_provider_data(self, transport):
        """Codex assistant message item ids/phases must survive transport normalization."""
        r = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    id="msg_abc",
                    phase="final_answer",
                    content=[SimpleNamespace(type="output_text", text="Hello world")],
                    status="completed",
                ),
            ],
            status="completed",
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  input_tokens_details=None, output_tokens_details=None),
        )
        nr = transport.normalize_response(r)
        assert nr.codex_message_items == [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello world"}],
                "id": "msg_abc",
                "phase": "final_answer",
            }
        ]

    def test_tool_call_response(self, transport):
        """Normalize a Codex response with tool calls."""
        r = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_abc123",
                    name="terminal",
                    arguments=json.dumps({"command": "ls"}),
                    id="fc_abc123",
                    status="completed",
                ),
            ],
            status="completed",
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=10, output_tokens=20,
                                  input_tokens_details=None, output_tokens_details=None),
        )
        nr = transport.normalize_response(r)
        assert nr.finish_reason == "tool_calls"
        assert len(nr.tool_calls) == 1
        tc = nr.tool_calls[0]
        assert tc.name == "terminal"
        assert '"command"' in tc.arguments



class TestCodexTransportTimeout:
    """Forward per-request timeout from build_kwargs to the SDK kwargs."""

    def test_positive_timeout_preserved(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            timeout=600.0,
        )
        assert kw.get("timeout") == 600.0



    def test_inf_timeout_dropped(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            timeout=float("inf"),
        )
        assert "timeout" not in kw




class TestCodexTransportXaiServiceTierStrip:
    """xAI Responses API rejects ``service_tier`` (#28490).

    ``resolve_fast_mode_overrides`` only returns ``service_tier`` for
    OpenAI fast-eligible models, so on paper the field should never
    reach a Grok request.  But ``self.service_tier`` lingers across
    model switches and can also be set directly via ``agent.service_tier``
    in config.yaml — both leak paths plumb through ``request_overrides``
    and would 400 against xAI's ``/v1/responses``.
    Strip defensively when targeting xAI.
    """

    @pytest.fixture
    def transport(self):
        from agent.transports.codex import ResponsesApiTransport
        return ResponsesApiTransport()

    def test_xai_strips_service_tier_from_request_overrides(self, transport):
        """Headline #28490 case: service_tier=priority leaks through
        request_overrides, must not reach the xAI request body."""
        kw = transport.build_kwargs(
            model="grok-4.3",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            is_xai_responses=True,
            request_overrides={"service_tier": "priority"},
        )
        assert "service_tier" not in kw, (
            f"service_tier must be stripped on xAI requests, "
            f"got {kw.get('service_tier')!r}"
        )

    def test_non_xai_codex_preserves_service_tier(self, transport):
        """The strip is xAI-only — native Codex DOES accept
        service_tier=priority (OpenAI Priority Processing).  Stripping
        it elsewhere would silently disable the user's fast-mode opt-in.
        """
        kw = transport.build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            is_xai_responses=False,
            is_codex_backend=True,
            request_overrides={"service_tier": "priority"},
        )
        assert kw.get("service_tier") == "priority", (
            "non-xAI codex_responses providers must keep service_tier"
        )

    def test_github_responses_preserves_service_tier(self, transport):
        """GitHub Models (Copilot) is another codex_responses surface
        that should not be affected by the xAI strip."""
        kw = transport.build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            is_github_responses=True,
            request_overrides={"service_tier": "priority"},
        )
        assert kw.get("service_tier") == "priority"


class TestPreflightSlashEnumStrip:
    """xAI Responses safety-net: strip slash-containing enum values
    when the model name indicates a Grok target (#28490).

    Native Codex accepts ``/``-containing enums; xAI rejects them with
    HTTP 400 "Invalid arguments passed to the model".  The main agent
    loop and the auxiliary client already sanitize at request-build
    time; this preflight catches any future code path that bypasses
    those — gated on model name so we don't unnecessarily strip on
    non-xAI providers.
    """

    def _make_kwargs(self, model: str, enum_values: list[str]) -> dict:
        return {
            "model": model,
            "instructions": "test",
            "input": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "name": "pick_model",
                    "description": "pick a model",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_id": {
                                "type": "string",
                                "enum": enum_values,
                            },
                        },
                    },
                },
            ],
        }

    def test_grok_model_strips_slash_enum_values(self):
        """When the model name is Grok-family, slash-containing enum
        values are stripped so xAI doesn't 400 on the tool schema."""
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs
        kwargs = self._make_kwargs(
            "grok-4.3",
            ["Qwen/Qwen3.5-0.8B", "openai/gpt-oss-20b", "plain-id"],
        )
        result = _preflight_codex_api_kwargs(kwargs)
        # The enum keyword itself is stripped (per strip_slash_enum's
        # semantics — it removes the constraint entirely when any value
        # contains /).
        params = result["tools"][0]["parameters"]
        assert "enum" not in params["properties"]["model_id"], (
            "slash-containing enum must be stripped on Grok"
        )

    def test_aggregator_prefixed_grok_also_strips(self):
        """Aggregator-prefixed (x-ai/grok-*) names hit the same path."""
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs
        kwargs = self._make_kwargs(
            "x-ai/grok-4.3",
            ["Qwen/Qwen3.5-0.8B"],
        )
        result = _preflight_codex_api_kwargs(kwargs)
        assert "enum" not in result["tools"][0]["parameters"]["properties"]["model_id"]

    def test_non_grok_model_preserves_slash_enum_values(self):
        """Native Codex / GitHub Models DO accept slash-containing
        enums.  The safety-net must NOT strip there or we silently
        degrade tool-schema constraints on every codex_responses
        provider that isn't xAI."""
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs
        kwargs = self._make_kwargs(
            "gpt-5.5",
            ["Qwen/Qwen3.5-0.8B", "plain-id"],
        )
        result = _preflight_codex_api_kwargs(kwargs)
        params = result["tools"][0]["parameters"]
        # The enum must survive on non-xAI providers.
        assert params["properties"]["model_id"].get("enum") == [
            "Qwen/Qwen3.5-0.8B", "plain-id"
        ]
