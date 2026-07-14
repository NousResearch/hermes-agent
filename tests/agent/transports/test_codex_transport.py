"""Tests for the ResponsesApiTransport (Codex)."""

import json
import pytest
from types import SimpleNamespace

from agent.codex_responses_adapter import _preflight_codex_api_kwargs
from agent.output_verbosity import supports_openai_output_verbosity
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

    def test_basic_kwargs(self, transport):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        kw = transport.build_kwargs(
            model="gpt-5.4",
            messages=messages,
            tools=[],
        )
        assert kw["model"] == "gpt-5.4"
        assert kw["instructions"] == "You are helpful."
        assert "input" in kw
        assert kw["store"] is False

    def test_system_extracted_from_messages(self, transport):
        messages = [
            {"role": "system", "content": "Custom system prompt"},
            {"role": "user", "content": "Hi"},
        ]
        kw = transport.build_kwargs(model="gpt-5.4", messages=messages, tools=[])
        assert kw["instructions"] == "Custom system prompt"

    def test_no_system_uses_default(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(model="gpt-5.4", messages=messages, tools=[])
        assert kw["instructions"]  # should be non-empty default

    def test_reasoning_config(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=messages, tools=[],
            reasoning_config={"effort": "high"},
        )
        assert kw.get("reasoning", {}).get("effort") == "high"

    @pytest.mark.parametrize("effort, wire_effort", [("max", "max"), ("ultra", "max")])
    def test_extended_reasoning_efforts_use_api_wire_value(self, transport, effort, wire_effort):
        kw = transport.build_kwargs(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert kw.get("reasoning", {}).get("effort") == wire_effort

    def test_reasoning_disabled(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=messages, tools=[],
            reasoning_config={"enabled": False},
        )
        assert "reasoning" not in kw or kw.get("include") == []

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

    def test_github_responses_no_cache_key(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=messages, tools=[],
            session_id="test-session",
            is_github_responses=True,
        )
        assert "prompt_cache_key" not in kw

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

    def test_github_responses_requires_literal_true(self, transport):
        messages = [
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
            is_github_responses="false",
        )

        message_item = next(item for item in kw["input"] if item.get("type") == "message")
        assert message_item["id"] == "msg_short_id"

    def test_github_preflight_drops_id_reintroduced_by_request_override(self, transport):
        injected = {
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [{"type": "output_text", "text": "pong"}],
            "id": "stale_short",
            "phase": "final_answer",
        }
        kw = transport.build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "continue"}],
            tools=[],
            is_github_responses=True,
            request_overrides={"input": [injected]},
        )

        preflight = transport.preflight_kwargs(
            kw,
            is_github_responses=True,
        )

        message_item = preflight["input"][0]
        assert "id" not in message_item
        assert message_item["status"] == "in_progress"
        assert message_item["phase"] == "final_answer"
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

    def test_max_tokens(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=messages, tools=[],
            max_tokens=4096,
        )
        assert kw.get("max_output_tokens") == 4096

    def test_codex_backend_no_max_output_tokens(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=messages, tools=[],
            max_tokens=4096,
            is_codex_backend=True,
        )
        assert "max_output_tokens" not in kw

    @pytest.mark.parametrize("verbosity", ["low", "medium", "high"])
    def test_codex_backend_sets_output_verbosity(self, transport, verbosity):
        kw = transport.build_kwargs(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            is_codex_backend=True,
            supports_output_verbosity=True,
            output_verbosity=verbosity,
        )

        assert kw["text"] == {"verbosity": verbosity}

        preflight = transport.preflight_kwargs(kw)
        assert preflight["text"] == {"verbosity": verbosity}

    def test_non_codex_responses_omit_output_verbosity(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            output_verbosity="low",
        )

        assert "text" not in kw

    def test_codex_backend_omits_invalid_output_verbosity(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            is_codex_backend=True,
            output_verbosity="extra-short",
        )

        assert "text" not in kw

    def test_output_verbosity_merges_with_request_override_text(self, transport):
        request_overrides = {"text": {"format": {"type": "text"}}}
        kw = transport.build_kwargs(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            is_codex_backend=True,
            supports_output_verbosity=True,
            output_verbosity="low",
            request_overrides=request_overrides,
        )

        assert kw["text"] == {
            "format": {"type": "text"},
            "verbosity": "low",
        }
        assert request_overrides == {"text": {"format": {"type": "text"}}}

    def test_output_verbosity_explicit_request_override_wins(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            is_codex_backend=True,
            output_verbosity="low",
            request_overrides={"text": {"verbosity": "high"}},
        )

        assert kw["text"] == {"verbosity": "high"}

    @pytest.mark.parametrize(
        ("model", "hostname", "is_codex_backend", "expected"),
        [
            ("gpt-5.6-sol", "", True, True),
            ("gpt-5.5", "api.openai.com", False, True),
            ("gpt-4.1", "", True, False),
            ("gpt-5.5", "proxy.example", False, False),
            ("gpt-5.5", "api.x.ai", False, False),
        ],
    )
    def test_output_verbosity_capability_boundary(
        self, model, hostname, is_codex_backend, expected
    ):
        assert (
            supports_openai_output_verbosity(
                model,
                base_url_hostname=hostname,
                is_codex_backend=is_codex_backend,
            )
            is expected
        )

    def test_preflight_allows_output_verbosity_with_text_format(self):
        payload = _preflight_codex_api_kwargs(
            {
                "model": "gpt-5.5",
                "instructions": "system",
                "input": [{"role": "user", "content": "hi"}],
                "store": False,
                "text": {"verbosity": "HIGH", "format": {"type": "text"}},
            }
        )

        assert payload["text"] == {
            "verbosity": "high",
            "format": {"type": "text"},
        }

    def test_codex_backend_sets_cache_routing_headers(self, transport):
        """Codex backend sends session_id / x-client-request-id as HTTP
        headers (via extra_headers) for cache-scope routing."""
        messages = [{"role": "user", "content": "Hi"}]

        kw = transport.build_kwargs(
            model="gpt-5.4",
            messages=messages,
            tools=[],
            session_id="conv-codex-1",
            is_codex_backend=True,
        )

        headers = kw.get("extra_headers", {})
        assert headers.get("session_id") == "conv-codex-1"
        assert headers.get("x-client-request-id") == "conv-codex-1"

    def test_codex_backend_no_headers_without_session_id(self, transport):
        messages = [{"role": "user", "content": "Hi"}]

        kw = transport.build_kwargs(
            model="gpt-5.4",
            messages=messages,
            tools=[],
            is_codex_backend=True,
        )

        assert "extra_headers" not in kw

    def test_codex_backend_preserves_caller_extra_headers(self, transport):
        messages = [{"role": "user", "content": "Hi"}]

        kw = transport.build_kwargs(
            model="gpt-5.4",
            messages=messages,
            tools=[],
            session_id="conv-codex-1",
            is_codex_backend=True,
            request_overrides={"extra_headers": {"x-test": "1"}},
        )

        headers = kw.get("extra_headers", {})
        assert headers.get("x-test") == "1"
        assert headers.get("session_id") == "conv-codex-1"
        assert headers.get("x-client-request-id") == "conv-codex-1"

    def test_non_codex_responses_preserves_caller_extra_headers(self, transport):
        messages = [{"role": "user", "content": "Hi"}]

        kw = transport.build_kwargs(
            model="gpt-5.4",
            messages=messages,
            tools=[],
            is_codex_backend=False,
            request_overrides={"extra_headers": {"x-test": "1"}},
        )

        assert kw["extra_headers"] == {"x-test": "1"}

    def test_xai_headers(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="grok-3", messages=messages, tools=[],
            session_id="conv-123",
            is_xai_responses=True,
        )
        assert kw.get("extra_headers", {}).get("x-grok-conv-id") == "conv-123"

    def test_xai_headers_preserve_request_override_headers(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="grok-3", messages=messages, tools=[],
            session_id="conv-123",
            is_xai_responses=True,
            request_overrides={"extra_headers": {"X-Test": "1", "X-Trace": "abc"}},
        )
        assert kw.get("extra_headers") == {
            "X-Test": "1",
            "X-Trace": "abc",
            "x-grok-conv-id": "conv-123",
        }

    def test_minimal_effort_clamped(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=messages, tools=[],
            reasoning_config={"effort": "minimal"},
        )
        # "minimal" should be clamped to "low"
        assert kw.get("reasoning", {}).get("effort") == "low"

    def test_xai_reasoning_effort_passed(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="grok-4.3", messages=messages, tools=[],
            is_xai_responses=True,
            reasoning_config={"effort": "high"},
        )
        # xAI Responses receives reasoning.effort on the allowlisted models.
        assert kw.get("reasoning") == {"effort": "high"}
        # As of May 2026 (post-revert of PR #26644) we DO request
        # reasoning.encrypted_content back from xAI so we can replay it
        # across turns for cross-turn coherence — xAI explicitly relies
        # on this for their partnership integration.  See
        # tests/run_agent/test_codex_xai_oauth_recovery.py for the
        # full history.
        assert "reasoning.encrypted_content" in kw.get("include", [])

    @pytest.mark.parametrize("effort", ["xhigh", "max", "ultra"])
    def test_xai_stronger_generic_efforts_clamp_to_high(self, transport, effort):
        kw = transport.build_kwargs(
            model="grok-4.3",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            is_xai_responses=True,
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert kw.get("reasoning") == {"effort": "high"}

    def test_xai_injects_native_web_search_when_client_web_search_present(self, transport):
        """xAI path swaps a client-side ``web_search`` function for xAI's
        native server-side ``web_search`` built-in so grok server-side search
        runs to completion (otherwise the turn stalls as
        reasoning-with-no-answer -> false 'incomplete' -> 3 retries -> fail).
        Non-conflicting client tools are preserved.
        """
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

    def test_xai_drops_clientside_web_search_to_avoid_duplicate(self, transport):
        """When the client registers its own 'web_search' function, the xAI
        path must drop it and rely on the native built-in — otherwise xAI
        returns HTTP 400 'Duplicate tool names: web_search'."""
        messages = [{"role": "user", "content": "Search the web."}]
        kw = transport.build_kwargs(
            model="grok-composer-2.5-fast", messages=messages,
            tools=[{"type": "function", "function": {
                "name": "web_search", "description": "Search the web.",
                "parameters": {"type": "object",
                               "properties": {"query": {"type": "string"}}}}}],
            is_xai_responses=True,
        )
        tools = kw.get("tools", [])
        # Exactly one tool named/typed web_search, and it is the native built-in.
        web_search_entries = [
            t for t in tools
            if t.get("name") == "web_search" or t.get("type") == "web_search"
        ]
        assert len(web_search_entries) == 1
        assert web_search_entries[0] == {"type": "web_search"}
        # No client-side function form of web_search survives.
        assert not any(
            t.get("type") == "function" and t.get("name") == "web_search"
            for t in tools
        )

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

    def test_xai_reasoning_disabled_no_reasoning_key(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="grok-4.3", messages=messages, tools=[],
            is_xai_responses=True,
            reasoning_config={"enabled": False},
        )
        # When reasoning is disabled, do not send the reasoning key at all
        assert "reasoning" not in kw

    def test_xai_minimal_effort_clamped(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="grok-4.3", messages=messages, tools=[],
            is_xai_responses=True,
            reasoning_config={"effort": "minimal"},
        )
        # "minimal" should be clamped to "low" for xAI as well
        assert kw.get("reasoning", {}).get("effort") == "low"

    # --- Grok reasoning-effort capability allowlist ---
    # api.x.ai 400s with "Model X does not support parameter reasoningEffort"
    # on grok-4 / grok-4-fast / grok-3 / grok-code-fast / grok-4.20-0309-*.
    # Those models reason natively but don't expose the dial. The transport
    # must omit the `reasoning` key for them.  As of May 2026 we DO request
    # ``reasoning.encrypted_content`` back from xAI on every model —
    # see test_xai_reasoning_effort_passed for the rationale.

    def test_xai_grok_4_omits_reasoning_effort(self, transport):
        """grok-4 / grok-4-0709 reject reasoning.effort with HTTP 400."""
        messages = [{"role": "user", "content": "Hi"}]
        for model in ("grok-4", "grok-4-0709"):
            kw = transport.build_kwargs(
                model=model, messages=messages, tools=[],
                is_xai_responses=True,
                reasoning_config={"effort": "high"},
            )
            assert "reasoning" not in kw, (
                f"{model} must not receive a reasoning key (xAI rejects it)"
            )
            # Even without the effort dial we still ask xAI to echo back
            # encrypted reasoning content so it can be replayed next turn.
            assert "reasoning.encrypted_content" in kw.get("include", [])

    def test_xai_grok_4_fast_omits_reasoning_effort(self, transport):
        """grok-4-fast and grok-4-1-fast variants reject reasoning.effort."""
        messages = [{"role": "user", "content": "Hi"}]
        for model in (
            "grok-4-fast-reasoning",
            "grok-4-fast-non-reasoning",
            "grok-4-1-fast-reasoning",
            "grok-4-1-fast-non-reasoning",
        ):
            kw = transport.build_kwargs(
                model=model, messages=messages, tools=[],
                is_xai_responses=True,
                reasoning_config={"effort": "low"},
            )
            assert "reasoning" not in kw, (
                f"{model} must not receive a reasoning key (xAI rejects it)"
            )

    def test_xai_grok_3_non_mini_omits_reasoning_effort(self, transport):
        """Plain grok-3 rejects reasoning.effort — only grok-3-mini accepts it."""
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="grok-3", messages=messages, tools=[],
            is_xai_responses=True,
            reasoning_config={"effort": "medium"},
        )
        assert "reasoning" not in kw

    def test_xai_grok_3_mini_keeps_reasoning_effort(self, transport):
        """grok-3-mini and -fast variants do accept the effort dial."""
        messages = [{"role": "user", "content": "Hi"}]
        for model in ("grok-3-mini", "grok-3-mini-fast"):
            kw = transport.build_kwargs(
                model=model, messages=messages, tools=[],
                is_xai_responses=True,
                reasoning_config={"effort": "high"},
            )
            assert kw.get("reasoning") == {"effort": "high"}

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

    def test_xai_grok_4_20_multi_agent_keeps_reasoning_effort(self, transport):
        """grok-4.20-multi-agent-0309 is the one grok-4.20 variant that accepts effort."""
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="grok-4.20-multi-agent-0309", messages=messages, tools=[],
            is_xai_responses=True,
            reasoning_config={"effort": "low"},
        )
        assert kw.get("reasoning") == {"effort": "low"}

    def test_xai_grok_code_fast_omits_reasoning_effort(self, transport):
        """grok-code-fast-1 rejects reasoning.effort."""
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="grok-code-fast-1", messages=messages, tools=[],
            is_xai_responses=True,
            reasoning_config={"effort": "high"},
        )
        assert "reasoning" not in kw

    def test_xai_aggregator_prefix_stripped(self, transport):
        """`x-ai/grok-3-mini` (OpenRouter-style slug) still resolves correctly."""
        messages = [{"role": "user", "content": "Hi"}]
        # Effort-capable
        kw = transport.build_kwargs(
            model="x-ai/grok-3-mini", messages=messages, tools=[],
            is_xai_responses=True,
            reasoning_config={"effort": "high"},
        )
        assert kw.get("reasoning") == {"effort": "high"}
        # Effort-incapable
        kw = transport.build_kwargs(
            model="x-ai/grok-4-0709", messages=messages, tools=[],
            is_xai_responses=True,
            reasoning_config={"effort": "high"},
        )
        assert "reasoning" not in kw


class TestCodexValidateResponse:

    def test_none_response(self, transport):
        assert transport.validate_response(None) is False

    def test_empty_output(self, transport):
        r = SimpleNamespace(output=[], output_text=None)
        assert transport.validate_response(r) is False

    def test_valid_output(self, transport):
        r = SimpleNamespace(output=[{"type": "message", "content": []}])
        assert transport.validate_response(r) is True

    def test_output_text_fallback_not_valid(self, transport):
        """validate_response is strict — output_text doesn't make it valid.
        The caller handles output_text fallback with diagnostic logging."""
        r = SimpleNamespace(output=None, output_text="Some text")
        assert transport.validate_response(r) is False


class TestCodexMapFinishReason:

    def test_completed(self, transport):
        assert transport.map_finish_reason("completed") == "stop"

    def test_incomplete(self, transport):
        assert transport.map_finish_reason("incomplete") == "length"

    def test_failed(self, transport):
        assert transport.map_finish_reason("failed") == "stop"

    def test_unknown(self, transport):
        assert transport.map_finish_reason("unknown_status") == "stop"


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

    def test_zero_timeout_dropped(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            timeout=0,
        )
        assert "timeout" not in kw

    def test_none_timeout_omitted(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            timeout=None,
        )
        assert "timeout" not in kw

    def test_inf_timeout_dropped(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            timeout=float("inf"),
        )
        assert "timeout" not in kw

    def test_bool_timeout_dropped(self, transport):
        """``True`` is technically int but must not survive — caller bug guard."""
        kw = transport.build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            timeout=True,
        )
        assert "timeout" not in kw

    def test_request_overrides_can_supply_timeout(self, transport):
        """request_overrides["timeout"] is honored when no explicit kwarg passed."""
        kw = transport.build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            request_overrides={"timeout": 450.0},
        )
        assert kw.get("timeout") == 450.0


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
