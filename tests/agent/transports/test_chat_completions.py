"""Tests for the ChatCompletionsTransport."""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from agent.transports import get_transport
from agent.transports.types import NormalizedResponse


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


class TestChatCompletionsBasic:
    @pytest.mark.parametrize(
        "choice",
        [SimpleNamespace(message=SimpleNamespace()), SimpleNamespace()],
    )
    def test_normalize_response_allows_missing_optional_message_fields(
        self, transport, choice
    ):
        response = SimpleNamespace(choices=[choice], usage=None)

        normalized = transport.normalize_response(response)

        assert normalized.content is None
        assert normalized.tool_calls is None
        assert normalized.finish_reason == "stop"

    def test_normalize_response_allows_sparse_tool_call_fields(self, transport):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(arguments='{"city":"Paris"}')
                            ),
                            SimpleNamespace(),
                            SimpleNamespace(
                                id="call-3",
                                function=SimpleNamespace(name="lookup"),
                            ),
                            SimpleNamespace(
                                id="call-4",
                                function=SimpleNamespace(name="", arguments="{}"),
                            ),
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="weather", arguments="{}"
                                )
                            ),
                        ]
                    )
                )
            ],
            usage=None,
        )

        normalized = transport.normalize_response(response)

        assert normalized.finish_reason == "stop"
        assert normalized.tool_calls is not None
        assert [tool.id for tool in normalized.tool_calls] == [
            "call-3",
            "call-4",
            None,
        ]
        assert [tool.name for tool in normalized.tool_calls] == [
            "lookup",
            "",
            "weather",
        ]
        assert [tool.arguments for tool in normalized.tool_calls] == [
            "{}",
            "{}",
            "{}",
        ]

    @pytest.mark.parametrize("provider", ["nous", "openrouter"])
    def test_gpt56_ultra_uses_max_wire_effort(self, transport, provider):
        from providers import get_provider_profile

        profile = get_provider_profile(provider)
        kw = transport.build_kwargs(
            model="openai/gpt-5.6-sol",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            reasoning_config={"enabled": True, "effort": "ultra"},
            supports_reasoning=True,
            provider_profile=profile,
            provider_name=provider,
            base_url=profile.base_url,
        )
        assert kw["extra_body"]["reasoning"] == {"enabled": True, "effort": "max"}


    def test_convert_messages_no_codex_leaks(self, transport):
        msgs = [{"role": "user", "content": "hi"}]
        result = transport.convert_messages(msgs)
        assert result is msgs  # no copy needed



    def _msg_with_extra_content(self):
        return [
            {"role": "assistant", "content": "ok",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "extra_content": {"google": {"thought_signature": "SIG_123"}},
                             "function": {"name": "t", "arguments": "{}"}}]},
        ]






    def test_convert_messages_strips_timestamp(self, transport):
        """Internal per-message ``timestamp`` metadata (stamped by
        ``_apply_persist_user_message_override`` to preserve platform event
        time without embedding it in content, and persisted to the SQLite
        store) is not part of the OpenAI Chat Completions schema. Strict
        providers like Mistral / Fireworks-backed endpoints reject it with
        HTTP 422 'Extra inputs are not permitted, field: messages[N].timestamp'.
        Regression test for #47868.
        """
        msgs = [
            {"role": "user", "content": "hi", "timestamp": 1781976577.0},
        ]
        result = transport.convert_messages(msgs)
        assert "timestamp" not in result[0]
        assert result[0]["content"] == "hi"
        assert result[0]["role"] == "user"
        # Original list untouched (deepcopy-on-demand)
        assert msgs[0]["timestamp"] == 1781976577.0

    def test_convert_messages_no_copy_without_timestamp(self, transport):
        """A timestamp-free message list needs no sanitize pass and is
        returned by identity (preserves the deepcopy-on-demand contract)."""
        msgs = [{"role": "user", "content": "hi"}]
        assert transport.convert_messages(msgs) is msgs

    def test_convert_messages_strips_internal_scaffolding_markers(self, transport):
        """Hermes-internal ``_``-prefixed markers must never reach the wire.

        The empty-response recovery path appends synthetic messages tagged
        with ``_empty_recovery_synthetic``; permissive providers ignore the
        unknown key, but strict gateways (opencode-go, codex.nekos.me)
        reject the request, poisoning every later turn in the session.
        """
        msgs = [
            {"role": "user", "content": "run the task"},
            {"role": "assistant", "content": "(empty)", "_empty_recovery_synthetic": True},
            {"role": "user", "content": "continue", "_empty_recovery_synthetic": True},
            {"role": "assistant", "content": "done", "_thinking_prefill": True,
             "_empty_terminal_sentinel": True},
        ]
        result = transport.convert_messages(msgs)
        for m in result:
            assert not any(k.startswith("_") for k in m), m
        # Visible content preserved
        assert result[1]["content"] == "(empty)"
        assert result[2]["content"] == "continue"
        # Original list untouched (deepcopy-on-demand)
        assert msgs[1]["_empty_recovery_synthetic"] is True


    def test_convert_messages_copy_on_write_for_dirty_history(self, transport):
        """Dirty provider metadata should not force a full-history deepcopy."""
        clean_tool_call = {
            "id": "call_clean",
            "type": "function",
            "function": {"name": "safe", "arguments": "{}"},
        }
        msgs = [
            {"role": "user", "content": "hi", "metadata": {"large": ["shared"]}},
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    clean_tool_call,
                    {
                        "id": "call_dirty",
                        "call_id": "call_dirty",
                        "response_item_id": "fc_dirty",
                        "extra_content": {"google": {"thought_signature": "SIG"}},
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    },
                ],
            },
        ]

        result = transport.convert_messages(msgs, model="gpt-4o")

        assert result is not msgs
        assert result[0] is msgs[0]
        assert result[1] is not msgs[1]
        assert result[1]["tool_calls"] is not msgs[1]["tool_calls"]
        assert result[1]["tool_calls"][0] is clean_tool_call
        assert result[1]["tool_calls"][1] is not msgs[1]["tool_calls"][1]
        assert "call_id" not in result[1]["tool_calls"][1]
        assert "response_item_id" not in result[1]["tool_calls"][1]
        assert "extra_content" not in result[1]["tool_calls"][1]
        assert "call_id" in msgs[1]["tool_calls"][1]
        assert "extra_content" in msgs[1]["tool_calls"][1]



class TestChatCompletionsBuildKwargs:

    def test_basic_kwargs(self, transport):
        msgs = [{"role": "user", "content": "Hello"}]
        kw = transport.build_kwargs(model="gpt-4o", messages=msgs, timeout=30.0)
        assert kw["model"] == "gpt-4o"
        assert kw["messages"][0]["content"] == "Hello"
        assert kw["timeout"] == 30.0



    def test_tools_included(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        kw = transport.build_kwargs(model="gpt-4o", messages=msgs, tools=tools)
        assert kw["tools"] == tools

    def test_openrouter_provider_prefs(self, transport):
        from providers import get_provider_profile
        profile = get_provider_profile("openrouter")
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o", messages=msgs,
            provider_profile=profile,
            provider_preferences={"only": ["openai"]},
        )
        assert kw["extra_body"]["provider"] == {"only": ["openai"]}






    def test_nous_tags(self, transport):
        from agent.portal_tags import nous_portal_tags
        from providers import get_provider_profile
        profile = get_provider_profile("nous")
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(model="gpt-4o", messages=msgs, provider_profile=profile)
        assert kw["extra_body"]["tags"] == nous_portal_tags()

    def test_reasoning_default(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o", messages=msgs,
            supports_reasoning=True,
        )
        assert kw["extra_body"]["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_nous_omits_disabled_reasoning(self, transport):
        from providers import get_provider_profile
        profile = get_provider_profile("nous")
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o", messages=msgs,
            provider_profile=profile,
            supports_reasoning=True,
            reasoning_config={"enabled": False},
        )
        # Nous rejects enabled=false; reasoning omitted entirely
        assert "reasoning" not in kw.get("extra_body", {})

    def test_ollama_num_ctx(self, transport):
        from providers import get_provider_profile
        profile = get_provider_profile("custom")
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="llama3", messages=msgs,
            provider_profile=profile,
            ollama_num_ctx=32768,
        )
        assert kw["extra_body"]["options"]["num_ctx"] == 32768

    def test_custom_think_false(self, transport):
        from providers import get_provider_profile
        profile = get_provider_profile("custom")
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="qwen3", messages=msgs,
            provider_profile=profile,
            reasoning_config={"effort": "none"},
        )
        assert kw["extra_body"]["think"] is False



    def test_gemini_openai_compat_flash_reasoning_maps_to_nested_google_thinking_config(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gemini-3-flash-preview",
            messages=msgs,
            provider_name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert "thinking_config" not in kw["extra_body"]
        assert kw["extra_body"]["extra_body"]["google"]["thinking_config"] == {
            "include_thoughts": True,
            "thinking_level": "high",
        }

    def test_gemini_ultra_thinking_raises_first_request_max_tokens(self, transport):
        from agent.gemini_native_adapter import GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
        from providers import get_provider_profile

        profile = get_provider_profile("gemini")
        kw = transport.build_kwargs(
            model="gemini-3.7-flash",
            messages=[{"role": "user", "content": "Hi"}],
            provider_profile=profile,
            provider_name="gemini",
            base_url=profile.base_url,
            max_tokens=4096,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
            reasoning_config={"enabled": True, "effort": "ultra"},
        )
        assert kw["max_tokens"] == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
        assert kw["extra_body"]["thinking_config"]["thinkingLevel"] == "high"

    def test_gemini_without_thinking_keeps_explicit_max_tokens(self, transport):
        from providers import get_provider_profile

        profile = get_provider_profile("gemini")
        kw = transport.build_kwargs(
            model="gemini-3.7-flash",
            messages=[{"role": "user", "content": "Hi"}],
            provider_profile=profile,
            provider_name="gemini",
            base_url=profile.base_url,
            max_tokens=4096,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        assert kw["max_tokens"] == 4096

















    def test_omit_temperature(self, transport):
        """Omit temperature is set via ProviderProfile with OMIT_TEMPERATURE sentinel."""
        from providers.base import ProviderProfile, OMIT_TEMPERATURE
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o", messages=msgs,
            provider_profile=ProviderProfile(name="_t", fixed_temperature=OMIT_TEMPERATURE),
        )
        assert "temperature" not in kw


class TestChatCompletionsKimi:
    """Regression tests for the Kimi/Moonshot quirks migrated into the transport."""






    def test_moonshot_tool_schemas_are_sanitized_by_model_name(self, transport):
        """Aggregator routes (Nous, OpenRouter) hit Moonshot by model name, not base URL."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q": {"description": "query"},  # missing type
                        },
                    },
                },
            },
        ]
        kw = transport.build_kwargs(
            model="moonshotai/kimi-k2.6",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        assert kw["tools"][0]["function"]["parameters"]["properties"]["q"]["type"] == "string"

    def test_moonshot_outgoing_schema_carries_required_array(self, transport):
        """Moonshot 400s on object schemas without an explicit `required` array
        (#66835). Assert the wire-level tool schema — what actually leaves the
        transport — carries `required: []` on a zero-required-param tool."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "browser_snapshot",
                    "description": "Snapshot",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        kw = transport.build_kwargs(
            model="moonshotai/kimi-k3",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        assert kw["tools"][0]["function"]["parameters"]["required"] == []

    def test_non_moonshot_tools_are_not_mutated(self, transport):
        """Other models don't go through the Moonshot sanitizer."""
        original_params = {
            "type": "object",
            "properties": {"q": {"description": "query"}},  # missing type
        }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": original_params,
                },
            },
        ]
        kw = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        # The parameters dict is passed through untouched (no synthetic type)
        assert "type" not in kw["tools"][0]["function"]["parameters"]["properties"]["q"]


class TestChatCompletionsLmStudioReasoning:
    """LM Studio publishes per-model reasoning ``allowed_options``. When the
    user requests an effort the model can't honor (e.g. ``high`` on a
    toggle-style ``["off","on"]`` model), the transport omits
    ``reasoning_effort`` so LM Studio falls back to the model's default —
    silently downgrading "high" to "low" would mislead the user.
    """

    def test_omits_effort_when_high_not_allowed_toggle(self, transport):
        kw = transport.build_kwargs(
            model="gpt-oss", messages=[{"role": "user", "content": "Hi"}],
            is_lmstudio=True,
            supports_reasoning=True,
            reasoning_config={"effort": "high"},
            lmstudio_reasoning_options=["off", "on"],
        )
        assert "reasoning_effort" not in kw


    def test_passes_through_when_effort_allowed(self, transport):
        kw = transport.build_kwargs(
            model="gpt-oss", messages=[{"role": "user", "content": "Hi"}],
            is_lmstudio=True,
            supports_reasoning=True,
            reasoning_config={"effort": "high"},
            lmstudio_reasoning_options=["off", "low", "medium", "high"],
        )
        assert kw["reasoning_effort"] == "high"





class TestChatCompletionsValidate:

    def test_none(self, transport):
        assert transport.validate_response(None) is False



    def test_valid(self, transport):
        r = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
        assert transport.validate_response(r) is True


class TestChatCompletionsNormalize:

    def test_text_response(self, transport):
        r = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Hello", tool_calls=None, reasoning_content=None),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        nr = transport.normalize_response(r)
        assert isinstance(nr, NormalizedResponse)
        assert nr.content == "Hello"
        assert nr.finish_reason == "stop"
        assert nr.tool_calls is None

    def test_tool_call_response(self, transport):
        tc = SimpleNamespace(
            id="call_123",
            function=SimpleNamespace(name="terminal", arguments='{"command": "ls"}'),
        )
        r = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tc], reasoning_content=None),
                finish_reason="tool_calls",
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        nr = transport.normalize_response(r)
        assert len(nr.tool_calls) == 1
        assert nr.tool_calls[0].name == "terminal"
        assert nr.tool_calls[0].id == "call_123"



    def test_empty_reasoning_content_preserved(self, transport):
        """DeepSeek can require an explicit empty reasoning_content replay field."""
        r = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content="",
                ),
                finish_reason="stop",
            )],
            usage=None,
        )
        nr = transport.normalize_response(r)
        assert nr.provider_data == {"reasoning_content": ""}
        assert nr.reasoning_content == ""



    def test_refusal_none_is_noop(self, transport):
        """The common case: ``refusal`` is None → behavior unchanged."""
        r = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="hello", tool_calls=None, reasoning_content=None,
                    refusal=None,
                ),
                finish_reason="stop",
            )],
            usage=None,
        )
        nr = transport.normalize_response(r)
        assert nr.finish_reason == "stop"
        assert nr.content == "hello"
        assert nr.provider_data is None






class TestChatCompletionsCacheStats:

    def test_no_usage(self, transport):
        r = SimpleNamespace(usage=None)
        assert transport.extract_cache_stats(r) is None



    def test_deepseek_native_top_level_cache_hit_tokens(self, transport):
        """DeepSeek's native API (api.deepseek.com) reports cache hits as
        top-level prompt_cache_hit_tokens, not the OpenAI nested shape —
        the extractor must read it or direct DeepSeek sessions show 0%
        cache hit rate (#61871)."""
        r = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens_details=None,
                prompt_cache_hit_tokens=1500,
                prompt_cache_miss_tokens=500,
            )
        )
        result = transport.extract_cache_stats(r)
        assert result == {"cached_tokens": 1500, "creation_tokens": 0}



class TestChatCompletionsGeminiNativeExtraBodyStrip:
    """Profile extra_body (e.g. Nous portal tags) must not reach a native
    Gemini endpoint — Google's REST API rejects unknown fields with HTTP 400.
    """

    def _nous_profile(self):
        from providers import get_provider_profile
        return get_provider_profile("nous")

    def test_tags_stripped_when_endpoint_is_native_gemini(self, transport):
        kw = transport.build_kwargs(
            "anthropic/claude-sonnet-4.6",
            [{"role": "user", "content": "hi"}],
            None,
            provider_profile=self._nous_profile(),
            base_url="https://generativelanguage.googleapis.com/v1beta",
            session_id="s1",
            max_tokens=None,
        )
        eb = kw.get("extra_body")
        assert not eb or "tags" not in eb

    def test_tags_preserved_on_nous_endpoint(self, transport):
        kw = transport.build_kwargs(
            "hermes-3-405b",
            [{"role": "user", "content": "hi"}],
            None,
            provider_profile=self._nous_profile(),
            base_url="https://inference.nousresearch.com/v1",
            session_id="s1",
            max_tokens=None,
        )
        eb = kw.get("extra_body")
        assert eb and "tags" in eb

    def test_tags_pass_through_on_gemini_openai_compat(self, transport):
        # /openai compat endpoint is not "native" — unchanged behavior.
        kw = transport.build_kwargs(
            "anthropic/claude-sonnet-4.6",
            [{"role": "user", "content": "hi"}],
            None,
            provider_profile=self._nous_profile(),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            session_id="s1",
            max_tokens=None,
        )
        eb = kw.get("extra_body")
        assert eb and "tags" in eb


class TestPromptCacheKeyCapability:
    """Chat Completions cache routing is opt-in and body-safe."""

    @staticmethod
    def _messages(instructions="You are stable."):
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": "hello"},
        ]

    @staticmethod
    def _tools(name="lookup"):
        return [{
            "type": "function",
            "function": {
                "name": name,
                "description": "Look something up.",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

    def _request_body(self, kwargs, *, stream=False):
        captured = {}

        def handler(request):
            captured.update(json.loads(request.content))
            if stream:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=(
                        'data: {"id":"chatcmpl_1","object":"chat.completion.chunk",'
                        '"choices":[{"index":0,"delta":{"content":"ok"},'
                        '"finish_reason":null}]}\n\n'
                        "data: [DONE]\n\n"
                    ),
                )
            return httpx.Response(200, json={
                "id": "chatcmpl_1",
                "object": "chat.completion",
                "created": 0,
                "model": kwargs["model"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            })

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAI(
                api_key="test-key",
                base_url="https://cache-capable.test/v1",
                http_client=http_client,
            )
            result = client.chat.completions.create(**kwargs, stream=stream)
            if stream:
                list(result)
        return captured

    def test_profile_capability_emits_content_key_in_nonstream_request_body(self, transport):
        from providers.base import ProviderProfile

        kwargs = transport.build_kwargs(
            model="cache-model",
            messages=self._messages(),
            tools=self._tools(),
            session_id="cron_job_2026-07-15T10:00:00Z",
            provider_profile=ProviderProfile(
                name="cache-capable", supports_prompt_cache_key=True,
            ),
        )

        body = self._request_body(kwargs)

        assert body["prompt_cache_key"].startswith("pck_")
        assert body["prompt_cache_key"] == kwargs["prompt_cache_key"]

    def test_legacy_capability_emits_same_key_in_streaming_request_body(self, transport):
        kwargs = transport.build_kwargs(
            model="cache-model",
            messages=self._messages(),
            tools=self._tools(),
            session_id="cron_job_2026-07-15T10:05:00Z",
            supports_prompt_cache_key=True,
        )

        body = self._request_body(kwargs, stream=True)

        assert body["prompt_cache_key"] == kwargs["prompt_cache_key"]

    def test_openai_api_base_url_implies_capability(self, transport):
        """api.openai.com gets the key WITHOUT an explicit flag (exact host)."""
        kwargs = transport.build_kwargs(
            model="gpt-cache-model",
            messages=self._messages(),
            tools=self._tools(),
            session_id="cron_job_2026-07-15T10:07:00Z",
            base_url="https://api.openai.com/v1",
        )

        assert kwargs["prompt_cache_key"].startswith("pck_")

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://myproxy.example.com/api.openai.com/v1",  # host embedded in path
            "https://api.openai.com.evil.example/v1",  # prefix-spoofed host
            "https://eastus.api.cognitive.microsoft.com/openai/v1",  # Azure
        ],
    )
    def test_non_openai_hosts_do_not_imply_capability(self, transport, base_url):
        kwargs = transport.build_kwargs(
            model="strict-model",
            messages=self._messages(),
            tools=self._tools(),
            session_id="cron_job_2026-07-15T10:08:00Z",
            base_url=base_url,
        )

        assert "prompt_cache_key" not in kwargs

    @pytest.mark.parametrize("provider", [None, "anthropic", "custom"])
    def test_default_off_never_leaks_unknown_body_field(self, transport, provider):
        from providers import get_provider_profile

        kwargs = transport.build_kwargs(
            model="strict-model",
            messages=self._messages(),
            tools=self._tools(),
            session_id="cron_job_2026-07-15T10:00:00Z",
            provider_profile=(get_provider_profile(provider) if provider else None),
        )

        body = self._request_body(kwargs)

        assert "prompt_cache_key" not in kwargs
        assert "prompt_cache_key" not in body

    def test_explicit_top_level_and_extra_body_overrides_are_preserved(self, transport):
        from providers.base import ProviderProfile

        profile = ProviderProfile(name="cache-capable", supports_prompt_cache_key=True)
        top_level = transport.build_kwargs(
            model="cache-model", messages=self._messages(), tools=self._tools(),
            provider_profile=profile,
            request_overrides={"prompt_cache_key": "caller-top-level"},
        )
        in_extra_body = transport.build_kwargs(
            model="cache-model", messages=self._messages(), tools=self._tools(),
            provider_profile=profile,
            request_overrides={"extra_body": {"prompt_cache_key": "caller-extra-body"}},
        )

        assert top_level["prompt_cache_key"] == "caller-top-level"
        assert "prompt_cache_key" not in top_level.get("extra_body", {})
        assert "prompt_cache_key" not in in_extra_body
        assert in_extra_body["extra_body"]["prompt_cache_key"] == "caller-extra-body"

    def test_cron_ids_share_static_prefix_key_and_content_changes_invalidate(self, transport):
        def key(session_id, *, instructions="You are stable.", tool_name="lookup"):
            return transport.build_kwargs(
                model="cache-model",
                messages=self._messages(instructions),
                tools=self._tools(tool_name),
                session_id=session_id,
                supports_prompt_cache_key=True,
            )["prompt_cache_key"]

        first = key("cron_job_20260715_100000")
        second = key("cron_job_20260715_100500")

        assert first == second
        assert first != key("cron_job_20260715_100500", instructions="You are different.")
        assert first != key("cron_job_20260715_100500", tool_name="search")

    def test_unrelated_sessions_get_distinct_keys(self, transport):
        """#78941: identical static prefix across unrelated (non-cron) sessions
        must not collapse onto one shared prompt_cache_key."""
        kw1 = transport.build_kwargs(
            model="cache-model",
            messages=self._messages("You are stable."),
            tools=self._tools("lookup"),
            session_id="session_alice_1",
            supports_prompt_cache_key=True,
        )
        kw2 = transport.build_kwargs(
            model="cache-model",
            messages=self._messages("You are stable."),
            tools=self._tools("lookup"),
            session_id="session_bob_1",
            supports_prompt_cache_key=True,
        )
        assert kw1["prompt_cache_key"] != kw2["prompt_cache_key"]

    def test_stale_profile_without_supports_prompt_cache_key_does_not_crash(self, transport):
        """A ProviderProfile from a stale sys.modules cache (pre-#f4fb23f3d)
        won't have the ``supports_prompt_cache_key`` field. Accessing it via
        ``profile.supports_prompt_cache_key`` raises AttributeError and crashes
        every API call. Use getattr with a False default so it degrades to
        "no prompt cache key" instead of crashing.

        Regression: 'NousProfile' object has no attribute
        'supports_prompt_cache_key' (Aug 2026, after partial update).
        """
        from providers.base import ProviderProfile

        # Simulate a stale class that predates supports_prompt_cache_key
        # by creating a profile and deleting the attribute.
        profile = ProviderProfile(name="stale-provider")
        del profile.supports_prompt_cache_key

        # Must not raise AttributeError — should fall back to False.
        kwargs = transport.build_kwargs(
            model="stale-model",
            messages=self._messages(),
            tools=self._tools(),
            provider_profile=profile,
        )
        assert "prompt_cache_key" not in kwargs
