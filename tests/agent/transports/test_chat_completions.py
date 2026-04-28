"""Tests for the ChatCompletionsTransport."""

import pytest
from types import SimpleNamespace

from agent.transports import get_transport
from agent.transports.types import NormalizedResponse


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


class TestChatCompletionsBasic:

    def test_api_mode(self, transport):
        assert transport.api_mode == "chat_completions"

    def test_registered(self, transport):
        assert transport is not None

    def test_convert_tools_identity(self, transport):
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        assert transport.convert_tools(tools) is tools

    def test_convert_messages_no_codex_leaks(self, transport):
        msgs = [{"role": "user", "content": "hi"}]
        result = transport.convert_messages(msgs)
        assert result is msgs  # no copy needed

    def test_convert_messages_strips_codex_fields(self, transport):
        msgs = [
            {"role": "assistant", "content": "ok", "codex_reasoning_items": [{"id": "rs_1"}],
             "codex_message_items": [{"id": "msg_1", "type": "message"}],
             "tool_calls": [{"id": "call_1", "call_id": "call_1", "response_item_id": "fc_1",
                            "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        ]
        result = transport.convert_messages(msgs)
        assert "codex_reasoning_items" not in result[0]
        assert "codex_message_items" not in result[0]
        assert "call_id" not in result[0]["tool_calls"][0]
        assert "response_item_id" not in result[0]["tool_calls"][0]
        # Original list untouched (deepcopy-on-demand)
        assert "codex_reasoning_items" in msgs[0]
        assert "codex_message_items" in msgs[0]


class TestChatCompletionsBuildKwargs:

    def test_basic_kwargs(self, transport):
        msgs = [{"role": "user", "content": "Hello"}]
        kw = transport.build_kwargs(model="gpt-4o", messages=msgs, timeout=30.0)
        assert kw["model"] == "gpt-4o"
        assert kw["messages"][0]["content"] == "Hello"
        assert kw["timeout"] == 30.0

    def test_developer_role_swap(self, transport):
        msgs = [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(model="gpt-5.4", messages=msgs, model_lower="gpt-5.4")
        assert kw["messages"][0]["role"] == "developer"

    def test_no_developer_swap_for_non_gpt5(self, transport):
        msgs = [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(model="claude-sonnet-4", messages=msgs, model_lower="claude-sonnet-4")
        assert kw["messages"][0]["role"] == "system"

    def test_tools_included(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        kw = transport.build_kwargs(model="gpt-4o", messages=msgs, tools=tools)
        assert kw["tools"] == tools

    def test_openrouter_provider_prefs(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o", messages=msgs,
            is_openrouter=True,
            provider_preferences={"only": ["openai"]},
        )
        assert kw["extra_body"]["provider"] == {"only": ["openai"]}

    def test_nous_tags(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(model="gpt-4o", messages=msgs, is_nous=True)
        assert kw["extra_body"]["tags"] == ["product=hermes-agent"]

    def test_reasoning_default(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o", messages=msgs,
            supports_reasoning=True,
        )
        assert kw["extra_body"]["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_nous_omits_disabled_reasoning(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o", messages=msgs,
            supports_reasoning=True,
            is_nous=True,
            reasoning_config={"enabled": False},
        )
        # Nous rejects enabled=false; reasoning omitted entirely
        assert "reasoning" not in kw.get("extra_body", {})

    def test_ollama_num_ctx(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="llama3", messages=msgs,
            ollama_num_ctx=32768,
        )
        assert kw["extra_body"]["options"]["num_ctx"] == 32768

    def test_custom_think_false(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="qwen3", messages=msgs,
            is_custom_provider=True,
            reasoning_config={"effort": "none"},
        )
        assert kw["extra_body"]["think"] is False

    def test_gemini_without_explicit_reasoning_config_keeps_existing_behavior(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gemini-3-flash-preview",
            messages=msgs,
            provider_name="gemini",
        )
        assert "thinking_config" not in kw.get("extra_body", {})

    def test_gemini_flash_reasoning_maps_to_thinking_config(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gemini-3-flash-preview",
            messages=msgs,
            provider_name="gemini",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert kw["extra_body"]["thinking_config"] == {
            "includeThoughts": True,
            "thinkingLevel": "high",
        }

    def test_gemini_25_reasoning_only_enables_visible_thoughts(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gemini-2.5-flash",
            messages=msgs,
            provider_name="gemini",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert kw["extra_body"]["thinking_config"] == {
            "includeThoughts": True,
        }

    def test_gemini_pro_reasoning_clamps_to_supported_levels(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="google/gemini-3.1-pro-preview",
            messages=msgs,
            provider_name="gemini",
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert kw["extra_body"]["thinking_config"] == {
            "includeThoughts": True,
            "thinkingLevel": "low",
        }

    def test_gemini_disabled_reasoning_hides_thoughts(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gemini-3-flash-preview",
            messages=msgs,
            provider_name="gemini",
            reasoning_config={"enabled": False},
        )
        assert kw["extra_body"]["thinking_config"] == {
            "includeThoughts": False,
        }

    def test_gemini_xhigh_clamps_to_high(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gemini-3-flash-preview",
            messages=msgs,
            provider_name="gemini",
            reasoning_config={"enabled": True, "effort": "xhigh"},
        )
        assert kw["extra_body"]["thinking_config"]["thinkingLevel"] == "high"

    def test_gemini_flash_minimal_clamps_to_low(self, transport):
        # Gemini 3 Flash documents low/medium/high; "minimal" isn't accepted,
        # so clamp it down to "low" rather than forwarding it verbatim.
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gemini-3-flash-preview",
            messages=msgs,
            provider_name="gemini",
            reasoning_config={"enabled": True, "effort": "minimal"},
        )
        assert kw["extra_body"]["thinking_config"] == {
            "includeThoughts": True,
            "thinkingLevel": "low",
        }

    def test_max_tokens_with_fn(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o", messages=msgs,
            max_tokens=4096,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        assert kw["max_tokens"] == 4096

    def test_ephemeral_overrides_max_tokens(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o", messages=msgs,
            max_tokens=4096,
            ephemeral_max_output_tokens=2048,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        assert kw["max_tokens"] == 2048

    def test_nvidia_default_max_tokens(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="glm-4.7", messages=msgs,
            is_nvidia_nim=True,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        # NVIDIA default: 16384
        assert kw["max_tokens"] == 16384

    def test_qwen_default_max_tokens(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="qwen3-coder-plus", messages=msgs,
            is_qwen_portal=True,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        # Qwen default: 65536
        assert kw["max_tokens"] == 65536

    def test_anthropic_max_output_for_claude_on_aggregator(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6", messages=msgs,
            is_openrouter=True,
            anthropic_max_output=64000,
        )
        # Set as plain max_tokens (not via fn) because the aggregator proxies to
        # Anthropic Messages API which requires the field.
        assert kw["max_tokens"] == 64000

    def test_request_overrides_last(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o", messages=msgs,
            request_overrides={"service_tier": "priority"},
        )
        assert kw["service_tier"] == "priority"

    def test_fixed_temperature(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(model="gpt-4o", messages=msgs, fixed_temperature=0.6)
        assert kw["temperature"] == 0.6

    def test_omit_temperature(self, transport):
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(model="gpt-4o", messages=msgs, omit_temperature=True, fixed_temperature=0.5)
        # omit wins
        assert "temperature" not in kw


class TestChatCompletionsKimi:
    """Regression tests for the Kimi/Moonshot quirks migrated into the transport."""

    def test_kimi_max_tokens_default(self, transport):
        kw = transport.build_kwargs(
            model="kimi-k2", messages=[{"role": "user", "content": "Hi"}],
            is_kimi=True,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        # Kimi CLI default: 32000
        assert kw["max_tokens"] == 32000

    def test_kimi_reasoning_effort_top_level(self, transport):
        kw = transport.build_kwargs(
            model="kimi-k2", messages=[{"role": "user", "content": "Hi"}],
            is_kimi=True,
            reasoning_config={"effort": "high"},
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        # Kimi requires reasoning_effort as a top-level parameter
        assert kw["reasoning_effort"] == "high"

    def test_kimi_reasoning_effort_omitted_when_thinking_disabled(self, transport):
        kw = transport.build_kwargs(
            model="kimi-k2", messages=[{"role": "user", "content": "Hi"}],
            is_kimi=True,
            reasoning_config={"enabled": False},
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        # Mirror Kimi CLI: omit reasoning_effort entirely when thinking off
        assert "reasoning_effort" not in kw

    def test_kimi_thinking_enabled_extra_body(self, transport):
        kw = transport.build_kwargs(
            model="kimi-k2", messages=[{"role": "user", "content": "Hi"}],
            is_kimi=True,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        assert kw["extra_body"]["thinking"] == {"type": "enabled"}

    def test_kimi_thinking_disabled_extra_body(self, transport):
        kw = transport.build_kwargs(
            model="kimi-k2", messages=[{"role": "user", "content": "Hi"}],
            is_kimi=True,
            reasoning_config={"enabled": False},
            max_tokens_param_fn=lambda n: {"max_tokens": n},
        )
        assert kw["extra_body"]["thinking"] == {"type": "disabled"}

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

    def test_omits_effort_when_high_not_allowed_minimal_low(self, transport):
        kw = transport.build_kwargs(
            model="gpt-oss", messages=[{"role": "user", "content": "Hi"}],
            is_lmstudio=True,
            supports_reasoning=True,
            reasoning_config={"effort": "high"},
            lmstudio_reasoning_options=["off", "minimal", "low"],
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

    def test_passes_through_aliased_on_for_toggle(self, transport):
        # User has reasoning enabled at the default "medium"; toggle model
        # publishes ["off","on"] which aliases to {"none","medium"}, so the
        # default request is honorable and gets sent.
        kw = transport.build_kwargs(
            model="gpt-oss", messages=[{"role": "user", "content": "Hi"}],
            is_lmstudio=True,
            supports_reasoning=True,
            reasoning_config={"effort": "medium"},
            lmstudio_reasoning_options=["off", "on"],
        )
        assert kw["reasoning_effort"] == "medium"

    def test_disabled_keeps_none_when_off_allowed(self, transport):
        kw = transport.build_kwargs(
            model="gpt-oss", messages=[{"role": "user", "content": "Hi"}],
            is_lmstudio=True,
            supports_reasoning=True,
            reasoning_config={"enabled": False},
            lmstudio_reasoning_options=["off", "on"],
        )
        assert kw["reasoning_effort"] == "none"

    def test_no_options_falls_back_to_legacy_behavior(self, transport):
        # When the probe failed or returned nothing, allowed_options is unknown;
        # send whatever the user picked rather than blocking the request.
        kw = transport.build_kwargs(
            model="gpt-oss", messages=[{"role": "user", "content": "Hi"}],
            is_lmstudio=True,
            supports_reasoning=True,
            reasoning_config={"effort": "high"},
            lmstudio_reasoning_options=None,
        )
        assert kw["reasoning_effort"] == "high"


class TestChatCompletionsValidate:

    def test_none(self, transport):
        assert transport.validate_response(None) is False

    def test_no_choices(self, transport):
        r = SimpleNamespace(choices=None)
        assert transport.validate_response(r) is False

    def test_empty_choices(self, transport):
        r = SimpleNamespace(choices=[])
        assert transport.validate_response(r) is False

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

    def test_tool_call_extra_content_preserved(self, transport):
        """Gemini 3 thinking models attach extra_content with thought_signature
        on tool_calls.  Without this replay on the next turn, the API rejects
        the request with 400.  The transport MUST surface extra_content so the
        agent loop can write it back into the assistant message."""
        tc = SimpleNamespace(
            id="call_gem",
            function=SimpleNamespace(name="terminal", arguments='{"command": "ls"}'),
            extra_content={"google": {"thought_signature": "SIG_ABC123"}},
        )
        r = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tc], reasoning_content=None),
                finish_reason="tool_calls",
            )],
            usage=None,
        )
        nr = transport.normalize_response(r)
        assert nr.tool_calls[0].provider_data == {
            "extra_content": {"google": {"thought_signature": "SIG_ABC123"}}
        }

    def test_reasoning_content_preserved_separately(self, transport):
        """DeepSeek/Moonshot use reasoning_content distinct from reasoning.
        Don't merge them — the thinking-prefill retry check reads each field
        separately."""
        r = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=None, tool_calls=None,
                    reasoning="summary text",
                    reasoning_content="detailed scratchpad",
                ),
                finish_reason="stop",
            )],
            usage=None,
        )
        nr = transport.normalize_response(r)
        assert nr.reasoning == "summary text"
        assert nr.provider_data == {"reasoning_content": "detailed scratchpad"}


class TestChatCompletionsCacheStats:

    def test_no_usage(self, transport):
        r = SimpleNamespace(usage=None)
        assert transport.extract_cache_stats(r) is None

    def test_no_details(self, transport):
        r = SimpleNamespace(usage=SimpleNamespace(prompt_tokens_details=None))
        assert transport.extract_cache_stats(r) is None

    def test_with_cache(self, transport):
        details = SimpleNamespace(cached_tokens=500, cache_write_tokens=100)
        r = SimpleNamespace(usage=SimpleNamespace(prompt_tokens_details=details))
        result = transport.extract_cache_stats(r)
        assert result == {"cached_tokens": 500, "creation_tokens": 100}
