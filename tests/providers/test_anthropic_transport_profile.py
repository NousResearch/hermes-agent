"""Anthropic transport ProviderProfile hook wiring tests (GH-75445).

Pin the contract between ``AnthropicTransport.build_kwargs()`` and the three
ProviderProfile request hooks, mirroring the chat_completions transport:

1. ``prepare_messages()`` runs once before Anthropic message conversion.
2. ``build_api_kwargs_extras()`` / ``build_extra_body()`` run once each with
   ``api_mode="anthropic_messages"`` in their context.
3. SDK-supported top-level fields stay top-level; unknown fields are projected
   onto ``extra_body`` so the Anthropic SDK never raises ``TypeError``.
4. Merge precedence is deterministic and ``provider_profile=None`` preserves
   the adapter-only request shape.
"""

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs
from agent.transports.anthropic import AnthropicTransport
from providers.base import ProviderProfile


class _SyntheticProfile(ProviderProfile):
    """Recordable profile whose hooks return fixed payloads."""

    def __init__(
        self,
        *,
        body=None,
        api_extra=None,
        api_top=None,
        prepare=None,
        hook_error=None,
    ):
        super().__init__(name="synthetic-test")
        self.body = body or {}
        self.api_extra = api_extra or {}
        self.api_top = api_top or {}
        self._prepare = prepare
        self.hook_error = hook_error
        self.prepare_calls = 0
        self.body_calls = 0
        self.api_kwargs_calls = 0
        self.body_context = None
        self.body_session_id = None
        self.api_context = None
        self.api_reasoning_config = None

    def prepare_messages(self, messages):
        self.prepare_calls += 1
        if self._prepare is not None:
            return self._prepare(messages)
        return messages

    def build_extra_body(self, *, session_id=None, **context):
        self.body_calls += 1
        self.body_context = context
        self.body_session_id = session_id
        if self.hook_error == "body":
            raise RuntimeError("body hook boom")
        return self.body

    def build_api_kwargs_extras(self, *, reasoning_config=None, **context):
        self.api_kwargs_calls += 1
        self.api_context = context
        self.api_reasoning_config = reasoning_config
        if self.hook_error == "api":
            raise RuntimeError("api hook boom")
        return self.api_extra, self.api_top


def _simple_messages():
    return [{"role": "user", "content": "hello"}]


class TestHookExecution:
    """Each ProviderProfile hook runs exactly once in the documented order."""

    def test_prepare_messages_runs_once_before_conversion(self):
        profile = _SyntheticProfile(
            prepare=lambda msgs: [{"role": "user", "content": "prepared"}]
        )
        kw = AnthropicTransport().build_kwargs(
            model="claude-sonnet-4-6",
            messages=_simple_messages(),
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            provider_profile=profile,
        )
        assert profile.prepare_calls == 1
        assert kw["messages"][0]["content"] == "prepared"

    def test_kwargs_hooks_run_once_with_api_mode_context(self):
        profile = _SyntheticProfile(
            body={"x_tag": "b"},
            api_extra={"x_extra": "e"},
            api_top={"service_tier": "standard_only"},
        )
        kw = AnthropicTransport().build_kwargs(
            model="claude-sonnet-4-6",
            messages=_simple_messages(),
            tools=None,
            max_tokens=1024,
            reasoning_config={"enabled": True, "effort": "high"},
            session_id="sess-1",
            provider_profile=profile,
        )
        assert profile.body_calls == 1
        assert profile.api_kwargs_calls == 1
        assert profile.body_context["api_mode"] == "anthropic_messages"
        assert profile.api_context["api_mode"] == "anthropic_messages"
        assert profile.body_session_id == "sess-1"
        assert profile.api_context["session_id"] == "sess-1"
        assert profile.api_reasoning_config["effort"] == "high"
        assert kw["extra_body"]["x_tag"] == "b"
        assert kw["extra_body"]["x_extra"] == "e"


class TestSdkProjection:
    """Profile top-level output lands on valid SDK kwargs or extra_body."""

    def test_sdk_top_level_field_stays_top_level(self):
        profile = _SyntheticProfile(api_top={"service_tier": "standard_only"})
        kw = AnthropicTransport().build_kwargs(
            model="claude-sonnet-4-6",
            messages=_simple_messages(),
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            provider_profile=profile,
        )
        assert kw["service_tier"] == "standard_only"
        assert "service_tier" not in kw.get("extra_body", {})

    def test_unknown_top_level_field_moves_to_extra_body(self):
        profile = _SyntheticProfile(api_top={"parallel_tool_calls": True})
        kw = AnthropicTransport().build_kwargs(
            model="claude-sonnet-4-6",
            messages=_simple_messages(),
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            provider_profile=profile,
        )
        assert "parallel_tool_calls" not in kw
        assert kw["extra_body"]["parallel_tool_calls"] is True

    def test_extra_headers_merge_with_adapter_beta_headers(self):
        profile = _SyntheticProfile(
            api_top={"extra_headers": {"x-custom": "1"}}
        )
        kw = AnthropicTransport().build_kwargs(
            model="claude-opus-4-6",
            messages=_simple_messages(),
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            fast_mode=True,
            provider_profile=profile,
        )
        assert kw["extra_headers"]["x-custom"] == "1"
        assert "anthropic-beta" in kw["extra_headers"]


class TestMergePrecedence:
    """Adapter defaults < build_extra_body < build_api_kwargs_extras body
    < extension fields projected from the top-level result."""

    def test_api_kwargs_extras_body_overrides_build_extra_body(self):
        profile = _SyntheticProfile(body={"dup": "body"}, api_extra={"dup": "api"})
        kw = AnthropicTransport().build_kwargs(
            model="claude-sonnet-4-6",
            messages=_simple_messages(),
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            provider_profile=profile,
        )
        assert kw["extra_body"]["dup"] == "api"

    def test_projected_extension_overrides_profile_body(self):
        profile = _SyntheticProfile(body={"dup": "body"}, api_top={"dup": "top"})
        kw = AnthropicTransport().build_kwargs(
            model="claude-sonnet-4-6",
            messages=_simple_messages(),
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            provider_profile=profile,
        )
        assert kw["extra_body"]["dup"] == "top"


class TestBaselineAndFailures:
    """No-profile parity and visible hook failures."""

    def test_no_profile_matches_adapter_result(self):
        transport = AnthropicTransport()
        messages = _simple_messages()
        assert transport.build_kwargs(
            model="claude-sonnet-4-6",
            messages=messages,
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
        ) == build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=messages,
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
        )

    def test_hook_exception_propagates(self):
        profile = _SyntheticProfile(hook_error="body")
        with pytest.raises(RuntimeError, match="body hook boom"):
            AnthropicTransport().build_kwargs(
                model="claude-sonnet-4-6",
                messages=_simple_messages(),
                tools=None,
                max_tokens=1024,
                reasoning_config=None,
                provider_profile=profile,
            )


class TestBundledProfileProtocolGates:
    """Dual-wire bundled profiles stay OpenAI-shaped on chat_completions and
    contribute nothing OpenAI-specific under api_mode='anthropic_messages'."""

    def test_nous_profile_messages_wire_shape(self):
        from providers import get_provider_profile

        nous = get_provider_profile("nous")
        prefs = {"only": ["anthropic"]}
        body = nous.build_extra_body(
            session_id="sess-1",
            provider_preferences=prefs,
            api_mode="anthropic_messages",
        )
        assert body["session_id"] == "sess-1"
        assert "tags" in body
        assert "provider" not in body
        extras, top = nous.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            supports_reasoning=True,
            api_mode="anthropic_messages",
        )
        assert extras == {}
        assert top == {}

    def test_nous_profile_keeps_chat_completions_shape(self):
        from providers import get_provider_profile

        nous = get_provider_profile("nous")
        prefs = {"only": ["anthropic"]}
        body = nous.build_extra_body(
            session_id="sess-1", provider_preferences=prefs
        )
        assert body["provider"] == prefs
        extras, top = nous.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            supports_reasoning=True,
        )
        assert extras == {"reasoning": {"enabled": True, "effort": "high"}}
        assert top == {}

    def test_kimi_profile_messages_wire_shape(self):
        from providers import get_provider_profile

        kimi = get_provider_profile("kimi-coding")
        extras, top = kimi.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            api_mode="anthropic_messages",
        )
        assert extras == {}
        assert top == {}

    def test_kimi_profile_keeps_chat_completions_shape(self):
        from providers import get_provider_profile

        kimi = get_provider_profile("kimi-coding")
        extras, top = kimi.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}
        )
        assert top == {"reasoning_effort": "high"}
        assert extras == {}
        extras_disabled, _ = kimi.build_api_kwargs_extras(reasoning_config=None)
        assert extras_disabled == {"thinking": {"type": "enabled"}}

    def test_custom_profile_messages_wire_shape(self):
        from providers import get_provider_profile

        custom = get_provider_profile("custom")
        extras, top = custom.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            api_mode="anthropic_messages",
        )
        assert extras == {}
        assert top == {}

    def test_custom_profile_keeps_chat_completions_shape(self):
        from providers import get_provider_profile

        custom = get_provider_profile("custom")
        extras, top = custom.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}
        )
        assert top == {"reasoning_effort": "high"}
        assert extras == {}
        extras_disabled, top_disabled = custom.build_api_kwargs_extras(
            reasoning_config={"enabled": False}
        )
        assert top_disabled == {"reasoning_effort": "none"}
        assert extras_disabled == {"think": False}

    def test_zai_profile_messages_wire_shape(self):
        from providers import get_provider_profile

        zai = get_provider_profile("zai")
        extras, top = zai.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="glm-5.2",
            api_mode="anthropic_messages",
        )
        assert extras == {}
        assert top == {}

    def test_zai_messages_request_uses_adapter_reasoning_shape(self):
        from providers import get_provider_profile

        zai = get_provider_profile("zai")
        kw = AnthropicTransport().build_kwargs(
            model="glm-5.2",
            messages=_simple_messages(),
            tools=None,
            max_tokens=1024,
            reasoning_config={"enabled": True, "effort": "high"},
            provider_profile=zai,
            base_url="https://open.bigmodel.cn/api/anthropic",
        )
        assert "thinking" in kw
        assert "reasoning_effort" not in kw.get("extra_body", {})
        assert "thinking" not in kw.get("extra_body", {})
