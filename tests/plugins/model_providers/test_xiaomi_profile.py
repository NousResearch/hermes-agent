"""Unit tests for the Xiaomi MiMo provider profile's reasoning wiring.

MiMo (``api.xiaomimimo.com/v1``, OpenAI-compatible) reasons by default. Turning
reasoning off (``/reasoning none`` -> reasoning_config ``{"enabled": False}``)
must send ``extra_body={"thinking": {"type": "disabled"}}`` so ``reasoning_tokens``
drop to 0. Every other state leaves the server default untouched — MiMo rejects a
top-level ``reasoning_effort`` (HTTP 400), so there is no effort granularity to map.

Mirrors ``tests/plugins/model_providers/test_kimi_profile.py``.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def xiaomi_profile():
    """Resolve the registered Xiaomi profile via the provider registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    Xiaomi profile. Going through ``get_provider_profile`` keeps the test honest:
    if the registered class is ever swapped for a plain ``ProviderProfile`` the
    disable assertion below collapses.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("xiaomi")
    assert profile is not None, "xiaomi provider profile must be registered"
    return profile


class TestXiaomiReasoningWireShape:
    def test_no_config_leaves_server_default(self, xiaomi_profile):
        extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
            reasoning_config=None
        )
        assert extra_body == {}
        assert top_level == {}

    def test_disabled_sends_thinking_disabled(self, xiaomi_profile):
        extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh"])
    def test_enabled_leaves_default(self, xiaomi_profile, effort):
        extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert extra_body == {}
        assert top_level == {}


class TestXiaomiFullKwargsIntegration:
    """End-to-end: the transport's full kwargs carry the reasoning wiring.

    The wire shape tests above call ``build_api_kwargs_extras`` in isolation;
    these drive the real profile-to-transport merge in
    ``ChatCompletionsTransport.build_kwargs`` so a regression in the merge (not
    just the profile) is caught. Mirrors ``TestZaiFullKwargsIntegration`` in
    ``tests/plugins/model_providers/test_zai_profile.py``.
    """

    def test_disabled_reaches_the_wire(self, xiaomi_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="mimo",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=xiaomi_profile,
            reasoning_config={"enabled": False},
            base_url="https://api.xiaomimimo.com/v1",
            provider_name="xiaomi",
        )
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    def test_no_preference_keeps_wire_clean(self, xiaomi_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="mimo",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=xiaomi_profile,
            reasoning_config=None,
            base_url="https://api.xiaomimimo.com/v1",
            provider_name="xiaomi",
        )
        assert "thinking" not in kwargs.get("extra_body", {})

    @pytest.mark.parametrize("effort", ["low", "high", "xhigh"])
    def test_enabled_keeps_wire_clean(self, xiaomi_profile, effort):
        # MiMo has no effort granularity (rejects top-level reasoning_effort with
        # HTTP 400), so any enabled level must leave the server default untouched.
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="mimo",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=xiaomi_profile,
            reasoning_config={"enabled": True, "effort": effort},
            base_url="https://api.xiaomimimo.com/v1",
            provider_name="xiaomi",
        )
        assert "thinking" not in kwargs.get("extra_body", {})
        assert "reasoning_effort" not in kwargs
