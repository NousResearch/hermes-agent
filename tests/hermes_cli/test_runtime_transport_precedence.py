"""Runtime transport precedence: declared provider transport is the fallback.

The Coatue data-residency report (2026-07): pointing ``openai-api`` at
``us.api.openai.com`` silently fell back to ``chat_completions`` — every
tool-calling turn 400'd — because the runtime resolvers defaulted to
``chat_completions`` and consulted URL detection only, never the transport
the provider overlay itself declares.

Contract pinned here: when URL detection has no opinion, the runtime falls
back to ``providers.determine_api_mode(provider, base_url, model)`` (the
provider's declared transport), and lands on ``chat_completions`` for
genuinely unknown providers/endpoints and for a declared ``anthropic_messages``
off the provider's own endpoint (#76836). Covers the explicit-runtime path and
the API-key-provider path; the pool-entry path shares the same helper.
"""

from __future__ import annotations

from unittest.mock import patch as mock_patch

import pytest

from hermes_cli.runtime_provider import _fallback_api_mode


class TestFallbackApiMode:
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.openai.com/v1",
            "https://us.api.openai.com/v1",
            "https://eu.api.openai.com/v1",
        ],
    )
    def test_openai_api_official_hosts_resolve_codex_responses(self, base_url):
        assert _fallback_api_mode("openai-api", base_url) == "codex_responses"

    def test_openai_api_unknown_custom_proxy_still_uses_declared_transport(self):
        # Explicitly selected openai-api against a custom proxy keeps the
        # provider's declared transport (mirrors determine_api_mode semantics;
        # host identity is a separate question from provider selection).
        assert (
            _fallback_api_mode("openai-api", "https://proxy.corp.test/v1")
            == "codex_responses"
        )

    @pytest.fixture
    def catalog_defaults(self, monkeypatch):
        """Pin the models.dev default endpoints the predicate compares against, so the contract
        holds without the network (a cold cache leaves ``ProviderDef.base_url`` empty)."""
        import dataclasses
        from hermes_cli import runtime_provider
        defaults = {
            "minimax": "https://api.minimax.io/anthropic/v1",
            "minimax-cn": "https://api.minimaxi.com/anthropic/v1",
            "tencent-tokenplan": "https://api.lkeap.cloud.tencent.com/plan/anthropic",
            "anthropic": "https://api.anthropic.com",
        }
        real = runtime_provider.get_provider

        def seeded(name, *, allow_network=True):
            # allow_network is dropped so the fixture can never reach models.dev; replace()
            # rather than mutate so the fixture never depends on get_provider's object identity.
            # Seed unconditionally: a warm disk cache must not swap in whatever models.dev
            # publishes today.
            pdef = real(name, allow_network=False)
            if pdef is not None and name in defaults:
                pdef = dataclasses.replace(pdef, base_url=defaults[name])
            return pdef

        monkeypatch.setattr(runtime_provider, "get_provider", seeded)

    @pytest.mark.usefixtures("catalog_defaults")
    @pytest.mark.parametrize("provider, base_url", [
        ("minimax", "http://127.0.0.1:8787/v1"),        # OpenAI-compatible relay in front of MiniMax
        ("minimax-cn", "https://egress.corp.test/v1"),  # corporate egress gateway
        ("minimax", "https://api.minimax.io/v1"),       # provider's own OpenAI-compatible path
        ("minimax", "https://api.minimax.io/anthropic-compat/v1"),  # lookalike path, not /anthropic/
    ])
    def test_anthropic_transport_is_not_assumed_off_the_provider_endpoint(self, provider, base_url):
        # #76836: the declared anthropic_messages transport is a statement about the provider's
        # own /anthropic endpoint. Anywhere else the override speaks chat/completions; Messages-
        # shaped requests with x-api-key there 401 on every turn (or silently reroute to a fallback).
        assert _fallback_api_mode(provider, base_url, "MiniMax-M3") == "chat_completions"

    @pytest.mark.usefixtures("catalog_defaults")
    @pytest.mark.parametrize("provider, base_url", [
        ("minimax", "https://api.minimax.io"),             # bare host, no /anthropic hint (#53054 kept)
        ("minimax", "https://api.minimax.io/anthropic"),   # URL-detected native path
        ("minimax-cn", "https://api.minimaxi.com/anthropic/v1"),
        ("minimax", "https://api.minimax.io/anthropic/v1/messages"),  # under the default path
        ("tencent-tokenplan", "https://api.lkeap.cloud.tencent.com/plan/anthropic/v2"),  # non-/anthropic default
        ("minimax", ""),  # nothing to compare: keep the declared transport
        ("anthropic", "http://127.0.0.1:4000/v1"),  # bare-host default: relays serve /v1/messages
    ])
    def test_anthropic_transport_holds_on_the_provider_endpoint(self, provider, base_url):
        assert _fallback_api_mode(provider, base_url, "MiniMax-M3") == "anthropic_messages"

    def test_lookalike_host_is_not_treated_as_official(self):
        # The spoof host must not be detected AS OpenAI by the URL lane —
        # the provider-declared transport may still apply, but host-derived
        # detection must return None for it.
        from hermes_cli.runtime_provider import _detect_api_mode_for_url

        assert _detect_api_mode_for_url("https://api.openai.com.attacker.test/v1") is None

    def test_openrouter_stays_chat_completions(self):
        assert _fallback_api_mode("openrouter", "https://openrouter.ai/api/v1") == "chat_completions"


    def test_unknown_provider_defaults_chat_completions(self):
        assert _fallback_api_mode("some-unknown", "https://example.test/v1") == "chat_completions"

    def test_url_detection_wins_over_provider_declaration(self):
        # /anthropic suffix on any provider routes anthropic_messages —
        # URL detection stays the higher-priority signal.
        assert (
            _fallback_api_mode("openai-api", "https://gateway.test/anthropic")
            == "anthropic_messages"
        )


class TestExplicitRuntimeIntegration:
    """The explicit-runtime path resolves regional OpenAI to codex_responses."""

    def test_explicit_openai_api_regional_host(self):
        from hermes_cli.runtime_provider import _resolve_explicit_runtime

        with mock_patch(
            "hermes_cli.runtime_provider._get_model_config",
            return_value={"provider": "openai-api", "default": "gpt-5.6-terra"},
        ):
            result = _resolve_explicit_runtime(
                provider="openai-api",
                requested_provider="openai-api",
                explicit_api_key="sk-test",
                explicit_base_url="https://us.api.openai.com/v1",
                model_cfg={"provider": "openai-api", "default": "gpt-5.6-terra"},
            )
        assert result is not None
        assert result["api_mode"] == "codex_responses"
        assert result["base_url"] == "https://us.api.openai.com/v1"
