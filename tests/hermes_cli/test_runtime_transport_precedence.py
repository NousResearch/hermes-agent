"""Runtime transport precedence: declared provider transport is the fallback.

The Coatue data-residency report (2026-07): pointing ``openai-api`` at
``us.api.openai.com`` silently fell back to ``chat_completions`` — every
tool-calling turn 400'd — because the runtime resolvers defaulted to
``chat_completions`` and consulted URL detection only, never the transport
the provider overlay itself declares.

Contract pinned here: when URL detection has no opinion, the runtime falls
back to ``providers.determine_api_mode(provider, base_url, model)`` (the
provider's declared transport) **only while the endpoint sits on the
provider's own default host**, and lands on ``chat_completions`` for
genuinely unknown providers/endpoints. A ``model.base_url`` override that
points at a different endpoint (OpenAI-compatible relay in front of MiniMax,
corporate egress gateway, LiteLLM, ...) is protocol-unknown: the declared
transport must not be assumed, or a proxied ``minimax`` setup gets
Anthropic-shaped requests / ``x-api-key`` against a chat/completions
endpoint and 401s (#76836). Covers the explicit-runtime path and the
API-key-provider path; the pool-entry path shares the same helper.
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

    def test_openai_api_custom_proxy_override_uses_generic_chat(self):
        # Explicitly selected openai-api against a custom proxy does NOT keep
        # the provider's declared transport: the override points at a
        # different endpoint whose wire protocol is unknown, so the safe
        # generic chat_completions default applies (pre-33a2f29a6 behavior).
        # Regression class of #76836 — the declared transport is a statement
        # about the provider's own endpoint, not about foreign relays.
        assert (
            _fallback_api_mode("openai-api", "https://proxy.corp.test/v1")
            == "chat_completions"
        )

    def test_minimax_proxy_override_falls_back_to_chat_completions(self):
        # #76836: minimax with model.base_url overridden to an
        # OpenAI-compatible relay (token-optimization proxy, LiteLLM,
        # corporate egress) must resolve chat_completions — not the
        # provider's declared anthropic_messages transport — or every request
        # 401s (x-api-key sent to an Authorization: Bearer endpoint).
        assert (
            _fallback_api_mode("minimax", "http://127.0.0.1:8787/v1", "MiniMax-M3")
            == "chat_completions"
        )

    def test_minimax_cn_proxy_override_falls_back_to_chat_completions(self):
        assert (
            _fallback_api_mode("minimax-cn", "http://127.0.0.1:8788/v1")
            == "chat_completions"
        )

    def test_declared_transport_applies_on_provider_own_host(self):
        # The declared non-chat transport still holds on the provider's own
        # default host — the #53054 fix is preserved for non-URL-detected
        # endpoints (bare host, no /anthropic hint).
        assert (
            _fallback_api_mode("minimax", "https://api.minimax.io")
            == "anthropic_messages"
        )
        assert (
            _fallback_api_mode("minimax-cn", "https://api.minimaxi.com")
            == "anthropic_messages"
        )

    def test_lookalike_host_is_not_treated_as_official(self):
        # The spoof host must not be detected AS OpenAI by the URL lane —
        # the provider-declared transport may still apply, but host-derived
        # detection must return None for it.
        from hermes_cli.runtime_provider import _detect_api_mode_for_url

        assert _detect_api_mode_for_url("https://api.openai.com.attacker.test/v1") is None

    def test_openrouter_stays_chat_completions(self):
        assert _fallback_api_mode("openrouter", "https://openrouter.ai/api/v1") == "chat_completions"

    def test_minimax_declared_anthropic_transport_honored(self):
        # Same latent bug class: minimax declares an Anthropic-compatible
        # transport but previously fell back to chat_completions when the
        # URL carried no /anthropic hint.
        from hermes_cli.providers import determine_api_mode

        expected = determine_api_mode("minimax", "https://api.minimax.io")
        assert _fallback_api_mode("minimax", "https://api.minimax.io") == expected
        assert expected != "chat_completions" or expected == determine_api_mode("minimax", "")

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
