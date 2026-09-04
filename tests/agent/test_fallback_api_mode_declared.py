"""Regression tests for fallback api_mode resolution (Volcengine Ark #B).

``try_activate_fallback`` used to leave ``fb_api_mode`` at the bare
``chat_completions`` default whenever its own detection chain matched
nothing, so a provider that speaks a non-OpenAI wire silently degraded the
moment it was demoted from primary model into the fallback chain.
Volcengine Ark's coding plan is the concrete case: it declares
``anthropic_messages`` via its plugin profile and 404s on
``/chat/completions``.

These cover ``_declared_fallback_api_mode``, the helper that consults the
same two authorities the primary path already honours.
"""

from unittest.mock import patch

from agent.chat_completion_helpers import _declared_fallback_api_mode


class TestDeclaredFallbackApiMode:
    """The helper answers 'what wire does this provider speak?' or abstains."""

    def test_config_block_api_mode_wins(self):
        """A custom provider pins its wire with providers.<name>.api_mode.

        ``resolve_user_provider`` only reads the ``transport`` key, so
        ``determine_api_mode`` cannot see this — the config lookup must.
        """
        cfg = {"providers": {"volcano-agent": {"api_mode": "codex_responses"}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="chat_completions"):
            assert _declared_fallback_api_mode(
                "volcano-agent", "https://example.test/api/plan/v3", "ark-code-latest",
            ) == "codex_responses"

    def test_provider_registry_transport_consulted(self):
        """With config silent, the registry/plugin transport decides."""
        with patch("hermes_cli.config.load_config_readonly", return_value={}), \
             patch("hermes_cli.providers.determine_api_mode", return_value="anthropic_messages"):
            assert _declared_fallback_api_mode(
                "volcano", "https://ark.cn-beijing.volces.com/api/coding", "glm-5.3",
            ) == "anthropic_messages"

    def test_chat_completions_is_no_opinion(self):
        """Plain OpenAI-wire providers must not be disturbed."""
        with patch("hermes_cli.config.load_config_readonly", return_value={}), \
             patch("hermes_cli.providers.determine_api_mode", return_value="chat_completions"):
            assert _declared_fallback_api_mode(
                "deepseek", "https://api.deepseek.com", "deepseek-chat",
            ) == ""

    def test_provider_lookup_is_case_insensitive(self):
        """Config keys keep their display casing; chain entries are lowercased."""
        cfg = {"providers": {"QwenAI_Kimi": {"api_mode": "anthropic_messages"}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="chat_completions"):
            assert _declared_fallback_api_mode(
                "qwenai_kimi", "https://example.test/v1", "kimi-k2",
            ) == "anthropic_messages"

    def test_blank_config_api_mode_falls_through_to_registry(self):
        """An empty api_mode key is not an opinion — keep looking."""
        cfg = {"providers": {"volcano": {"api_mode": "   "}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="anthropic_messages"):
            assert _declared_fallback_api_mode(
                "volcano", "https://ark.cn-beijing.volces.com/api/coding", "glm-5.3",
            ) == "anthropic_messages"

    def test_config_read_failure_is_non_fatal(self):
        """A broken config must not break failover — fall through to the registry."""
        with patch("hermes_cli.config.load_config_readonly", side_effect=OSError("boom")), \
             patch("hermes_cli.providers.determine_api_mode", return_value="anthropic_messages"):
            assert _declared_fallback_api_mode(
                "volcano", "https://ark.cn-beijing.volces.com/api/coding", "glm-5.3",
            ) == "anthropic_messages"

    def test_registry_failure_abstains(self):
        """If both authorities fail the caller keeps its own default."""
        with patch("hermes_cli.config.load_config_readonly", return_value={}), \
             patch("hermes_cli.providers.determine_api_mode", side_effect=RuntimeError("boom")):
            assert _declared_fallback_api_mode("mystery", "https://example.test/v1", "m") == ""

    def test_empty_provider_still_consults_registry(self):
        """No provider name is not a crash — just skip the config lookup."""
        with patch("hermes_cli.providers.determine_api_mode", return_value="chat_completions"):
            assert _declared_fallback_api_mode("", "https://example.test/v1", "m") == ""
