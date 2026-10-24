"""Regression tests for fallback api_mode resolution (Volcengine Ark).

``try_activate_fallback`` used to leave ``fb_api_mode`` at the bare ``chat_completions``
default whenever its own detection chain matched nothing, so a provider that speaks a
non-OpenAI wire silently degraded the moment it was demoted from primary model into the
fallback chain. Volcengine Ark's coding plan is the concrete case: it declares
``anthropic_messages`` via its plugin profile, and ``/chat/completions`` on that host 404s.

``_fallback_api_mode_resolved`` is that tail. It now asks ``_fallback_api_mode_declared``
what wire the provider declares for itself before falling back to ``chat_completions``.
"""

from unittest.mock import patch

from agent.chat_completion_helpers import (
    _fallback_api_mode_declared,
    _fallback_api_mode_resolved,
)


class _StubAgent:
    """Only the URL-shape predicates _fallback_api_mode_resolved consults."""

    def _is_azure_openai_url(self, url: str) -> bool:
        return "azure" in (url or "").lower()

    def _is_direct_openai_url(self, url: str) -> bool:
        return False

    def _provider_model_requires_responses_api(self, model: str, provider: str = "") -> bool:
        return False


class TestDeclaredFallbackApiMode:
    """The helper answers "what wire does this provider speak?" or abstains."""

    def test_config_transport_wins_over_the_registry(self):
        """A custom provider pins its wire with providers.<name>.transport.

        ``resolve_user_provider`` — the reader of that section — honours only that key,
        and ``determine_api_mode`` never sees user config at all.
        """
        cfg = {"providers": {"volcano-coding-plan": {"transport": "anthropic_messages"}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="codex_responses"):
            assert _fallback_api_mode_declared(
                "volcano-coding-plan", "https://ark.cn-beijing.volces.com/api/coding", "glm-5.3",
            ) == "anthropic_messages"

    def test_config_transport_beats_the_legacy_alias(self):
        """Both keys present: ``transport`` is the one ``resolve_user_provider`` reads."""
        cfg = {"providers": {"volcano": {"transport": "anthropic_messages",
                                         "api_mode": "codex_responses"}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="codex_responses"):
            assert _fallback_api_mode_declared("volcano", "", "m") == "anthropic_messages"

    def test_legacy_api_mode_alias_is_still_read(self):
        """Configs predating the ``transport`` rename carry only ``api_mode``."""
        cfg = {"providers": {"volcano": {"api_mode": "anthropic_messages"}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="codex_responses"):
            assert _fallback_api_mode_declared("volcano", "", "m") == "anthropic_messages"

    def test_legacy_spellings_are_canonicalized(self):
        """``anthropic`` is an accepted alias for anthropic_messages, as in _API_MODE_ALIASES."""
        cfg = {"providers": {"volcano": {"transport": "anthropic"}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="codex_responses"):
            assert _fallback_api_mode_declared("volcano", "", "m") == "anthropic_messages"

    def test_unknown_transport_falls_through_to_the_registry(self):
        """A spelling that maps to no known wire is not an opinion — keep looking."""
        cfg = {"providers": {"volcano": {"transport": "telepathy"}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="codex_responses"):
            assert _fallback_api_mode_declared("volcano", "", "m") == "codex_responses"

    def test_config_chat_completions_is_an_explicit_answer(self):
        """Same rule as _fallback_api_mode_hint: explicit wins, even chat_completions."""
        cfg = {"providers": {"volcano": {"transport": "openai_chat"}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="anthropic_messages"):
            assert _fallback_api_mode_declared("volcano", "", "m") == "chat_completions"

    def test_provider_lookup_is_case_insensitive(self):
        """Config keys keep their display casing; chain entries are lowercased."""
        cfg = {"providers": {"QwenAI_Kimi": {"transport": "anthropic_messages"}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="codex_responses"):
            assert _fallback_api_mode_declared("qwenai_kimi", "", "kimi-k2") == "anthropic_messages"

    def test_blank_config_transport_falls_through_to_the_registry(self):
        """An empty transport is not an opinion — keep looking."""
        cfg = {"providers": {"volcano": {"transport": "   "}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="anthropic_messages"):
            assert _fallback_api_mode_declared("volcano", "", "m") == "anthropic_messages"

    def test_registry_transport_is_consulted(self):
        """With config silent, the registry / plugin-profile transport decides."""
        with patch("hermes_cli.config.load_config_readonly", return_value={}), \
             patch("hermes_cli.providers.determine_api_mode", return_value="anthropic_messages"):
            assert _fallback_api_mode_declared(
                "volcano", "https://ark.cn-beijing.volces.com/api/coding", "glm-5.3",
            ) == "anthropic_messages"

    def test_registry_chat_completions_is_no_opinion(self):
        """Plain OpenAI-wire providers must not be disturbed."""
        with patch("hermes_cli.config.load_config_readonly", return_value={}), \
             patch("hermes_cli.providers.determine_api_mode", return_value="chat_completions"):
            assert _fallback_api_mode_declared(
                "deepseek", "https://api.deepseek.com", "deepseek-chat",
            ) == ""

    def test_config_read_failure_is_non_fatal(self):
        """A broken config must not break failover — fall through to the registry."""
        with patch("hermes_cli.config.load_config_readonly", side_effect=OSError("boom")), \
             patch("hermes_cli.providers.determine_api_mode", return_value="anthropic_messages"):
            assert _fallback_api_mode_declared("volcano", "", "m") == "anthropic_messages"

    def test_registry_failure_abstains(self):
        """If both authorities fail the caller keeps its own default."""
        with patch("hermes_cli.config.load_config_readonly", return_value={}), \
             patch("hermes_cli.providers.determine_api_mode", side_effect=RuntimeError("boom")):
            assert _fallback_api_mode_declared("mystery", "https://example.test/v1", "m") == ""

    def test_empty_provider_still_consults_the_registry(self):
        """No provider name is not a crash — just skip the config lookup."""
        with patch("hermes_cli.providers.determine_api_mode", return_value="chat_completions"):
            assert _fallback_api_mode_declared("", "https://example.test/v1", "m") == ""


class TestResolvedTail:
    """The tail of _fallback_api_mode_resolved actually consults the declaration."""

    def test_declared_wire_replaces_the_bare_default(self):
        cfg = {"providers": {"volcano-coding-plan": {"transport": "anthropic_messages"}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="chat_completions"):
            assert _fallback_api_mode_resolved(
                _StubAgent(), "volcano-coding-plan", "ark-code-latest",
                "https://ark.cn-beijing.volces.com/api/coding",
            ) == "anthropic_messages"

    def test_plain_openai_wire_keeps_the_default(self):
        with patch("hermes_cli.config.load_config_readonly", return_value={}), \
             patch("hermes_cli.providers.determine_api_mode", return_value="chat_completions"):
            assert _fallback_api_mode_resolved(
                _StubAgent(), "deepseek", "deepseek-chat", "https://api.deepseek.com",
            ) == "chat_completions"

    def test_url_shape_still_outranks_the_declaration(self):
        """The checks above the tail keep priority: a /anthropic URL is never overridden."""
        cfg = {"providers": {"proxy": {"transport": "chat_completions"}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
             patch("hermes_cli.providers.determine_api_mode", return_value="chat_completions"):
            assert _fallback_api_mode_resolved(
                _StubAgent(), "proxy", "m", "https://gateway.example.test/anthropic",
            ) == "anthropic_messages"
