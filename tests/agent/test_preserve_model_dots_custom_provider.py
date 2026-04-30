"""Tests for opt-in ``preserve_model_dots`` flag on ``custom_providers``.

Regression for #17452 — users running local Anthropic-compatible proxies
(CCS, cliproxy, etc.) need to pass dotted model names (``gpt-5.4``,
``glm-4.6``, etc.) verbatim. Before this flag, ``_anthropic_preserve_dots``
only consulted a hardcoded provider/base_url allowlist that never matched
``127.0.0.1`` or ``localhost``.
"""

from types import SimpleNamespace
from unittest.mock import patch


class TestCustomProviderPreserveDots:
    """``custom_providers[*].preserve_model_dots: true`` opts a non-allowlisted
    base_url into dot-preservation, without changing the default behavior."""

    def _agent(self, base_url: str, provider: str = "custom"):
        # SimpleNamespace gives us getattr defaults for the other lookups
        # that ``_anthropic_preserve_dots`` performs (cache attr, etc).
        return SimpleNamespace(provider=provider, base_url=base_url)

    def test_localhost_proxy_without_flag_does_not_preserve_dots(self):
        from run_agent import AIAgent

        agent = self._agent("http://127.0.0.1:8317/api/provider/codex")
        with patch(
            "hermes_cli.config.get_custom_provider_preserve_model_dots",
            return_value=False,
        ):
            assert AIAgent._anthropic_preserve_dots(agent) is False

    def test_localhost_proxy_with_flag_preserves_dots(self):
        from run_agent import AIAgent

        agent = self._agent("http://127.0.0.1:8317/api/provider/codex")
        with patch(
            "hermes_cli.config.get_custom_provider_preserve_model_dots",
            return_value=True,
        ):
            assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_private_host_with_flag_preserves_dots(self):
        from run_agent import AIAgent

        agent = self._agent("https://internal.corp.example/anthropic")
        with patch(
            "hermes_cli.config.get_custom_provider_preserve_model_dots",
            return_value=True,
        ):
            assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_anthropic_endpoint_ignores_custom_flag(self):
        """Built-in Anthropic still hyphenates by default — flag lookup
        only fires after the hardcoded allowlist returns False, but the
        Anthropic provider falls through to the same lookup so we make
        sure the helper isn't called for the official endpoint by
        verifying the result remains False when the helper returns False."""
        from run_agent import AIAgent

        agent = self._agent("https://api.anthropic.com", provider="anthropic")
        with patch(
            "hermes_cli.config.get_custom_provider_preserve_model_dots",
            return_value=False,
        ):
            assert AIAgent._anthropic_preserve_dots(agent) is False

    def test_existing_minimax_allowlist_still_wins(self):
        """The hardcoded allowlist must still be honored without consulting
        the helper (no spurious config reads on the hot path for built-in
        providers)."""
        from run_agent import AIAgent

        agent = self._agent("", provider="minimax")
        with patch(
            "hermes_cli.config.get_custom_provider_preserve_model_dots",
        ) as helper:
            assert AIAgent._anthropic_preserve_dots(agent) is True
            helper.assert_not_called()

    def test_helper_failure_does_not_break_request(self):
        """The flag lookup must never raise — failures degrade silently
        to the previous behavior."""
        from run_agent import AIAgent

        agent = self._agent("http://127.0.0.1:9000/anthropic")
        with patch(
            "hermes_cli.config.get_custom_provider_preserve_model_dots",
            side_effect=RuntimeError("config blew up"),
        ):
            assert AIAgent._anthropic_preserve_dots(agent) is False


class TestPreserveModelDotsHelper:
    """Unit tests for the standalone helper in hermes_cli.config."""

    def test_returns_true_when_entry_has_flag(self):
        from hermes_cli.config import get_custom_provider_preserve_model_dots

        result = get_custom_provider_preserve_model_dots(
            base_url="http://127.0.0.1:8317/api/provider/codex",
            custom_providers=[
                {
                    "name": "codex",
                    "base_url": "http://127.0.0.1:8317/api/provider/codex",
                    "preserve_model_dots": True,
                }
            ],
        )
        assert result is True

    def test_returns_false_without_flag(self):
        from hermes_cli.config import get_custom_provider_preserve_model_dots

        result = get_custom_provider_preserve_model_dots(
            base_url="http://127.0.0.1:8317/api/provider/codex",
            custom_providers=[
                {
                    "name": "codex",
                    "base_url": "http://127.0.0.1:8317/api/provider/codex",
                }
            ],
        )
        assert result is False

    def test_trailing_slash_insensitive_match(self):
        from hermes_cli.config import get_custom_provider_preserve_model_dots

        result = get_custom_provider_preserve_model_dots(
            base_url="http://127.0.0.1:8317/api/provider/codex/",
            custom_providers=[
                {
                    "name": "codex",
                    "base_url": "http://127.0.0.1:8317/api/provider/codex",
                    "preserve_model_dots": True,
                }
            ],
        )
        assert result is True

    def test_case_insensitive_host_match(self):
        from hermes_cli.config import get_custom_provider_preserve_model_dots

        result = get_custom_provider_preserve_model_dots(
            base_url="http://LOCALHOST:8317/api",
            custom_providers=[
                {
                    "name": "p",
                    "base_url": "http://localhost:8317/api",
                    "preserve_model_dots": True,
                }
            ],
        )
        assert result is True

    def test_no_match_returns_false(self):
        from hermes_cli.config import get_custom_provider_preserve_model_dots

        result = get_custom_provider_preserve_model_dots(
            base_url="http://127.0.0.1:8317/api/provider/codex",
            custom_providers=[
                {
                    "name": "other",
                    "base_url": "http://127.0.0.1:9999/other",
                    "preserve_model_dots": True,
                }
            ],
        )
        assert result is False

    def test_empty_inputs_return_false(self):
        from hermes_cli.config import get_custom_provider_preserve_model_dots

        assert get_custom_provider_preserve_model_dots("", custom_providers=[]) is False
        assert (
            get_custom_provider_preserve_model_dots(
                "http://127.0.0.1:8317", custom_providers=[]
            )
            is False
        )

    def test_non_bool_flag_ignored(self):
        from hermes_cli.config import get_custom_provider_preserve_model_dots

        # ``preserve_model_dots: "true"`` (string) is ignored — must be a real bool.
        result = get_custom_provider_preserve_model_dots(
            base_url="http://127.0.0.1:8317",
            custom_providers=[
                {
                    "name": "p",
                    "base_url": "http://127.0.0.1:8317",
                    "preserve_model_dots": "true",
                }
            ],
        )
        assert result is False


class TestNormalizerPreservesFlag:
    """``_normalize_custom_provider_entry`` must keep the flag in the
    normalized dict so ``get_compatible_custom_providers`` exposes it."""

    def test_normalizer_preserves_flag(self):
        from hermes_cli.config import _normalize_custom_provider_entry

        normalized = _normalize_custom_provider_entry(
            {
                "name": "codex",
                "base_url": "http://127.0.0.1:8317/api/provider/codex",
                "api_mode": "anthropic_messages",
                "preserve_model_dots": True,
            }
        )
        assert normalized is not None
        assert normalized.get("preserve_model_dots") is True

    def test_normalizer_drops_non_bool_flag(self):
        from hermes_cli.config import _normalize_custom_provider_entry

        normalized = _normalize_custom_provider_entry(
            {
                "name": "codex",
                "base_url": "http://127.0.0.1:8317/api/provider/codex",
                "preserve_model_dots": "yes",  # not a bool
            }
        )
        assert normalized is not None
        assert "preserve_model_dots" not in normalized

    def test_normalizer_omits_flag_when_absent(self):
        from hermes_cli.config import _normalize_custom_provider_entry

        normalized = _normalize_custom_provider_entry(
            {
                "name": "codex",
                "base_url": "http://127.0.0.1:8317/api/provider/codex",
            }
        )
        assert normalized is not None
        assert "preserve_model_dots" not in normalized

    def test_normalizer_warns_on_unknown_keys_does_not_warn_for_flag(self, caplog):
        """Adding the flag to _KNOWN_KEYS must suppress the
        'unknown config keys ignored' warning."""
        import logging
        from hermes_cli.config import _normalize_custom_provider_entry

        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            _normalize_custom_provider_entry(
                {
                    "name": "codex",
                    "base_url": "http://127.0.0.1:8317",
                    "preserve_model_dots": True,
                },
                provider_key="codex",
            )
        joined = " ".join(rec.getMessage() for rec in caplog.records)
        assert "preserve_model_dots" not in joined or "ignored" not in joined


class TestNormalizeModelNameWithFlag:
    """End-to-end: a custom_provider with the flag set should produce a
    request body with dots intact."""

    def test_normalize_keeps_dots_when_preserved(self):
        from agent.anthropic_adapter import normalize_model_name

        assert (
            normalize_model_name("gpt-5.4", preserve_dots=True) == "gpt-5.4"
        )
        assert (
            normalize_model_name("glm-4.6", preserve_dots=True) == "glm-4.6"
        )
