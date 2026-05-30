"""Regression tests for custom_providers per-model max_tokens resolution.

Covers #28046 / #28782 — a per-model ``custom_providers[].models.<id>.max_tokens``
output cap must reach ``agent.max_tokens`` at startup AND be re-resolved on
mid-session /model switch and fallback, the same way context_length is.
"""

from unittest.mock import MagicMock, patch

import pytest

import agent.agent_init as agent_init
from hermes_cli.config import get_custom_provider_max_tokens

URL = "https://example.invalid/v1"


def _cfg(max_tokens, *, base_url=URL, model="m"):
    return [{"base_url": base_url, "models": {model: {"max_tokens": max_tokens}}}]


class TestGetCustomProviderMaxTokens:
    def test_returns_override_for_matching_entry(self):
        assert get_custom_provider_max_tokens("m", URL, _cfg(131_072)) == 131_072

    @pytest.mark.parametrize("entry_url, query_url", [
        (URL + "/", URL),
        (URL, URL + "/"),
    ])
    def test_trailing_slash_insensitive(self, entry_url, query_url):
        assert get_custom_provider_max_tokens("m", query_url, _cfg(32_000, base_url=entry_url)) == 32_000

    def test_returns_none_when_url_does_not_match(self):
        assert get_custom_provider_max_tokens("m", URL, _cfg(32_000, base_url="https://other.invalid/v1")) is None

    def test_returns_none_when_model_missing(self):
        assert get_custom_provider_max_tokens("m", URL, _cfg(32_000, model="other")) is None

    def test_numeric_string_is_coerced(self):
        assert get_custom_provider_max_tokens("m", URL, _cfg("16000")) == 16_000

    @pytest.mark.parametrize("bad", [0, -1, "0", "-5"])
    def test_zero_and_negative_skipped(self, bad):
        assert get_custom_provider_max_tokens("m", URL, _cfg(bad)) is None

    def test_non_numeric_skipped(self):
        assert get_custom_provider_max_tokens("m", URL, _cfg("32K")) is None

    def test_bool_rejected(self):
        # bool is an int subclass — must be rejected, not coerced to 1.
        assert get_custom_provider_max_tokens("m", URL, _cfg(True)) is None

    @pytest.mark.parametrize("model, url, providers", [
        ("", URL, []),
        ("m", "", []),
        ("m", URL, None),
    ])
    def test_empty_inputs_guarded(self, model, url, providers):
        assert get_custom_provider_max_tokens(model, url, providers) is None


def _bare_agent(custom_providers, *, model="m", base_url=URL, max_tokens=None):
    """Minimal agent stub exercising only the max_tokens resolution block."""
    a = MagicMock()
    a.model = model
    a.base_url = base_url
    a.max_tokens = max_tokens
    a._custom_providers = custom_providers
    a._session_init_model_config = {}
    return a


class TestStartupPrecedence:
    def test_per_model_lands_on_agent(self):
        a = _bare_agent(_cfg(131_072))
        agent_init._resolve_max_tokens(a, {"model": {}})
        assert a.max_tokens == 131_072
        assert a._config_max_tokens == 131_072
        assert a._max_tokens_explicit is False

    def test_per_model_beats_global(self):
        a = _bare_agent(_cfg(131_072))
        agent_init._resolve_max_tokens(a, {"model": {"max_tokens": 8000}})
        assert a.max_tokens == 131_072

    def test_global_used_when_no_per_model(self):
        a = _bare_agent([])
        agent_init._resolve_max_tokens(a, {"model": {"max_tokens": 8000}})
        assert a.max_tokens == 8000

    def test_explicit_constructor_wins_and_marks_explicit(self):
        a = _bare_agent(_cfg(131_072), max_tokens=5000)
        agent_init._resolve_max_tokens(a, {"model": {"max_tokens": 8000}})
        assert a.max_tokens == 5000
        assert a._max_tokens_explicit is True
        assert a._config_max_tokens is None  # explicit is not config-derived

    def test_invalid_per_model_warns_and_falls_through(self):
        a = _bare_agent(_cfg("32K"))
        with patch.object(agent_init, "_ra") as mock_ra:
            agent_init._resolve_max_tokens(a, {"model": {}})
            assert mock_ra.return_value.logger.warning.called
        assert a.max_tokens is None
        assert a._config_max_tokens is None

    def test_invalid_global_warns_and_falls_through(self):
        a = _bare_agent([])
        with patch.object(agent_init, "_ra") as mock_ra:
            agent_init._resolve_max_tokens(a, {"model": {"max_tokens": "lots"}})
            assert mock_ra.return_value.logger.warning.called
        assert a.max_tokens is None
