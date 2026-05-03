"""Tests for OpenRouter response caching header injection."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# build_or_headers
# ---------------------------------------------------------------------------

class TestBuildOrHeaders:
    """Test the build_or_headers() helper in agent/auxiliary_client.py."""

    def test_base_attribution_always_present(self):
        """Attribution headers must always be included regardless of cache setting."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": False})
        assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
        assert headers["X-OpenRouter-Title"] == "Hermes Agent"
        assert headers["X-OpenRouter-Categories"] == "productivity,cli-agent"

    def test_cache_enabled(self):
        """When response_cache is True, X-OpenRouter-Cache header is set."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": True})
        assert headers["X-OpenRouter-Cache"] == "true"

    def test_cache_disabled(self):
        """When response_cache is False, no cache header is sent."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": False})
        assert "X-OpenRouter-Cache" not in headers
        assert "X-OpenRouter-Cache-TTL" not in headers

    def test_cache_disabled_by_default_empty_config(self):
        """Empty config dict means no cache headers (response_cache defaults to False)."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={})
        assert "X-OpenRouter-Cache" not in headers

    def test_ttl_default(self):
        """Default TTL (300) is included when cache is enabled."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": True, "response_cache_ttl": 300})
        assert headers["X-OpenRouter-Cache-TTL"] == "300"

    def test_ttl_custom(self):
        """Custom TTL values within range are sent."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": True, "response_cache_ttl": 3600})
        assert headers["X-OpenRouter-Cache-TTL"] == "3600"

    def test_ttl_max(self):
        """Maximum TTL (86400) is accepted."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": True, "response_cache_ttl": 86400})
        assert headers["X-OpenRouter-Cache-TTL"] == "86400"

    def test_ttl_out_of_range_too_high(self):
        """TTL above 86400 is silently ignored (no TTL header sent)."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": True, "response_cache_ttl": 100000})
        assert "X-OpenRouter-Cache-TTL" not in headers
        # But cache is still enabled
        assert headers["X-OpenRouter-Cache"] == "true"

    def test_ttl_out_of_range_zero(self):
        """TTL of 0 is below minimum — no TTL header sent."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": True, "response_cache_ttl": 0})
        assert "X-OpenRouter-Cache-TTL" not in headers

    def test_ttl_negative(self):
        """Negative TTL is ignored."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": True, "response_cache_ttl": -5})
        assert "X-OpenRouter-Cache-TTL" not in headers

    def test_ttl_not_a_number(self):
        """Non-numeric TTL is ignored."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": True, "response_cache_ttl": "five"})
        assert "X-OpenRouter-Cache-TTL" not in headers

    def test_ttl_float_truncated(self):
        """Float TTL values are truncated to int."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": True, "response_cache_ttl": 600.7})
        assert headers["X-OpenRouter-Cache-TTL"] == "600"

    def test_returns_fresh_dict(self):
        """Each call returns a new dict so mutations don't leak."""
        from agent.auxiliary_client import build_or_headers

        cfg = {"response_cache": True}
        h1 = build_or_headers(or_config=cfg)
        h2 = build_or_headers(or_config=cfg)
        assert h1 is not h2
        assert h1 == h2

    def test_none_config_falls_back_to_load_config(self):
        """When or_config is None, build_or_headers reads from load_config()."""
        from agent.auxiliary_client import build_or_headers

        fake_cfg = {
            "openrouter": {"response_cache": True, "response_cache_ttl": 900},
        }
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            headers = build_or_headers(or_config=None)
        assert headers["X-OpenRouter-Cache"] == "true"
        assert headers["X-OpenRouter-Cache-TTL"] == "900"

    def test_none_config_load_config_fails_gracefully(self):
        """When load_config() fails, build_or_headers still returns base headers."""
        from agent.auxiliary_client import build_or_headers

        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            headers = build_or_headers(or_config=None)
        # Should have base attribution but no cache headers
        assert "HTTP-Referer" in headers
        assert "X-OpenRouter-Cache" not in headers


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

class TestEnvVarOverrides:
    """Test env var precedence over config.yaml for response caching."""

    def test_env_enables_cache(self, monkeypatch):
        """HERMES_OPENROUTER_CACHE=true enables cache even when config disables it."""
        from agent.auxiliary_client import build_or_headers

        monkeypatch.setenv("HERMES_OPENROUTER_CACHE", "true")
        headers = build_or_headers(or_config={"response_cache": False})
        assert headers["X-OpenRouter-Cache"] == "true"

    def test_env_disables_cache(self, monkeypatch):
        """HERMES_OPENROUTER_CACHE=false disables cache even when config enables it."""
        from agent.auxiliary_client import build_or_headers

        monkeypatch.setenv("HERMES_OPENROUTER_CACHE", "false")
        headers = build_or_headers(or_config={"response_cache": True})
        assert "X-OpenRouter-Cache" not in headers

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes", "on"])
    def test_truthy_values(self, monkeypatch, value):
        """Various truthy strings enable caching."""
        from agent.auxiliary_client import build_or_headers

        monkeypatch.setenv("HERMES_OPENROUTER_CACHE", value)
        headers = build_or_headers(or_config={})
        assert headers["X-OpenRouter-Cache"] == "true"

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "maybe", ""])
    def test_non_truthy_values(self, monkeypatch, value):
        """Non-truthy strings do not enable caching (empty falls through to config)."""
        from agent.auxiliary_client import build_or_headers

        monkeypatch.setenv("HERMES_OPENROUTER_CACHE", value)
        # Empty string falls through to config; others are explicitly non-truthy
        if value == "":
            # Empty env var falls through to config default (False)
            headers = build_or_headers(or_config={"response_cache": False})
        else:
            headers = build_or_headers(or_config={"response_cache": True})
        assert "X-OpenRouter-Cache" not in headers

    def test_env_ttl_overrides_config(self, monkeypatch):
        """HERMES_OPENROUTER_CACHE_TTL overrides config TTL."""
        from agent.auxiliary_client import build_or_headers

        monkeypatch.setenv("HERMES_OPENROUTER_CACHE", "true")
        monkeypatch.setenv("HERMES_OPENROUTER_CACHE_TTL", "1800")
        headers = build_or_headers(or_config={"response_cache_ttl": 300})
        assert headers["X-OpenRouter-Cache-TTL"] == "1800"

    @pytest.mark.parametrize("ttl", ["0", "86401", "abc", "-1", "12.5"])
    def test_invalid_env_ttl_dropped(self, monkeypatch, ttl):
        """Invalid TTL env values are ignored; cache still enabled without TTL."""
        from agent.auxiliary_client import build_or_headers

        monkeypatch.setenv("HERMES_OPENROUTER_CACHE", "1")
        monkeypatch.setenv("HERMES_OPENROUTER_CACHE_TTL", ttl)
        headers = build_or_headers(or_config={})
        assert headers["X-OpenRouter-Cache"] == "true"
        assert "X-OpenRouter-Cache-TTL" not in headers

    @pytest.mark.parametrize("ttl", ["1", "300", "86400"])
    def test_valid_env_ttl_boundaries(self, monkeypatch, ttl):
        """Boundary TTL values (1, 300, 86400) are accepted."""
        from agent.auxiliary_client import build_or_headers

        monkeypatch.setenv("HERMES_OPENROUTER_CACHE", "yes")
        monkeypatch.setenv("HERMES_OPENROUTER_CACHE_TTL", ttl)
        assert build_or_headers(or_config={})["X-OpenRouter-Cache-TTL"] == ttl

    def test_no_env_vars_falls_through_to_config(self, monkeypatch):
        """Without env vars, config.yaml controls behavior."""
        from agent.auxiliary_client import build_or_headers

        monkeypatch.delenv("HERMES_OPENROUTER_CACHE", raising=False)
        monkeypatch.delenv("HERMES_OPENROUTER_CACHE_TTL", raising=False)
        headers = build_or_headers(or_config={"response_cache": True, "response_cache_ttl": 600})
        assert headers["X-OpenRouter-Cache"] == "true"
        assert headers["X-OpenRouter-Cache-TTL"] == "600"

class TestDefaultConfig:
    """Verify the openrouter config section is in DEFAULT_CONFIG."""

    def test_openrouter_section_exists(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert "openrouter" in DEFAULT_CONFIG
        or_cfg = DEFAULT_CONFIG["openrouter"]
        assert or_cfg["response_cache"] is True
        assert or_cfg["response_cache_ttl"] == 300


# ---------------------------------------------------------------------------
# _check_openrouter_cache_status
# ---------------------------------------------------------------------------

class TestCheckOpenrouterCacheStatus:
    """Test the _check_openrouter_cache_status method on AIAgent."""

    def _make_agent(self):
        """Create a minimal AIAgent-like object with just the method under test."""
        from run_agent import AIAgent

        # Use object.__new__ to skip __init__, then set the attributes we need
        agent = object.__new__(AIAgent)
        agent._or_cache_hits = 0
        return agent

    def test_hit_increments_counter(self):
        agent = self._make_agent()
        resp = SimpleNamespace(headers={"x-openrouter-cache-status": "HIT"})
        agent._check_openrouter_cache_status(resp)
        assert agent._or_cache_hits == 1
        # Second hit increments
        agent._check_openrouter_cache_status(resp)
        assert agent._or_cache_hits == 2

    def test_miss_does_not_increment(self):
        agent = self._make_agent()
        resp = SimpleNamespace(headers={"x-openrouter-cache-status": "MISS"})
        agent._check_openrouter_cache_status(resp)
        assert getattr(agent, "_or_cache_hits", 0) == 0

    def test_no_header_is_noop(self):
        agent = self._make_agent()
        resp = SimpleNamespace(headers={})
        agent._check_openrouter_cache_status(resp)
        assert getattr(agent, "_or_cache_hits", 0) == 0

    def test_none_response_is_safe(self):
        agent = self._make_agent()
        agent._check_openrouter_cache_status(None)  # no crash

    def test_no_headers_attr_is_safe(self):
        agent = self._make_agent()
        agent._check_openrouter_cache_status(object())  # no crash

    def test_case_insensitive(self):
        agent = self._make_agent()
        resp = SimpleNamespace(headers={"x-openrouter-cache-status": "hit"})
        agent._check_openrouter_cache_status(resp)
        assert agent._or_cache_hits == 1
