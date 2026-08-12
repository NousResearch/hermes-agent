"""Tests for OpenRouter response caching header injection."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# build_or_headers
# ---------------------------------------------------------------------------

class TestBuildOrHeaders:
    """Test the build_or_headers() helper in agent/auxiliary_client.py."""





    def test_ttl_default(self):
        """Default TTL (300) is included when cache is enabled."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": True, "response_cache_ttl": 300})
        assert headers["X-OpenRouter-Cache-TTL"] == "300"





    def test_ttl_negative(self):
        """Negative TTL is ignored."""
        from agent.auxiliary_client import build_or_headers

        headers = build_or_headers(or_config={"response_cache": True, "response_cache_ttl": -5})
        assert "X-OpenRouter-Cache-TTL" not in headers



    def test_returns_fresh_dict(self):
        """Each call returns a new dict so mutations don't leak."""
        from agent.auxiliary_client import build_or_headers

        cfg = {"response_cache": True}
        h1 = build_or_headers(or_config=cfg)
        h2 = build_or_headers(or_config=cfg)
        assert h1 is not h2
        assert h1 == h2


    def test_none_config_load_config_fails_gracefully(self):
        """When load_config() fails, build_or_headers still returns base headers."""
        from agent.auxiliary_client import build_or_headers

        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")), patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("boom")):
            headers = build_or_headers(or_config=None)
        # Should have base attribution but no cache headers
        assert "HTTP-Referer" in headers
        assert "X-OpenRouter-Cache" not in headers


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

class TestEnvVarOverrides:
    """Test env var precedence over config.yaml for response caching."""






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



    def test_no_header_is_noop(self):
        agent = self._make_agent()
        resp = SimpleNamespace(headers={})
        agent._check_openrouter_cache_status(resp)
        assert getattr(agent, "_or_cache_hits", 0) == 0



    def test_case_insensitive(self):
        agent = self._make_agent()
        resp = SimpleNamespace(headers={"x-openrouter-cache-status": "hit"})
        agent._check_openrouter_cache_status(resp)
        assert agent._or_cache_hits == 1
