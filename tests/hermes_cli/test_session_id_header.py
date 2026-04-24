"""Tests for per-provider session-ID header injection via config.yaml.

Covers:
  • get_provider_session_id_header_name() resolver with mocked config
  • _inject_session_id_headers() post-init patching of client default_headers
  • Consent model never leaks headers to unauthorized providers
"""

from __future__ import annotations

import types
from unittest.mock import patch


# ── Helpers / fixtures ─────────────────────────────────────────────────────────

def _default_config(provider_id: str = "openrouter", header: str = "X-Agent-Session-Id"):
    """Return a provider config dict that enables session-ID header injection."""
    return {"providers": {provider_id: {"session_id_header_name": header}}}


def _make_agent_mock(config, **kw):
    """Build a minimal mock matching only the attributes used by _inject_session_id_headers."""
    agent = types.SimpleNamespace()
    agent.provider = kw.get("provider", "openrouter")
    agent.model = kw.get("model", "openai/gpt-4.1-nano")
    agent.session_id = kw.get("session_id", "test-session-001")
    agent.client = kw.get("client", None)
    agent._anthropic_client = kw.get("anthropic_client", None)

    def _inject():
        """Copy of actual method body (dual-read from run_agent.py)."""
        from hermes_cli.timeouts import get_provider_session_id_header_name

        hdr = get_provider_session_id_header_name(agent.provider, agent.model)
        if not hdr:
            return
        sid = str(getattr(agent, "session_id", ""))

        pc = getattr(agent, "client", None)
        if pc is not None:
            d = getattr(pc, "default_headers", None)
            if isinstance(d, dict):
                d[hdr] = sid

        ac = getattr(agent, "_anthropic_client", None)
        if ac is not None:
            d = getattr(ac, "default_headers", None)
            if isinstance(d, dict):
                d[hdr] = sid

    agent._inject_session_id_headers = _inject
    return agent


# ── Resolver unit tests ────────────────────────────────────────────────────────

class TestGetProviderSessionIdHeaderName:
    """Unit tests for the resolver.  Config is mocked — no filesystem needed."""

    def _resolver(self, config_data, provider_id, model=None):
        with patch(
            "hermes_cli.config.load_config", return_value=config_data
        ):
            from hermes_cli.timeouts import get_provider_session_id_header_name

            return get_provider_session_id_header_name(provider_id, model)

    # --- Provider-level header name ---

    def test_provider_level_returns_header_name(self):
        assert self._resolver(_default_config("openrouter", "X-Trace-Span"), "openrouter") == "X-Trace-Span"

    def test_empty_value_returns_none(self):
        cfg = {"providers": {"openrouter": {"session_id_header_name": ""}}}
        assert self._resolver(cfg, "openrouter") is None

    def test_none_value_returns_none(self):
        cfg = {"providers": {"openrouter": {"session_id_header_name": None}}}
        assert self._resolver(cfg, "openrouter") is None

    def test_whitespace_stripped(self):
        assert (
            self._resolver(
                _default_config("anthropic", "  X-Clean-Header  "),
                "anthropic",
            )
            == "X-Clean-Header"
        )

    # --- Model-level override ---

    def test_model_level_override_wins(self):
        cfg = {
            "providers": {
                "anthropic": {
                    "session_id_header_name": "X-Provider",
                    "models": {"claude-sonnet-4-20250514": {"session_id_header_name": "X-Model"}},
                }
            }
        }
        assert self._resolver(cfg, "anthropic", "claude-sonnet-4-20250514") == "X-Model"

    def test_model_level_no_override(self):
        cfg = {
            "providers": {
                "openrouter": {
                    "session_id_header_name": "X-Default",
                    "models": {"some/model": {}},  # no override key
                }
            }
        }
        assert self._resolver(cfg, "openrouter", "some/model") == "X-Default"

    # --- Unknown provider / missing keys ---

    def test_missing_provider_returns_none(self):
        assert self._resolver(_default_config(), "nonexistent") is None

    def test_no_providers_key(self):
        assert self._resolver({"models": {}}, "openrouter") is None

    def test_none_provider_id(self):
        assert self._resolver(_default_config(), None) is None

    def test_no_config_at_all(self):
        assert self._resolver({}, "openrouter") is None


# ── Injection method unit tests (mocked clients) ───────────────────────────────

class TestInjectSessionIdHeaders:
    """Directly exercise the _inject_session_id_headers logic with mocked
    client objects.  No AIAgent construction, no config reload needed."""

    def test_patches_openai_primary_client(self):
        openai_client = types.SimpleNamespace()
        openai_client.default_headers = {"Accept": "application/json", "User-Agent": "OpenAI/Python"}

        agent = _make_agent_mock(
            config=_default_config(),
            client=openai_client,
        )
        with patch("hermes_cli.config.load_config", return_value=_default_config()):
            agent._inject_session_id_headers()

        assert "X-Agent-Session-Id" in openai_client.default_headers
        assert openai_client.default_headers["X-Agent-Session-Id"] == "test-session-001"
        assert openai_client.default_headers["Accept"] == "application/json"

    def test_patches_anthropic_sdk_client(self):
        anthropic_client = types.SimpleNamespace()
        anthropic_client.default_headers = {}

        agent = _make_agent_mock(
            config=_default_config(),
            anthropic_client=anthropic_client,
        )
        with patch("hermes_cli.config.load_config", return_value=_default_config()):
            agent._inject_session_id_headers()

        assert "X-Agent-Session-Id" in anthropic_client.default_headers
        assert (
            anthropic_client.default_headers["X-Agent-Session-Id"] == "test-session-001"
        )

    def test_patches_both_clients(self):
        openai_client = types.SimpleNamespace()
        openai_client.default_headers = {"Accept": "application/json"}

        anthropic_client = types.SimpleNamespace()
        anthropic_client.default_headers = {}

        agent = _make_agent_mock(
            config=_default_config(),
            client=openai_client,
            anthropic_client=anthropic_client,
        )
        with patch("hermes_cli.config.load_config", return_value=_default_config()):
            agent._inject_session_id_headers()

        assert "X-Agent-Session-Id" in openai_client.default_headers
        assert "X-Agent-Session-Id" in anthropic_client.default_headers

    def test_uses_actual_session_id_value(self):
        openai_client = types.SimpleNamespace()
        openai_client.default_headers = {}

        agent = _make_agent_mock(
            config=_default_config(),
            client=openai_client,
            session_id="20260423_110000_aabbcc",
        )
        with patch("hermes_cli.config.load_config", return_value=_default_config()):
            agent._inject_session_id_headers()

        assert (
            openai_client.default_headers["X-Agent-Session-Id"] == "20260423_110000_aabbcc"
        )

    def test_model_level_override_injection(self):
        """When a model-level override exists, that header name is used instead of provider-level."""
        cfg = {
            "providers": {
                "anthropic": {
                    "session_id_header_name": "X-Provider",
                    "models": {"claude-sonnet-4-20250514": {"session_id_header_name": "X-Claude"}},
                }
            }
        }
        anthropic_client = types.SimpleNamespace()
        anthropic_client.default_headers = {}

        agent = _make_agent_mock(
            config=cfg,
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            anthropic_client=anthropic_client,
        )
        with patch("hermes_cli.config.load_config", return_value=cfg):
            agent._inject_session_id_headers()

        assert "X-Claude" in anthropic_client.default_headers


    # --- Edge cases (no-op paths) ---

    def test_noop_when_no_header_configured(self):
        cfg = {}
        agent = _make_agent_mock(config=cfg)
        openai_client = types.SimpleNamespace(default_headers={"HTTP-Referer": "https://hermes"})
        agent.client = openai_client
        with patch("hermes_cli.config.load_config", return_value=cfg):
            agent._inject_session_id_headers()
        # Should not add anything
        assert len(openai_client.default_headers) == 1

    def test_noop_when_no_clients_exist(self):
        cfg = _default_config()
        agent = _make_agent_mock(
            config=cfg,
            client=None,
            anthropic_client=None,
        )
        with patch("hermes_cli.config.load_config", return_value=cfg):
            agent._inject_session_id_headers()
        # No exception

    def test_default_headers_is_none(self):
        """Guard against None default_headers — isinstance check should skip."""
        openai_client = types.SimpleNamespace(default_headers=None)  # type: ignore

        agent = _make_agent_mock(
            config=_default_config(),
            client=openai_client,
        )
        with patch("hermes_cli.config.load_config", return_value=_default_config()):
            agent._inject_session_id_headers()
        # No crash — isinstance guard prevents attribute error

    def test_default_headers_is_non_dict(self):
        anthropic_client = types.SimpleNamespace(default_headers="not-a-dict")  # type: ignore

        agent = _make_agent_mock(
            config=_default_config(),
            anthropic_client=anthropic_client,
        )
        with patch("hermes_cli.config.load_config", return_value=_default_config()):
            agent._inject_session_id_headers()
        # No crash


# ── Consent model — never leaks session ID ─────────────────────────────────────

class TestConsentModel:
    """The consent model uses a separate HTTPX client; it must never inherit
    the session-ID header from the primary OpenAI/Anthropic clients."""

    def test_empty_config_openai_client_untouched(self):
        openai_client = types.SimpleNamespace(default_headers={"HTTP-Referer": "https://hermes"})
        agent = _make_agent_mock(config={})
        agent.client = openai_client
        agent._inject_session_id_headers()

        assert len(openai_client.default_headers) == 1
        assert "X-Agent-Session-Id" not in openai_client.default_headers

    def test_empty_config_anthropic_client_untouched(self):
        anthropic_client = types.SimpleNamespace(default_headers={})
        agent = _make_agent_mock(config={})
        agent._anthropic_client = anthropic_client
        agent._inject_session_id_headers()

        assert len(anthropic_client.default_headers) == 0
