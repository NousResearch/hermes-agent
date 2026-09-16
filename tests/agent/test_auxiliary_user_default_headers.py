"""Tests for user-configured ``model.default_headers`` in the auxiliary client.

Companion to ``tests/agent/test_provider_attribution_headers.py`` (which
covers the main agent client). The main agent turn and the auxiliary client
(title generation, context compression, vision routing) build separate OpenAI
clients, so a ``custom`` endpoint behind a gateway/WAF that rejects the OpenAI
SDK's identifying headers needs the ``model.default_headers`` override applied
on BOTH paths — otherwise the main turn succeeds but auxiliary calls to the
same endpoint still fail with an opaque 4xx/502. (#40033)
"""

from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so load_config() reads our test config.yaml."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    import yaml
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.dump(config_dict))


class TestApplyUserDefaultHeadersHelper:
    """Direct unit tests for the merge helper."""

    def test_user_headers_merged_and_win(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"User-Agent": "curl/8.7.1", "X-Extra": "1"}},
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers({"User-Agent": "OpenAI/Python 2.24.0"})
        assert merged["User-Agent"] == "curl/8.7.1"  # user wins
        assert merged["X-Extra"] == "1"




    def test_none_values_skipped(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"User-Agent": "curl/8.7.1", "X-Drop": None}},
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers({})
        assert merged == {"User-Agent": "curl/8.7.1"}
        assert "X-Drop" not in merged


class TestAuxClientHonorsUserDefaultHeaders:
    """Integration: resolve_provider_client must pass overridden headers to OpenAI."""

    def test_custom_provider_overrides_sdk_user_agent(self, tmp_path):
        """The #40033 reproduction on the auxiliary path."""
        _write_config(tmp_path, {
            "model": {
                "default": "my-custom-model",
                "provider": "custom",
                "base_url": "http://localhost:8080/v1",
                "default_headers": {"User-Agent": "curl/8.7.1", "X-Extra": "1"},
            },
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("main", "my-custom-model")

        assert client is not None
        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {})
        assert headers.get("User-Agent") == "curl/8.7.1"
        assert headers.get("X-Extra") == "1"

    def test_custom_provider_no_override_sends_no_user_agent(self, tmp_path):
        """Without config, the aux client injects nothing — SDK defaults apply."""
        _write_config(tmp_path, {
            "model": {
                "default": "my-custom-model",
                "provider": "custom",
                "base_url": "http://localhost:8080/v1",
            },
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("main", "my-custom-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert "User-Agent" not in headers

    def test_named_custom_provider_honors_override(self, tmp_path):
        """A `custom_providers:` entry's aux calls also honor model.default_headers.

        This is a distinct construction path (_extra2) from the config-level
        `model.provider: custom` path — both must apply the global override.
        """
        _write_config(tmp_path, {
            "model": {
                "default": "test-model",
                "default_headers": {"User-Agent": "curl/8.7.1"},
            },
            "custom_providers": [
                {"name": "my-gw", "base_url": "http://my-gw.local/v1", "api_key": "k"},
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("my-gw", "test-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("User-Agent") == "curl/8.7.1"


class TestPerProviderExtraHeadersReachAuxPath:
    """A provider entry's OWN ``extra_headers`` must apply to auxiliary calls.

    The main agent client already reads per-provider ``extra_headers``. The
    auxiliary client historically read only the global ``model.default_headers``
    / ``model.extra_headers``, so a provider that REQUIRES a custom client
    header (AgentRouter enforces ``User-Agent: claude-cli/2.0.0``; Kimi,
    Copilot and NVIDIA NIM each needed a hardcoded branch for the same reason)
    worked for main turns but failed every auxiliary call with
    ``unauthorized client detected``. The aux path then misreported that as a
    payment error and marked the provider unhealthy for 600s.

    Reading the provider entry — rather than adding another hardcoded branch —
    keeps the fix SCOPED: only a provider that declares a header gets one.
    """

    _AR = {
        "name": "areg",
        "base_url": "https://agentrouter.org/v1",
        "key_env": "AGENTROUTER_API_KEY",
        "extra_headers": {"User-Agent": "claude-cli/2.0.0 (external, cli)"},
    }

    def test_provider_extra_headers_applied_for_matching_base_url(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "m"}, "custom_providers": [self._AR]})
        from agent.auxiliary_client import _apply_user_default_headers
        assert _apply_user_default_headers(None, "https://agentrouter.org/v1") == {
            "User-Agent": "claude-cli/2.0.0 (external, cli)"
        }

    def test_provider_headers_not_applied_to_other_base_urls(self, tmp_path):
        """The scoping invariant: declaring a header for one provider must not
        leak it onto every other provider."""
        _write_config(tmp_path, {"model": {"default": "m"}, "custom_providers": [self._AR]})
        from agent.auxiliary_client import _apply_user_default_headers
        assert _apply_user_default_headers(None, "https://api.deepseek.com/v1") is None
        assert _apply_user_default_headers(None, "https://openrouter.ai/api/v1") is None

    def test_explicit_model_headers_still_win_over_provider_headers(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"User-Agent": "curl/8.7.1"}},
            "custom_providers": [self._AR],
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers(None, "https://agentrouter.org/v1")
        assert merged is not None
        assert merged["User-Agent"] == "curl/8.7.1"

    def test_no_base_url_argument_preserves_previous_behaviour(self, tmp_path):
        """Callers that pass no base_url keep the old global-only semantics."""
        _write_config(tmp_path, {"model": {"default": "m"}, "custom_providers": [self._AR]})
        from agent.auxiliary_client import _apply_user_default_headers
        assert _apply_user_default_headers(None) is None
        assert _apply_user_default_headers({"X-Keep": "1"}) == {"X-Keep": "1"}

    def test_provider_and_user_headers_merge(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"X-User": "u"}},
            "custom_providers": [self._AR],
        })
        from agent.auxiliary_client import _apply_user_default_headers
        assert _apply_user_default_headers({"X-Existing": "e"}, "https://agentrouter.org/v1") == {
            "X-Existing": "e",
            "User-Agent": "claude-cli/2.0.0 (external, cli)",
            "X-User": "u",
        }
