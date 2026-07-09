"""Tests for user-configured ``model.default_headers`` in the auxiliary client.

Companion to ``tests/run_agent/test_provider_attribution_headers.py`` (which
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

    def test_no_config_is_noop_returns_original(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "m"}})
        from agent.auxiliary_client import _apply_user_default_headers
        original = {"User-Agent": "OpenAI/Python"}
        merged = _apply_user_default_headers(original)
        assert merged == original

    def test_none_headers_with_config_creates_dict(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"User-Agent": "curl/8.7.1"}},
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers(None)
        assert merged == {"User-Agent": "curl/8.7.1"}

    def test_none_headers_no_config_returns_none(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "m"}})
        from agent.auxiliary_client import _apply_user_default_headers
        assert _apply_user_default_headers(None) is None

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


class TestAuxClientHonorsPerProviderExtraHeaders:
    """A named ``custom_providers``/``providers`` entry's own ``extra_headers``
    (e.g. gateway auth like ``x-gateway-auth``) must reach the auxiliary
    client, not just the main agent client. Regression for the auxiliary
    named-custom-provider path silently dropping ``custom_entry["extra_headers"]``.
    """

    def test_entry_extra_headers_applied_to_aux_client(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"x-gateway-auth": "token"},
                },
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("my-gw", "test-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("x-gateway-auth") == "token"

    def test_entry_extra_headers_win_over_top_level_default_headers(self, tmp_path):
        """Precedence must match the main client: per-provider extra_headers
        (applied last by ``apply_custom_provider_extra_headers_to_client_kwargs``
        in agent_init.py/run_agent.py) win over ``model.default_headers``.
        """
        _write_config(tmp_path, {
            "model": {
                "default": "test-model",
                "default_headers": {"User-Agent": "curl/8.7.1", "x-gateway-auth": "should-lose"},
            },
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"x-gateway-auth": "token"},
                },
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("my-gw", "test-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        # Non-conflicting top-level header still present...
        assert headers.get("User-Agent") == "curl/8.7.1"
        # ...but the per-provider entry wins on the overlapping key.
        assert headers.get("x-gateway-auth") == "token"

    def test_entry_extra_headers_applied_on_anthropic_messages_sdk_missing_fallback(self, tmp_path):
        """When api_mode=anthropic_messages but the anthropic SDK import/call
        fails, the code falls back to an OpenAI-wire client (see
        agent/auxiliary_client.py around the ImportError handler). That
        fallback path builds its own headers dict and must also honor the
        entry's extra_headers.
        """
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/anthropic",
                    "api_key": "k",
                    "api_mode": "anthropic_messages",
                    "extra_headers": {"x-gateway-auth": "token"},
                },
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai, patch(
            "agent.anthropic_adapter.build_anthropic_client",
            side_effect=ImportError("anthropic package not installed"),
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("my-gw", "test-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("x-gateway-auth") == "token"
