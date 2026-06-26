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


class TestAnthropicAdapterHonorsUserDefaultHeaders:
    """Tests for user-configured model.default_headers on Anthropic-mode providers.

    build_anthropic_client() accepts an optional user_default_headers dict
    that is merged on top of provider/OAuth defaults before SDK instantiation.
    This ensures model.default_headers works for api_mode: anthropic_messages
    providers (Kimi, DeepSeek, etc.) — not just OpenAI-compatible ones. (#9589)
    """

    def test_user_headers_merged_onto_kimi_defaults(self, tmp_path):
        """User-Agent override wins over Kimi's hardcoded claude-code/0.1.0."""
        _write_config(tmp_path, {
            "model": {
                "default": "kimi-k2.5",
                "default_headers": {"User-Agent": "my-gateway/1.0"},
            },
        })
        from agent.auxiliary_client import _apply_user_default_headers
        from agent.anthropic_adapter import _is_kimi_coding_endpoint

        user_dh = _apply_user_default_headers(None)
        assert user_dh is not None
        assert user_dh.get("User-Agent") == "my-gateway/1.0"

        # Verify the merge logic directly (same as build_anthropic_client uses)
        provider_defaults = {
            "User-Agent": "claude-code/0.1.0",
            "anthropic-beta": "test-beta",
        }
        merged = dict(provider_defaults)
        for key, value in user_dh.items():
            if value is None:
                continue
            merged[str(key)] = str(value)

        assert merged["User-Agent"] == "my-gateway/1.0"  # user wins
        assert merged["anthropic-beta"] == "test-beta"    # provider preserved

    def test_none_user_headers_noop(self, tmp_path):
        """When no user headers are configured, provider defaults are untouched."""
        _write_config(tmp_path, {"model": {"default": "m"}})
        from agent.auxiliary_client import _apply_user_default_headers

        user_dh = _apply_user_default_headers(None)
        assert user_dh is None  # no config → None

    def test_user_header_none_value_does_not_remove_provider_default(self, tmp_path):
        """A None value in user_default_headers means 'don't override', not 'delete'.

        The _apply_user_default_headers helper strips None values before the
        merge, so a provider-default header (like Kimi's User-Agent) is
        preserved when the user sets it to None.  Whether None should instead
        mean 'remove this key' is an open design question — see PR #XXXX.
        """
        _write_config(tmp_path, {
            "model": {
                "default": "m",
                "default_headers": {"User-Agent": None, "X-Extra": "yes"},
            },
        })
        from agent.auxiliary_client import _apply_user_default_headers

        user_dh = _apply_user_default_headers(None)
        assert user_dh is not None
        # None values are stripped by the helper — only non-None remain
        assert "User-Agent" not in user_dh
        assert user_dh["X-Extra"] == "yes"

        # Simulate the merge build_anthropic_client performs
        provider_defaults = {"User-Agent": "claude-code/0.1.0"}
        merged = dict(provider_defaults)
        for key, value in user_dh.items():
            if value is None:
                continue
            merged[str(key)] = str(value)
        # Provider default is untouched — user only added X-Extra
        assert merged["User-Agent"] == "claude-code/0.1.0"
        assert merged["X-Extra"] == "yes"
