"""Tests for per-provider ``custom_providers[].extra_headers`` in the auxiliary client.

The main agent client merges ``custom_providers[].extra_headers`` into
``default_headers`` by matching ``base_url`` (see
``hermes_cli.config.apply_custom_provider_extra_headers_to_client_kwargs``).
The auxiliary client (title generation, compression, vision, web_extract)
builds separate OpenAI clients and must apply the same merge — otherwise an
OpenAI-compatible gateway/WAF that filters on the SDK's identifying
``User-Agent`` lets the main turn through while every auxiliary call to the
same endpoint gets rejected (403). Per-provider headers are the most specific
configuration level: they override both provider/SDK defaults and the global
``model.default_headers`` merge.
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


class TestProviderExtraHeadersHelper:
    """Direct unit tests for the base_url-aware merge helper."""

    def test_provider_extra_headers_applied_on_base_url_match(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m"},
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers({}, base_url="http://my-gw.local/v1")
        assert merged["User-Agent"] == "curl/8.7.1"

    def test_no_matching_provider_injects_nothing(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m"},
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers(None, base_url="http://other.local/v1")
        assert merged is None

    def test_provider_extra_headers_beat_global_default_headers(self, tmp_path):
        """Per-provider extra_headers are the most specific level and win."""
        _write_config(tmp_path, {
            "model": {
                "default": "m",
                "default_headers": {"User-Agent": "global-ua/1.0", "X-Global": "g"},
            },
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers({}, base_url="http://my-gw.local/v1")
        assert merged["User-Agent"] == "curl/8.7.1"  # provider wins over global
        assert merged["X-Global"] == "g"  # unrelated global keys still merge

    def test_default_base_url_none_preserves_legacy_behavior(self, tmp_path):
        """Omitting base_url must behave exactly as before (global merge only)."""
        _write_config(tmp_path, {
            "model": {
                "default": "m",
                "default_headers": {"User-Agent": "global-ua/1.0"},
            },
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers({})
        assert merged == {"User-Agent": "global-ua/1.0"}

    def test_lookup_exception_falls_back_to_global_headers(self, tmp_path):
        """Lookup exception must be swallowed, defaulting provider_headers to {} while keeping global headers."""
        _write_config(tmp_path, {
            "model": {
                "default": "m",
                "default_headers": {"User-Agent": "global-ua/1.0", "X-Global": "g"},
            },
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        from agent.auxiliary_client import _apply_user_default_headers
        with patch("hermes_cli.config.get_custom_provider_extra_headers", side_effect=RuntimeError("lookup failure")):
            merged = _apply_user_default_headers({}, base_url="http://my-gw.local/v1")
        assert merged is not None
        assert merged["User-Agent"] == "global-ua/1.0"
        assert merged["X-Global"] == "g"

    def test_non_dict_return_falls_back_to_global_headers(self, tmp_path):
        """Non-dict return from lookup must fall back to {} while keeping global headers."""
        _write_config(tmp_path, {
            "model": {
                "default": "m",
                "default_headers": {"User-Agent": "global-ua/1.0", "X-Global": "g"},
            },
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        from agent.auxiliary_client import _apply_user_default_headers
        with patch("hermes_cli.config.get_custom_provider_extra_headers", return_value="not-a-dict"):
            merged = _apply_user_default_headers({}, base_url="http://my-gw.local/v1")
        assert merged is not None
        assert merged["User-Agent"] == "global-ua/1.0"
        assert merged["X-Global"] == "g"


class TestAuxClientHonorsProviderExtraHeaders:
    """Integration: resolve_provider_client must pass per-provider headers to OpenAI."""

    def test_anonymous_custom_provider_gets_extra_headers(self, tmp_path):
        """Anonymous ``model.provider: custom`` branch (base_url match)."""
        _write_config(tmp_path, {
            "model": {
                "default": "my-custom-model",
                "provider": "custom",
                "base_url": "http://my-gw.local/v1",
            },
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("main", "my-custom-model")

        assert client is not None
        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("User-Agent") == "curl/8.7.1"

    def test_named_custom_provider_gets_extra_headers(self, tmp_path):
        """Named custom provider branch (L6653 path)."""
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("my-gw", "test-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("User-Agent") == "curl/8.7.1"

    def test_anthropic_fallback_branch_gets_extra_headers(self, tmp_path):
        """anthropic_messages with SDK missing falls back to OpenAI-wire (L6678)."""
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "api_mode": "anthropic_messages",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai, \
             patch.dict("sys.modules", {"agent.anthropic_adapter": None}):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("my-gw", "test-model")

        assert client is not None
        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("User-Agent") == "curl/8.7.1"

    def test_no_matching_provider_sends_no_user_agent(self, tmp_path):
        """A custom_providers entry for a DIFFERENT base_url must not leak in."""
        _write_config(tmp_path, {
            "model": {
                "default": "my-custom-model",
                "provider": "custom",
                "base_url": "http://unrelated.local/v1",
            },
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("main", "my-custom-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert "User-Agent" not in headers

    @pytest.mark.asyncio
    async def test_async_path_gets_extra_headers(self, tmp_path):
        """The sync->async conversion (L6174) must carry provider headers too."""
        _write_config(tmp_path, {
            "model": {
                "default": "my-custom-model",
                "provider": "custom",
                "base_url": "http://my-gw.local/v1",
            },
            "custom_providers": [
                {
                    "name": "my-gw",
                    "base_url": "http://my-gw.local/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        sync_instance = MagicMock()
        sync_instance.api_key = "k"
        sync_instance.base_url = "http://my-gw.local/v1"
        captured_async_kwargs = {}
        real_async_openai = None
        with patch("agent.auxiliary_client.OpenAI", return_value=sync_instance):
            import openai as _openai_mod
            real_async_openai = _openai_mod.AsyncOpenAI

            def _capture_async(**kwargs):
                captured_async_kwargs.update(kwargs)
                return MagicMock()

            _openai_mod.AsyncOpenAI = _capture_async
            try:
                from agent.auxiliary_client import resolve_provider_client
                client, model = resolve_provider_client(
                    "main", "my-custom-model", async_mode=True
                )
            finally:
                _openai_mod.AsyncOpenAI = real_async_openai

        assert client is not None
        assert captured_async_kwargs, "AsyncOpenAI constructor was not called"
        headers = captured_async_kwargs.get("default_headers", {}) or {}
        assert headers.get("User-Agent") == "curl/8.7.1"
