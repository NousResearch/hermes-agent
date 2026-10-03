"""Auxiliary clients must apply ``providers.<name>.extra_headers`` on the resolved route.

Companion to ``tests/hermes_cli/test_custom_provider_extra_headers.py`` (pure matching
helpers) and ``tests/agent/test_auxiliary_user_default_headers.py`` (model-level headers).
The main agent client merges per-provider ``extra_headers`` in ``agent/agent_init.py`` and
``agent/client_lifecycle.py``; auxiliary calls (title generation, context compression,
vision, approvals) build separate OpenAI clients through
``agent.auxiliary_client._create_openai_client`` and silently dropped those route headers —
breaking zero-data-retention enforcement, WAF/gateway auth and similar per-route header
config on every auxiliary request.
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


_PROVIDERS = [
    {
        "name": "gw",
        "base_url": "http://gw.local/v1",
        "api_key": "k",
        "extra_headers": {"X-ZDR": "1", "X-Route": "my-route"},
    },
]


class TestAuxClientAppliesCustomProviderExtraHeaders:
    def test_client_creation_merges_matching_provider_headers(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "m"}, "custom_providers": _PROVIDERS})
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _create_openai_client
            _create_openai_client(api_key="k", base_url="http://gw.local/v1")

        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {})
        assert headers.get("X-ZDR") == "1"
        assert headers.get("X-Route") == "my-route"

    def test_merge_preserves_existing_default_headers(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "m"}, "custom_providers": _PROVIDERS})
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _create_openai_client
            _create_openai_client(
                api_key="k",
                base_url="http://gw.local/v1",
                default_headers={"User-Agent": "curl/8.7.1"},
            )

        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {})
        assert headers.get("User-Agent") == "curl/8.7.1"  # untouched defaults preserved
        assert headers.get("X-ZDR") == "1"                 # provider header added

    def test_no_matching_provider_adds_nothing(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "m"}, "custom_providers": _PROVIDERS})
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _create_openai_client
            _create_openai_client(api_key="k", base_url="http://other.example.com/v1")

        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert "X-ZDR" not in headers

    def test_named_custom_provider_resolution_applies_headers(self, tmp_path):
        """End-to-end: resolve_provider_client funnels through the fixed choke point."""
        _write_config(tmp_path, {"model": {"default": "test-model"}, "custom_providers": _PROVIDERS})
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("gw", "my-model")

        assert client is not None
        assert model == "my-model"
        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("X-ZDR") == "1"
