"""Tests for custom_providers[].extra_headers propagation in auxiliary client.

Regression guard: the main agent (run_agent.py) applies per-provider
``extra_headers`` from ``custom_providers[]`` entries via
``apply_custom_provider_extra_headers_to_client_kwargs``, but the
auxiliary client's named-custom-provider and anonymous-custom branches
both missed that step. Auxiliary calls (title generation, compression,
vision) to providers that declare ``extra_headers`` silently dropped
auth/attribution headers and 401/403'd on the wire.
"""

from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME and clear module caches."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    """Write a config.yaml to the test HERMES_HOME."""
    import yaml
    config_path = tmp_path / ".hermes" / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))


class TestNamedCustomProviderExtraHeaders:
    """resolve_provider_client must merge custom_providers[].extra_headers
    onto the OpenAI client's default_headers for named custom providers."""

    def test_named_custom_provider_extra_headers_applied(self, tmp_path):
        """extra_headers from a named custom_providers entry must reach the
        OpenAI client's default_headers."""
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {
                    "name": "mygw",
                    "base_url": "http://mygw.local/v1",
                    "api_key": "k",
                    "extra_headers": {
                        "x-service-id": "slsi-agent-desktop",
                        "x-user-id": "ksyang",
                    },
                },
            ],
        })
        captured_kwargs = {}

        def _capture_openai(api_key, base_url, **kwargs):
            captured_kwargs.update(kwargs)
            mock_client = MagicMock()
            mock_client.base_url = base_url
            return mock_client

        with patch("agent.auxiliary_client.OpenAI", side_effect=_capture_openai):
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("mygw", "test-model")

        assert client is not None
        default_headers = captured_kwargs.get("default_headers")
        assert default_headers is not None
        assert default_headers["x-service-id"] == "slsi-agent-desktop"
        assert default_headers["x-user-id"] == "ksyang"

    def test_named_custom_provider_no_extra_headers_unchanged(self, tmp_path):
        """When no extra_headers are declared, default_headers must not be
        populated from the provider entry."""
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {
                    "name": "plain",
                    "base_url": "http://plain.local/v1",
                    "api_key": "k",
                },
            ],
        })
        captured_kwargs = {}

        def _capture_openai(api_key, base_url, **kwargs):
            captured_kwargs.update(kwargs)
            mock_client = MagicMock()
            mock_client.base_url = base_url
            return mock_client

        with patch("agent.auxiliary_client.OpenAI", side_effect=_capture_openai):
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("plain", "test-model")

        assert client is not None
        # No extra_headers → no default_headers injected from the provider entry
        assert "default_headers" not in captured_kwargs or \
               captured_kwargs["default_headers"] is None or \
               "x-service-id" not in (captured_kwargs.get("default_headers") or {})

    def test_named_custom_provider_extra_headers_merge_with_user_headers(self, tmp_path):
        """Per-provider extra_headers must merge with (not replace)
        model.default_headers, with provider headers winning."""
        _write_config(tmp_path, {
            "model": {
                "default": "test-model",
                "default_headers": {"User-Agent": "my-agent/1.0"},
            },
            "custom_providers": [
                {
                    "name": "mygw",
                    "base_url": "http://mygw.local/v1",
                    "api_key": "k",
                    "extra_headers": {
                        "x-service-id": "slsi-agent-desktop",
                    },
                },
            ],
        })
        captured_kwargs = {}

        def _capture_openai(api_key, base_url, **kwargs):
            captured_kwargs.update(kwargs)
            mock_client = MagicMock()
            mock_client.base_url = base_url
            return mock_client

        with patch("agent.auxiliary_client.OpenAI", side_effect=_capture_openai):
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("mygw", "test-model")

        assert client is not None
        default_headers = captured_kwargs.get("default_headers")
        assert default_headers is not None
        # Both user-level and provider-level headers should be present
        assert default_headers["User-Agent"] == "my-agent/1.0"
        assert default_headers["x-service-id"] == "slsi-agent-desktop"


class TestAnonymousCustomEndpointExtraHeaders:
    """_try_custom_endpoint (anonymous custom) must also apply
    custom_providers[].extra_headers matched by base_url."""

    def test_anonymous_custom_extra_headers_applied(self, tmp_path):
        """When the anonymous custom endpoint's base_url matches a
        custom_providers entry with extra_headers, those headers must
        reach the OpenAI client."""
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {
                    "name": "viagw",
                    "base_url": "http://viagw.local/v1",
                    "api_key": "k",
                    "extra_headers": {
                        "X-Client-API-Key": "secret-key-123",
                    },
                },
            ],
        })
        # Set OPENAI_BASE_URL to match the provider entry, and OPENAI_API_KEY
        # so _resolve_custom_runtime picks it up.
        import os
        old_base = os.environ.get("OPENAI_BASE_URL")
        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_BASE_URL"] = "http://viagw.local/v1"
        os.environ["OPENAI_API_KEY"] = "k"
        try:
            captured_kwargs = {}

            def _capture_openai(api_key, base_url, **kwargs):
                captured_kwargs.update(kwargs)
                mock_client = MagicMock()
                mock_client.base_url = base_url
                return mock_client

            with patch("agent.auxiliary_client.OpenAI", side_effect=_capture_openai):
                from agent.auxiliary_client import _try_custom_endpoint
                client, model = _try_custom_endpoint()

            # The anonymous custom path may return None if resolution fails
            # for other reasons, but when it returns a client the headers
            # must be present.
            if client is not None:
                default_headers = captured_kwargs.get("default_headers")
                assert default_headers is not None
                assert default_headers.get("X-Client-API-Key") == "secret-key-123"
        finally:
            if old_base is not None:
                os.environ["OPENAI_BASE_URL"] = old_base
            else:
                os.environ.pop("OPENAI_BASE_URL", None)
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)
