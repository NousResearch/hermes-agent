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
    import hermes_yaml as yaml
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.safe_dump(config_dict))


class TestApplyUserDefaultHeadersHelper:
    """Direct unit tests for the merge helper."""





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


class TestAuxClientEndpointMatchedProviderHeaders:
    """#127823 — endpoint-matched ``providers.<name>.extra_headers`` reach aux clients.

    The main client applies them on every build (client_lifecycle); a built-in provider
    routed through a proxy whose URL equals ``providers.<name>.api`` must carry them on
    auxiliary calls too, or compression/title generation 404 against the same proxy the
    main turn just used.
    """

    def test_endpoint_matched_headers_reach_client(self, tmp_path):
        """The #127823 reproduction at the single construction point."""
        _write_config(tmp_path, {
            "model": {"default": "glm-5.3-flash", "provider": "zai"},
            "providers": {
                "zai": {
                    "api": "http://127.0.0.1:8790",
                    "extra_headers": {
                        "x-headroom-base-url": "https://api.z.ai/api/coding/paas/v4",
                        "x-headroom-original-path": "/chat/completions",
                    },
                },
            },
        })
        from agent.auxiliary_client import _create_openai_client
        client = _create_openai_client(api_key="k", base_url="http://127.0.0.1:8790")
        headers = getattr(client, "default_headers", {}) or {}
        assert headers.get("x-headroom-base-url") == "https://api.z.ai/api/coding/paas/v4"
        assert headers.get("x-headroom-original-path") == "/chat/completions"

    def test_endpoint_matched_headers_never_override_codex_identity(self, tmp_path):
        """A ``providers:`` entry keyed on the official Codex URL must not strip the
        required identity headers: ``apply_required_codex_headers`` is documented as
        landing AFTER user/provider overrides (AI review ordering catch — merging the
        endpoint headers later let ``originator`` be overridden, diverging from both
        base and the async twin)."""
        _write_config(tmp_path, {
            "model": {"default": "gpt-5.2", "provider": "codex"},
            "providers": {
                "codex": {
                    "api": "https://chatgpt.com/backend-api/codex",
                    "extra_headers": {
                        "originator": "evil-override",
                        "User-Agent": "curl/8.7.1",
                    },
                },
            },
        })
        from agent.auxiliary_client import _create_openai_client
        client = _create_openai_client(
            api_key="k", base_url="https://chatgpt.com/backend-api/codex")
        headers = getattr(client, "default_headers", {}) or {}
        assert headers.get("originator") == "hermes-agent"
        assert headers.get("User-Agent") != "curl/8.7.1"

    def test_other_endpoint_gets_no_headers(self, tmp_path):
        """Matching is by exact normalized URL: an entry declaring another endpoint
        must not leak its headers onto this client."""
        _write_config(tmp_path, {
            "model": {"default": "glm-5.3-flash", "provider": "zai"},
            "providers": {
                "zai": {
                    "api": "http://127.0.0.1:8790",
                    "extra_headers": {"x-headroom-base-url": "https://api.z.ai/api/coding/paas/v4"},
                },
            },
        })
        from agent.auxiliary_client import _create_openai_client
        client = _create_openai_client(api_key="k", base_url="http://127.0.0.1:9999")
        headers = getattr(client, "default_headers", {}) or {}
        assert "x-headroom-base-url" not in headers

    def test_builtin_provider_aux_resolve_carries_headers(self, tmp_path, monkeypatch):
        """The reporter path end to end: an auxiliary lane pinned to a built-in provider
        (``auxiliary.compression.provider: zai``) resolves a client that carries the
        endpoint's routing headers."""
        monkeypatch.setenv("GLM_API_KEY", "test-key")
        monkeypatch.setenv("GLM_BASE_URL", "http://127.0.0.1:8790")
        _write_config(tmp_path, {
            "model": {"default": "glm-5.3-flash", "provider": "zai"},
            "providers": {
                "zai": {
                    "api": "http://127.0.0.1:8790",
                    "extra_headers": {"x-headroom-base-url": "https://api.z.ai/api/coding/paas/v4"},
                },
            },
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("zai", "glm-5.3-flash")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("x-headroom-base-url") == "https://api.z.ai/api/coding/paas/v4"
