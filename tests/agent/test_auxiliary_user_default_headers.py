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
from types import SimpleNamespace

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

    def test_provider_headers_apply_only_to_matching_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {
                "default": "m",
                "default_headers": {"X-Global": "global"},
                "provider_headers": {
                    "gmi": {"X-Provider": "gmi", "X-Global": "gmi"},
                },
            },
        })
        from agent.auxiliary_client import _apply_user_default_headers

        matched = _apply_user_default_headers({"User-Agent": "HermesAgent/test"}, provider="gmi")
        assert matched is not None
        assert matched["User-Agent"] == "HermesAgent/test"
        assert matched["X-Global"] == "gmi"
        assert matched["X-Provider"] == "gmi"

        other = _apply_user_default_headers({}, provider="openrouter")
        assert other == {"X-Global": "global"}

    def test_provider_headers_can_match_by_base_url_for_auto_async(self, tmp_path):
        _write_config(tmp_path, {
            "model": {
                "default": "m",
                "provider_headers": {
                    "gmi": {"X-Provider": "gmi"},
                },
            },
        })
        from agent.auxiliary_client import _apply_user_default_headers

        merged = _apply_user_default_headers(
            {},
            provider="auto",
            base_url="https://api.gmi-serving.com/v1",
        )
        assert merged == {"X-Provider": "gmi"}

    def test_provider_headers_can_match_custom_provider_by_base_url(self, tmp_path):
        _write_config(tmp_path, {
            "model": {
                "default": "m",
                "provider_headers": {
                    "gmi": {"X-Provider": "gmi"},
                },
            },
        })
        from agent.auxiliary_client import _apply_user_default_headers

        merged = _apply_user_default_headers(
            {},
            provider="custom",
            base_url="https://api.gmi-serving.com/v1",
        )
        assert merged == {"X-Provider": "gmi"}

    def test_base_url_inference_prefers_most_specific_profile_path(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {
            "model": {
                "default": "m",
                "provider_headers": {
                    "wide": {"X-Billing": "wide"},
                    "specific": {"X-Billing": "specific"},
                },
            },
        })
        profiles = [
            SimpleNamespace(name="wide", base_url="https://example.test/root/v1"),
            SimpleNamespace(name="specific", base_url="https://example.test/root/specific/v1"),
        ]
        monkeypatch.setattr("providers.list_providers", lambda: profiles)
        from agent.auxiliary_client import _apply_user_default_headers

        merged = _apply_user_default_headers(
            {},
            provider="auto",
            base_url="https://example.test/root/specific/v1",
        )
        assert merged == {"X-Billing": "specific"}


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

    def test_provider_scoped_header_reaches_api_key_aux_client(self, tmp_path):
        """Provider-scoped headers apply to matching API-key auxiliary clients only."""
        _write_config(tmp_path, {
            "model": {
                "default": "gmi-test-model",
                "provider_headers": {
                    "gmi": {"X-Provider": "gmi"},
                },
            },
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client(
                "gmi",
                "gmi-test-model",
                explicit_api_key="gmi-test-key",
                explicit_base_url="https://api.gmi-serving.com/v1",
            )

        assert client is not None
        assert model == "gmi-test-model"
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("X-Provider") == "gmi"

    def test_auto_resolution_remembers_concrete_main_provider(self, tmp_path, monkeypatch):
        """_resolve_auto keeps the provider identity without changing its two-tuple API."""
        from agent import auxiliary_client as aux

        sync_client = SimpleNamespace(
            api_key="gmi-test-key",
            base_url="https://unregistered-gmi-gateway.example/v1",
        )

        def fake_resolve_provider_client(provider, model, **_kwargs):
            assert provider == "gmi"
            assert model == "gmi-test-model"
            return sync_client, model

        monkeypatch.setattr(aux, "resolve_provider_client", fake_resolve_provider_client)
        client, model, provider = aux._resolve_auto_with_provider(
            main_runtime={
                "provider": "gmi",
                "model": "gmi-test-model",
                "base_url": "https://unregistered-gmi-gateway.example/v1",
                "api_key": "gmi-test-key",
            },
        )

        assert client is sync_client
        assert model == "gmi-test-model"
        assert provider == "gmi"

    def test_auto_async_infers_provider_for_scoped_headers(self, tmp_path):
        """Auto-routed async clients must keep the resolved provider's scoped headers."""
        _write_config(tmp_path, {
            "model": {
                "default": "gmi-test-model",
                "provider_headers": {
                    "gmi": {"X-Provider": "gmi"},
                },
            },
        })
        sync_client = SimpleNamespace(
            api_key="gmi-test-key",
            base_url="https://unregistered-gmi-gateway.example/v1",
        )

        def fake_resolve_auto_with_provider(*_args, **_kwargs):
            return sync_client, "gmi-test-model", "gmi"

        with patch("agent.auxiliary_client._resolve_auto_with_provider", side_effect=fake_resolve_auto_with_provider), \
             patch("openai.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("auto", async_mode=True)

        assert client is not None
        assert model == "gmi-test-model"
        headers = mock_async_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("X-Provider") == "gmi"

    def test_openrouter_aux_client_honors_provider_scoped_headers(self, tmp_path, monkeypatch):
        """Direct OpenRouter aux fallback clients should merge scoped headers too."""
        _write_config(tmp_path, {
            "model": {
                "provider_headers": {
                    "openrouter": {"X-Provider": "openrouter"},
                },
            },
        })
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _try_openrouter
            client, model = _try_openrouter()

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("X-Provider") == "openrouter"

    def test_nous_aux_client_honors_provider_scoped_headers(self, tmp_path):
        """Direct Nous aux fallback clients should merge scoped headers too."""
        _write_config(tmp_path, {
            "model": {
                "provider_headers": {
                    "nous": {"X-Provider": "nous"},
                },
            },
        })

        with patch("agent.auxiliary_client.OpenAI") as mock_openai, \
             patch("agent.auxiliary_client._read_nous_auth", return_value={}), \
             patch(
                 "agent.auxiliary_client._resolve_nous_runtime_api",
                 return_value=("nous-test-key", "https://inference-api.nousresearch.com/v1"),
             ), \
             patch("hermes_cli.models.get_nous_recommended_aux_model", return_value="nous-test-model"):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _try_nous
            client, model = _try_nous()

        assert client is not None
        assert model == "nous-test-model"
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("X-Provider") == "nous"

    def test_auto_vision_async_finalize_keeps_resolved_provider_headers(self, tmp_path, monkeypatch):
        """Vision auto async conversion must preserve the provider selected by auto."""
        _write_config(tmp_path, {
            "model": {
                "provider_headers": {
                    "gmi": {"X-Provider": "gmi"},
                },
            },
        })
        import agent.auxiliary_client as aux

        sync_client = SimpleNamespace(
            api_key="gmi-key",
            base_url="https://unregistered-gmi-gateway.example/v1",
        )
        monkeypatch.setattr(
            aux,
            "_resolve_task_provider_model",
            lambda *_args, **_kwargs: ("auto", "gmi-vision", None, None, None),
        )
        monkeypatch.setattr(aux, "_read_main_provider", lambda: "gmi")
        monkeypatch.setattr(aux, "_read_main_model", lambda: "gmi-vision")
        monkeypatch.setattr(aux, "_main_model_supports_vision", lambda provider, model: True)
        monkeypatch.setattr(
            aux,
            "resolve_provider_client",
            lambda provider, model, **_kwargs: (sync_client, model),
        )

        with patch("openai.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()
            provider, client, model = aux.resolve_vision_provider_client(
                "auto",
                "gmi-vision",
                async_mode=True,
            )

        assert provider == "gmi"
        assert client is not None
        assert model == "gmi-vision"
        headers = mock_async_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("X-Provider") == "gmi"

    def test_nous_runtime_refresh_async_keeps_provider_scoped_headers(self, tmp_path, monkeypatch):
        """Refreshing Nous async clients must preserve Nous-scoped headers."""
        _write_config(tmp_path, {
            "model": {
                "provider_headers": {
                    "nous": {"X-Provider": "nous"},
                },
            },
        })
        import agent.auxiliary_client as aux

        sync_client = SimpleNamespace(
            api_key="nous-key",
            base_url="https://inference-api.nousresearch.com/v1",
        )
        monkeypatch.setattr(
            aux,
            "_resolve_nous_runtime_api",
            lambda force_refresh=False: ("nous-key", "https://inference-api.nousresearch.com/v1"),
        )

        with patch("agent.auxiliary_client.OpenAI", return_value=sync_client), \
             patch("openai.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()
            client, model = aux._refresh_nous_auxiliary_client(
                cache_provider="nous",
                model="nous-model",
                async_mode=True,
            )

        assert client is not None
        assert model == "nous-model"
        headers = mock_async_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("X-Provider") == "nous"
