"""Tests for _resolve_task_provider_model credential lookup when a named
provider is configured with a custom base_url but no explicit api_key.

Regression for #16290: ZAI_API_KEY (and any PROVIDER_REGISTRY env var) was
silently ignored when auxiliary.vision.base_url was set because the function
returned "custom" without consulting the provider's credential pool.
"""

from unittest.mock import patch


def _make_task_config(provider, base_url, api_key=""):
    return {
        "provider": provider,
        "model": "glm-4v",
        "base_url": base_url,
        "api_key": api_key,
    }


def _patch_aux_config(task_cfg, monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client._get_auxiliary_task_config",
        lambda task: task_cfg if task == "vision" else {},
    )


class TestResolveTaskProviderModelCredentialLookup:
    def test_named_provider_with_base_url_resolves_env_key(self, monkeypatch):
        """ZAI_API_KEY must be used when provider=zai, base_url set, api_key empty."""
        from agent.auxiliary_client import _resolve_task_provider_model

        _patch_aux_config(
            _make_task_config("zai", "https://open.bigmodel.cn/api/paas/v4"),
            monkeypatch,
        )

        fake_creds = {"api_key": "zai-test-key-123"}
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value=fake_creds,
        ):
            provider, model, base_url, api_key, api_mode = _resolve_task_provider_model("vision")

        assert provider == "custom"
        assert base_url == "https://open.bigmodel.cn/api/paas/v4"
        assert api_key == "zai-test-key-123"

    def test_explicit_api_key_in_config_takes_precedence(self, monkeypatch):
        """An explicit api_key in config.yaml must not be overwritten."""
        from agent.auxiliary_client import _resolve_task_provider_model

        _patch_aux_config(
            _make_task_config("zai", "https://open.bigmodel.cn/api/paas/v4", api_key="hardcoded-key"),
            monkeypatch,
        )

        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "env-key-should-not-win"},
        ) as mock_creds:
            provider, model, base_url, api_key, api_mode = _resolve_task_provider_model("vision")

        mock_creds.assert_not_called()
        assert api_key == "hardcoded-key"

    def test_unknown_provider_in_registry_leaves_key_none(self, monkeypatch):
        """Providers not in PROVIDER_REGISTRY must not raise; api_key returns None."""
        from agent.auxiliary_client import _resolve_task_provider_model

        _patch_aux_config(
            _make_task_config("my-local-llm", "http://localhost:11434/v1"),
            monkeypatch,
        )

        provider, model, base_url, api_key, api_mode = _resolve_task_provider_model("vision")

        assert provider == "custom"
        assert base_url == "http://localhost:11434/v1"
        assert api_key is None

    def test_custom_provider_name_skips_registry_lookup(self, monkeypatch):
        """provider=custom with base_url set must not attempt registry lookup."""
        from agent.auxiliary_client import _resolve_task_provider_model

        _patch_aux_config(
            _make_task_config("custom", "http://custom-endpoint.example.com/v1"),
            monkeypatch,
        )

        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
        ) as mock_creds:
            provider, model, base_url, api_key, api_mode = _resolve_task_provider_model("vision")

        mock_creds.assert_not_called()
        assert provider == "custom"
        assert api_key is None

    def test_missing_credentials_do_not_raise(self, monkeypatch):
        """If the credential pool has no key for the provider, api_key returns None."""
        from agent.auxiliary_client import _resolve_task_provider_model

        _patch_aux_config(
            _make_task_config("zai", "https://open.bigmodel.cn/api/paas/v4"),
            monkeypatch,
        )

        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": ""},
        ):
            provider, model, base_url, api_key, api_mode = _resolve_task_provider_model("vision")

        assert provider == "custom"
        assert api_key is None

    def test_fix_applies_to_non_vision_tasks_too(self, monkeypatch):
        """Same fix applies to compression/review/other auxiliary tasks."""
        from agent.auxiliary_client import _resolve_task_provider_model

        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: _make_task_config("deepseek", "https://api.deepseek.com/v1") if task == "compression" else {},
        )

        fake_creds = {"api_key": "deepseek-key-456"}
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value=fake_creds,
        ):
            provider, model, base_url, api_key, api_mode = _resolve_task_provider_model("compression")

        assert provider == "custom"
        assert base_url == "https://api.deepseek.com/v1"
        assert api_key == "deepseek-key-456"
