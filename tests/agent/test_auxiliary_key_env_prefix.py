"""Test key_env / api_key_env field resolution for auxiliary task config.

When auxiliary.<task>.key_env (or its api_key_env alias) names an
environment variable, _resolve_task_provider_model() must resolve it to
the actual key value. This mirrors the same pattern used by
providers.<name>.key_env in resolve_provider_client() and fallback
entries in agent_init.py / chat_completion_helpers.py.

Without this fix, configuring e.g.:

  auxiliary:
    title_generation:
      provider: openrouter
      base_url: https://openrouter.ai/api/v1
      key_env: OPENROUTER_API_KEY

would silently ignore the key_env field — the task gets no api_key,
causing HTTP 401 errors on auth-required endpoints.
"""

import os
import pytest
from unittest.mock import patch


class TestAuxiliaryKeyEnvResolution:
    """Tests for key_env / api_key_env resolution in _resolve_task_provider_model."""

    def test_key_env_resolved_with_base_url(self):
        """key_env field resolves to the actual env var value.

        When both base_url and key_env (resolved to api_key) are set,
        _resolve_task_provider_model returns provider="custom" with the
        resolved api_key.
        """
        from agent.auxiliary_client import _resolve_task_provider_model

        task_config = {
            "provider": "openrouter",
            "model": "nvidia/nemotron-3-nano-30b-a3b:free",
            "base_url": "https://openrouter.ai/api/v1",
            "key_env": "OPENROUTER_API_KEY",
        }

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key-12345"}, clear=False):
            with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
                provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                    task="title_generation"
                )

        assert provider == "custom"
        assert model == "nvidia/nemotron-3-nano-30b-a3b:free"
        assert base_url == "https://openrouter.ai/api/v1"
        assert api_key == "sk-test-key-12345"

    def test_api_key_env_alias_resolved(self):
        """api_key_env (snake_case alias) resolves identically to key_env."""
        from agent.auxiliary_client import _resolve_task_provider_model

        task_config = {
            "provider": "openrouter",
            "model": "google/gemini-2.5-flash",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        }

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key-99999"}, clear=False):
            with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
                provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                    task="title_generation"
                )

        assert provider == "custom"
        assert api_key == "sk-test-key-99999"

    def test_key_env_without_base_url(self):
        """When key_env resolves but no base_url, named provider is returned.

        The provider resolves its own credentials from provider-level env vars
        (e.g. OPENROUTER_API_KEY). The resolved api_key is still passed through
        for resolve_provider_client to use.
        """
        from agent.auxiliary_client import _resolve_task_provider_model

        task_config = {
            "provider": "openrouter",
            "model": "nvidia/nemotron-3-nano-30b-a3b:free",
            "base_url": "",
            "key_env": "OPENROUTER_API_KEY",
        }

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key-12345"}, clear=False):
            with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
                provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                    task="title_generation"
                )

        assert provider == "openrouter"
        assert api_key == "sk-test-key-12345"

    def test_key_env_missing_env_var(self):
        """When the env var referenced by key_env does not exist, api_key
        resolves to None — not the literal env var name."""
        from agent.auxiliary_client import _resolve_task_provider_model

        task_config = {
            "provider": "openrouter",
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "base_url": "https://openrouter.ai/api/v1",
            "key_env": "NONEXISTENT_KEY_12345",
        }

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NONEXISTENT_KEY_12345", None)
            with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
                provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                    task="compression"
                )

        # Missing env var → cfg_api_key resolves to None.
        # With base_url set but no api_key, falls to the
        # cfg_base_url + cfg_provider != "auto" branch.
        assert provider == "openrouter"
        assert base_url == "https://openrouter.ai/api/v1"
        assert api_key is None

    def test_explicit_api_key_takes_precedence_over_key_env(self):
        """An explicit api_key value takes precedence over key_env.

        When both api_key and key_env are set, api_key wins — matching
        the provider-level precedence in resolve_provider_client().
        """
        from agent.auxiliary_client import _resolve_task_provider_model

        task_config = {
            "provider": "openrouter",
            "model": "some-model",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-explicit-key",
            "key_env": "OPENROUTER_API_KEY",
        }

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-from-env"}, clear=False):
            with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
                provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                    task="title_generation"
                )

        assert provider == "custom"
        assert api_key == "sk-explicit-key"

    def test_key_env_empty_string_env_var(self):
        """key_env: VAR where the env var exists but is empty-string
        should resolve to None (not the empty string)."""
        from agent.auxiliary_client import _resolve_task_provider_model

        task_config = {
            "provider": "openrouter",
            "model": "nvidia/nemotron-3-nano-30b-a3b:free",
            "base_url": "https://openrouter.ai/api/v1",
            "key_env": "EMPTY_API_KEY",
        }

        with patch.dict(os.environ, {"EMPTY_API_KEY": ""}, clear=False):
            with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
                provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                    task="compression"
                )

        # Empty string in env → .strip() yields "" → or None → None
        assert api_key is None

    def test_no_key_env_no_api_key(self):
        """When neither api_key nor key_env is set, api_key stays None.
        Provider-level credential resolution handles auth."""
        from agent.auxiliary_client import _resolve_task_provider_model

        task_config = {
            "provider": "openrouter",
            "model": "some-model",
        }

        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="web_extract"
            )

        assert provider == "openrouter"
        assert api_key is None
