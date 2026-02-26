"""Unit tests for hermes_cli.main._has_any_provider_configured."""

import json
import sys
import types

import pytest

import hermes_cli.config as config_module

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

import hermes_cli.main as main


PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GLM_API_KEY",
    "ZAI_API_KEY",
    "Z_AI_API_KEY",
    "KIMI_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in PROVIDER_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


def _patch_config_paths(monkeypatch, tmp_path, *, env_text=None, auth_payload=None):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    env_path = hermes_home / ".env"
    if env_text is not None:
        env_path.write_text(env_text)

    auth_path = hermes_home / "auth.json"
    if auth_payload is not None:
        auth_path.write_text(json.dumps(auth_payload))

    monkeypatch.setattr(config_module, "get_env_path", lambda: env_path)
    monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)


def test_has_any_provider_configured_uses_runtime_env_keys(monkeypatch, tmp_path):
    _patch_config_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("GLM_API_KEY", "glm-live-key")

    def _unexpected_load_config():
        raise AssertionError("load_config should not run on early env-key detection")

    monkeypatch.setattr(config_module, "load_config", _unexpected_load_config)
    assert main._has_any_provider_configured() is True


def test_has_any_provider_configured_reads_provider_vars_from_dotenv_file(monkeypatch, tmp_path):
    _patch_config_paths(
        monkeypatch,
        tmp_path,
        env_text="\n# comments are ignored\nKIMI_API_KEY='from-dotenv'\n",
    )

    def _unexpected_load_config():
        raise AssertionError("load_config should not run when .env already has provider key")

    monkeypatch.setattr(config_module, "load_config", _unexpected_load_config)
    assert main._has_any_provider_configured() is True


def test_has_any_provider_configured_accepts_custom_base_url_only(monkeypatch, tmp_path):
    _patch_config_paths(
        monkeypatch,
        tmp_path,
        env_text="\nOPENAI_BASE_URL='https://custom.example/v1'\n",
    )

    def _unexpected_load_config():
        raise AssertionError("load_config should not run when .env already has custom base URL")

    monkeypatch.setattr(config_module, "load_config", _unexpected_load_config)
    assert main._has_any_provider_configured() is True


def test_has_any_provider_configured_uses_configured_custom_profile(monkeypatch, tmp_path):
    _patch_config_paths(monkeypatch, tmp_path, env_text="UNRELATED_KEY=1\n")
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {
            "model": {
                "profiles": [
                    {
                        "name": "custom-profile",
                        "provider": "custom",
                        "model": "my-model",
                        "base_url": "http://localhost:8000/v1",
                        "enabled": True,
                    }
                ],
                "active_profile": "custom-profile",
                "scoped_profiles": ["custom-profile"],
            }
        },
    )
    assert main._has_any_provider_configured() is True


def test_has_any_provider_configured_uses_auth_file_tokens_when_config_unavailable(monkeypatch, tmp_path):
    _patch_config_paths(
        monkeypatch,
        tmp_path,
        auth_payload={
            "active_provider": "nous",
            "providers": {"nous": {"refresh_token": "refresh-token"}},
        },
    )

    def _load_config_failure():
        raise RuntimeError("config not available")

    monkeypatch.setattr(config_module, "load_config", _load_config_failure)
    assert main._has_any_provider_configured() is True


def test_has_any_provider_configured_returns_false_when_no_sources_are_present(monkeypatch, tmp_path):
    _patch_config_paths(monkeypatch, tmp_path, env_text="UNRELATED_KEY=1\n")

    def _load_config_failure():
        raise RuntimeError("config not available")

    monkeypatch.setattr(config_module, "load_config", _load_config_failure)
    assert main._has_any_provider_configured() is False


def test_sync_oauth_active_provider_deactivates_when_active_profile_is_non_nous(monkeypatch):
    calls = {"count": 0}

    monkeypatch.setattr(
        "hermes_cli.auth.deactivate_provider",
        lambda: calls.__setitem__("count", calls["count"] + 1),
    )

    main._sync_oauth_active_provider(
        {
            "default": "model-a",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "profiles": [
                {
                    "name": "openrouter-default",
                    "provider": "openrouter",
                    "model": "model-a",
                    "base_url": "https://openrouter.ai/api/v1",
                    "enabled": True,
                }
            ],
            "active_profile": "openrouter-default",
            "scoped_profiles": ["openrouter-default"],
        }
    )
    assert calls["count"] == 1

    main._sync_oauth_active_provider(
        {
            "default": "nous/model",
            "provider": "nous",
            "base_url": "https://inference.nousresearch.com/v1",
            "profiles": [
                {
                    "name": "nous-default",
                    "provider": "nous",
                    "model": "nous/model",
                    "base_url": "https://inference.nousresearch.com/v1",
                    "enabled": True,
                }
            ],
            "active_profile": "nous-default",
            "scoped_profiles": ["nous-default"],
        }
    )
    assert calls["count"] == 1
