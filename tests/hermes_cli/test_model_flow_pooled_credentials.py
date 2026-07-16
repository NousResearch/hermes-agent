"""Pool-only credentials must be visible to interactive model setup flows."""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.model_setup_flows import _existing_api_key_for_model_flow


class _PoolEntry:
    access_token = "pool-secret"
    runtime_api_key = ""


class _AvailablePool:
    def has_credentials(self) -> bool:
        return True

    def peek(self):
        return _PoolEntry()


class _ExhaustedPool:
    def has_credentials(self) -> bool:
        return True

    def peek(self):
        return None


def test_existing_key_precedence_is_dotenv_then_process_then_pool(tmp_path, monkeypatch):
    pconfig = PROVIDER_REGISTRY["deepseek"]
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-secret")
    (hermes_home / ".env").write_text("DEEPSEEK_API_KEY=dotenv-secret\n", encoding="utf-8")

    with patch("agent.credential_pool.load_pool", return_value=_AvailablePool()):
        assert _existing_api_key_for_model_flow("deepseek", pconfig) == (
            "dotenv-secret",
            "DEEPSEEK_API_KEY",
        )

    (hermes_home / ".env").write_text("", encoding="utf-8")
    with patch("agent.credential_pool.load_pool", return_value=_AvailablePool()):
        assert _existing_api_key_for_model_flow("deepseek", pconfig) == (
            "process-secret",
            "DEEPSEEK_API_KEY",
        )

    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with patch("agent.credential_pool.load_pool", return_value=_AvailablePool()):
        assert _existing_api_key_for_model_flow("deepseek", pconfig) == (
            "pool-secret",
            "credential_pool:deepseek",
        )


def test_exhausted_pool_is_not_an_existing_key(monkeypatch):
    pconfig = PROVIDER_REGISTRY["deepseek"]
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with (
        patch("hermes_cli.config.get_env_value", return_value=""),
        patch("agent.credential_pool.load_pool", return_value=_ExhaustedPool()),
    ):
        assert _existing_api_key_for_model_flow("deepseek", pconfig) == ("", "")


def test_generic_api_key_flow_passes_pool_key_to_existing_key_prompt(monkeypatch):
    from hermes_cli.model_setup_flows import _model_flow_api_key_provider

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    captured: dict[str, str] = {}

    def capture_prompt(_pconfig, existing_key, **_kwargs):
        captured["existing_key"] = existing_key
        return existing_key, True

    with (
        patch("hermes_cli.config.get_env_value", return_value=""),
        patch("agent.credential_pool.load_pool", return_value=_AvailablePool()),
        patch("hermes_cli.main._prompt_api_key", side_effect=capture_prompt),
    ):
        _model_flow_api_key_provider({}, "deepseek")

    assert captured["existing_key"] == "pool-secret"


def test_kimi_flow_passes_pool_key_to_existing_key_prompt(monkeypatch):
    from hermes_cli.model_setup_flows import _model_flow_kimi

    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    captured: dict[str, str] = {}

    def capture_prompt(_pconfig, existing_key, **_kwargs):
        captured["existing_key"] = existing_key
        return existing_key, True

    with (
        patch("hermes_cli.config.get_env_value", return_value=""),
        patch("agent.credential_pool.load_pool", return_value=_AvailablePool()),
        patch("hermes_cli.main._prompt_api_key", side_effect=capture_prompt),
    ):
        _model_flow_kimi({})

    assert captured["existing_key"] == "pool-secret"


def test_exhausted_pool_still_uses_first_time_prompt(monkeypatch):
    from hermes_cli.model_setup_flows import _model_flow_api_key_provider

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    captured: dict[str, str] = {}

    def capture_prompt(_pconfig, existing_key, **_kwargs):
        captured["existing_key"] = existing_key
        return "", True

    with (
        patch("hermes_cli.config.get_env_value", return_value=""),
        patch("agent.credential_pool.load_pool", return_value=_ExhaustedPool()),
        patch("hermes_cli.main._prompt_api_key", side_effect=capture_prompt),
    ):
        _model_flow_api_key_provider({}, "deepseek")

    assert captured["existing_key"] == ""
