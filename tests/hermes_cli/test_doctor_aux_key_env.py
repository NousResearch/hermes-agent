"""Verify that doctor accepts auxiliary providers configured through key pointers.

These tests are needed because doctor must validate auxiliary routing using the
same credential forms accepted by the delegation runtime.
"""

import hermes_cli.config as hermes_config
import hermes_cli.runtime_provider as runtime_provider

from hermes_cli.doctor_config import _validate_auxiliary_config


def _clear_possible_credentials(monkeypatch):
    for name in ("OPENAI_API_KEY", "LITELLM_MASTER_KEY", "POINTER_KEY_ENV"):
        monkeypatch.delenv(name, raising=False)


def _write_auxiliary_config(tmp_path, credential_line):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "auxiliary:\n"
        "  test-task:\n"
        "    provider: litellm\n"
        "    base_url: https://litellm.internal/v1\n"
        "    model: some-model\n"
        f"    {credential_line}\n",
        encoding="utf-8",
    )
    return config_path


def test_auxiliary_key_env_is_not_reported_without_credentials(monkeypatch, tmp_path, capsys):
    """A routed auxiliary block with only key_env resolves successfully."""
    _clear_possible_credentials(monkeypatch)
    fake_key = "test-key-not-a-secret"
    monkeypatch.setattr(hermes_config, "get_env_value_prefer_dotenv", lambda name: fake_key)

    def fake_resolve_runtime_provider(
        *, requested, target_model, explicit_api_key, explicit_base_url
    ):
        assert explicit_api_key == fake_key
        return {
            "provider": "custom",
            "api_key": explicit_api_key or "",
            "base_url": explicit_base_url,
            "command": None,
        }

    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )
    config_path = _write_auxiliary_config(tmp_path, "key_env: POINTER_KEY_ENV")
    issues = []

    _validate_auxiliary_config(str(config_path), issues)

    output = capsys.readouterr().out
    assert "without credentials" not in output
    assert issues == []


def test_auxiliary_explicit_api_key_behavior_is_unchanged(monkeypatch, tmp_path, capsys):
    """A routed auxiliary block with an explicit API key remains valid."""
    _clear_possible_credentials(monkeypatch)
    monkeypatch.setattr(
        hermes_config,
        "get_env_value_prefer_dotenv",
        lambda name: "test-key-not-a-secret",
    )

    def fake_resolve_runtime_provider(
        *, requested, target_model, explicit_api_key, explicit_base_url
    ):
        assert explicit_api_key == "test-explicit-key"
        return {
            "provider": "custom",
            "api_key": explicit_api_key or "",
            "base_url": explicit_base_url,
            "command": None,
        }

    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )
    config_path = _write_auxiliary_config(tmp_path, "api_key: test-explicit-key")
    issues = []

    _validate_auxiliary_config(str(config_path), issues)

    output = capsys.readouterr().out
    assert "without credentials" not in output
    assert issues == []
