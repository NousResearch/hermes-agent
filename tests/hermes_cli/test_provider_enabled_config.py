"""Tests for _is_provider_enabled() — providers.<name>.enabled config gate."""

import pytest


def _write_config(tmp_path, config: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    import yaml
    (hermes_home / "config.yaml").write_text(yaml.dump(config))


# ── _is_provider_enabled ──────────────────────────────────────────────────


def test_enabled_false_disables_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"providers": {"copilot": {"enabled": False}}})

    from hermes_cli.auth import _is_provider_enabled
    assert _is_provider_enabled("copilot") is False


def test_enabled_true_does_not_disable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"providers": {"copilot": {"enabled": True}}})

    from hermes_cli.auth import _is_provider_enabled
    assert _is_provider_enabled("copilot") is True


def test_no_config_is_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    from hermes_cli.auth import _is_provider_enabled
    assert _is_provider_enabled("copilot") is True


def test_empty_providers_is_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {})

    from hermes_cli.auth import _is_provider_enabled
    assert _is_provider_enabled("copilot") is True


def test_missing_enabled_key_is_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"providers": {"copilot": {}}})

    from hermes_cli.auth import _is_provider_enabled
    assert _is_provider_enabled("copilot") is True


def test_other_provider_unmentioned_is_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"providers": {"copilot": {"enabled": False}}})

    from hermes_cli.auth import _is_provider_enabled
    assert _is_provider_enabled("deepseek") is True


# ── _resolve_api_key_provider_secret integration ──────────────────────────


def test_disabled_provider_skips_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"providers": {"copilot": {"enabled": False}}})

    from hermes_cli.auth import _resolve_api_key_provider_secret, PROVIDER_REGISTRY
    pconfig = PROVIDER_REGISTRY["copilot"]
    result = _resolve_api_key_provider_secret("copilot", pconfig)
    assert result == ("", "")


def test_enabled_provider_resolves_normally(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "«redacted:sk-test»")

    from hermes_cli.auth import _resolve_api_key_provider_secret, PROVIDER_REGISTRY
    pconfig = PROVIDER_REGISTRY["deepseek"]
    result = _resolve_api_key_provider_secret("deepseek", pconfig)
    assert result[0] == "«redacted:sk-test»"
    assert result[1] == "DEEPSEEK_API_KEY"
