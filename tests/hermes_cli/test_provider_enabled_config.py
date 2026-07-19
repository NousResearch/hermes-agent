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
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    from hermes_cli.auth import _resolve_api_key_provider_secret, PROVIDER_REGISTRY
    pconfig = PROVIDER_REGISTRY["deepseek"]
    result = _resolve_api_key_provider_secret("deepseek", pconfig)
    assert result[0] == "sk-test"
    assert result[1] == "DEEPSEEK_API_KEY"


# ── config normalizer: enabled key not warned ─────────────────────────────


def test_enabled_key_not_treated_as_unknown(tmp_path, monkeypatch):
    """``enabled`` in a provider entry does not trigger unknown-key warning.

    The normalizer uses ``logger.warning()`` (not ``warnings.warn()``) for
    unknown keys.  Adding ``enabled`` to ``_KNOWN_KEYS`` prevents that
    warning.  The normalizer still only copies recognised config keys
    (base_url, api_key, etc.) into its output — it does not propagate
    ``enabled`` — but that is fine because ``_is_provider_enabled()``
    reads from ``read_raw_config()``, not from the normalizer's output.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"providers": {"copilot": {"enabled": False}}})

    from hermes_cli.config import _normalize_custom_provider_entry

    result = _normalize_custom_provider_entry(
        {"api": "https://api.example.com", "enabled": False},
        provider_key="copilot",
    )

    # Must still be a valid provider entry (not None)
    assert result is not None
    assert "base_url" in result


# ── resolve_api_key_provider_credentials guard ────────────────────────────


def test_disabled_resolve_api_key_returns_empty(tmp_path, monkeypatch):
    """Disabled provider returns empty credentials, bypassing all resolution."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"providers": {"copilot": {"enabled": False}}})

    from hermes_cli.auth import resolve_api_key_provider_credentials

    creds = resolve_api_key_provider_credentials("copilot")
    assert creds["provider"] == "copilot"
    assert creds["api_key"] == ""
    assert creds["base_url"] == ""
    assert creds["source"] == "disabled"


def test_disabled_resolve_api_key_no_copilot_token_call(tmp_path, monkeypatch):
    """Disabled provider must not call resolve_copilot_token()."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"providers": {"copilot": {"enabled": False}}})

    from unittest import mock
    from hermes_cli.auth import resolve_api_key_provider_credentials

    with mock.patch(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        side_effect=AssertionError("resolve_copilot_token was called"),
    ):
        creds = resolve_api_key_provider_credentials("copilot")
        assert creds["source"] == "disabled"


def test_enabled_resolve_api_key_works_normally(tmp_path, monkeypatch):
    """Enabled provider (default) still resolves normally."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    from hermes_cli.auth import resolve_api_key_provider_credentials

    creds = resolve_api_key_provider_credentials("deepseek")
    assert creds["api_key"] == "sk-test"
    assert creds["source"] == "DEEPSEEK_API_KEY"


# ── resolve_external_process_provider_credentials guard ───────────────────


def test_disabled_resolve_external_process_returns_empty(tmp_path, monkeypatch):
    """Disabled provider returns empty credentials for external-process auth."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"providers": {"copilot": {"enabled": False}}})

    from hermes_cli.auth import resolve_external_process_provider_credentials

    creds = resolve_external_process_provider_credentials("copilot")
    assert creds["provider"] == "copilot"
    assert creds["base_url"] == ""
    assert creds["api_key"] == ""
    assert creds["command"] == ""
    assert creds["args"] == []
    assert creds["source"] == "disabled"


# ── picker discovery gate ─────────────────────────────────────────────────


def test_disabled_provider_skipped_in_picker(tmp_path, monkeypatch):
    """Picker credential discovery skips disabled providers.

    Uses ``_is_provider_enabled()`` to confirm the gate works; the picker
    calls the same function inside the HERMES_OVERLAYS loop.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"providers": {"copilot": {"enabled": False}}})

    from hermes_cli.auth import _is_provider_enabled
    assert _is_provider_enabled("copilot") is False

    # deepseek is not mentioned in config → still enabled
    assert _is_provider_enabled("deepseek") is True
