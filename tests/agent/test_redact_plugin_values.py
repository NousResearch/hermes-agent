"""A plugin's redaction values live apart from the vault-fill values and cannot disturb them."""

import pytest

from agent import redact_vault_values as rv
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(rv, "_VAULT_REDACTION_VALUES", {})
    monkeypatch.setattr(rv, "_PLUGIN_REDACTION_VALUES", {})


def test_plugin_values_never_evict_vault_fill_values():
    rv.register_vault_redaction_value("vault-password-1")
    for i in range(rv._PLUGIN_REDACTION_MAX_PER_PROFILE * 2):
        rv.register_plugin_redaction_value(f"plugin-value-{i:04d}")

    scrubbed = rv.redact_registered_vault_values("x vault-password-1 y plugin-value-0127 z plugin-value-0000")
    assert "vault-password-1" not in scrubbed  # the vault value survived the flood
    assert "plugin-value-0127" not in scrubbed  # the newest plugin value is scrubbed
    assert "plugin-value-0000" in scrubbed  # the oldest plugin value fell off its own cap


def test_a_short_or_non_string_value_is_refused():
    for bad in ("", "a", "1234567", None, 12345678):
        with pytest.raises(ValueError):
            rv.register_plugin_redaction_value(bad)
    assert rv.redact_registered_vault_values("a 1234567") == "a 1234567"
    rv.register_plugin_redaction_value("12345678")
    assert "12345678" not in rv.redact_registered_vault_values("n 12345678")


def test_plugin_values_are_scoped_to_the_profile(tmp_path):
    token = set_hermes_home_override(str(tmp_path / "a"))
    try:
        rv.register_plugin_redaction_value("profile-a-secret")
        assert "profile-a-secret" not in rv.redact_registered_vault_values("profile-a-secret")
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(str(tmp_path / "b"))
    try:
        assert rv.redact_registered_vault_values("profile-a-secret") == "profile-a-secret"
    finally:
        reset_hermes_home_override(token)


def test_clearing_the_profile_drops_plugin_values_too():
    rv.register_plugin_redaction_value("plugin-secret-1")
    rv.clear_vault_redaction_values()
    assert rv.redact_registered_vault_values("plugin-secret-1") == "plugin-secret-1"
