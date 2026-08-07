"""MATRIX_ENCRYPTION status must accept shared truthy aliases."""

from __future__ import annotations

from hermes_cli.gateway import _platform_status


def _matrix_plat():
    return {
        "key": "matrix",
        "token_var": "MATRIX_ACCESS_TOKEN",
    }


def test_platform_status_matrix_e2ee_on_alias(monkeypatch):
    values = {
        "MATRIX_ACCESS_TOKEN": "syt_test",
        "MATRIX_HOMESERVER": "https://matrix.example.org",
        "MATRIX_ENCRYPTION": "on",
    }
    monkeypatch.setattr(
        "hermes_cli.gateway.get_env_value",
        lambda key: values.get(key),
    )
    assert _platform_status(_matrix_plat()) == "configured + E2EE"


def test_platform_status_matrix_e2ee_off_alias(monkeypatch):
    values = {
        "MATRIX_ACCESS_TOKEN": "syt_test",
        "MATRIX_HOMESERVER": "https://matrix.example.org",
        "MATRIX_ENCRYPTION": "off",
    }
    monkeypatch.setattr(
        "hermes_cli.gateway.get_env_value",
        lambda key: values.get(key),
    )
    assert _platform_status(_matrix_plat()) == "configured"
