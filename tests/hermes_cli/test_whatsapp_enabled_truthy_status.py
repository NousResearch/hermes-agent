"""WHATSAPP_ENABLED status/setup must accept shared truthy aliases."""

from __future__ import annotations

from hermes_cli.gateway import _platform_status


def test_platform_status_whatsapp_accepts_truthy_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.gateway.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.gateway.get_env_value",
        lambda key: "on" if key == "WHATSAPP_ENABLED" else None,
    )

    status = _platform_status({"token_var": "WHATSAPP_ENABLED", "key": "whatsapp"})
    assert status == "enabled, not paired"


def test_platform_status_whatsapp_paired_when_creds_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.gateway.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.gateway.get_env_value",
        lambda key: "1" if key == "WHATSAPP_ENABLED" else None,
    )
    creds = tmp_path / "whatsapp" / "session" / "creds.json"
    creds.parent.mkdir(parents=True)
    creds.write_text("{}", encoding="utf-8")

    status = _platform_status({"token_var": "WHATSAPP_ENABLED", "key": "whatsapp"})
    assert status == "configured + paired"


def test_platform_status_whatsapp_rejects_falsy(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.gateway.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.gateway.get_env_value",
        lambda key: "off" if key == "WHATSAPP_ENABLED" else None,
    )

    status = _platform_status({"token_var": "WHATSAPP_ENABLED", "key": "whatsapp"})
    assert status == "not configured"
