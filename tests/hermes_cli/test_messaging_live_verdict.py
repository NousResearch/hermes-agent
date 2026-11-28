"""The Desktop/dashboard save verdict: an enabled platform without credentials needs setup, not a restart.

Enabling a platform before its credentials are saved makes the running gateway try it and fail. A
restart would fail the same way, yet the save answered ``applied: false`` and both UIs put up a sticky
"Restart the messaging gateway" banner. The gateway's own "configured" verdict decides instead.
"""
from hermes_cli.web_routers import messaging


def _verdict(tmp_path, monkeypatch, dotenv):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: true\n", encoding="utf-8")
    (home / ".env").write_text(dotenv, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    answer = {"applied": True, "changed": True, "started": ["telegram"], "restarted": [], "stopped": [],
              "platforms": {"telegram": "unavailable"}, "reasons": {}}
    monkeypatch.setattr("hermes_cli.gateway_platform_live.reconcile_running_gateway_platforms",
                        lambda name, home: answer)
    return messaging._apply_platform_live(None, "telegram")


def test_an_enabled_platform_without_credentials_needs_setup_not_a_restart(tmp_path, monkeypatch):
    live = _verdict(tmp_path, monkeypatch, "")

    assert live["applied"] is True and live["state"] == "not_configured", live


def test_a_configured_platform_the_gateway_could_not_start_still_needs_a_restart(tmp_path, monkeypatch):
    live = _verdict(tmp_path, monkeypatch, "TELEGRAM_BOT_TOKEN=123:abc\n")

    assert live["applied"] is False and live["state"] == "unavailable", live
