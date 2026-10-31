"""The Channels page must say WHY Team Gateway turned a platform off (#141175).

With the ``GATEWAY_RELAY_URL`` env stamp, ``gateway/config_env.py::_relay`` disables every
directly-connected messaging platform, explicitly enabled ones included, unless
``GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS`` is set. The dashboard used to see only the
resulting ``enabled: False`` and rendered a plain "Disabled": the user flips the toggle on,
restarts, and it reverts with no explanation. The rule is kept; the payload now carries the
reason so the page can name it.
"""
import pytest


_VALID_BOT_TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_1234"


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    import hermes_state
    import hermes_constants
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    home = get_hermes_home()
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    for var in ("TELEGRAM_BOT_TOKEN", "GATEWAY_RELAY_URL", "GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS"):
        monkeypatch.delenv(var, raising=False)
    (home / ".env").write_text(f"TELEGRAM_BOT_TOKEN={_VALID_BOT_TOKEN}\n", encoding="utf-8")
    (home / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: true\n", encoding="utf-8")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _telegram(client) -> dict:
    resp = client.get("/api/messaging/platforms")
    assert resp.status_code == 200, resp.text
    return next(p for p in resp.json()["platforms"] if p["id"] == "telegram")


def test_relay_stamp_reports_the_override_not_a_bare_disabled(client, monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_URL", "https://relay.example.test")

    telegram = _telegram(client)

    assert telegram["enabled"] is False
    assert telegram["state"] == "disabled"
    assert telegram["disabled_by_relay"] is True


def test_direct_platforms_opt_out_and_no_stamp_carry_no_relay_reason(client, monkeypatch):
    assert _telegram(client)["disabled_by_relay"] is False

    monkeypatch.setenv("GATEWAY_RELAY_URL", "https://relay.example.test")
    monkeypatch.setenv("GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS", "true")
    telegram = _telegram(client)
    assert telegram["enabled"] is True
    assert telegram["disabled_by_relay"] is False


def test_user_disabled_platform_is_not_blamed_on_the_relay(client, monkeypatch):
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(
        "platforms:\n  telegram:\n    enabled: false\n", encoding="utf-8"
    )
    monkeypatch.setenv("GATEWAY_RELAY_URL", "https://relay.example.test")

    telegram = _telegram(client)
    assert telegram["state"] == "disabled"
    assert telegram["disabled_by_relay"] is False
