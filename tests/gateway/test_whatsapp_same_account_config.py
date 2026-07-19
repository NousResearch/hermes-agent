"""Same-account group opt-in reaches the bridge without crossing profile boundaries."""

import os

import pytest

from agent import secret_scope as ss
from gateway.config import Platform, load_gateway_config
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


@pytest.fixture(autouse=True)
def _enable_whatsapp(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ENABLED", "true")


@pytest.mark.parametrize("configured", [None, False, True])
@pytest.mark.parametrize("override", [None, "true", "false"])
def test_same_account_config_reaches_bridge(tmp_path, monkeypatch, configured, override):
    text = "whatsapp: {}\n" if configured is None else (
        f"whatsapp:\n  process_from_me_groups: {str(configured).lower()}\n"
    )
    (tmp_path / "config.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("WHATSAPP_PROCESS_FROM_ME_GROUPS", raising=False)
    if override is not None:
        monkeypatch.setenv("WHATSAPP_PROCESS_FROM_ME_GROUPS", override)

    config = load_gateway_config()
    adapter = WhatsAppAdapter(config.platforms[Platform.WHATSAPP])
    expected = override if override is not None else str(bool(configured)).lower()
    env = adapter._bridge_env()

    assert env["WHATSAPP_PROCESS_FROM_ME_GROUPS"] == expected
    assert env["HERMES_BRIDGE_RUNTIME_HASH"] == adapter._bridge_env()["HERMES_BRIDGE_RUNTIME_HASH"]


@pytest.mark.parametrize("launch_value", ["true", "false"])
def test_same_account_config_stays_profile_local(tmp_path, monkeypatch, launch_value):
    monkeypatch.setenv("WHATSAPP_PROCESS_FROM_ME_GROUPS", launch_value)
    ss.set_multiplex_active(True)
    try:
        for name, enabled in (("a", True), ("b", False), ("a-again", True)):
            home = tmp_path / name
            home.mkdir()
            (home / "config.yaml").write_text(
                f"whatsapp:\n  process_from_me_groups: {str(enabled).lower()}\n",
                encoding="utf-8",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))
            token = ss.set_secret_scope(ss.build_profile_secret_scope(home))
            try:
                config = load_gateway_config()
                adapter = WhatsAppAdapter(config.platforms[Platform.WHATSAPP])
                assert adapter._bridge_env()["WHATSAPP_PROCESS_FROM_ME_GROUPS"] == str(enabled).lower()
                assert os.environ["WHATSAPP_PROCESS_FROM_ME_GROUPS"] == launch_value
            finally:
                ss.reset_secret_scope(token)
    finally:
        ss.set_multiplex_active(False)
