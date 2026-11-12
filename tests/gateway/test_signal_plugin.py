"""Signal ships as a bundled platform plugin: the registry entry carries what core used to hard-wire
(requirements probe, config gate, out-of-process sender, chunking cap, setup wizard)."""

from types import SimpleNamespace

from gateway.config import PlatformConfig


def test_register_declares_the_former_core_wiring(monkeypatch):
    from plugins.platforms.signal import adapter
    from plugins.platforms.signal.register import register

    captured = {}
    register(SimpleNamespace(register_platform=lambda **kw: captured.update(kw)))

    assert captured["name"] == "signal" and captured["label"] == "Signal" and captured["emoji"] == "📡"
    assert captured["adapter_factory"] is adapter.SignalAdapter
    assert captured["max_message_length"] == adapter.MAX_MESSAGE_LENGTH == 8000
    assert captured["allowed_users_env"] == "SIGNAL_ALLOWED_USERS"
    assert captured["allow_all_env"] == "SIGNAL_ALLOW_ALL_USERS"
    assert captured["cron_deliver_env_var"] == "SIGNAL_HOME_CHANNEL"
    assert captured["pii_safe"] is True and captured["allow_update_command"] is True
    assert callable(captured["standalone_sender_fn"]) and callable(captured["setup_fn"])
    for k in ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"):
        monkeypatch.delenv(k, raising=False)
    validate = captured["validate_config"]
    assert validate(PlatformConfig(enabled=True)) is False
    assert validate(PlatformConfig(enabled=True, extra={"http_url": "http://127.0.0.1:1", "account": "+15550001"}))
