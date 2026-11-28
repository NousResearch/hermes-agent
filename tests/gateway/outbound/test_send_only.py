"""``StandaloneSendPolicy.sends_while_disabled``: a send-only home (credentials, platform disabled) delivers
through the plugin's registered standalone sender, with a fake sender and no platform SDK."""

import json

import pytest


@pytest.mark.parametrize(("policy", "connected", "sent"), [
    ({"sends_while_disabled": True, "all_sends": True}, True, True),
    ({"sends_while_disabled": True, "all_sends": True}, False, False),
    ({"sends_while_disabled": True, "all_sends": True}, RuntimeError("probe broke"), False),
    ({"sends_while_disabled": True}, True, False),  # without all_sends a live adapter/handler could run
    ({"all_sends": True}, True, False),
])
def test_a_send_only_home_delivers_through_the_plugins_standalone_sender(monkeypatch, tmp_path, policy, connected, sent):
    """Credentials but ``platforms.<name>.enabled: false``: send_message reaches the plugin's registered
    standalone sender with the platform's OWN config (api_key kept), only when the policy opts in with
    ``all_sends`` and ``is_connected`` sees credentials; else, or when the probe raises, a JSON error."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.outbound import send as send_mod
    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.platforms.send_targets import StandaloneSendPolicy

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []

    async def sender(pconfig, chat_id, message, **kw):
        calls.append((pconfig.enabled, pconfig.api_key, pconfig.extra, chat_id, message))
        return {"success": True}

    def is_connected(cfg):
        if isinstance(connected, Exception):
            raise connected
        return connected

    name = "sendonlyprobe"
    platform_registry.register(PlatformEntry(
        name=name, label="Send-only probe", adapter_factory=lambda cfg: None, check_fn=lambda: True,
        source="plugin", is_connected=is_connected, standalone_sender_fn=sender,
        target_ref_syntax_fn=lambda ref: (ref.strip(), None),
        standalone_send=StandaloneSendPolicy(**policy)))
    config = GatewayConfig(platforms={Platform(name): PlatformConfig(enabled=False, api_key="k1", extra={"a": "b"})})
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    try:
        result = json.loads(send_mod.send_message_tool({"action": "send", "target": f"{name}:chat-1", "message": "hi"}))
    finally:
        platform_registry.unregister(name)
    if sent:
        assert result.get("success") is True
        assert calls == [(True, "k1", {"a": "b"}, "chat-1", "hi")]
    else:
        assert calls == [] and "not configured" in result["error"].lower()
