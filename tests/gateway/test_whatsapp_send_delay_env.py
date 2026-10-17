"""WHATSAPP_SEND_DELAY_MS must ride _BRIDGE_PASSTHROUGH_ENV.

The Node bridge reads its optional pre-send throttle delay (WHATSAPP_SEND_DELAY_MS,
see createSendThrottle in scripts/whatsapp-bridge/bridge_helpers.js) from its own
process env, and the adapter only forwards vars named in the passthrough tuple.
A var wired in the bridge but missing from the tuple silently keeps every
outgoing send at machine speed on env-configured installs.
"""


def test_send_delay_rides_the_bridge_passthrough_env():
    from plugins.platforms.whatsapp.adapter import _BRIDGE_PASSTHROUGH_ENV

    assert "WHATSAPP_SEND_DELAY_MS" in _BRIDGE_PASSTHROUGH_ENV