"""Outbound recipient gate for the WhatsApp bridge send endpoints.

Inbound traffic has always been filtered by WHATSAPP_ALLOWED_USERS, but the
Node.js bridge's HTTP send endpoints (/send, /send-media, /send-poll,
/send-location, /edit) trusted whatever chatId the caller passed.  Any local
process able to reach the loopback bridge — a stray curl, a debug script
working around the Python gateway — could message arbitrary contacts of the
*linked* WhatsApp account.

These tests pin the policy contract the bridge enforces (see
scripts/whatsapp-bridge/outbound_gate.js): recipients are limited to the
linked account itself, WHATSAPP_ALLOWED_USERS, WHATSAPP_HOME_CHANNEL,
WHATSAPP_GROUP_ALLOWED_USERS and WHATSAPP_OUTBOUND_ALLOWED entries; unknown
targets are denied fail-closed.  WHATSAPP_OUTBOUND_ALLOW_ALL=1 restores the
previous open behaviour for deployments that intentionally deliver to
arbitrary chats.

The Node-side semantics are covered directly by
scripts/whatsapp-bridge/outbound_gate.test.mjs; here we verify the Python
side passes the gate's env contract through to spawned bridges.
"""


from plugins.platforms.whatsapp.adapter import _BRIDGE_PASSTHROUGH_ENV


def test_outbound_gate_env_contract_is_passed_to_bridges():
    # A multiplexed bridge subprocess must see the outbound gate variables,
    # otherwise secondary profiles silently run with the wrong egress policy.
    assert "WHATSAPP_OUTBOUND_ALLOWED" in _BRIDGE_PASSTHROUGH_ENV
    assert "WHATSAPP_OUTBOUND_ALLOW_ALL" in _BRIDGE_PASSTHROUGH_ENV
    # The gate falls back to the home channel; without passthrough a forked
    # bridge could never deliver there.
    assert "WHATSAPP_HOME_CHANNEL" in _BRIDGE_PASSTHROUGH_ENV
