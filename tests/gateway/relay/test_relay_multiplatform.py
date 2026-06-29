"""Unit tests for Phase 1.5 multi-platform-per-agent (relay).

Covers the agent half of Shape A (gateway-gateway D-Q1.5b.1 / D-Q1.5c):
  - relay_platform_identities() parsing the GATEWAY_RELAY_PLATFORMS list +
    GATEWAY_RELAY_BOT_IDS keyed map (the cut-over shape — no scalar fallback),
  - relay_bot_username() reading the per-platform username,
  - self_provision_relay() looping one /relay/provision POST per platform under
    one gatewayId + one secret, partial-failure-tolerant,
  - the RelayAdapter stamping the per-frame egress platform on outbound from the
    chat's inbound source.platform.

The connector HTTP is monkeypatched; the cross-repo E2E exercises the real path.
"""

from __future__ import annotations

import json

import pytest

import gateway.relay as relay


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "GATEWAY_RELAY_URL",
        "GATEWAY_RELAY_ID",
        "GATEWAY_RELAY_SECRET",
        "GATEWAY_RELAY_DELIVERY_KEY",
        "GATEWAY_RELAY_PLATFORM",
        "GATEWAY_RELAY_BOT_ID",
        "GATEWAY_RELAY_PLATFORMS",
        "GATEWAY_RELAY_BOT_IDS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {}, raising=False)


# ─────────────────────────── identity parsing ───────────────────────────

def test_identities_default_relay_when_unconfigured():
    assert relay.relay_platform_identities() == [("relay", "")]
    # The primary helper mirrors the first identity.
    assert relay.relay_platform_identity() == ("relay", "")


def test_identities_single_platform(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")
    monkeypatch.setenv("GATEWAY_RELAY_BOT_IDS", json.dumps({"discord": {"botId": "app-1"}}))
    assert relay.relay_platform_identities() == [("discord", "app-1")]
    assert relay.relay_platform_identity() == ("discord", "app-1")


def test_identities_multi_platform_keyed_map(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord, telegram")
    monkeypatch.setenv(
        "GATEWAY_RELAY_BOT_IDS",
        json.dumps(
            {
                "discord": {"botId": "app-1"},
                "telegram": {"botId": "bot-9", "username": "@my_bot"},
            }
        ),
    )
    # Order preserved; whitespace in the list trimmed.
    assert relay.relay_platform_identities() == [("discord", "app-1"), ("telegram", "bot-9")]
    # The PRIMARY is the first listed platform.
    assert relay.relay_platform_identity() == ("discord", "app-1")
    # Username folded into the per-platform entry; the leading @ is stripped.
    assert relay.relay_bot_username("telegram") == "my_bot"
    assert relay.relay_bot_username("discord") is None


def test_identities_platform_missing_from_map_gets_empty_bot_id(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord,telegram")
    monkeypatch.setenv("GATEWAY_RELAY_BOT_IDS", json.dumps({"discord": {"botId": "app-1"}}))
    # telegram is listed but absent from the ids map ⇒ empty bot_id (the
    # connector rejects an unprovisioned platform with a structured failure).
    assert relay.relay_platform_identities() == [("discord", "app-1"), ("telegram", "")]


def test_bot_ids_malformed_json_degrades_to_empty(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")
    monkeypatch.setenv("GATEWAY_RELAY_BOT_IDS", "{not valid json")
    # A bad map must not crash boot — degrades to empty bot ids.
    assert relay.relay_platform_identities() == [("discord", "")]


# ─────────────────────────── provision loop ───────────────────────────

def _arm(monkeypatch, *, url="wss://connector.example/relay", token="nas-token"):
    monkeypatch.setattr(relay, "relay_url", lambda: url)
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: token)


def test_self_provision_loops_per_platform(monkeypatch):
    _arm(monkeypatch)
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord,telegram")
    monkeypatch.setenv(
        "GATEWAY_RELAY_BOT_IDS",
        json.dumps({"discord": {"botId": "app-1"}, "telegram": {"botId": "bot-9"}}),
    )
    calls = []

    def _fake(**kwargs):
        calls.append((kwargs["platform"], kwargs["bot_id"], kwargs["gateway_id"]))
        return {"secret": "s" * 64, "deliveryKey": "d" * 64, "tenant": "t", "gatewayId": kwargs["gateway_id"]}

    monkeypatch.setattr(relay, "_post_provision", _fake)
    assert relay.self_provision_relay() is True
    # One POST per fronted platform, all under the SAME gatewayId.
    assert [(p, b) for p, b, _ in calls] == [("discord", "app-1"), ("telegram", "bot-9")]
    assert len({gw for _, _, gw in calls}) == 1
    # The in-process secret is set once (from the first success).
    import os

    assert os.environ["GATEWAY_RELAY_SECRET"] == "s" * 64


def test_self_provision_partial_failure_tolerant(monkeypatch):
    _arm(monkeypatch)
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord,telegram")
    monkeypatch.setenv(
        "GATEWAY_RELAY_BOT_IDS",
        json.dumps({"discord": {"botId": "app-1"}, "telegram": {"botId": "bot-9"}}),
    )

    def _fake(**kwargs):
        if kwargs["platform"] == "telegram":
            raise RuntimeError("telegram provision boom")
        return {"secret": "s" * 64, "deliveryKey": "d" * 64, "tenant": "t", "gatewayId": kwargs["gateway_id"]}

    monkeypatch.setattr(relay, "_post_provision", _fake)
    # discord succeeds, telegram fails ⇒ still True (at least one fronted).
    assert relay.self_provision_relay() is True


def test_self_provision_all_fail_returns_false(monkeypatch):
    _arm(monkeypatch)
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord,telegram")

    def _fake(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(relay, "_post_provision", _fake)
    assert relay.self_provision_relay() is False


# ─────────────────────────── per-frame egress (adapter) ───────────────────────────

@pytest.mark.asyncio
async def test_adapter_stamps_per_frame_platform_from_inbound(monkeypatch):
    """An inbound from a concrete platform makes the reply egress tagged for it."""
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
    from gateway.session import SessionSource

    from tests.gateway.relay.stub_connector import StubConnector

    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="relay",
        label="Relay",
        max_message_length=4096,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=False,
        markdown_dialect="plain",
        len_unit="chars",
    )
    stub = StubConnector(descriptor)
    # This gateway fronts both discord and telegram.
    stub._identities = [("discord", "app-1"), ("telegram", "bot-9")]
    adapter = RelayAdapter(PlatformConfig(), descriptor, transport=stub)
    await adapter.connect()

    # A telegram inbound for chat "tg-1".
    await stub.push_inbound(
        MessageEvent(
            text="hi",
            message_type=MessageType.TEXT,
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="tg-1", chat_type="dm", user_id="u-1"),
        )
    )
    await adapter.send("tg-1", "a telegram reply")
    # The reply was tagged for telegram (per-frame egress).
    assert stub.sent_platforms[-1] == "telegram"

    # A discord inbound for chat "dc-1".
    await stub.push_inbound(
        MessageEvent(
            text="yo",
            message_type=MessageType.TEXT,
            source=SessionSource(platform=Platform.DISCORD, chat_id="dc-1", chat_type="channel", guild_id="g-1"),
        )
    )
    await adapter.send("dc-1", "a discord reply")
    assert stub.sent_platforms[-1] == "discord"


@pytest.mark.asyncio
async def test_adapter_untagged_when_chat_platform_unknown(monkeypatch):
    """A reply to a chat we never saw inbound for carries no per-frame platform
    (the connector falls back to the session default)."""
    from gateway.config import Platform, PlatformConfig
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

    from tests.gateway.relay.stub_connector import StubConnector

    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="relay",
        label="Relay",
        max_message_length=4096,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=False,
        markdown_dialect="plain",
        len_unit="chars",
    )
    stub = StubConnector(descriptor)
    adapter = RelayAdapter(PlatformConfig(), descriptor, transport=stub)
    await adapter.connect()
    await adapter.send("never-seen", "reply")
    assert stub.sent_platforms[-1] is None
