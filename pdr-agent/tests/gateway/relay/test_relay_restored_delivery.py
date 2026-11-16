"""A session revived from durable state delivers through the Relay that fronts its platform.

A relayed source keeps the LOGICAL platform (``discord``) and marks relay provenance with the
wire-invisible ``delivered_via_upstream_relay`` flag, which is never persisted. The Relay is
registered under ``Platform.RELAY``. After a restart the restored source therefore matched no
adapter in ``_delivery_adapter_for`` and startup auto-resume skipped the session with "adapter not
ready", leaving it ``resume_pending`` for good (heartbeats and plugin injection used the same
selector). Reported by @andrexibiza in the #128797 review.

Real objects throughout: ``GatewayRunner(config)``, ``WebSocketRelayTransport``, ``RelayAdapter``,
``SessionStore.mark_resume_pending`` and a second runner over the same home running the real
scheduler; only the model turn is replaced by a reply sent through the dispatched adapter.
"""

import asyncio

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.ws_transport import WebSocketRelayTransport
from gateway.session import SessionSource
from gateway.session_identity import resolve_identity

CHAT, OWNER = "chan-1", "owner-1"


class _NativeDiscord(BasePlatformAdapter):
    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="native")

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}


@pytest.fixture
def gateway_home(tmp_path, monkeypatch):
    home = tmp_path / "hh"
    home.mkdir()
    # Multiplexed authorization reads the receiving bot's home; standalone reads the env.
    (home / ".env").write_text(f"DISCORD_ALLOWED_USERS={OWNER}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", OWNER)
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    return home


def _runner(home, *, multiplex, native=False):
    platforms = {Platform.RELAY: PlatformConfig(enabled=True)}
    if native:
        platforms[Platform.DISCORD] = PlatformConfig(enabled=True)
    config = GatewayConfig(platforms=platforms, multiplex_profiles=multiplex)
    config.sessions_dir = home / "sessions"
    runner = gateway_run.GatewayRunner(config)
    transport = WebSocketRelayTransport(
        "wss://connector.example/relay", "discord", "app-1", identities=[("discord", "app-1")])
    relay = RelayAdapter(PlatformConfig(enabled=True), CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="discord", label="Discord", max_message_length=2000,
        supports_draft_streaming=False, supports_edit=True, supports_threads=False,
        markdown_dialect="plain", len_unit="chars",
    ), transport=transport)
    runner._publish_primary_adapter(Platform.RELAY, relay)
    discord = None
    if native:
        discord = _NativeDiscord(PlatformConfig(enabled=True), Platform.DISCORD)
        runner._publish_primary_adapter(Platform.DISCORD, discord)
    return runner, relay, discord


def _interrupt_relayed_session(home, *, multiplex):
    """A previous process received a relayed DM and was restarted mid-turn."""
    runner, relay, _ = _runner(home, multiplex=multiplex)
    source = SessionSource(platform=Platform.DISCORD, chat_id=CHAT, chat_type="dm", user_id=OWNER,
                           delivered_via_upstream_relay=True)
    resolve_identity(source, runner=runner, adapter=relay)
    entry = runner.session_store.get_or_create_session(source)
    assert runner.session_store.mark_resume_pending(entry.session_key, "restart_timeout")
    return entry.session_key


def _record_dispatch(adapter, dispatched):
    async def handle_message(event):
        dispatched.append((adapter, event))
        # The turn's final reply leaves through the adapter that runs it.
        await adapter.send(event.source.chat_id, "resumed")
    adapter.handle_message = handle_message


@pytest.mark.asyncio
@pytest.mark.parametrize("multiplex", [False, True], ids=["standalone", "multiplexed-primary"])
@pytest.mark.parametrize("trigger", ["startup", "relay-reconnect"])
async def test_restored_relay_session_resumes_and_replies_through_the_relay(gateway_home, multiplex, trigger):
    key = _interrupt_relayed_session(gateway_home, multiplex=multiplex)
    runner, relay, _ = _runner(gateway_home, multiplex=multiplex)
    frames, dispatched = [], []

    async def send_outbound(action, *, platform=None):
        frames.append((platform, action))
        return {"success": True, "message_id": "relayed"}

    relay._transport.send_outbound = send_outbound
    _record_dispatch(relay, dispatched)

    scheduled = runner._schedule_resume_pending_sessions(
        platform=Platform.RELAY if trigger == "relay-reconnect" else None)
    await asyncio.gather(*list(runner._background_tasks))

    assert scheduled == 1, "a restored relayed session is resumed through the Relay that fronts it"
    assert [(adapter, runner._session_key_for_source(event.source)) for adapter, event in dispatched] \
        == [(relay, key)]
    # Cold egress caches after the restart: the reply must still name its platform and owner, or
    # the connector's tenant guard declines it.
    assert [(platform, action["chat_id"], action["metadata"].get("user_id")) for platform, action in frames] \
        == [("discord", CHAT, OWNER)]


@pytest.mark.asyncio
async def test_restored_delivery_never_takes_the_relay_over_a_native_bot_or_for_a_secondary(
    gateway_home, monkeypatch,
):
    # A live native Discord adapter outranks a Relay that also fronts Discord.
    _interrupt_relayed_session(gateway_home, multiplex=False)
    runner, relay, discord = _runner(gateway_home, multiplex=False, native=True)
    dispatched = []
    _record_dispatch(relay, dispatched)
    _record_dispatch(discord, dispatched)
    assert runner._schedule_resume_pending_sessions() == 1
    await asyncio.gather(*list(runner._background_tasks))
    assert [adapter for adapter, _event in dispatched] == [discord]

    # A lane received by a secondary profile's bot, now offline, never borrows the primary's Relay.
    secondary = gateway_home / "profiles" / "team_b"
    secondary.mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir",
                        lambda name: gateway_home if name == "default" else gateway_home / "profiles" / name)
    key = _interrupt_relayed_session(gateway_home, multiplex=True)
    runner, relay, _ = _runner(gateway_home, multiplex=True)
    runner._profile_adapters = {"team_b": {}}
    runner.session_store._ensure_loaded()
    entry = runner.session_store._entries[key]
    entry.transport_profile = "team_b"
    dispatched = []
    _record_dispatch(relay, dispatched)
    assert runner._delivery_adapter_for(runner._restored_source(entry)) is None
    assert runner._schedule_resume_pending_sessions() == 0
    assert dispatched == []
