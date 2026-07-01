"""Phase 5 (drain-window recovery): incident-reproduction E2E — the real proof.

Two variants, each with a mutation gate that must go RED when the fix is
removed:

- Variant A — the OBSERVED incident: a bot-post channel (#logs-like) whose
  only prior activity was an outbound send, receives a user message that lands
  in channel.history but never in the transcript. After reconnect the backfill
  must process it exactly once (through the auth path), and a second reconnect
  must recover nothing. Mutation: restart_backfill off → 0 recovered.

- Variant B — cold channel recovered SOLELY via the inbound drain-mark: a
  channel with zero prior activity is messaged DURING the drain (handler drops
  it). The channel lands in the durable active map only because the inbound
  mark ran before the drop (D-8); on reconnect the backfill scans it and
  recovers the message exactly once. Mutation: no inbound mark (empty map) →
  0 recovered.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2e"))


class _FakeAuthor:
    def __init__(self, author_id, name="alice", bot=False):
        self.id = author_id
        self.name = name
        self.display_name = name
        self.bot = bot

    def __eq__(self, other):
        return getattr(other, "id", None) == self.id

    def __hash__(self):
        return hash(self.id)


class _FakeChannel:
    def __init__(self, channel_id, messages=None, guild_id=44444, name="logs"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(id=guild_id, name="srv")
        self._messages = messages or []
        self.history_calls = []
        self.send = AsyncMock(return_value=SimpleNamespace(id=999, channel=None))

    def history(self, *, after=None, oldest_first=True, limit=50):
        self.history_calls.append({"after": after, "limit": limit})
        msgs = list(self._messages)
        if not oldest_first:
            msgs = list(reversed(msgs))

        async def _gen():
            for m in msgs[:limit]:
                yield m
        return _gen()


def _make_user_msg(msg_id, author, channel, content="are you there?"):
    import discord
    return SimpleNamespace(
        id=msg_id, content=content, clean_content=content,
        author=author, channel=channel, guild=channel.guild,
        mentions=[], attachments=[], type=discord.MessageType.default,
        reference=None,
    )


@pytest.fixture()
def _tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _build(tmp_path):
    from tests.e2e.conftest import _make_discord_adapter_wired
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_state import SessionDB
    from plugins.platforms.discord.adapter import _DiscordRestartRecoveryState

    adapter, runner = _make_discord_adapter_wired()
    adapter._restart_recovery = _DiscordRestartRecoveryState(persist_interval_s=0.0)

    db = SessionDB(db_path=tmp_path / "state.db")
    config = GatewayConfig()
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    store._db = db
    store._loaded = True
    adapter._session_store = store
    return adapter, store, db


# ═══════════════════════════════════════════════════════════════════════════
# Variant A — the observed incident (outbound-marked bot-post channel)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_variantA_incident_recovers_exactly_once(_tmp_home):
    adapter, store, db = _build(_tmp_home)
    bot = _FakeAuthor(99999, "bot", bot=True)
    human = _FakeAuthor(11111, "ace")
    channel = _FakeChannel(1480528231286181948, name="alerts")

    adapter._client = SimpleNamespace(
        user=bot,
        get_channel=lambda _id: channel if int(_id) == channel.id else None,
        fetch_channel=AsyncMock(return_value=channel),
    )

    # (1) The bot posts an alert to the channel — a conversational outbound
    # send marks the channel active (the ONLY prior activity).
    from plugins.platforms.discord.adapter import DiscordAdapter
    await DiscordAdapter.send(adapter, chat_id=str(channel.id),
                              content="⚠️ something needs a look")
    assert str(channel.id) in adapter._restart_recovery.recent_channels(10_000)

    # (2) The user replies during the restart drain window — that message lands
    # in channel.history but is NEVER dispatched/persisted (gateway dropped it).
    user_msg = _make_user_msg(70100, human, channel, content="on it — what broke?")
    channel._messages = [user_msg]

    # Re-injection sink: spy on the full auth path.
    handled = []

    async def _handle(message, role_authorized=False):
        handled.append(message.id)
        # simulate downstream persistence into the transcript
        sid = adapter._resolve_session_id_for_message(message)
        if sid is None:
            # first contact: create the session so the 2nd sweep can dedup
            src = adapter.build_source(
                chat_id=str(channel.id), chat_name=channel.name, chat_type="group",
                user_id=str(message.author.id), guild_id=str(channel.guild.id),
            )
            key = store._generate_session_key(src)
            db.create_session(session_id="sessA", source="discord")
            from gateway.session import SessionEntry
            from gateway.config import Platform
            import datetime as _dt
            store._entries[key] = SessionEntry(
                session_key=key, session_id="sessA",
                created_at=_dt.datetime.now(), updated_at=_dt.datetime.now(),
                origin=src, display_name=channel.name,
                platform=Platform.DISCORD, chat_type="group",
            )
            sid = "sessA"
        db.append_message(session_id=sid, role="user", content=message.content,
                          platform_message_id=str(message.id))

    adapter._handle_message = _handle
    adapter._dispatch_incoming_message = DiscordAdapter._dispatch_incoming_message.__get__(adapter)
    adapter._is_allowed_user = lambda uid, a=None, guild=None, is_dm=False: True

    # (3) Reconnect → backfill sweep recovers the message exactly once.
    with patch.dict(os.environ, {"DISCORD_IGNORE_NO_MENTION": "false"}):
        await adapter._backfill_missed_messages()
    assert handled == [70100], "the dropped incident message must be recovered once"

    # (4) A SECOND reconnect recovers nothing (now present in the transcript).
    with patch.dict(os.environ, {"DISCORD_IGNORE_NO_MENTION": "false"}):
        await adapter._backfill_missed_messages()
    assert handled == [70100], "a second reconnect must not re-process it"


@pytest.mark.asyncio
async def test_variantA_mutation_backfill_off_recovers_nothing(_tmp_home):
    """Mutation gate: with restart_backfill disabled, the incident message is
    NOT recovered (proves the recovery is what does the work)."""
    adapter, store, db = _build(_tmp_home)
    bot = _FakeAuthor(99999, "bot", bot=True)
    human = _FakeAuthor(11111, "ace")
    channel = _FakeChannel(1480528231286181948, name="alerts")
    adapter._client = SimpleNamespace(
        user=bot, get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    from plugins.platforms.discord.adapter import DiscordAdapter
    await DiscordAdapter.send(adapter, chat_id=str(channel.id), content="⚠️ alert")
    channel._messages = [_make_user_msg(70100, human, channel)]

    handled = []

    async def _handle(message, role_authorized=False):
        handled.append(message.id)
    adapter._handle_message = _handle
    adapter._dispatch_incoming_message = DiscordAdapter._dispatch_incoming_message.__get__(adapter)
    adapter._is_allowed_user = lambda uid, a=None, guild=None, is_dm=False: True

    with patch.dict(os.environ, {"DISCORD_RESTART_BACKFILL": "false",
                                 "DISCORD_IGNORE_NO_MENTION": "false"}):
        await adapter._backfill_missed_messages()
    assert handled == [], "backfill off ⇒ nothing recovered (mutation RED)"


# ═══════════════════════════════════════════════════════════════════════════
# Variant B — cold channel recovered SOLELY via the inbound drain-mark
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_variantB_cold_channel_recovered_via_inbound_mark(_tmp_home):
    adapter, store, db = _build(_tmp_home)
    bot = _FakeAuthor(99999, "bot", bot=True)
    human = _FakeAuthor(11111, "ace")
    # Cold channel: ZERO prior activity (no outbound send, nothing in the map).
    channel = _FakeChannel(555123, name="cold-general")

    adapter._client = SimpleNamespace(
        user=bot, get_channel=lambda _id: channel if int(_id) == channel.id else None,
        fetch_channel=AsyncMock(return_value=channel),
    )
    adapter._is_allowed_user = lambda uid, a=None, guild=None, is_dm=False: True

    # (1) A user messages the cold channel DURING the drain. The gateway drops
    # it (handler raises, as during drain), but the inbound mark at the top of
    # _handle_message fires FIRST (D-8).
    async def _dropping_handler(event):
        raise RuntimeError("draining — dropped")
    adapter.set_message_handler(_dropping_handler)
    adapter.send = AsyncMock()

    user_msg = _make_user_msg(70200, human, channel, content=f"<@{bot.id}> urgent")
    user_msg.mentions = [bot]
    with patch.dict(os.environ, {"DISCORD_AUTO_THREAD": "false"}):
        try:
            await adapter._handle_message(user_msg, role_authorized=False)
        except Exception:
            pass

    # (2) The channel is in the durable active map SOLELY because of the inbound
    # mark — even though the message was never dispatched or persisted.
    assert str(channel.id) in adapter._restart_recovery.recent_channels(10_000), \
        "cold channel must be in the map via the inbound drain-mark (D-8)"

    # (3) On reconnect the backfill scans that channel (because it's marked) and
    # recovers the message exactly once.
    channel._messages = [user_msg]
    handled = []

    async def _handle(message, role_authorized=False):
        handled.append(message.id)
    adapter._handle_message = _handle
    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter._dispatch_incoming_message = DiscordAdapter._dispatch_incoming_message.__get__(adapter)
    # reset dedup so the reinjection isn't swallowed as a replay of step 1
    adapter._dedup._seen.clear() if hasattr(adapter._dedup, "_seen") else None

    await adapter._backfill_missed_messages()
    assert handled == [70200], "cold-channel message recovered via the inbound mark"


@pytest.mark.asyncio
async def test_variantB_mutation_no_inbound_mark_recovers_nothing(_tmp_home):
    """Mutation gate: if the inbound mark never ran (channel absent from the
    map — equivalent to moving the mark below the drain-drop), the cold channel
    is not scanned and nothing is recovered."""
    adapter, store, db = _build(_tmp_home)
    bot = _FakeAuthor(99999, "bot", bot=True)
    human = _FakeAuthor(11111, "ace")
    channel = _FakeChannel(555123, name="cold-general")
    channel._messages = [_make_user_msg(70200, human, channel)]

    adapter._client = SimpleNamespace(
        user=bot, get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    handled = []

    async def _handle(message, role_authorized=False):
        handled.append(message.id)
    adapter._handle_message = _handle
    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter._dispatch_incoming_message = DiscordAdapter._dispatch_incoming_message.__get__(adapter)

    # NO mark_channel_active call → the map is empty (simulates the mark being
    # below the drop / never firing).
    await adapter._backfill_missed_messages()
    assert handled == [], "no inbound mark ⇒ cold channel not scanned (mutation RED)"
    assert channel.history_calls == [], "an unmarked channel is never scanned"
