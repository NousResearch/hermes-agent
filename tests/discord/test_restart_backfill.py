"""Phase 3 (drain-window recovery): the reconnect backfill sweep.

Covers the core recovery engine _backfill_missed_messages + the INV-9
answerability gate, the shared partition filter, single-flight, caps,
ordering, fail-open, and the transcript-authority dedup. The full incident
E2E (Variants A/B with mutation gates) lives in test_restart_backfill_e2e.py
(Phase 5); this file unit-proves each invariant in isolation.
"""
import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2e"))


# ── Fakes ───────────────────────────────────────────────────────────────────

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
    """A discord text channel double whose .history() yields a fixed list."""
    def __init__(self, channel_id, messages, guild_id=44444, name="logs"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(id=guild_id, name="srv")
        self._messages = messages
        self.history_calls = []

    def history(self, *, after=None, oldest_first=True, limit=50):
        self.history_calls.append({"after": after, "oldest_first": oldest_first, "limit": limit})
        msgs = list(self._messages)
        if not oldest_first:
            msgs = list(reversed(msgs))
        msgs = msgs[:limit]

        async def _gen():
            for m in msgs:
                yield m
        return _gen()


def _make_msg(msg_id, author, channel, content="hi", mtype=None):
    import discord
    return SimpleNamespace(
        id=msg_id,
        content=content,
        clean_content=content,
        author=author,
        channel=channel,
        guild=getattr(channel, "guild", None),
        mentions=[],
        attachments=[],
        type=mtype if mtype is not None else discord.MessageType.default,
        reference=None,
    )


@pytest.fixture()
def _tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _build_adapter_with_store(tmp_path):
    """Adapter wired to a REAL SessionStore + SessionDB on a temp path."""
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

    # Re-inject sink: capture dispatched messages instead of running the gateway.
    dispatched = []

    async def _capture(msg):
        dispatched.append(msg)

    adapter._dispatch_incoming_message = _capture
    adapter._dispatched = dispatched
    return adapter, store, db


def _seed_session_for_channel(adapter, store, db, channel, author):
    """Create a session mapping for `channel` and return its session_id."""
    msg = _make_msg(1, author, channel)
    source = adapter.build_source(
        chat_id=str(channel.id),
        chat_name=channel.name,
        chat_type="group",
        user_id=str(author.id),
        user_name=author.display_name,
        guild_id=str(channel.guild.id),
    )
    session_key = store._generate_session_key(source)
    session_id = f"sess-{channel.id}"
    db.create_session(session_id=session_id, source="discord")
    from gateway.session import SessionEntry
    from gateway.config import Platform
    store._entries[session_key] = SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=__import__("datetime").datetime.now(),
        updated_at=__import__("datetime").datetime.now(),
        origin=source,
        display_name=channel.name,
        platform=Platform.DISCORD,
        chat_type="group",
    )
    return session_id


@pytest.mark.asyncio
async def test_normal_turn_persisted_id_makes_backfill_skip_it(_tmp_home):
    """B1 / RC#1 (Opus code review): the exactly-once guarantee for a message
    that was NORMALLY processed just before the restart. Prove end-to-end that:
    (a) a user turn persisted via the REAL flush path carries a non-NULL
    platform_message_id in its transcript row (the normal build_turn_context
    stamp, NOT the interrupted-turn override), and (b) backfill then SKIPS it
    (recovered=0, skipped_in_transcript=1) instead of re-processing it into a
    duplicate reply.

    This closes the reviewer's B1: without it, the exactly-once claim is a
    happy-path assertion. A completed-pre-SIGTERM message with a NULL id would
    read as 'absent' and get re-answered.
    """
    from run_agent import AIAgent
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    author = _FakeAuthor(11111)
    channel = _FakeChannel(987500, [])
    msg = _make_msg(70500, author, channel, content="a normally-processed message")
    channel._messages = [msg]
    session_id = _seed_session_for_channel(adapter, store, db, channel, author)

    # (a) Persist the user turn through the REAL _flush_messages_to_session_db,
    # with the id stamped on the user dict exactly as build_turn_context does on
    # the NORMAL path (no _persist_user_message_* override attrs set → the
    # override arm is a no-op; the id rides on the dict itself).
    class _StubAgent:
        _session_db = db
        _session_db_created = True
        _last_flushed_db_idx = 0
        _flushed_db_message_session_id = None
        _persist_user_message_idx = None
        _persist_user_message_override = None
        _persist_user_message_timestamp = None
        _persist_user_message_platform_id = None  # NORMAL path: no override

        def _ensure_db_session(self):
            pass

        _apply_persist_user_message_override = AIAgent._apply_persist_user_message_override
        _flush_messages_to_session_db = AIAgent._flush_messages_to_session_db

    stub = _StubAgent()
    stub.session_id = session_id
    # build_turn_context stamps user_msg["platform_message_id"] on every turn.
    normal_turn = [{"role": "user", "content": "a normally-processed message",
                    "platform_message_id": "70500"}]
    stub._flush_messages_to_session_db(normal_turn, conversation_history=[])

    # Row carries the id (not NULL) — the authority answers True.
    assert db.has_platform_message_id(session_id, "70500") is True

    # (b) Backfill over the same channel must SKIP it — recovered nothing.
    adapter._restart_recovery.mark_channel_active(str(channel.id))
    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    await adapter._backfill_missed_messages()
    assert adapter._dispatched == [], "a normally-processed message must not be re-injected"


# ── Tests ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_backfill_recovers_message_absent_from_transcript(_tmp_home):
    """AC-3b: a message in channel.history but absent from the transcript IS
    re-injected."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    author = _FakeAuthor(11111)
    channel = _FakeChannel(987001, [])
    msg = _make_msg(70001, author, channel, content="did you see this?")
    channel._messages = [msg]
    _seed_session_for_channel(adapter, store, db, channel, author)

    adapter._restart_recovery.mark_channel_active(str(channel.id))
    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )

    await adapter._backfill_missed_messages()
    assert [m.id for m in adapter._dispatched] == [70001]


@pytest.mark.asyncio
async def test_backfill_skips_message_already_in_transcript(_tmp_home):
    """AC-3: a message already persisted (has_platform_message_id True) is NOT
    re-injected."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    author = _FakeAuthor(11111)
    channel = _FakeChannel(987002, [])
    msg = _make_msg(70002, author, channel)
    channel._messages = [msg]
    session_id = _seed_session_for_channel(adapter, store, db, channel, author)

    # Persist the message into the transcript WITH its platform_message_id.
    db.append_message(
        session_id=session_id, role="user", content="hi",
        platform_message_id="70002",
    )

    adapter._restart_recovery.mark_channel_active(str(channel.id))
    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )

    await adapter._backfill_missed_messages()
    assert adapter._dispatched == []


@pytest.mark.asyncio
async def test_backfill_when_transcript_authority_unavailable_does_not_double_inject(_tmp_home):
    """AC-15 / INV-9: when the authority cannot answer (lookup raises), the
    channel's recovery is skipped — never recover on 'don't know'."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    author = _FakeAuthor(11111)
    channel = _FakeChannel(987003, [])
    channel._messages = [_make_msg(70003, author, channel)]
    _seed_session_for_channel(adapter, store, db, channel, author)

    # Make the authority UNANSWERABLE: the DB lookup raises.
    def _boom(*a, **k):
        raise RuntimeError("db down")
    db.has_platform_message_id = _boom

    adapter._restart_recovery.mark_channel_active(str(channel.id))
    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )

    await adapter._backfill_missed_messages()
    assert adapter._dispatched == [], "must not recover when the authority can't answer"


@pytest.mark.asyncio
async def test_backfill_skips_self_and_banners(_tmp_home):
    """AC-5 / INV-2: the bot's own messages, system messages, and banners are
    not re-injected."""
    import discord
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    bot = _FakeAuthor(99999, "bot", bot=True)
    human = _FakeAuthor(11111)
    channel = _FakeChannel(987004, [])
    _seed_session_for_channel(adapter, store, db, channel, human)

    own = _make_msg(70010, bot, channel, content="my own reply")
    system = _make_msg(70011, human, channel, content="joined", mtype=99)
    real = _make_msg(70012, human, channel, content="a real question")
    channel._messages = [own, system, real]

    adapter._restart_recovery.mark_channel_active(str(channel.id))
    adapter._client = SimpleNamespace(
        user=bot, get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    await adapter._backfill_missed_messages()
    assert [m.id for m in adapter._dispatched] == [70012]


@pytest.mark.asyncio
async def test_backfill_preserves_order(_tmp_home):
    """AC-8 / INV-5: a multi-message burst re-injects oldest-first."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    author = _FakeAuthor(11111)
    channel = _FakeChannel(987005, [])
    _seed_session_for_channel(adapter, store, db, channel, author)
    channel._messages = [
        _make_msg(70020, author, channel, content="first"),
        _make_msg(70021, author, channel, content="second"),
        _make_msg(70022, author, channel, content="third"),
    ]
    adapter._restart_recovery.mark_channel_active(str(channel.id))
    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    await adapter._backfill_missed_messages()
    assert [m.id for m in adapter._dispatched] == [70020, 70021, 70022]
    # oldest_first must be requested
    assert channel.history_calls[0]["oldest_first"] is True


@pytest.mark.asyncio
async def test_backfill_respects_per_channel_and_aggregate_caps(_tmp_home, monkeypatch):
    """AC-7 / INV-3: per-channel limit + aggregate max_channels hold."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    author = _FakeAuthor(11111)
    monkeypatch.setenv("DISCORD_RESTART_BACKFILL_LIMIT", "2")
    monkeypatch.setenv("DISCORD_RESTART_BACKFILL_MAX_CHANNELS", "1")

    channels = {}
    _now = __import__("time").time()
    for idx, cid in enumerate((987006, 987007)):
        ch = _FakeChannel(cid, [])
        ch._messages = [_make_msg(cid * 10 + i, author, ch, content=f"m{i}") for i in range(5)]
        _seed_session_for_channel(adapter, store, db, ch, author)
        channels[cid] = ch
        # Recent (within lookback); 987007 marked newest.
        adapter._restart_recovery.mark_channel_active(str(cid), now=_now - 10 + idx)

    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: channels.get(int(_id)),
        fetch_channel=AsyncMock(),
    )
    await adapter._backfill_missed_messages()
    # aggregate cap 1 → only the newest channel scanned; per-channel limit 2.
    assert len(adapter._dispatched) == 2


@pytest.mark.asyncio
async def test_backfill_history_calls_are_sequential(_tmp_home):
    """INV-3: history calls run sequentially (no parallel burst)."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    author = _FakeAuthor(11111)
    concurrent = {"now": 0, "max": 0}

    class _SlowChannel(_FakeChannel):
        def history(self, *, after=None, oldest_first=True, limit=50):
            outer = self

            async def _gen():
                concurrent["now"] += 1
                concurrent["max"] = max(concurrent["max"], concurrent["now"])
                await asyncio.sleep(0.01)
                for m in outer._messages[:limit]:
                    yield m
                concurrent["now"] -= 1
            return _gen()

    channels = {}
    for cid in (987030, 987031, 987032):
        ch = _SlowChannel(cid, [])
        ch._messages = [_make_msg(cid, author, ch)]
        _seed_session_for_channel(adapter, store, db, ch, author)
        channels[cid] = ch
        adapter._restart_recovery.mark_channel_active(str(cid))

    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: channels.get(int(_id)),
        fetch_channel=AsyncMock(),
    )
    await adapter._backfill_missed_messages()
    assert concurrent["max"] == 1, "history scans must not overlap"


@pytest.mark.asyncio
async def test_backfill_idempotent_across_double_reconnect(_tmp_home):
    """AC-1 tail / INV-6: a second sweep recovers nothing new (dedup marks it)."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    author = _FakeAuthor(11111)
    channel = _FakeChannel(987008, [])
    channel._messages = [_make_msg(70040, author, channel)]
    _seed_session_for_channel(adapter, store, db, channel, author)
    adapter._restart_recovery.mark_channel_active(str(channel.id))
    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )

    # First sweep recovers. Use the REAL dispatch to exercise the dedup cache,
    # then simulate the transcript now containing it.
    real_dispatched = []

    async def _dispatch(msg):
        real_dispatched.append(msg.id)
        # Simulate downstream persistence: mark in transcript.
        db.append_message(
            session_id=f"sess-{channel.id}", role="user",
            content="x", platform_message_id=str(msg.id),
        )
    adapter._dispatch_incoming_message = _dispatch

    await adapter._backfill_missed_messages()
    await adapter._backfill_missed_messages()
    assert real_dispatched == [70040], "second reconnect must recover nothing new"


@pytest.mark.asyncio
async def test_backfill_reinject_failure_leaves_message_recoverable(_tmp_home):
    """AC-9 / INV-8: a re-inject that raises leaves the message absent (retried
    next sweep); reinject_failed is observability-only, not a correctness gate."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    author = _FakeAuthor(11111)
    channel = _FakeChannel(987009, [])
    channel._messages = [_make_msg(70050, author, channel)]
    _seed_session_for_channel(adapter, store, db, channel, author)
    adapter._restart_recovery.mark_channel_active(str(channel.id))
    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )

    calls = {"n": 0}

    async def _flaky(msg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        # second sweep succeeds

    adapter._dispatch_incoming_message = _flaky

    # Sweep 1: raises → still absent (no transcript write). Must not crash.
    await adapter._backfill_missed_messages()
    # Sweep 2: recovers it (message still absent from transcript).
    await adapter._backfill_missed_messages()
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_backfill_first_boot_no_state_scans_nothing(_tmp_home):
    """AC-10 tail / D-9: empty active map ⇒ 0 channels scanned, no history call,
    no crash on missing shutdown_ts."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    called = {"history": 0}

    class _Watch(_FakeChannel):
        def history(self, **k):
            called["history"] += 1
            return super().history(**k)

    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: _Watch(1, []),
        fetch_channel=AsyncMock(),
    )
    # No mark_channel_active calls → empty map.
    await adapter._backfill_missed_messages()
    assert called["history"] == 0
    assert adapter._dispatched == []


@pytest.mark.asyncio
async def test_backfill_killswitch_off_no_history_call(_tmp_home, monkeypatch):
    """AC-10 / D-9: restart_backfill=false → no channel.history call at all."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    monkeypatch.setenv("DISCORD_RESTART_BACKFILL", "false")
    author = _FakeAuthor(11111)
    channel = _FakeChannel(987011, [])
    channel._messages = [_make_msg(70060, author, channel)]
    _seed_session_for_channel(adapter, store, db, channel, author)
    adapter._restart_recovery.mark_channel_active(str(channel.id))
    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    await adapter._backfill_missed_messages()
    assert channel.history_calls == []
    assert adapter._dispatched == []


@pytest.mark.asyncio
async def test_backfill_reinjected_message_hits_full_auth_path(_tmp_home, monkeypatch):
    """AC-6 / INV-4: re-injection goes through the REAL _dispatch_incoming_message
    (the extracted on_message body), so an unauthorized user is rejected exactly
    as a live message would be — backfill grants no bypass."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    # Restore the REAL dispatch path (the fixture stubbed it with a capture).
    from plugins.platforms.discord.adapter import DiscordAdapter
    dispatched_to_handler = []

    async def _handler_spy(message, role_authorized=False):
        dispatched_to_handler.append(message.id)

    adapter._handle_message = _handler_spy
    adapter._dispatch_incoming_message = DiscordAdapter._dispatch_incoming_message.__get__(adapter)

    # Lock the allowlist: only user 11111 is allowed.
    adapter._is_allowed_user = lambda uid, author, guild=None, is_dm=False: str(uid) == "11111"

    bot = _FakeAuthor(99999, "bot", bot=True)
    allowed = _FakeAuthor(11111, "alice")
    blocked = _FakeAuthor(22222, "mallory")
    channel = _FakeChannel(987012, [])
    _seed_session_for_channel(adapter, store, db, channel, allowed)
    channel._messages = [
        _make_msg(70070, blocked, channel, content="unauthorized"),
        _make_msg(70071, allowed, channel, content="authorized"),
    ]
    adapter._restart_recovery.mark_channel_active(str(channel.id))
    adapter._client = SimpleNamespace(
        user=bot, get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    # DM-only free path off; require plain group handling.
    with patch.dict(os.environ, {"DISCORD_IGNORE_NO_MENTION": "false"}):
        await adapter._backfill_missed_messages()

    # Only the authorized user's message reaches _handle_message.
    assert dispatched_to_handler == [70071]


@pytest.mark.asyncio
async def test_backfill_thread_scoped_to_thread(_tmp_home):
    """D-6: the recovery scan runs on the exact channel object returned for the
    marked channel id (a thread's own history returns only that thread's
    messages), and recovered messages come from that object."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    author = _FakeAuthor(11111)

    # A thread-like channel keyed by the thread id 933001. Session resolution
    # falls back to the vacuously-absent path (no seeded session) which still
    # recovers via the normal auth path — the point here is scan-scoping.
    thread_channel = _FakeChannel(933001, [], name="a-thread")
    thread_channel._messages = [_make_msg(70080, author, thread_channel, content="in thread")]

    adapter._restart_recovery.mark_channel_active("933001")
    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: thread_channel if int(_id) == 933001 else None,
        fetch_channel=AsyncMock(return_value=thread_channel),
    )
    await adapter._backfill_missed_messages()
    # The thread's OWN history was scanned and its message recovered.
    assert [m.id for m in adapter._dispatched] == [70080]
    assert thread_channel.history_calls, "the marked channel's history must be scanned"


@pytest.mark.asyncio
async def test_backfill_no_session_store_skips_entirely(_tmp_home):
    """Greptile P2 / INV-9: with no session store wired, the transcript
    authority is entirely unavailable — the sweep must recover NOTHING (fail
    toward no-dup), never blind-re-inject. Guards an init-race where
    _session_store is absent at backfill time."""
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    adapter._session_store = None  # authority gone
    author = _FakeAuthor(11111)
    channel = _FakeChannel(987600, [])
    channel._messages = [_make_msg(70600, author, channel)]
    adapter._restart_recovery.mark_channel_active(str(channel.id))
    adapter._client = SimpleNamespace(
        user=_FakeAuthor(99999, "bot", bot=True),
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    await adapter._backfill_missed_messages()
    assert adapter._dispatched == [], "no session store ⇒ recover nothing (INV-9)"
    assert channel.history_calls == [], "must not even scan when the authority is absent"


def test_anchor_uses_shutdown_margin_not_lookback(_tmp_home):
    """Greptile P2: the shutdown_ts branch of the anchor must be load-bearing.
    A recent shutdown_ts anchors at shutdown_ts - MARGIN (300s), which is
    NEWER (tighter) than the now-lookback floor (900s) — proving the branch
    isn't dead code that always collapses to the floor."""
    from plugins.platforms.discord.adapter import (
        _RESTART_BACKFILL_ANCHOR_MARGIN_S, _DiscordRestartRecoveryState,
    )
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    now = 2_000_000_000.0
    lookback = 900.0
    # A shutdown ~30s ago (recent restart): anchor should be shutdown_ts - 300,
    # NOT now - 900. Prove the two differ and the tighter (newer) one wins.
    adapter._restart_recovery = _DiscordRestartRecoveryState(persist_interval_s=0.0)
    adapter._restart_recovery.flush(shutdown_ts=now - 30)

    anchor = adapter._restart_backfill_anchor_snowflake(now, lookback)
    # Reverse the snowflake back to an epoch to compare.
    anchor_epoch = ((anchor.id >> 22) + 1420070400000) / 1000.0
    expected = (now - 30) - _RESTART_BACKFILL_ANCHOR_MARGIN_S
    floor = now - lookback
    assert abs(anchor_epoch - expected) < 1.0, "recent shutdown must anchor at shutdown_ts - margin"
    assert anchor_epoch > floor, "the shutdown_ts branch must be tighter than the floor (not dead)"


def test_anchor_falls_back_to_floor_on_long_outage(_tmp_home):
    """The floor bounds a long outage: a shutdown_ts far in the past makes
    shutdown_ts - margin older than now - lookback, so the floor takes over."""
    from plugins.platforms.discord.adapter import _DiscordRestartRecoveryState
    adapter, store, db = _build_adapter_with_store(_tmp_home)
    now = 2_000_000_000.0
    lookback = 900.0
    adapter._restart_recovery = _DiscordRestartRecoveryState(persist_interval_s=0.0)
    adapter._restart_recovery.flush(shutdown_ts=now - 100_000)  # ~28h ago

    anchor = adapter._restart_backfill_anchor_snowflake(now, lookback)
    anchor_epoch = ((anchor.id >> 22) + 1420070400000) / 1000.0
    floor = now - lookback
    assert abs(anchor_epoch - floor) < 1.0, "a long outage must clamp to the now-lookback floor"


