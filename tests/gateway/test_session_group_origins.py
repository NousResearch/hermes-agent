"""Session groups (#79198), adapter side: two chats on different platforms share one session key.

The group itself comes from ``gateway.session.session_group_source`` (patched here; the
config/hook side fills it in). These tests pin what the gateway guarantees once two chats share a
key: no group means nothing changes; only a source a live adapter of its own platform built can
join a group; the shared conversation survives a restart from either chat; and every turn has
exactly one origin, so messages from the two chats never merge and each reply goes back to the
chat it answers.

Real: ``BasePlatformAdapter`` dispatch/lock/delivery, the ``GatewayRunner`` busy path wired by
``_wire_adapter_handlers``, ``SessionStore`` over a real ``SessionDB``. Fake: the agent turn.
"""

import asyncio
import weakref

import pytest

import gateway.session as gateway_session
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key, key_source_for
from hermes_state import SessionDB

GROUP = SessionSource(platform=Platform.TELEGRAM, chat_id="tg-chat", chat_type="dm")


class _Adapter(BasePlatformAdapter):
    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="t"), platform)
        self.sent: list[tuple[str, str]] = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id=f"out-{len(self.sent)}")


def _group(source: SessionSource):
    """Telegram ``tg-chat`` and Discord ``dc-chat`` form one group keyed on the Telegram chat."""
    if (source.platform, source.chat_id) in {(Platform.TELEGRAM, "tg-chat"), (Platform.DISCORD, "dc-chat")}:
        return GROUP
    return None


@pytest.fixture
def grouped(monkeypatch):
    monkeypatch.setattr(gateway_session, "session_group_source", _group)


def _lines():
    tg, dc = _Adapter(Platform.TELEGRAM), _Adapter(Platform.DISCORD)
    return tg, dc, tg.build_source(chat_id="tg-chat", user_id="u1"), dc.build_source(chat_id="dc-chat", user_id="u1")


def _store(tmp_path, db):
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    store._db = db
    return store


def test_no_group_keeps_every_key_and_row_unchanged(tmp_path):
    tg, _, tg_src, dc_src = _lines()
    assert key_source_for(tg_src) is tg_src and key_source_for(dc_src) is dc_src
    with SessionDB(db_path=tmp_path / "state.db") as db:
        store = _store(tmp_path, db)
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        key = build_session_key(dc_src)
        assert store._generate_session_key(dc_src) == runner._session_key_for_source(dc_src) == key
        assert tg._source_session_key(tg_src) == build_session_key(tg_src)
        entry = store.get_or_create_session(dc_src)
        assert entry.key_source is None and "key_source" not in entry.to_dict()
        assert db.get_session(entry.session_id)["source"] == "discord"
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=tg_src)
    assert tg._group_turn_lock(event, "k") is None


def test_only_a_live_source_of_its_own_platform_joins_a_group(grouped):
    tg, dc, tg_src, dc_src = _lines()
    shared = build_session_key(GROUP)
    assert dc._source_session_key(dc_src) == tg._source_session_key(tg_src) == shared
    assert dc_src.platform == Platform.DISCORD and dc_src.chat_id == "dc-chat"  # origin untouched

    wire = dc_src.to_dict()
    wire.update(key_source=GROUP.to_dict(), _transport_adapter_ref="forged")
    restored = SessionSource.from_dict(wire)
    assert key_source_for(restored) is restored
    assert build_session_key(restored) == build_session_key(SessionSource(
        platform=Platform.DISCORD, chat_id="dc-chat", chat_type="dm")) != shared

    # A live reference to an adapter of another platform cannot vouch for this source.
    spoofed = SessionSource(platform=Platform.DISCORD, chat_id="dc-chat", chat_type="dm")
    spoofed._transport_adapter_ref = weakref.ref(tg)
    assert key_source_for(spoofed) is spoofed


def test_restart_recovers_the_shared_session_from_either_chat(tmp_path, grouped):
    _, _, tg_src, dc_src = _lines()
    key = build_session_key(GROUP)
    with SessionDB(db_path=tmp_path / "state.db") as db:
        store = _store(tmp_path, db)
        original = store.get_or_create_session(dc_src)
        assert original.session_key == key
        assert original.origin.platform == Platform.DISCORD
        assert original.key_source.platform == Platform.TELEGRAM
        assert db.get_session(original.session_id)["source"] == "telegram"

        # Routing index lost: the durable row is found again from both chats.
        for src in (tg_src, dc_src):
            store._entries.clear()
            assert store.get_or_create_session(src).session_id == original.session_id

        # Restart: the persisted entry keeps both the delivery origin and the key source, and the
        # startup stale-entry pass reopens the conversation instead of pruning it.
        restarted = _store(tmp_path, db)
        restarted._ensure_loaded()
        entry = restarted._entries[key]
        assert entry.origin.platform == Platform.DISCORD
        assert entry.key_source == SessionEntry.from_dict(original.to_dict()).key_source
        db.end_session(original.session_id, "agent_close")
        assert restarted._stale_entry_verdict(key, entry, db.get_session(original.session_id)) is None
        assert db.get_session(original.session_id)["end_reason"] is None

        restarted.update_session(key)
        assert db.get_session(original.session_id)["source"] == "telegram"
        reset = restarted.reset_session(key)
        assert reset.key_source.platform == Platform.TELEGRAM
        assert db.get_session(reset.session_id)["source"] == "telegram"


class _Agent:
    def __init__(self):
        self.steered: list[str] = []

    def steer(self, text):
        self.steered.append(text)
        return True


class _Turns:
    """Stands in for the agent turn: one call per turn, the first held open until released."""

    def __init__(self, runner, key):
        self.runner, self.key = runner, key
        self.order: list[tuple[str, str, str]] = []
        self.agent = _Agent()
        self.release_first = asyncio.Event()
        self.first_running = asyncio.Event()
        self.active = self.max_active = 0

    async def __call__(self, event: MessageEvent):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        turn = self.runner._session_state(self.key).turn
        turn.event, turn.agent = event, self.agent
        self.order.append((event.source.platform.value, event.source.chat_id, event.text))
        try:
            if len(self.order) == 1:
                self.first_running.set()
                await self.release_first.wait()
            return f"reply to {event.text}"
        finally:
            turn.event = turn.agent = None
            self.active -= 1


def _gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    tg, dc, tg_src, dc_src = _lines()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.TELEGRAM: tg, Platform.DISCORD: dc}
    runner._draining = False
    runner._restart_requested = False
    runner._busy_input_mode = "steer"  # without the origin rule, the Discord text steers into Telegram's turn
    runner._is_user_authorized = lambda _source: True
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=runner.config)
    runner.session_store._db = SessionDB(db_path=tmp_path / "state.db")
    key = build_session_key(GROUP)
    turns = _Turns(runner, key)
    for adapter in (tg, dc):
        runner._wire_adapter_handlers(adapter, message_handler=turns)
        adapter.gateway_runner = runner
    return tg, dc, tg_src, dc_src, turns, key


def _event(source: SessionSource, text: str) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id=f"in-{text}")


async def _until_idle(*adapters, key):
    for _ in range(50):
        owners = [a._session_tasks.get(key) for a in adapters]
        live = [t for t in owners if t is not None and not t.done()]
        if not live:
            return
        await asyncio.wait_for(asyncio.gather(*(asyncio.shield(t) for t in live)), timeout=5)


@pytest.mark.asyncio
async def test_two_chats_sharing_a_key_never_merge_and_each_reply_goes_home(tmp_path, monkeypatch, grouped):
    tg, dc, tg_src, dc_src, turns, key = _gateway(tmp_path, monkeypatch)
    try:
        await tg.handle_message(_event(tg_src, "A1"))
        await asyncio.wait_for(turns.first_running.wait(), timeout=5)
        # Both Discord messages arrive while Telegram's turn owns the key.
        await dc.handle_message(_event(dc_src, "B1"))
        await dc.handle_message(_event(dc_src, "B2"))
        await asyncio.sleep(0)
        assert turns.order == [("telegram", "tg-chat", "A1")]

        turns.release_first.set()
        await _until_idle(tg, dc, key=key)

        assert turns.order == [
            ("telegram", "tg-chat", "A1"), ("discord", "dc-chat", "B1"), ("discord", "dc-chat", "B2"),
        ], "each message is its own turn, in arrival order"
        assert turns.max_active == 1, "two turns ran on one shared key at once"
        assert turns.agent.steered == [], "a Discord message was steered into Telegram's turn"
        assert tg.sent == [("tg-chat", "reply to A1")]
        assert dc.sent == [("dc-chat", "reply to B1"), ("dc-chat", "reply to B2")]
    finally:
        turns.release_first.set()
        await tg.cancel_background_tasks()
        await dc.cancel_background_tasks()


def test_each_adapter_drains_only_its_own_queued_followups(tmp_path, monkeypatch, grouped):
    """The /queue overflow FIFO is per key, so a group shares it: a lane promotes only its own."""
    tg, dc, tg_src, dc_src, turns, key = _gateway(tmp_path, monkeypatch)
    runner = turns.runner
    b3, a3, b4 = _event(dc_src, "B3"), _event(tg_src, "A3"), _event(dc_src, "B4")
    runner._session_state(key).conversation.queued_events.extend([b3, a3, b4])
    assert runner._promote_queued_event(key, tg, None) is a3
    assert runner._rescue_orphaned_overflow(key, dc) is b3
    assert dc._pending_messages[key] is b4 and runner._overflow_queue(key) == []


@pytest.mark.asyncio
async def test_other_chat_commands_act_now_and_the_runner_never_steers_across_chats(
    tmp_path, monkeypatch, grouped
):
    tg, dc, tg_src, dc_src, turns, key = _gateway(tmp_path, monkeypatch)
    try:
        await tg.handle_message(_event(tg_src, "A1"))
        await asyncio.wait_for(turns.first_running.wait(), timeout=5)
        # A control command from the other chat is dispatched at once, not parked behind the turn.
        await dc.handle_message(_event(dc_src, "/stop"))
        assert ("discord", "dc-chat", "/stop") in turns.order

        # A message that reaches the runner's running-session path from the other chat queues on
        # its own adapter instead of steering the running turn.
        b1 = _event(dc_src, "B1")
        turn = turns.runner._session_state(key).turn
        turn.event, turn.agent = _event(tg_src, "A1"), turns.agent
        assert await turns.runner._hm_handle_running_session_message(b1, dc_src, key) is None
        assert turns.agent.steered == [] and dc._pending_messages[key] is b1
    finally:
        turns.release_first.set()
        await tg.cancel_background_tasks()
        await dc.cancel_background_tasks()
