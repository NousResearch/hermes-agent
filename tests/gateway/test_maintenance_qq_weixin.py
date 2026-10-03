"""Synthetic authenticated adapter events through gateway admission and durable intake."""

import asyncio
import json
from types import SimpleNamespace
import pytest

from agent.session_persistence import _db_flush_row
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageType
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.qqbot.adapter import QQAdapter
from gateway.platforms.weixin import WeixinAdapter, ITEM_TEXT
from gateway.run import GatewayRunner
from gateway.run_inbound import GatewayInboundMixin
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from hermes_maintenance_source import pending_sources, read_committed_source
from hermes_state import SessionDB
from _maintenance_intake import MaintenanceIntake


class Ingress(GatewayInboundMixin):
    def __init__(self, home, adapter, key, authorized=True):
        self.home, self.adapter, self.key, self.authorized = home, adapter, key, authorized

    def _scale_to_zero_note_real_inbound(self): pass
    async def _hm_pre_gateway_dispatch_hook(self, event, source): return event
    def _is_user_authorized_for_source(self, source): return self.authorized
    def _admit_bot_message_for_source(self, source): return True
    def _intake_adapter_for(self, source): return self.adapter
    def _resolve_profile_home_for_source(self, source): return self.home
    def _session_key_for_source(self, source): return self.key
    def _get_unauthorized_dm_behavior(self, *args, **kwargs): return "ignore"
    async def _hm_report_ignored_dm(self, source): pass
    def _get_executor(self): return None
    _run_in_executor_with_context = GatewayRunner._run_in_executor_with_context


def persist(db, event, session_id, session_key, change=None):
    runner = object.__new__(TurnRunner)
    runner._ctx = TurnContext(source=event.source, session_id=session_id, session_key=session_key,
                              message=event.text, inbound_message_id=event.message_id)
    runner._native_image_run_message = lambda: event.text
    agent = SimpleNamespace()

    def run(message, **kwargs):
        row = _db_flush_row(agent, {"role": "user", "content": message,
                                    "platform_message_id": kwargs["persist_user_platform_id"]}, True)
        if change:
            row.update(change)
        db.append_messages_batch(session_id, [row])
        return row["_row_id"]

    agent.run_conversation = run
    return runner._run_conversation_with_approval(agent, [], None, None, None)


def qq_adapter(extra=None):
    adapter = object.__new__(QQAdapter)
    adapter.platform = Platform.QQBOT
    adapter.config = PlatformConfig(extra=dict(extra or {}))
    adapter.gateway_runner = None
    adapter._app_id = "qq-receiver"
    adapter._chat_type_map = {}
    adapter._is_group_allowed = lambda *args: True
    adapter._is_dm_intake_allowed = lambda *args: True
    async def attachments(_):
        return {"voice_transcripts": [], "attachment_info": "", "image_urls": [], "image_media_types": []}
    async def quote(_):
        return {"quote_block": "", "image_urls": [], "image_media_types": []}
    adapter._process_attachments = attachments
    adapter._process_quoted_context = quote
    adapter._detect_message_type = lambda *args: MessageType.TEXT
    seen = []
    async def handle(event): seen.append(event)
    adapter.handle_message = handle
    return adapter, seen


def weixin_adapter(extra=None):
    adapter = object.__new__(WeixinAdapter)
    adapter.platform = Platform.WEIXIN
    adapter.config = PlatformConfig(extra=dict(extra or {}))
    adapter.gateway_runner = None
    adapter._poll_session = object()
    adapter._account_id = "wx-receiver"
    adapter._token = ""
    adapter._dedup = MessageDeduplicator(ttl_seconds=60)
    adapter._typing_cache = {}
    adapter._is_group_allowed = lambda *args: True
    adapter._is_dm_intake_allowed = lambda *args: True
    async def set_token(*args): pass
    async def collect(*args): pass
    adapter._token_store = SimpleNamespace(set=set_token)
    adapter._collect_media = collect
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._drop_unresolved = lambda event: False
    adapter._text_batch_key = lambda event: event.source.chat_id
    adapter._text_batch_delay_seconds = 100
    adapter._text_batch_split_delay_seconds = 100
    seen = []
    async def handle(event): seen.append(event)
    adapter.handle_message = handle
    return adapter, seen


def wx_message(sender, event_id, text, *, room=None):
    return {"from_user_id": sender, "to_user_id": "wx-receiver", "message_id": event_id,
            "room_id": room, "item_list": [{"type": ITEM_TEXT, "text_item": {"text": text}}]}


def store(home, platform, actor, chat, chat_type, key):
    home.mkdir()
    (home / "mem0.json").write_text(json.dumps({"governed": {"maintenance_intake": True}}))
    db = SessionDB(db_path=home / "state.db")
    db.create_session("s1", platform, user_id=actor, chat_id=chat,
                      chat_type=chat_type, session_key=key)
    return db


def test_qq_group_keeps_raw_author_and_owner(tmp_path):
    async def scenario():
        adapter, seen = qq_adapter()
        await adapter._handle_group_message(
            {"id": "qq-1", "content": "@bot I prefer blue", "group_openid": "room",
             "author": {"member_openid": "alice"}}, "qq-1", "@bot I prefer blue",
            {"member_openid": "alice"}, "2026-09-30T18:00:00+00:00")
        assert len(seen) == 1
        event = seen[0]
        assert event.text == "I prefer blue"
        home, key = tmp_path / "qq", "agent:main:qqbot:group:room"
        db = store(home, "qqbot", "alice", "room", "group", key)
        try:
            ingress = Ingress(home, adapter, key)
            assert await ingress._hm_admit_event(event)
            row_id = await ingress._run_in_executor_with_context(persist, db, event, "s1", key)
            source_id, = pending_sources(home)
            source = read_committed_source(home, source_id)
            assert (source["raw_text"], source["actor"], source["message_row_id"]) == (
                "@bot I prefer blue", "alice", row_id)
            assert source["extraction_text"] == "I prefer blue"
            intake = MaintenanceIntake(home, "owner-alice", {("qqbot", "alice"): "owner-alice",
                                                         ("qqbot", "bob"): "owner-bob"}, enabled=True)
            task, = intake.pump()
            assert task["actor"] == "alice" and task["owner_id"] == "owner-alice"
            assert intake.read(task["task_id"], "owner-alice")
        finally:
            db.close()
    asyncio.run(scenario())


def test_weixin_batch_has_two_sources_and_no_cross_actor_owner(tmp_path, monkeypatch):
    async def scenario():
        adapter, seen = weixin_adapter()
        await adapter._process_message(wx_message("alice", "wx-1", "blue", room="room"))
        await adapter._process_message(wx_message("bob", "wx-2", "green", room="room"))
        await adapter._process_message(wx_message("bob", "wx-2", "green", room="room"))
        for task in adapter._pending_text_batch_tasks.values(): task.cancel()
        await adapter._flush_text_batch_now("room")
        assert len(seen) == 1
        event = seen[0]
        assert len(event._maintenance_sources) == 2
        home, key = tmp_path / "wx", "agent:main:weixin:group:room"
        db = store(home, "weixin", "alice", "room", "group", key)
        try:
            ingress = Ingress(home, adapter, key)
            assert await ingress._hm_admit_event(event)
            with monkeypatch.context() as patch:
                def fail(*args, **kwargs): raise RuntimeError("rollback batch")
                patch.setattr(db, "_bump_session_counters", fail)
                with pytest.raises(RuntimeError, match="rollback batch"):
                    await ingress._run_in_executor_with_context(persist, db, event, "s1", key)
            assert pending_sources(home) == [] and db.get_messages("s1") == []
            row_id = await ingress._run_in_executor_with_context(persist, db, event, "s1", key)
            sources = [read_committed_source(home, source_id) for source_id in pending_sources(home)]
            assert {(source["actor"], source["event_id"], source["raw_text"])
                    for source in sources} == {("alice", "wx-1", "blue"), ("bob", "wx-2", "green")}
            assert {source["extraction_text"] for source in sources} == {"blue", "green"}
            assert {source["message_row_id"] for source in sources} == {row_id}
            alice = MaintenanceIntake(home, "owner-alice", {("weixin", "alice"): "owner-alice",
                                                           ("weixin", "bob"): "owner-bob"}, enabled=True)
            bob = MaintenanceIntake(home, "owner-bob", {("weixin", "alice"): "owner-alice",
                                                       ("weixin", "bob"): "owner-bob"}, enabled=True)
            assert {task["actor"] for task in alice.pump()} == {"alice"}
            assert {task["actor"] for task in bob.pump()} == {"bob"}
        finally:
            db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["unauthorized", "internal", "actor", "event",
                                     "receiver", "rewritten", "forged", "row-event", "row-text"])
def test_changed_or_untrusted_qq_event_cannot_create_source(tmp_path, change):
    async def scenario():
        adapter, seen = qq_adapter()
        await adapter._handle_group_message(
            {"id": "qq-negative", "content": "@bot private preference", "group_openid": "room"},
            "qq-negative", "@bot private preference", {"member_openid": "alice"},
            "2026-09-30T18:00:00+00:00")
        event = seen[0]
        home, key = tmp_path / "negative", "agent:main:qqbot:group:room"
        db = store(home, "qqbot", "alice", "room", "group", key)
        try:
            if change == "internal": event.internal = True
            if change == "actor": event.source.user_id = "mallory"
            if change == "event": event.message_id = "changed-event"
            if change == "receiver": adapter._app_id = "different-receiver"
            if change == "rewritten": event.text += " injected"
            if change == "forged": event._maintenance_sources = ({"role": "user"},)
            ingress = Ingress(home, adapter, key, authorized=change != "unauthorized")
            await ingress._hm_admit_event(event)
            row_change = {"row-event": {"platform_message_id": "other"},
                          "row-text": {"content": "other"}}.get(change)
            await ingress._run_in_executor_with_context(persist, db, event, "s1", key, row_change)
            assert pending_sources(home) == []
        finally:
            db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["qq-c2c", "qq-guild-channel", "qq-guild-dm", "weixin-dm"])
def test_other_text_modes_reach_the_same_durable_intake(tmp_path, mode):
    async def scenario():
        if mode == "weixin-dm":
            adapter, seen = weixin_adapter()
            await adapter._process_message(wx_message("alice", "wx-dm-1", "one preference"))
            for task in adapter._pending_text_batch_tasks.values(): task.cancel()
            await adapter._flush_text_batch_now("alice")
            platform, chat, chat_type = "weixin", "alice", "dm"
        else:
            adapter, seen = qq_adapter()
            data = {"id": "qq-mode-1", "content": "one preference"}
            author = {"user_openid": "alice", "id": "alice", "username": "Alice"}
            if mode == "qq-c2c":
                handler, chat, chat_type = adapter._handle_c2c_message, "alice", "dm"
            elif mode == "qq-guild-channel":
                data.update(channel_id="channel", guild_id="guild")
                handler, chat, chat_type = adapter._handle_guild_message, "channel", "group"
            else:
                data.update(guild_id="guild")
                handler, chat, chat_type = adapter._handle_dm_message, "guild", "dm"
            await handler(data, "qq-mode-1", "one preference", author,
                          "2026-09-30T18:00:00+00:00")
            platform = "qqbot"
        event, = seen
        home, key = tmp_path / mode, f"agent:main:{platform}:{chat_type}:{chat}"
        db = store(home, platform, "alice", chat, chat_type, key)
        try:
            ingress = Ingress(home, adapter, key)
            assert await ingress._hm_admit_event(event)
            await ingress._run_in_executor_with_context(persist, db, event, "s1", key)
            source_id, = pending_sources(home)
            source = read_committed_source(home, source_id)
            assert source["actor"] == "alice" and source["event_id"] == event.message_id
            intake = MaintenanceIntake(home, "owner-alice", {(platform, "alice"): "owner-alice"}, enabled=True)
            task, = intake.pump()
            assert task["source_id"] == source_id
        finally:
            db.close()
    asyncio.run(scenario())


def test_weixin_dm_batch_preserves_each_original_event(tmp_path):
    async def scenario():
        adapter, seen = weixin_adapter()
        await adapter._process_message(wx_message("alice", "dm-1", "first"))
        await adapter._process_message(wx_message("alice", "dm-2", "second"))
        for task in adapter._pending_text_batch_tasks.values(): task.cancel()
        await adapter._flush_text_batch_now("alice")
        event, = seen
        home, key = tmp_path / "dm-batch", "agent:main:weixin:dm:alice"
        db = store(home, "weixin", "alice", "alice", "dm", key)
        try:
            ingress = Ingress(home, adapter, key)
            assert await ingress._hm_admit_event(event)
            await ingress._run_in_executor_with_context(persist, db, event, "s1", key)
            assert {(source["event_id"], source["raw_text"]) for source in
                    (read_committed_source(home, sid) for sid in pending_sources(home))} == {
                        ("dm-1", "first"), ("dm-2", "second")}
            intake = MaintenanceIntake(home, "owner", {("weixin", "alice"): "owner"}, enabled=True)
            assert len(intake.pump()) == 2
        finally:
            db.close()
    asyncio.run(scenario())
