"""Feishu text-family sources through the real gateway/state boundary."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.session_persistence import _db_flush_row
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_inbound import GatewayInboundMixin
from gateway.session import build_session_key
from hermes_maintenance_source import pending_sources, read_committed_source
from hermes_state import SessionDB
from _maintenance_intake import MaintenanceIntake
from plugins.platforms.feishu.adapter import FeishuAdapter


FEISHU_MODES = {
    "dm:peer": {"chat_info": "private", "event_chat": "p2p", "thread": None, "extra": {}},
    "dm_thread:peer": {"chat_info": "private", "event_chat": "p2p", "thread": "thread-dm", "extra": {}},
    "group:per_actor": {"chat_info": "group", "event_chat": "group", "thread": None,
                        "extra": {"group_sessions_per_user": True}},
    "group:shared": {"chat_info": "group", "event_chat": "group", "thread": None,
                     "extra": {"group_sessions_per_user": False}},
    "group_thread:per_actor": {"chat_info": "group", "event_chat": "group", "thread": "thread-group",
                                "extra": {"group_sessions_per_user": True, "thread_sessions_per_user": True}},
    "group_thread:shared": {"chat_info": "group", "event_chat": "group", "thread": "thread-group",
                             "extra": {"group_sessions_per_user": True, "thread_sessions_per_user": False}},
}

INPUT_PATHS = ("text", "text_batch", "post", "reply_reference")


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


def make_adapter(extra=None):
    adapter = object.__new__(FeishuAdapter)
    adapter.platform = Platform.FEISHU
    adapter.config = PlatformConfig(extra=dict(extra or {}))
    adapter.gateway_runner = None
    adapter._bot_open_id, adapter._bot_user_id, adapter._bot_name = "receiver-bot", "", "Hermes"
    adapter._message_text_cache = {}
    adapter._message_quote_author_cache = {}
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_text_batch_counts = {}
    adapter._text_batch_max_messages = 10
    adapter._text_batch_max_chars = 10_000
    adapter._text_batch_delay_seconds = 0
    adapter._text_batch_split_delay_seconds = 0
    adapter._dispatched = []

    async def dispatch(event):
        adapter._dispatched.append(event)

    async def handle(event):
        adapter._dispatched.append(event)

    adapter._dispatch_inbound_event = dispatch
    adapter._handle_message_with_guards = handle
    adapter.get_chat_info = AsyncMock(return_value={"name": "feishu-chat", "type": "group"})
    adapter._resolve_sender_profile = AsyncMock(return_value={
        "user_id": "alice", "user_name": "Alice", "user_id_alt": None,
    })
    return adapter


async def emit(adapter, *, actor, text, kind, chat_id, mode, event_id, reply_id=None,
               reply_text=None, reply_author=None):
    if kind == "post":
        content = json.dumps({"zh_cn": {"content": [[{"tag": "text", "text": text}]]}})
        message_type = "post"
    else:
        content = json.dumps({"text": text})
        message_type = "text"
    sender_id = SimpleNamespace(open_id=actor, user_id="", union_id="")
    message = SimpleNamespace(
        content=content, message_type=message_type, message_id=event_id, mentions=[],
        chat_id=chat_id, chat_type=mode["event_chat"], thread_id=mode["thread"],
        parent_id=reply_id, upper_message_id=None, root_id=reply_id,
    )
    if reply_id:
        adapter._message_quote_author_cache[reply_id] = (reply_author or "", "Parent")
    adapter.get_chat_info.return_value = {"name": "feishu-chat", "type": mode["chat_info"]}
    adapter._resolve_sender_profile.return_value = {
        "user_id": actor, "user_name": actor.title(), "user_id_alt": None,
    }
    adapter._fetch_message_text = AsyncMock(return_value=reply_text)
    data = SimpleNamespace(event=SimpleNamespace(
        message=message,
        sender=SimpleNamespace(sender_type="user", sender_id=sender_id),
    ))
    await adapter._process_inbound_message(
        data=data, message=message, sender_id=sender_id,
        chat_type=mode["event_chat"], message_id=event_id, is_bot=False,
    )
    return adapter._dispatched[-1]


async def events_for(adapter, *, mode, kind, chat_id, actor="alice", prefix="e"):
    if kind == "text_batch":
        first = await emit(adapter, actor=actor, text="first preference", kind="text",
                           chat_id=chat_id, mode=mode, event_id=prefix + "-1")
        second = await emit(adapter, actor=actor, text="second preference", kind="text",
                            chat_id=chat_id, mode=mode, event_id=prefix + "-2")
        key = adapter._text_batch_key(first)
        await adapter._enqueue_text_event(first)
        await adapter._enqueue_text_event(second)
        for task in list(adapter._pending_text_batch_tasks.values()):
            task.cancel()
        before = len(adapter._dispatched)
        await adapter._flush_text_batch_now(key)
        assert len(adapter._dispatched) == before + 1
        return adapter._dispatched[-1], ("first preference", "second preference")
    if kind == "reply_reference":
        event = await emit(adapter, actor=actor, text="new reply", kind="text", chat_id=chat_id,
                           mode=mode, event_id=prefix + "-reply", reply_id=prefix + "-parent",
                           reply_text="quoted old text", reply_author="parent-author")
        return event, ("new reply",)
    event = await emit(adapter, actor=actor, text="one preference", kind=kind,
                       chat_id=chat_id, mode=mode, event_id=prefix + "-1")
    return event, ("one preference",)


def store(home, event, key):
    home.mkdir(parents=True, exist_ok=True)
    (home / "mem0.json").write_text(json.dumps({"governed": {"maintenance_intake": True}}), encoding="utf-8")
    db = SessionDB(db_path=home / "state.db")
    db.create_session("s1", "feishu", user_id=event.source.user_id, chat_id=event.source.chat_id,
                      chat_type=event.source.chat_type, thread_id=event.source.thread_id,
                      session_key=key)
    return db


def persist(db, event, key):
    from hermes_inbound_evidence import authenticated_turn
    with authenticated_turn("s1", key):
        row = _db_flush_row(SimpleNamespace(), {
            "role": "user", "content": event.text, "platform_message_id": event.message_id,
        }, True)
        db.append_messages_batch("s1", [row])
    return row


def source_owner(actor):
    return "owner-" + actor


@pytest.mark.parametrize("mode_name", list(FEISHU_MODES))
@pytest.mark.parametrize("input_path", INPUT_PATHS)
def test_feishu_text_family_modes_reach_committed_tasks(tmp_path, mode_name, input_path):
    async def scenario():
        mode = FEISHU_MODES[mode_name]
        adapter = make_adapter(mode["extra"])
        chat_id = "chat-" + mode_name.replace(":", "-")
        event, expected = await events_for(
            adapter, mode=mode, kind=input_path, chat_id=chat_id,
            prefix=f"{mode_name}-{input_path}".replace(":", "-"),
        )
        key = adapter._event_session_key(event)
        home = tmp_path / f"{mode_name.replace(':', '-')}-{input_path}"
        db = store(home, event, key)
        try:
            ingress = Ingress(home, adapter, key)
            assert await ingress._hm_admit_event(event)
            row = persist(db, event, key)
            sources = [read_committed_source(home, sid) for sid in pending_sources(home)]
            assert sources and all(source is not None for source in sources)
            assert {source["actor"] for source in sources} == {"alice"}
            prefix = f"{mode_name}-{input_path}".replace(":", "-")
            expected_ids = ({
                prefix + "-1", prefix + "-2",
            } if input_path == "text_batch" else {
                prefix + "-reply" if input_path == "reply_reference" else prefix + "-1",
            })
            assert {source["event_id"] for source in sources} == expected_ids
            if input_path == "text_batch":
                assert len(sources) == 2
                assert {source["extraction_text"] for source in sources} == set(expected)
                assert all(source["display_text"] == "first preference\nsecond preference" for source in sources)
                assert all(source["batch_size"] == 2 for source in sources)
            elif input_path == "post":
                assert sources[0]["message_type"] == "post"
                assert sources[0]["event_structure"]
            elif input_path == "reply_reference":
                assert sources[0]["input_kind"] == "reply_reference"
                assert sources[0]["extraction_text"] == "new reply"
                assert sources[0]["reply_to_text"] == "quoted old text"
                assert sources[0]["reply_to_author_id"] == "parent-author"
            else:
                assert sources[0]["extraction_text"] == "one preference"
            assert all(source["message_row_id"] == row["_row_id"] for source in sources)
            owners = sorted({source["actor"] for source in sources})
            mapping = {("feishu", actor): source_owner(actor) for actor in owners}
            tasks = []
            for source in sources:
                owner = source_owner(source["actor"])
                tasks.append(MaintenanceIntake(home, owner, mapping, enabled=True).accept(source["source_id"]))
            assert len(tasks) == len(sources)
        finally:
            db.close()
    asyncio.run(scenario())


def test_feishu_shared_batch_keeps_each_author_and_unmapped_author_pending(tmp_path):
    async def scenario():
        mode = FEISHU_MODES["group:shared"]
        adapter = make_adapter(mode["extra"])
        alice = await emit(adapter, actor="alice", text="alice fact", kind="text",
                           chat_id="shared-chat", mode=mode, event_id="shared-1")
        bob = await emit(adapter, actor="bob", text="bob fact", kind="text",
                         chat_id="shared-chat", mode=mode, event_id="shared-2")
        key = adapter._text_batch_key(alice)
        await adapter._enqueue_text_event(alice)
        await adapter._enqueue_text_event(bob)
        for task in list(adapter._pending_text_batch_tasks.values()):
            task.cancel()
        before = len(adapter._dispatched)
        await adapter._flush_text_batch_now(key)
        assert len(adapter._dispatched) == before + 1
        event = adapter._dispatched[-1]
        home = tmp_path / "shared-author-batch"
        db = store(home, event, key)
        try:
            ingress = Ingress(home, adapter, key)
            assert await ingress._hm_admit_event(event)
            row = persist(db, event, key)
            sources = [read_committed_source(home, sid) for sid in pending_sources(home)]
            assert {source["actor"] for source in sources} == {"alice", "bob"}
            assert {source["extraction_text"] for source in sources} == {"alice fact", "bob fact"}
            assert all(source["display_text"] == "alice fact\nbob fact" for source in sources)
            assert {source["message_row_id"] for source in sources} == {row["_row_id"]}
            mapping = {("feishu", "alice"): "owner-alice"}
            alice_tasks = MaintenanceIntake(home, "owner-alice", mapping, enabled=True).pump()
            assert {task["actor"] for task in alice_tasks} == {"alice"}
            remaining = [read_committed_source(home, sid) for sid in pending_sources(home)]
            assert {source["actor"] for source in remaining} == {"bob"}
        finally:
            db.close()
    asyncio.run(scenario())


def test_feishu_reply_reference_does_not_turn_quote_into_user_text(tmp_path):
    async def scenario():
        mode = FEISHU_MODES["dm:peer"]
        adapter = make_adapter(mode["extra"])
        event, _ = await events_for(adapter, mode=mode, kind="reply_reference",
                                    chat_id="dm-reply", prefix="reply")
        key = adapter._event_session_key(event)
        home = tmp_path / "reply-reference"
        db = store(home, event, key)
        try:
            ingress = Ingress(home, adapter, key)
            assert await ingress._hm_admit_event(event)
            row_content = GatewayInboundMixin._prepend_inbound_reply_context(event, event.source, event.text)
            row = _db_flush_row(SimpleNamespace(), {
                "role": "user", "content": row_content, "platform_message_id": event.message_id,
            }, True)
            db.append_messages_batch("s1", [row])
            source = read_committed_source(home, pending_sources(home)[0])
            assert source["extraction_text"] == "new reply"
            assert source["raw_text"] == "new reply"
            assert source["display_text"] == row_content
            assert source["reply_to_text"] == "quoted old text"
            assert source["reply_to_author_id"] == "parent-author"
            assert "quoted old text" not in source["extraction_text"]
        finally:
            db.close()
    asyncio.run(scenario())


def test_feishu_duplicate_delivery_is_idempotent(tmp_path):
    async def scenario():
        mode = FEISHU_MODES["group:shared"]
        adapter = make_adapter(mode["extra"])
        event = (await events_for(adapter, mode=mode, kind="text", chat_id="dupe-chat",
                                  prefix="dupe"))[0]
        key = adapter._event_session_key(event)
        home = tmp_path / "duplicate"
        db = store(home, event, key)
        try:
            ingress = Ingress(home, adapter, key)
            assert await ingress._hm_admit_event(event)
            first = persist(db, event, key)
            source_id = pending_sources(home)[0]
            second = persist(db, event, key)
            assert second["_row_id"] != first["_row_id"]
            assert pending_sources(home) == [source_id]
            assert read_committed_source(home, source_id)["message_row_id"] == first["_row_id"]
        finally:
            db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["unauthorized", "internal", "receiver", "actor", "row"])
def test_feishu_untrusted_or_tampered_event_creates_no_source(tmp_path, change):
    async def scenario():
        mode = FEISHU_MODES["group:shared"]
        adapter = make_adapter(mode["extra"])
        event, _ = await events_for(adapter, mode=mode, kind="text", chat_id="negative-chat",
                                    prefix="negative")
        if change == "internal":
            event.internal = True
        if change == "receiver":
            adapter._bot_open_id = "different-receiver"
        if change == "actor":
            event.source.user_id = "mallory"
        key = adapter._event_session_key(event)
        home = tmp_path / ("negative-" + change)
        db = store(home, event, key)
        try:
            ingress = Ingress(home, adapter, key, authorized=change != "unauthorized")
            await ingress._hm_admit_event(event)
            if change == "row":
                row = _db_flush_row(SimpleNamespace(), {
                    "role": "user", "content": event.text, "platform_message_id": event.message_id,
                }, True)
                row["content"] = "tampered row"
                db.append_messages_batch("s1", [row])
            elif change != "unauthorized":
                persist(db, event, key)
            assert pending_sources(home) == []
        finally:
            db.close()
    asyncio.run(scenario())
