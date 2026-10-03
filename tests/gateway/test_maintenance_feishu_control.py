"""Feishu control-source matrix across the four canonical control modes.

Synthetic authenticated Feishu events against the real SessionDB and isolated
SQLite journals. These are not live-channel acceptance tests.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.session_persistence import _db_flush_row
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.run_inbound import GatewayInboundMixin
from hermes_maintenance_source import (
    FEISHU_CONTROL_CONTRACT, admit_reused_feishu_source, begin_host_question_delivery,
    commit_feishu_control_source, committed_source_for_receipt, control_sources,
    derive_feishu_control_source, mint_feishu_text, pending_sources, read_committed_source,
    record_host_question_delivery, rehydrate_source_receipt, source_for_session_row,
)
from hermes_state import SessionDB
from _maintenance_intake import MaintenanceIntake
from plugins.platforms.feishu.adapter import FeishuAdapter


MODES = {
    "dm:peer": {"chat_info": "private", "event_chat": "p2p", "thread": None, "extra": {}},
    "dm_thread:peer": {"chat_info": "private", "event_chat": "p2p", "thread": "dm-thread", "extra": {}},
    "group:per_actor": {"chat_info": "group", "event_chat": "group", "thread": None,
                        "extra": {"group_sessions_per_user": True}},
    "group_thread:shared": {"chat_info": "group", "event_chat": "group", "thread": "group-thread",
                             "extra": {"group_sessions_per_user": True, "thread_sessions_per_user": False}},
}
CONTROLS = ("typed_clarify_response", "explicit_queue", "explicit_steer", "retry_previous_input")


class Ingress(GatewayInboundMixin):
    def __init__(self, home, adapter, key, authorized=True):
        self.home, self.adapter, self.key, self.authorized = home, adapter, key, authorized

    def _scale_to_zero_note_real_inbound(self):
        pass

    async def _hm_pre_gateway_dispatch_hook(self, event, source):
        return event

    def _is_user_authorized_for_source(self, source):
        return self.authorized

    def _admit_bot_message_for_source(self, source):
        return True

    def _intake_adapter_for(self, source):
        return self.adapter

    def _resolve_profile_home_for_source(self, source):
        return self.home

    def _session_key_for_source(self, source):
        return self.key

    def _get_unauthorized_dm_behavior(self, *args, **kwargs):
        return "ignore"

    async def _hm_report_ignored_dm(self, source):
        pass

    def _get_executor(self):
        return None

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


async def emit(adapter, *, text, chat_id, mode, event_id, actor="alice"):
    sender_id = SimpleNamespace(open_id=actor, user_id="", union_id="")
    message = SimpleNamespace(
        content=json.dumps({"text": text}), message_type="text", message_id=event_id,
        mentions=[], chat_id=chat_id, chat_type=mode["event_chat"], thread_id=mode["thread"],
        parent_id=None, upper_message_id=None, root_id=None,
    )
    adapter.get_chat_info.return_value = {"name": "feishu-chat", "type": mode["chat_info"]}
    adapter._resolve_sender_profile.return_value = {
        "user_id": actor, "user_name": actor.title(), "user_id_alt": None,
    }
    data = SimpleNamespace(event=SimpleNamespace(
        message=message,
        sender=SimpleNamespace(sender_type="user", sender_id=sender_id),
    ))
    await adapter._process_inbound_message(
        data=data, message=message, sender_id=sender_id,
        chat_type=mode["event_chat"], message_id=event_id, is_bot=False,
    )
    return adapter._dispatched[-1]


def store(home, *, user_id, chat_id, chat_type, thread_id, key):
    home.mkdir(parents=True, exist_ok=True)
    (home / "mem0.json").write_text(json.dumps({
        "governed": {"maintenance_intake": True},
    }), encoding="utf-8")
    db = SessionDB(db_path=home / "state.db")
    db.create_session("s1", "feishu", user_id=user_id, chat_id=chat_id,
                      chat_type=chat_type, thread_id=thread_id, session_key=key)
    return db


def persist(db, event):
    from hermes_inbound_evidence import authenticated_turn
    session = db.get_session("s1")
    key = session["session_key"]
    with authenticated_turn("s1", key):
        row = _db_flush_row(SimpleNamespace(), {
            "role": "user", "content": event.text,
            "platform_message_id": event.message_id,
        }, True)
        db.append_messages_batch("s1", [row])
    return row


def owner_map(actor="alice"):
    return {("feishu", actor): "owner-" + actor}


async def admit(ingress, event):
    admitted = await ingress._hm_admit_event(event)
    assert admitted is not None
    return admitted


@pytest.mark.parametrize("mode_name", list(MODES))
@pytest.mark.parametrize("control", CONTROLS)
def test_feishu_control_matrix_source_identity_and_task_boundary(tmp_path, mode_name, control):
    async def scenario():
        mode = MODES[mode_name]
        adapter = make_adapter(mode["extra"])
        chat_id = "chat-" + mode_name.replace(":", "-")
        seed = await emit(adapter, text="seed", chat_id=chat_id, mode=mode, event_id="seed")
        key = adapter._event_session_key(seed)
        source_chat_type = "dm" if mode["event_chat"] == "p2p" else "group"
        home = tmp_path / (mode_name.replace(":", "-") + "-" + control)
        db = store(home, user_id="alice", chat_id=chat_id, chat_type=source_chat_type,
                   thread_id=mode["thread"], key=key)
        try:
            ingress = Ingress(home, adapter, key)
            if control == "retry_previous_input":
                original = await emit(adapter, text="retry fact", chat_id=chat_id, mode=mode,
                                      event_id="retry-original")
                await admit(ingress, original)
                row = persist(db, original)
                original_source_id = committed_source_for_receipt(
                    home, "s1", original._maintenance_source)["source_id"]
                retry = await emit(adapter, text="/retry", chat_id=chat_id, mode=mode,
                                   event_id="retry-command")
                await admit(ingress, retry)
                resolved_source_id = source_for_session_row(home, "s1", row["_row_id"])
                reused = rehydrate_source_receipt(home, resolved_source_id)
                assert resolved_source_id == original_source_id
                assert admit_reused_feishu_source(reused, retry, adapter, home, key) is reused
                assert pending_sources(home) == [original_source_id]
                intake = MaintenanceIntake(home, "owner-alice", owner_map(), enabled=True)
                first_task = intake.accept(original_source_id)
                retried_task = intake.accept(resolved_source_id)
                assert retried_task["task_id"] == first_task["task_id"]
                return

            if control == "typed_clarify_response":
                event = await emit(adapter, text="the project database", chat_id=chat_id,
                                   mode=mode, event_id="clarify-answer")
                await admit(ingress, event)
                binding = {"task_id": "task-1", "item_id": "item-1", "proposal_revision": "1"}
                base = mint_feishu_text(event, adapter, home, key)
                receipt = derive_feishu_control_source(
                    base, kind=control, text=event.text, binding=binding)
            else:
                command = "/queue queue fact" if control == "explicit_queue" else "/steer steer fact"
                event = await emit(adapter, text=command, chat_id=chat_id, mode=mode,
                                   event_id=control + "-command")
                await admit(ingress, event)
                base = mint_feishu_text(event, adapter, home, key)
                receipt = derive_feishu_control_source(
                    base, kind=control,
                    text=event.get_command_args().strip())
            source_id = commit_feishu_control_source(home, "s1", receipt)
            assert source_id is not None
            source = read_committed_source(home, source_id)
            assert source["schema"] == FEISHU_CONTROL_CONTRACT
            assert source["control_kind"] == control
            assert source["actor"] == "alice"
            assert source["event_id"] == receipt.event_id
            if control == "typed_clarify_response":
                assert source["control_binding_json"] == json.dumps(
                    binding, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                assert control_sources(home) == [source_id]
                with pytest.raises(ValueError, match="MAINTENANCE_INTAKE_DISABLED|SOURCE_NOT_ENROLLED|SOURCE_LEGACY_OWNED"):
                    MaintenanceIntake(home, "owner-alice", owner_map(), enabled=True).accept(source_id)
            else:
                task = MaintenanceIntake(home, "owner-alice", owner_map(), enabled=True).accept(source_id)
                assert task["source_id"] == source_id
                assert task["actor"] == "alice"
        finally:
            db.close()
    asyncio.run(scenario())


def test_bound_clarify_requires_one_host_candidate():
    async def scenario():
        from tools import clarify_gateway
        key = "feishu:control-ambiguous"
        clarify_gateway.clear_session(key)
        clarify_gateway.register("c1", key, "q1", None,
                                 control_binding={"task_id": "t1", "item_id": "i1", "proposal_revision": "1"})
        clarify_gateway.register("c2", key, "q2", None,
                                 control_binding={"task_id": "t2", "item_id": "i2", "proposal_revision": "1"})
        selected, candidates = clarify_gateway.get_pending_control_for_session(key)
        assert selected is None and len(candidates) == 2
        clarify_gateway.clear_session(key)
    asyncio.run(scenario())


def test_control_negative_internal_wrong_author_event_and_tamper(tmp_path):
    async def scenario():
        mode = MODES["group:per_actor"]
        adapter = make_adapter(mode["extra"])
        chat_id = "negative"
        seed = await emit(adapter, text="seed", chat_id=chat_id, mode=mode, event_id="neg-seed")
        key = adapter._event_session_key(seed)
        home = tmp_path / "negative"
        db = store(home, user_id="alice", chat_id=chat_id, chat_type="group",
                   thread_id=None, key=key)
        try:
            ingress = Ingress(home, adapter, key)
            internal = await emit(adapter, text="/queue hidden", chat_id=chat_id, mode=mode,
                                  event_id="neg-internal")
            internal.internal = True
            await admit(ingress, internal)
            assert getattr(internal, "_maintenance_source", None) is None

            event = await emit(adapter, text="/queue visible", chat_id=chat_id, mode=mode,
                               event_id="neg-visible")
            event.source.user_id = "mallory"
            await ingress._hm_admit_event(event)
            assert getattr(event, "_maintenance_source", None) is None
            assert pending_sources(home) == []

            event = await emit(adapter, text="/queue visible", chat_id=chat_id, mode=mode,
                               event_id="neg-event")
            event.message_id = "neg-forged-event"
            await ingress._hm_admit_event(event)
            assert getattr(event, "_maintenance_source", None) is None
            assert pending_sources(home) == []

            event = await emit(adapter, text="/queue visible", chat_id=chat_id, mode=mode,
                               event_id="neg-tamper")
            await admit(ingress, event)
            receipt = derive_feishu_control_source(event._maintenance_source,
                                                   kind="explicit_queue", text="visible")
            source_id = commit_feishu_control_source(home, "s1", receipt)
            import sqlite3
            con = sqlite3.connect(home / "state.db")
            con.execute("UPDATE maintenance_sources_v1 SET raw_text=? WHERE source_id=?",
                        (b"tampered", source_id))
            con.commit()
            con.close()
            assert read_committed_source(home, source_id) is None
        finally:
            db.close()
    asyncio.run(scenario())


def make_runner(home, adapter, key, captured):
    runner = object.__new__(GatewayRunner)

    class Store:
        def __init__(self):
            self._store = self

        async def get_or_create_session(self, source):
            return SimpleNamespace(session_id="s1")

    store = Store()
    runner._delivery_adapter_for = lambda source: adapter
    runner._resolve_profile_home_for_source = lambda source: home
    runner._session_key_for_source = lambda source: key
    runner.session_store = store
    runner._async_session_store = store
    runner._enqueue_fifo = lambda quick_key, event, selected_adapter: captured.append(event)
    runner._queue_depth = lambda quick_key, adapter=None: len(captured)
    runner._peek_session_state = lambda quick_key: None
    return runner


@pytest.mark.parametrize("control,text", [
    ("explicit_queue", "/queue queue fact"),
    ("explicit_steer", "/steer steer fact"),
])
def test_busy_queue_and_steer_commit_carry_source(tmp_path, control, text):
    async def scenario():
        mode = MODES["dm_thread:peer"]
        adapter = make_adapter(mode["extra"])
        chat_id = "busy-" + control
        seed = await emit(adapter, text="seed", chat_id=chat_id, mode=mode, event_id="busy-seed")
        key = adapter._event_session_key(seed)
        home = tmp_path / ("busy-" + control)
        db = store(home, user_id="alice", chat_id=chat_id, chat_type="dm",
                   thread_id=mode["thread"], key=key)
        try:
            event = await emit(adapter, text=text, chat_id=chat_id, mode=mode,
                               event_id=control + "-busy")
            await admit(Ingress(home, adapter, key), event)
            captured = []
            runner = make_runner(home, adapter, key, captured)
            if control == "explicit_queue":
                await runner._busy_queue_command(event, key, event.source)
            else:
                await runner._busy_steer_command(event, key, event.source)
            assert len(captured) == 1
            reuse = captured[0]._maintenance_source_reuse
            assert reuse is not None and reuse.control_kind == control
            source = committed_source_for_receipt(home, "s1", reuse)
            assert source is not None and source["control_kind"] == control
            assert source["raw_text"] == text.split(maxsplit=1)[1]
        finally:
            db.close()
    asyncio.run(scenario())


def test_typed_clarify_entry_commits_bound_source(tmp_path):
    async def scenario():
        from tools import clarify_gateway
        mode = MODES["group:per_actor"]
        adapter = make_adapter(mode["extra"])
        chat_id = "clarify-entry"
        seed = await emit(adapter, text="seed", chat_id=chat_id, mode=mode, event_id="clarify-seed")
        key = adapter._event_session_key(seed)
        home = tmp_path / "clarify-entry"
        db = store(home, user_id="alice", chat_id=chat_id, chat_type="group",
                   thread_id=None, key=key)
        try:
            event = await emit(adapter, text="the project database", chat_id=chat_id,
                               mode=mode, event_id="clarify-entry-answer")
            await admit(Ingress(home, adapter, key), event)
            binding = {"task_id": "task-1", "item_id": "item-1", "proposal_revision": "1"}
            clarify_gateway.clear_session(key)
            entry = clarify_gateway.register("clarify-control", key, "which database?", None,
                                             control_binding=binding)
            adapter.resume_typing_for_chat = lambda chat_id: None
            runner = make_runner(home, adapter, key, [])
            assert await runner._hm_clarify_reply(event, event.source, key) == ""
            assert entry.event.is_set() and entry.response == "the project database"
            sources = control_sources(home)
            assert len(sources) == 1
            source = read_committed_source(home, sources[0])
            assert source["control_kind"] == "typed_clarify_response"
            assert source["control_binding_json"] == json.dumps(
                binding, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            clarify_gateway.clear_session(key)
        finally:
            db.close()
    asyncio.run(scenario())


def test_retry_entry_reuses_original_source(tmp_path):
    async def scenario():
        mode = MODES["dm_thread:peer"]
        adapter = make_adapter(mode["extra"])
        chat_id = "retry-entry"
        seed = await emit(adapter, text="seed", chat_id=chat_id, mode=mode, event_id="retry-entry-seed")
        key = adapter._event_session_key(seed)
        home = tmp_path / "retry-entry"
        db = store(home, user_id="alice", chat_id=chat_id, chat_type="dm",
                   thread_id=mode["thread"], key=key)
        try:
            original = await emit(adapter, text="retry fact", chat_id=chat_id, mode=mode,
                                  event_id="retry-entry-original")
            await admit(Ingress(home, adapter, key), original)
            row = persist(db, original)
            original_source_id = committed_source_for_receipt(
                home, "s1", original._maintenance_source)["source_id"]
            retry = await emit(adapter, text="/retry", chat_id=chat_id, mode=mode,
                               event_id="retry-entry-command")
            captured = []

            class Store:
                def __init__(self):
                    self._store = self

                async def get_or_create_session(self, source):
                    return SimpleNamespace(session_id="s1")

                async def load_transcript(self, session_id):
                    return [{"role": "user", "content": "retry fact"}]

                async def rewrite_transcript(self, *args, **kwargs):
                    return True

            retry_store = Store()
            runner = object.__new__(GatewayRunner)
            runner.session_store = retry_store
            runner._async_session_store = retry_store
            runner._session_db = SimpleNamespace(
                latest_message_row_id=lambda session_id, role="user": row["_row_id"])
            runner._resolve_profile_home_for_source = lambda source: home
            runner._record_model_friction = lambda *args, **kwargs: None

            async def handle(event):
                captured.append(event)
                return "ok"

            runner._handle_message = handle
            assert await runner._handle_retry_command(retry) == "ok"
            assert len(captured) == 1
            reuse = captured[0]._maintenance_source_reuse
            assert reuse is not None
            assert committed_source_for_receipt(home, "s1", reuse)["source_id"] == original_source_id
            assert pending_sources(home) == [original_source_id]
        finally:
            db.close()
    asyncio.run(scenario())


def test_group_retry_cannot_reuse_another_authors_source(tmp_path):
    async def scenario():
        mode = MODES["group_thread:shared"]
        adapter = make_adapter(mode["extra"])
        chat_id = "shared-retry-author"
        seed = await emit(adapter, text="seed", chat_id=chat_id, mode=mode, event_id="shared-seed")
        key = adapter._event_session_key(seed)
        home = tmp_path / "shared-retry-author"
        db = store(home, user_id="alice", chat_id=chat_id, chat_type="group",
                   thread_id=mode["thread"], key=key)
        try:
            original = await emit(adapter, text="alice fact", chat_id=chat_id, mode=mode,
                                  event_id="shared-original", actor="alice")
            await admit(Ingress(home, adapter, key), original)
            row = persist(db, original)
            original_source_id = committed_source_for_receipt(
                home, "s1", original._maintenance_source)["source_id"]
            retry = await emit(adapter, text="/retry", chat_id=chat_id, mode=mode,
                               event_id="shared-retry", actor="bob")
            resolved_source_id = source_for_session_row(home, "s1", row["_row_id"])
            assert resolved_source_id == original_source_id
            reused = rehydrate_source_receipt(home, resolved_source_id)
            assert admit_reused_feishu_source(reused, retry, adapter, home, key) is None
        finally:
            db.close()
    asyncio.run(scenario())



def test_persisted_host_delivery_hydrates_one_typed_control_source(tmp_path):
    async def scenario():
        from tools import clarify_gateway

        mode = MODES["dm:peer"]
        adapter = make_adapter(mode["extra"])
        chat_id = "persisted-host-delivery"
        seed = await emit(adapter, text="seed", chat_id=chat_id, mode=mode,
                          event_id="persisted-host-seed")
        key = adapter._event_session_key(seed)
        home = tmp_path / "persisted-host-delivery"
        db = store(home, user_id="alice", chat_id=chat_id, chat_type="dm",
                   thread_id=None, key=key)
        try:
            answer = await emit(adapter, text="the project database", chat_id=chat_id,
                                mode=mode, event_id="persisted-host-answer")
            await admit(Ingress(home, adapter, key), answer)
            binding = {"task_id": "task-1", "item_id": "item-1", "proposal_revision": "1"}
            begin_host_question_delivery(
                home, "Q-HOST-1", "owner-alice", key, "Which database?", None,
                binding, expires_at=time.time() + 3600)
            # PREPARED is not answerable; only the real host send receipt arms it.
            clarify_gateway.clear_session(key)
            runner = make_runner(home, adapter, key, [])
            assert await runner._hm_clarify_reply(answer, answer.source, key) is None
            assert control_sources(home) == []

            record_host_question_delivery(
                home, "Q-HOST-1", channel="feishu", session_key=key,
                message_ref="message-host-1")
            runner = make_runner(home, adapter, key, [])
            assert await runner._hm_clarify_reply(answer, answer.source, key) == ""
            source_id, = control_sources(home)
            source = read_committed_source(home, source_id)
            assert source is not None and source["control_kind"] == "typed_clarify_response"
            assert json.loads(source["control_binding_json"]) == binding
            assert clarify_gateway.get_pending_control_for_session(key) == (None, ())
        finally:
            clarify_gateway.clear_session(key)
            db.close()

    asyncio.run(scenario())
