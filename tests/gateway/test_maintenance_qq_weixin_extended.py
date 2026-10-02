"""QQ quoted/control and Weixin mixed/control matrix through real SessionDB."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner
from gateway.platforms.qqbot.adapter import QQAdapter
from gateway.platforms.weixin import ITEM_IMAGE
from hermes_maintenance_source import (
    FEISHU_CONTROL_CONTRACT, committed_source_for_receipt, control_sources,
    pending_sources, read_committed_source,
)
from _maintenance_intake import MaintenanceIntake
from tests.gateway.test_maintenance_qq_weixin import (
    Ingress, persist, qq_adapter, store, weixin_adapter,
)
from tools import clarify_gateway


QQ_QUOTE_MODES = {
    "c2c_dm:peer": {"extra": {}, "kind": "c2c"},
    "group_at:per_actor": {"extra": {"group_sessions_per_user": True}, "kind": "group"},
    "group_at:shared": {"extra": {"group_sessions_per_user": False}, "kind": "group"},
    "guild_channel:per_actor": {"extra": {"group_sessions_per_user": True}, "kind": "guild"},
    "guild_channel:shared": {"extra": {"group_sessions_per_user": False}, "kind": "guild"},
    "guild_dm:peer": {"extra": {}, "kind": "guild_dm"},
}
QQ_CONTROL_MODES = {
    "c2c_dm": QQ_QUOTE_MODES["c2c_dm:peer"],
    "group_at": QQ_QUOTE_MODES["group_at:per_actor"],
    "guild_channel": QQ_QUOTE_MODES["guild_channel:per_actor"],
    "guild_dm": QQ_QUOTE_MODES["guild_dm:peer"],
}
WEIXIN_MIXED_MODES = {
    "dm:peer": {"extra": {}, "room": None, "chat_type": "dm"},
    "group:per_actor": {"extra": {"group_sessions_per_user": True}, "room": "wx-room", "chat_type": "group"},
    "group:shared": {"extra": {"group_sessions_per_user": False}, "room": "wx-room", "chat_type": "group"},
}
WEIXIN_CONTROL_MODES = {
    "dm": WEIXIN_MIXED_MODES["dm:peer"],
    "group": WEIXIN_MIXED_MODES["group:per_actor"],
}
CONTROLS = ("typed_clarify_response", "explicit_queue", "explicit_steer", "retry_previous_input")


def _qq_target(mode):
    kind = mode["kind"]
    if kind == "c2c":
        return {"handler": "_handle_c2c_message", "data": {}, "chat": "alice", "chat_type": "dm", "prefix": ""}
    if kind == "group":
        return {"handler": "_handle_group_message", "data": {"group_openid": "qq-room"},
                "chat": "qq-room", "chat_type": "group", "prefix": "@bot "}
    if kind == "guild":
        return {"handler": "_handle_guild_message",
                "data": {"channel_id": "qq-channel", "guild_id": "qq-guild"},
                "chat": "qq-channel", "chat_type": "group", "prefix": ""}
    return {"handler": "_handle_dm_message", "data": {"guild_id": "qq-guild-dm"},
            "chat": "qq-guild-dm", "chat_type": "dm", "prefix": ""}


def _qq_key(event, target, mode):
    per_actor = mode["extra"].get("group_sessions_per_user", True)
    key = f"agent:main:qqbot:{target['chat_type']}:{target['chat']}"
    return key + (":" + event.source.user_id if target["chat_type"] == "group" and per_actor else "")


async def emit_qq(adapter, mode_name, event_id, text, *, actor="alice", quote=None):
    mode = QQ_QUOTE_MODES.get(mode_name) or QQ_CONTROL_MODES[mode_name]
    target = _qq_target(mode)
    data = {"id": event_id, "content": target["prefix"] + text, **target["data"]}
    if quote is not None:
        data["message_type"] = 103
        data["msg_elements"] = [{
            "content": quote["text"], "message_id": quote["event_id"], "author": {"id": quote["author"]},
        }]
        adapter._process_quoted_context = QQAdapter._process_quoted_context.__get__(adapter, QQAdapter)
    if target["handler"] == "_handle_c2c_message":
        author = {"user_openid": actor, "id": actor, "username": actor}
    elif target["handler"] == "_handle_group_message":
        author = {"member_openid": actor, "id": actor, "username": actor}
    else:
        author = {"id": actor, "username": actor}
    content = target["prefix"] + text
    await getattr(adapter, target["handler"])(data, event_id, content, author, "2026-09-30T18:00:00+00:00")
    return target


async def wx_mixed(adapter, *, sender, event_id, text, room=None):
    async def collect(item, media_paths, media_types):
        if item.get("type") == ITEM_IMAGE:
            media_paths.append("cached-image.jpg")
            media_types.append("image/jpeg")
    adapter._collect_media = collect
    message = {
        "from_user_id": sender, "to_user_id": "wx-receiver", "message_id": event_id,
        "room_id": room,
        "item_list": [
            {"type": 1, "text_item": {"text": text}},
            {"type": ITEM_IMAGE, "image_item": {"media": {"full_url": "https://example.invalid/image"}}},
        ],
    }
    await adapter._process_message(message)


async def drain_weixin(adapter, key):
    for task in list(adapter._pending_text_batch_tasks.values()):
        task.cancel()
    if key in adapter._pending_text_batches:
        await adapter._flush_text_batch_now(key)


async def flush_weixin(adapter):
    keys = list(adapter._pending_text_batches)
    for task in list(adapter._pending_text_batch_tasks.values()):
        task.cancel()
    for key in keys:
        await adapter._flush_text_batch_now(key)


def make_runner(home, adapter, key, captured):
    runner = object.__new__(GatewayRunner)

    class Store:
        def __init__(self):
            self._store = self

        async def get_or_create_session(self, source):
            return SimpleNamespace(session_id="s1")

        async def load_transcript(self, session_id):
            return [{"role": "user", "content": "retry fact"}]

        async def rewrite_transcript(self, *args, **kwargs):
            return True

    store_obj = Store()
    runner._delivery_adapter_for = lambda source: adapter
    runner._resolve_profile_home_for_source = lambda source: home
    runner._session_key_for_source = lambda source: key
    runner.session_store = store_obj
    runner._async_session_store = store_obj
    runner._enqueue_fifo = lambda quick_key, event, selected_adapter: captured.append(event)
    runner._queue_depth = lambda quick_key, adapter=None: len(captured)
    runner._peek_session_state = lambda quick_key: None
    runner._record_model_friction = lambda *args, **kwargs: None
    return runner


async def admit_and_persist(ingress, db, event, key):
    admitted = await ingress._hm_admit_event(event)
    assert admitted is not None
    return await ingress._run_in_executor_with_context(persist, db, event, "s1", key)


@pytest.mark.parametrize("mode_name", list(QQ_QUOTE_MODES))
def test_qq_quoted_message_modes_preserve_current_authorship(tmp_path, mode_name):
    async def scenario():
        mode = QQ_QUOTE_MODES[mode_name]
        adapter, seen = qq_adapter(mode["extra"])
        await emit_qq(adapter, mode_name, f"{mode_name}-reply", "new reply", quote={
            "text": "quoted old text", "event_id": "parent-event", "author": "parent-author"})
        event, = seen
        target = _qq_target(mode)
        key = _qq_key(event, target, mode)
        home = tmp_path / mode_name.replace(":", "-")
        db = store(home, "qqbot", "alice", target["chat"], target["chat_type"], key)
        try:
            await admit_and_persist(Ingress(home, adapter, key), db, event, key)
            source_id, = pending_sources(home)
            source = read_committed_source(home, source_id)
            assert source["event_id"] == event.message_id
            assert source["actor"] == "alice"
            assert source["input_kind"] == "reply_reference"
            assert source["extraction_text"] == "new reply"
            assert source["reply_to_text"] == "quoted old text"
            assert source["reply_to_author_id"] == "parent-author"
            assert source["reply_to_message_id"] == "parent-event"
            assert "quoted old text" not in source["extraction_text"]
        finally:
            db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode_name", list(WEIXIN_MIXED_MODES))
def test_weixin_mixed_items_modes_keep_only_text_as_source(tmp_path, mode_name):
    async def scenario():
        mode = WEIXIN_MIXED_MODES[mode_name]
        adapter, seen = weixin_adapter(mode["extra"])
        await wx_mixed(adapter, sender="alice", event_id=f"{mode_name}-mixed", text="typed fact", room=mode["room"])
        event, = seen
        key = adapter._event_session_key(event)
        chat = mode["room"] or "alice"
        home = tmp_path / mode_name.replace(":", "-")
        db = store(home, "weixin", "alice", chat, mode["chat_type"], key)
        try:
            await admit_and_persist(Ingress(home, adapter, key), db, event, key)
            source_id, = pending_sources(home)
            source = read_committed_source(home, source_id)
            assert source["input_kind"] == "mixed_items"
            assert source["actor"] == "alice" and source["event_id"] == event.message_id
            assert source["raw_text"] == "typed fact" and source["extraction_text"] == "typed fact"
            assert "image" not in source["raw_text"].lower()
            assert json.loads(source["event_structure"])[1]["type"] == ITEM_IMAGE
        finally:
            db.close()
    asyncio.run(scenario())


async def _control_scenario(home, adapter, event, key, db, control, target_chat, target_type, actor="alice"):
    ingress = Ingress(home, adapter, key)
    admitted = await ingress._hm_admit_event(event)
    assert admitted is not None
    captured = []
    runner = make_runner(home, adapter, key, captured)
    if control == "typed_clarify_response":
        binding = {"task_id": "task-1", "item_id": "item-1", "proposal_revision": "1"}
        clarify_gateway.clear_session(key)
        clarify_gateway.register("clarify-control", key, "which database?", None, control_binding=binding)
        adapter.resume_typing_for_chat = lambda chat_id: None
        assert await runner._hm_clarify_reply(event, event.source, key) == ""
        source_id, = control_sources(home)
        source = read_committed_source(home, source_id)
        assert source["schema"] == FEISHU_CONTROL_CONTRACT
        assert source["control_kind"] == "typed_clarify_response"
        assert source["control_binding_json"] == json.dumps(binding, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        assert source["actor"] == actor
        return
    if control == "explicit_queue":
        await runner._busy_queue_command(event, key, event.source)
    else:
        await runner._busy_steer_command(event, key, event.source)
    assert len(captured) == 1
    control_source = captured[0]._maintenance_source_reuse
    assert control_source is not None and control_source.control_kind == control
    source = committed_source_for_receipt(home, "s1", control_source)
    assert source is not None and source["control_kind"] == control
    assert source["actor"] == actor
    platform = source["authority"].split("-", 1)[0]
    owner = "owner-" + actor
    task = MaintenanceIntake(home, owner, {(platform, actor): owner}, enabled=True).accept(source["source_id"])
    assert task["source_id"] == source["source_id"] and task["actor"] == actor


@pytest.mark.parametrize("mode_name", list(QQ_CONTROL_MODES))
@pytest.mark.parametrize("control", CONTROLS)
def test_qq_control_matrix_uses_shared_source_contract(tmp_path, mode_name, control):
    async def scenario():
        mode = QQ_CONTROL_MODES[mode_name]
        adapter, seen = qq_adapter(mode["extra"])
        if control == "retry_previous_input":
            await emit_qq(adapter, mode_name, f"{mode_name}-seed", "retry fact")
            original = seen[-1]
            target = _qq_target(mode)
            key = _qq_key(original, target, mode)
            home = tmp_path / f"{mode_name}-{control}".replace(":", "-")
            db = store(home, "qqbot", "alice", target["chat"], target["chat_type"], key)
            try:
                ingress = Ingress(home, adapter, key)
                await admit_and_persist(ingress, db, original, key)
                original_source_id, = pending_sources(home)
                await emit_qq(adapter, mode_name, f"{mode_name}-retry", "/retry")
                retry = seen[-1]
                captured = []
                runner = make_runner(home, adapter, key, captured)
                async def handle(event):
                    captured.append(event)
                    return "ok"
                runner._handle_message = handle
                assert await runner._handle_retry_command(retry) == "ok"
                reused_event, = captured
                assert reused_event._maintenance_source_reuse is not None
                assert await ingress._hm_admit_event(reused_event) is not None
                await ingress._run_in_executor_with_context(persist, db, reused_event, "s1", key)
                assert committed_source_for_receipt(home, "s1", reused_event._maintenance_source)["source_id"] == original_source_id
                assert pending_sources(home) == [original_source_id]
            finally:
                db.close()
            return
        text = "the project database" if control == "typed_clarify_response" else (
            "/queue queue fact" if control == "explicit_queue" else "/steer steer fact")
        await emit_qq(adapter, mode_name, f"{mode_name}-{control}", text)
        event, = seen
        target = _qq_target(mode)
        key = _qq_key(event, target, mode)
        home = tmp_path / f"{mode_name}-{control}".replace(":", "-")
        db = store(home, "qqbot", "alice", target["chat"], target["chat_type"], key)
        try:
            await _control_scenario(home, adapter, event, key, db, control, target["chat"], target["chat_type"])
        finally:
            db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode_name", list(WEIXIN_CONTROL_MODES))
@pytest.mark.parametrize("control", CONTROLS)
def test_weixin_control_matrix_uses_shared_source_contract(tmp_path, mode_name, control):
    async def scenario():
        mode = WEIXIN_CONTROL_MODES[mode_name]
        adapter, seen = weixin_adapter(mode["extra"])
        if control == "retry_previous_input":
            await adapter._process_message({"from_user_id": "alice", "to_user_id": "wx-receiver",
                                            "message_id": f"{mode_name}-seed", "room_id": mode["room"],
                                            "item_list": [{"type": 1, "text_item": {"text": "retry fact"}}]})
            await flush_weixin(adapter)
            original = seen[-1]
            key = adapter._event_session_key(original)
            home = tmp_path / f"wx-{mode_name}-{control}"
            chat = mode["room"] or "alice"
            db = store(home, "weixin", "alice", chat, mode["chat_type"], key)
            try:
                ingress = Ingress(home, adapter, key)
                await admit_and_persist(ingress, db, original, key)
                original_source_id, = pending_sources(home)
                await adapter._process_message({"from_user_id": "alice", "to_user_id": "wx-receiver",
                                                "message_id": f"{mode_name}-retry", "room_id": mode["room"],
                                                "item_list": [{"type": 1, "text_item": {"text": "/retry"}}]})
                retry = seen[-1]
                captured = []
                runner = make_runner(home, adapter, key, captured)
                async def handle(event):
                    captured.append(event)
                    return "ok"
                runner._handle_message = handle
                assert await runner._handle_retry_command(retry) == "ok"
                reused_event, = captured
                assert reused_event._maintenance_source_reuse is not None
                assert await ingress._hm_admit_event(reused_event) is not None
                await ingress._run_in_executor_with_context(persist, db, reused_event, "s1", key)
                assert committed_source_for_receipt(home, "s1", reused_event._maintenance_source)["source_id"] == original_source_id
                assert pending_sources(home) == [original_source_id]
            finally:
                db.close()
            return
        text = "the project database" if control == "typed_clarify_response" else (
            "/queue queue fact" if control == "explicit_queue" else "/steer steer fact")
        await adapter._process_message({"from_user_id": "alice", "to_user_id": "wx-receiver",
                                        "message_id": f"{mode_name}-{control}", "room_id": mode["room"],
                                        "item_list": [{"type": 1, "text_item": {"text": text}}]})
        if control == "typed_clarify_response":
            await flush_weixin(adapter)
        event = seen[-1]
        key = adapter._event_session_key(event)
        home = tmp_path / f"wx-{mode_name}-{control}"
        chat = mode["room"] or "alice"
        db = store(home, "weixin", "alice", chat, mode["chat_type"], key)
        try:
            await _control_scenario(home, adapter, event, key, db, control, chat, mode["chat_type"])
        finally:
            db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["actor", "event", "receiver", "internal"])
def test_qq_structured_source_negative_matrix(tmp_path, change):
    async def scenario():
        adapter, seen = qq_adapter({})
        await emit_qq(adapter, "c2c_dm:peer", "negative", "private fact")
        event, = seen
        if change == "actor":
            event.source.user_id = "mallory"
        elif change == "event":
            event.message_id = "changed"
        elif change == "receiver":
            adapter._app_id = "different"
        else:
            event.internal = True
        home = tmp_path / change
        db = store(home, "qqbot", "alice", "alice", "dm", "agent:main:qqbot:dm:alice")
        try:
            ingress = Ingress(home, adapter, "agent:main:qqbot:dm:alice")
            await ingress._hm_admit_event(event)
            assert getattr(event, "_maintenance_source", None) is None
            await ingress._run_in_executor_with_context(
                persist, db, event, "s1", "agent:main:qqbot:dm:alice")
            assert pending_sources(home) == []
        finally:
            db.close()
    asyncio.run(scenario())


def test_qq_duplicate_delivery_is_idempotent(tmp_path):
    async def scenario():
        adapter, seen = qq_adapter({})
        key = "agent:main:qqbot:dm:alice"
        home = tmp_path / "duplicate"
        db = store(home, "qqbot", "alice", "alice", "dm", key)
        try:
            for _ in range(2):
                await emit_qq(adapter, "c2c_dm:peer", "same-event", "same fact")
                event = seen[-1]
                await admit_and_persist(Ingress(home, adapter, key), db, event, key)
            assert len(pending_sources(home)) == 1
        finally:
            db.close()
    asyncio.run(scenario())


def test_qq_control_source_tamper_is_rejected(tmp_path):
    async def scenario():
        adapter, seen = qq_adapter({})
        await emit_qq(adapter, "c2c_dm:peer", "tamper", "/queue tamper")
        event, = seen
        key = "agent:main:qqbot:dm:alice"
        home = tmp_path / "tamper"
        db = store(home, "qqbot", "alice", "alice", "dm", key)
        try:
            admitted = await Ingress(home, adapter, key)._hm_admit_event(event)
            assert admitted is not None
            captured = []
            await make_runner(home, adapter, key, captured)._busy_queue_command(event, key, event.source)
            source_id = committed_source_for_receipt(home, "s1", captured[0]._maintenance_source_reuse)["source_id"]
            with sqlite3.connect(home / "state.db") as conn:
                conn.execute("UPDATE maintenance_sources_v1 SET raw_text=? WHERE source_id=?", (b"tampered", source_id))
                conn.commit()
            assert read_committed_source(home, source_id) is None
        finally:
            db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["actor", "event", "receiver", "internal"])
def test_weixin_structured_source_negative_matrix(tmp_path, change):
    async def scenario():
        adapter, seen = weixin_adapter({})
        await wx_mixed(adapter, sender="alice", event_id="negative", text="private fact", room="wx-room")
        event, = seen
        if change == "actor":
            event.source.user_id = "mallory"
        elif change == "event":
            event.message_id = "changed"
        elif change == "receiver":
            adapter._account_id = "different"
        else:
            event.internal = True
        key = "agent:main:weixin:group:wx-room"
        home = tmp_path / ("wx-" + change)
        db = store(home, "weixin", "alice", "wx-room", "group", key)
        try:
            ingress = Ingress(home, adapter, key)
            await ingress._hm_admit_event(event)
            assert getattr(event, "_maintenance_source", None) is None
            await ingress._run_in_executor_with_context(persist, db, event, "s1", key)
            assert pending_sources(home) == []
        finally:
            db.close()
    asyncio.run(scenario())


def test_weixin_duplicate_delivery_is_idempotent(tmp_path):
    async def scenario():
        adapter, seen = weixin_adapter({})
        key = "agent:main:weixin:dm:alice"
        home = tmp_path / "wx-duplicate"
        db = store(home, "weixin", "alice", "alice", "dm", key)
        try:
            message = {"from_user_id": "alice", "to_user_id": "wx-receiver",
                       "message_id": "same-event", "room_id": None,
                       "item_list": [{"type": 1, "text_item": {"text": "same fact"}}]}
            await adapter._process_message(message)
            await adapter._process_message(message)
            await flush_weixin(adapter)
            event, = seen
            await admit_and_persist(Ingress(home, adapter, key), db, event, key)
            assert len(pending_sources(home)) == 1
        finally:
            db.close()
    asyncio.run(scenario())


def test_retry_latest_projection_owned_by_another_author_fails_closed(tmp_path):
    async def scenario():
        mode_name = "group_at:per_actor"
        mode = QQ_QUOTE_MODES[mode_name]
        adapter, seen = qq_adapter(mode["extra"])
        await emit_qq(adapter, mode_name, "author-seed", "alice fact", actor="alice")
        original = seen[-1]
        target = _qq_target(mode)
        key = _qq_key(original, target, mode)
        home = tmp_path / "retry-author"
        db = store(home, "qqbot", "alice", target["chat"], target["chat_type"], key)
        try:
            ingress = Ingress(home, adapter, key)
            await admit_and_persist(ingress, db, original, key)
            await emit_qq(adapter, mode_name, "author-retry", "/retry", actor="bob")
            retry = seen[-1]
            from hermes_maintenance_source import (admit_reused_platform_source,
                rehydrate_source_receipt, source_for_latest_user_message)
            source_id = source_for_latest_user_message(home, "s1")
            reused = rehydrate_source_receipt(home, source_id)
            assert reused is not None
            assert admit_reused_platform_source(reused, retry, adapter, home, key) is None
        finally:
            db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("control,text", [
    ("explicit_queue", "/queue queue fact"),
    ("explicit_steer", "/steer steer fact"),
])
def test_qq_idle_control_replaces_plain_carry_with_single_control_source(tmp_path, control, text):
    async def scenario():
        adapter, seen = qq_adapter({})
        await emit_qq(adapter, "c2c_dm:peer", control + "-idle", text)
        event, = seen
        key = "agent:main:qqbot:dm:alice"
        home = tmp_path / (control + "-idle")
        db = store(home, "qqbot", "alice", "alice", "dm", key)
        try:
            ingress = Ingress(home, adapter, key)
            assert await ingress._hm_admit_event(event) is not None
            runner = make_runner(home, adapter, key, [])
            handler = runner._hm_cmd_queue if control == "explicit_queue" else runner._hm_cmd_steer
            handled, result = await handler(event, event.source, key)
            assert handled is False and result is None
            assert event.text == text.split(maxsplit=1)[1]
            control_source = event._maintenance_source_reuse
            assert control_source is not None and control_source.control_kind == control
            await ingress._run_in_executor_with_context(persist, db, event, "s1", key)
            source_id, = pending_sources(home)
            committed = read_committed_source(home, source_id)
            assert committed["control_kind"] == control
            assert committed["raw_text"] == text.split(maxsplit=1)[1]
        finally:
            db.close()
    asyncio.run(scenario())
