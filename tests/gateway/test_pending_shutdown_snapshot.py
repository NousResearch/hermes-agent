"""Shutdown preserves ordered event context without authorising live replay."""

import asyncio
import json
from dataclasses import fields
from datetime import datetime

import pytest

from gateway.platforms.base_pending import release_pending_dispatch, reserve_pending_dispatch
from gateway.platforms.event import MessageType
from hermes_constants import get_hermes_home
from hermes_state_registry import close_all_under
from tests.gateway.test_active_session_text_merge import _make_event, _make_initialized_adapter
from tests.gateway.test_busy_followup_after_session_release import _QueueRunner


def _wire_event(event):
    value = {
        item.name: getattr(event, item.name)
        for item in fields(event)
        if item.init
        and not item.name.startswith("_")
        and item.name not in {"raw_message", "source"}
    }
    value["source"] = {
        item.name: getattr(event.source, item.name) for item in fields(event.source)
    }
    value["message_type"] = event.message_type.value
    value["source"]["platform"] = event.source.platform.value
    value["timestamp"] = event.timestamp.isoformat()
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("reservation", ["absent", "unclaimed", "claimed", "cancel-before", "claim-during", "late-provisional"])
async def test_shutdown_preserves_the_complete_ordered_pending_session(tmp_path, monkeypatch, reservation):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_initialized_adapter()
    runner = _QueueRunner(adapter)
    adapter.gateway_runner = runner
    adapter._event_session_key = lambda event: "shared"
    adapter._busy_text_debounce_seconds = 10
    events = [_make_event(str(i), user_id=f"user-{i}") for i in range(5)]
    for event in events:
        event.timestamp = datetime(2026, 10, 1, 12)
        event.metadata = {"hermes_plugin_id": "example", "nested": {"future": [1, False]}}
        event.reply_to_message_id, event.reply_to_text = "quote", "quoted text"
        event.reply_to_author_id, event.reply_to_author_name = "author", "Quoted author"
        event.reply_to_is_own_message = True
    events[1].message_type, events[1].text = MessageType.PHOTO, ""
    events[1].media_urls, events[1].media_types, events[1].media_text_inlined = [str(tmp_path / "image.jpg")], ["image/jpeg"], [False]
    events[2].internal, events[2].allow_gateway_control = True, False
    for event in events[1:3]:
        runner._enqueue_fifo("shared", event, adapter)
    for event in events[3:]:
        assert await adapter._queue_text_debounce("shared", event)
    if reservation != "absent":
        reserve_pending_dispatch(adapter, "shared", events[0])
        adapter._pending_dispatch_reservations["shared"].claimed = reservation == "claimed"
    if reservation in {"cancel-before", "claim-during", "late-provisional"}:
        entered = asyncio.Event()
        async def handler(event):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if reservation == "late-provisional" and event is events[0]:
                    later = adapter._pending_messages.pop("shared")
                    adapter._stage_next_queued_event("shared", later)
                    adapter._start_session_processing(later, "shared")
                if reservation == "claim-during":
                    release_pending_dispatch(adapter, "shared", event, claimed=True)
                    return None
                raise
        adapter.set_message_handler(handler)
        adapter._start_session_processing(events[0], "shared")
        if reservation in {"claim-during", "late-provisional"}:
            await asyncio.wait_for(entered.wait(), 2)
    expected = [_wire_event(event) for event in (events if reservation in {"unclaimed", "cancel-before", "late-provisional"} else events[1:])]

    await adapter.cancel_background_tasks()

    payloads = [json.loads(path.read_text()) for path in (tmp_path / "pending_messages").glob("*.json")]
    assert len(payloads) == 1
    payload = payloads[0]
    assert (payload["schema"], payload["version"], payload["session_key"],
            [record["event"] for record in payload["events"]],
            list(runner._overflow_queue("shared") or ()), adapter._pending_messages) == (
        "hermes.gateway.pending", 1, "shared", expected, [], {})


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["valid", "invalid-version", "copied-home", "write-failure", "projection-crash", "orphan", "invalid-event"])
async def test_snapshot_projection_keeps_records_in_the_owning_profile(tmp_path, monkeypatch, case):
    import weakref
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session import build_session_key
    from gateway.session_identity import RoutingIdentity
    from gateway.shutdown_flush import recover_pending_to_db
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    launch = tmp_path / "launch"
    launch.mkdir()
    homes = {profile: tmp_path / profile for profile in ("a", "b")}
    for home in homes.values():
        home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    active = is_multiplex_active()
    set_multiplex_active(True)
    dbs = {profile: SessionDB(db_path=home / "state.db") for profile, home in homes.items()}
    runner = None
    try:
        adapter = _make_initialized_adapter()
        runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
        runner.adapters[adapter.platform] = adapter
        adapter.gateway_runner = runner
        events = []
        for index, profile in enumerate(("a", "b", "a")):
            event = _make_event(f"input-{index}", user_id=f"user-{index}")
            event.source.profile = profile
            event.source._identity = RoutingIdentity("default", profile, launch, homes[profile],
                                                      transport=weakref.ref(adapter))
            events.append(event)
            runner._enqueue_fifo(build_session_key(event.source, profile=profile), event, adapter)
        if case == "orphan":
            key = build_session_key(events[1].source, profile="b")
            runner._session_state(key).conversation.queued_events.insert(0, adapter._pending_messages.pop(key))
        for profile, db in dbs.items():
            db.create_session(f"session-{profile}", "gateway")
        if case == "write-failure":
            from gateway.shutdown_flush import _write_payload
            def write(directory, payload):
                if get_hermes_home() == homes["a"]:
                    raise OSError("controlled full disk")
                return _write_payload(directory, payload)
            monkeypatch.setattr("gateway.shutdown_flush._write_payload", write)

        await adapter.cancel_background_tasks()
        paths = {profile: list((home / "pending_messages").glob("*.json")) for profile, home in homes.items()}
        assert {profile: len(files) for profile, files in paths.items()} == {"a": 0 if case == "write-failure" else 1, "b": 1}
        before = {path: json.loads(path.read_text()) for files in paths.values() for path in files}
        if case == "invalid-version":
            before[paths["a"][0]]["version"] = 999
            paths["a"][0].write_text(json.dumps(before[paths["a"][0]]))
        if case == "invalid-event":
            before[paths["a"][0]]["events"][0]["event"]["media_urls"] = "invalid attachment list"
            paths["a"][0].write_text(json.dumps(before[paths["a"][0]]))
        if case == "copied-home":
            copied = homes["b"] / "pending_messages" / "copied.json"
            copied.write_bytes(paths["a"][0].read_bytes())
            before[copied] = json.loads(copied.read_text())
        if case == "projection-crash":
            from utils import atomic_json_write
            failed = False
            def publish(path, payload, **kwargs):
                nonlocal failed
                if get_hermes_home() == homes["a"] and payload.get("projection") and not failed:
                    failed = True
                    raise OSError("controlled crash after SQLite commit")
                return atomic_json_write(path, payload, **kwargs)
            monkeypatch.setattr("utils.atomic_json_write", publish)
        counts = []
        for profile in ("a", "b", "a"):
            with runner._profile_scope_for_source(next(event.source for event in events if event.source.profile == profile)):
                counts.append(recover_pending_to_db(dbs[profile], session_resolver=lambda key, **kw: (
                    f"session-{profile}", dbs[profile])))
        contents = {profile: [row["content"] for row in db.get_messages(f"session-{profile}")] for profile, db in dbs.items()}
        assert (counts, contents, get_hermes_home()) == (
            [0 if case in {"invalid-version", "invalid-event", "write-failure", "projection-crash"} else 2, 1, 0],
            {"a": [] if case in {"invalid-version", "invalid-event", "write-failure"} else [
                "[Pending input preserved at gateway shutdown; not executed]\ninput-0",
                "[Pending input preserved at gateway shutdown; not executed]\ninput-2"],
             "b": ["[Pending input preserved at gateway shutdown; not executed]\ninput-1"]}, launch)
        assert {path: {key: value for key, value in json.loads(path.read_text()).items() if key != "projection"}
                for path in before} == before
        if case == "write-failure":
            from gateway.shutdown_pending import flush_runner_pending
            key = build_session_key(events[0].source, profile="a")
            retained_queue = runner._overflow_queue(key)
            assert retained_queue is not None
            assert list(retained_queue) == [events[0], events[2]]
            monkeypatch.setattr("gateway.shutdown_flush._write_payload", _write_payload)
            flush_runner_pending(runner)
            with runner._profile_scope_for_source(events[0].source):
                assert recover_pending_to_db(dbs["a"], session_resolver=lambda key, **kw: (
                    "session-a", dbs["a"])) == 2
            retried_queue = runner._overflow_queue(key)
            assert retried_queue is not None
            assert (list(retried_queue),
                    [row["content"] for row in dbs["a"].get_messages("session-a")],
                    list((launch / "pending_messages").glob("*.json"))) == (
                [], ["[Pending input preserved at gateway shutdown; not executed]\ninput-0",
                     "[Pending input preserved at gateway shutdown; not executed]\ninput-2"], [])
        assert not any(adapter._background_tasks)
    finally:
        if runner is not None:
            runner.session_store.close_all_db_handles()
        for db in dbs.values():
            db.close()
        set_multiplex_active(active)


@pytest.mark.asyncio
async def test_shutdown_recovers_queued_input_cancelled_in_post_admission_hook(tmp_path, monkeypatch):
    from gateway.session import SessionStore
    from tests.gateway.test_post_gateway_admission import _event, _register, _runner
    from tests.gateway.test_active_session_text_merge import _DummyAdapter
    from gateway.config import Platform, PlatformConfig

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    runner = _runner()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    adapter = _DummyAdapter(PlatformConfig(enabled=True), Platform.WHATSAPP)
    runner.adapters = {adapter.platform: adapter}
    runner._delivery_adapter_for = lambda _source: adapter
    runner._intake_adapter_for = lambda _source: adapter
    runner._enqueue_fifo = type(runner)._enqueue_fifo.__get__(runner)
    adapter.gateway_runner = runner
    adapter.set_message_handler(runner._handle_message)
    entered = asyncio.Event()

    async def pause(**_kwargs):
        entered.set()
        await asyncio.Event().wait()

    _register(pause)
    first = _event("original queued speech")
    first.reply_to_message_id, first.reply_to_text = "quote", "quoted text"
    first.reply_to_author_id, first.reply_to_author_name = "author", "Quoted author"
    first.media_urls, first.media_types, first.media_text_inlined = [str(tmp_path / "image.jpg")], ["image/jpeg"], [False]
    first.metadata = {"nested": {"future": [1, False]}}
    later = _event("later queued input")
    later.message_id = "m2"
    key = runner._session_key_for_source(first.source)
    runner._enqueue_fifo(key, first, adapter)
    runner._enqueue_fifo(key, later, adapter)
    started = adapter._pending_messages.pop(key)
    adapter._stage_next_queued_event(key, started)
    assert adapter._start_session_processing(started, key)
    try:
        await asyncio.wait_for(entered.wait(), 5)
    finally:
        await adapter.cancel_background_tasks()
        runner.session_store.close_all_db_handles()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in (tmp_path / "pending_messages").glob("*.json")]
    assert [[record["event"] for record in payload["events"]] for payload in payloads] == [
        [_wire_event(first), _wire_event(later)]
    ]
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["shutdown", "stop", "withdraw"])
@pytest.mark.parametrize("ownership", ["provisional", "durable", "other-profile", "read-error"])
async def test_pending_dispatch_preserves_only_inputs_without_durable_ownership(tmp_path, monkeypatch, action, ownership):
    from gateway.config import GatewayConfig
    from gateway.platforms.base_pending import pending_dispatch_record
    from gateway.run import _profile_runtime_scope
    from gateway.session import SessionStore
    from gateway.session_transcript import TranscriptReadError
    from hermes_state import SessionDB

    launch = tmp_path / "launch"
    homes = {profile: tmp_path / profile for profile in ("a", "b")}
    for home in (launch, *homes.values()):
        home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    adapter = _make_initialized_adapter()
    class _OwnedInputRunner(_QueueRunner):
        def __init__(self, adapter):
            super().__init__(adapter)
            self.session_store = SessionStore(launch / "sessions", GatewayConfig(multiplex_profiles=False))

        def _profile_scope_for_source(self, source):
            return _profile_runtime_scope(homes[source.profile])

    runner = _OwnedInputRunner(adapter)
    adapter.gateway_runner = runner
    dbs = {profile: SessionDB(db_path=home / "state.db") for profile, home in homes.items()}
    try:
        for index, profile in enumerate(("a", "b", "a")):
            session_id, owner = f"session-{index}", f"owner-{index}"
            for db in dbs.values():
                db.create_session(session_id, "gateway")
            if ownership in {"durable", "other-profile"}:
                target = profile if ownership == "durable" else ("b" if profile == "a" else "a")
                dbs[target].append_message(session_id, "user", f"input-{index}",
                                           display_metadata={"gateway_input_owner": owner})
            event = _make_event(f"input-{index}")
            event.source.profile = profile
            key = f"key-{index}"
            entered = asyncio.Event()

            async def handler(_event):
                record = pending_dispatch_record(adapter, key, _event)
                assert record is not None
                record.input_session_id, record.input_owner = session_id, owner
                entered.set()
                await asyncio.Event().wait()

            adapter.set_message_handler(handler)
            reserve_pending_dispatch(adapter, key, event)
            assert adapter._start_session_processing(event, key)
            try:
                await asyncio.wait_for(entered.wait(), 5)
                if ownership == "read-error":
                    def failed_read(*_args):
                        raise TranscriptReadError(session_id)
                    monkeypatch.setattr(runner.session_store, "has_input_owner", failed_read)
                depth = runner._queue_depth(key, adapter=adapter)
                if action == "withdraw":
                    assert event.message_id is not None
                    assert event.source.user_id is not None
                    found = adapter.withdraw_pending_message(event.message_id, chat_id=event.source.chat_id,
                                                             sender_id=event.source.user_id)
                    record = pending_dispatch_record(adapter, key, event)
                    assert record is not None
                    assert (found, record.withdrawn, event.text, depth) == (
                        ownership != "durable", ownership != "durable", f"input-{index}",
                        0 if ownership == "durable" else 1,
                    )
                if action == "shutdown":
                    await adapter.cancel_background_tasks()
                    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in (homes[profile] / "pending_messages").glob("*.json")]
                    captured = [record["event"] for payload in payloads for record in payload["events"]]
                    expected = [] if ownership == "durable" else [_wire_event(event)]
                    assert (captured, depth) == (expected, 0 if ownership == "durable" else 1)
                    for path in (homes[profile] / "pending_messages").glob("*.json"):
                        path.unlink()
                if action == "stop":
                    await adapter.cancel_session_processing(key, discard_pending=False)
                    assert (adapter._pending_messages, depth) == (
                        {} if ownership == "durable" else {key: event},
                        0 if ownership == "durable" else 1,
                    )
                    adapter._pending_messages.clear()
            finally:
                await adapter.cancel_background_tasks()
        assert list((launch / "pending_messages").glob("*.json")) == []
    finally:
        runner.session_store.close_all_db_handles()
        for db in dbs.values():
            db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["lease", "agent-start", "pins"])
@pytest.mark.parametrize("partial", [False, True])
async def test_shutdown_preserves_dispatch_waiting_for_the_turn_lease(tmp_path, monkeypatch, phase, partial):
    from unittest.mock import AsyncMock
    from tests.gateway.test_duplicate_user_message import _bootstrap, _event

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = _bootstrap(monkeypatch, tmp_path)
    from gateway.session import SessionStore
    from gateway.platforms.base_pending import pending_dispatch_record

    store = SessionStore(tmp_path / "sessions", runner.config)
    runner.session_store = store
    adapter = _make_initialized_adapter()
    runner.adapters = {adapter.platform: adapter}
    runner._delivery_adapter_for = lambda _source: adapter
    runner._intake_adapter_for = lambda _source: adapter
    runner._run_agent = AsyncMock()
    adapter.gateway_runner = runner
    adapter.set_message_handler(runner._handle_message)
    entered = asyncio.Event()

    async def acquire(*_args):
        entered.set()
        await asyncio.Event().wait()

    if phase == "lease":
        runner._hmwa_acquire_turn_lease = acquire
    if phase == "agent-start":
        async def hook(kind, *_args):
            if kind == "agent:start":
                await acquire()
        runner.hooks.emit = hook
    if phase == "pins":
        runner._persist_prompt_pins = acquire
    first = _event()
    first.reply_to_message_id, first.reply_to_text = "quote", "quoted text"
    first.reply_to_author_id, first.reply_to_author_name = "author", "Quoted author"
    first.metadata = {"nested": {"future": [1, False]}}
    later = _event()
    later.text, later.message_id = "later queued input", "later"
    key = runner._session_key_for_source(first.source)
    monkeypatch.setattr(adapter, "_event_session_key", lambda event: key)
    runner._enqueue_fifo(key, first, adapter)
    if partial:
        from gateway.platforms.base_pending_merge import merge_pending_message_event
        retained = _event()
        retained.text, retained.message_id = "retained contribution", "retained"
        merge_pending_message_event(adapter._pending_messages, key, retained, merge_text=True)
    runner._enqueue_fifo(key, later, adapter)
    started = adapter._pending_messages.pop(key)
    adapter._stage_next_queued_event(key, started)
    assert adapter._start_session_processing(started, key)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert first.message_id is not None
        assert first.source.user_id is not None
        record = pending_dispatch_record(adapter, key, first)
        assert record is not None
        if phase != "lease":
            assert record.input_session_id is not None
            assert record.input_owner is not None
            assert not store.has_input_owner(record.input_session_id, record.input_owner)
        original = _wire_event(first)
        found = adapter.withdraw_pending_message(first.message_id, chat_id=first.source.chat_id,
                                                 sender_id=first.source.user_id)
        assert (found, _wire_event(first)) == (False, original)
    finally:
        await adapter.cancel_background_tasks()
        store.close_all_db_handles()
        close_all_under(get_hermes_home())

    payloads = [json.loads(path.read_text()) for path in (tmp_path / "pending_messages").glob("*.json")]
    assert [[record["event"] for record in payload["events"]] for payload in payloads] == [
        [_wire_event(first), _wire_event(later)]
    ]
    runner._run_agent.assert_not_awaited()
