"""Trusted request identity reaches hooks, never inferred from notification text."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.turn_context import _collect_pre_llm_call_context
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionStore
from gateway.session_context import clear_session_vars, reset_session_vars


@pytest.fixture(autouse=True)
def clean_context():
    reset_session_vars()
    yield
    clear_session_vars([])


class BoundTurn(Exception):
    pass


async def prepare(event, tmp_path):
    """Exercise real turn binding, stopping before transcript/LLM work."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    store = SessionStore(tmp_path / "sessions", runner.config)
    entry = store.get_or_create_session(event.source)
    runner._hmwa_open_session = AsyncMock(return_value=(False, False))
    def stop(*args, **kwargs):
        raise BoundTurn
    runner._pinned_session_context_prompt = stop
    with pytest.raises(BoundTurn):
        await runner._hmwa_prepare_turn(event, event.source, entry, entry.session_key, entry.session_key, 1)
    return entry


def hook_origin(monkeypatch, session_id):
    captured = {}
    def hook(name, **kwargs):
        captured.update(kwargs)
        return []
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hook)
    _collect_pre_llm_call_context(
        SimpleNamespace(session_id=session_id, model="test", platform="telegram"),
        effective_task_id="task", turn_id="turn", original_user_message="opaque",
        messages=[], conversation_history=None,
    )
    assert "request_origin" in captured
    return captured["request_origin"]


def event_for(kind=MessageType.VOICE):
    from gateway.session import SessionSource
    return MessageEvent(text="request", message_type=kind, source=SessionSource(
        platform=Platform.TELEGRAM, chat_id="chat", chat_type="group",
        thread_id="topic", user_id="user", message_id="original", scope_id="scope",
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MessageType.VOICE, MessageType.TEXT])
@pytest.mark.parametrize("event_id", [None, "actual-event"])
async def test_turn_binds_exact_origin_and_hook_returns_session_checked_copy(tmp_path, monkeypatch, kind, event_id):
    event = event_for(kind)
    event.message_id = event_id
    expected_id = event_id or "original"
    entry = await prepare(event, tmp_path)
    origin = hook_origin(monkeypatch, entry.session_id)
    assert origin == {
        "session_id": entry.session_id, "session_key": entry.session_key,
        "platform": "telegram", "chat_id": "chat", "thread_id": "topic",
        "message_id": expected_id, "message_type": kind.value, "internal": False,
        "user_id": "user", "chat_type": "group", "profile": "", "scope_id": "scope",
    }
    origin["message_id"] = "mutation"
    assert hook_origin(monkeypatch, entry.session_id)["message_id"] == expected_id
    assert hook_origin(monkeypatch, "different-session") is None
    clear_session_vars([])
    monkeypatch.setenv("HERMES_REQUEST_ORIGIN", "forged")
    assert hook_origin(monkeypatch, entry.session_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("kind", [MessageType.VOICE, MessageType.TEXT])
async def test_dispatch_recovery_injection_and_chained_continuation_keep_original_trigger(tmp_path, monkeypatch, recovery, kind):
    import json
    import queue
    from tools import async_delegation as ad
    from tools.process_registry import process_registry
    from gateway.session_context import get_request_origin, scoped_current_session_id

    ad._reset_for_tests()
    event = event_for(kind)
    entry = await prepare(event, tmp_path)
    original = hook_origin(monkeypatch, entry.session_id)
    try:
        for generation in range(2):
            # Child construction may replace the ambient session id before dispatch.
            with scoped_current_session_id("constructed-child"):
                handle = ad.dispatch_async_delegation(
                    goal="work", context=None, toolsets=None, role="leaf", model="test",
                    session_key=entry.session_key, parent_session_id=entry.session_id,
                    runner=lambda: {"status": "completed", "summary": "done"},
                )
            assert handle["status"] == "dispatched"
            while True:
                completed = process_registry.completion_queue.get(timeout=10)
                if completed.get("delegation_id") == handle["delegation_id"]:
                    break
            assert completed.get("request_origin") == get_request_origin(entry.session_id)
            if recovery:
                # Rebuild from task_json, not the live record or terminal event_json.
                with ad._DB_LOCK, ad._transaction() as conn:
                    task = json.loads(conn.execute(
                        "SELECT task_json FROM async_delegations WHERE delegation_id=?",
                        (handle["delegation_id"],),
                    ).fetchone()[0])
                    assert task["request_origin"] == get_request_origin(entry.session_id)
                    conn.execute("UPDATE async_delegations SET state='running', owner_pid=-1, event_json=NULL WHERE delegation_id=?",
                                 (handle["delegation_id"],))
                assert ad.recover_abandoned_delegations() == 1
                restored = queue.Queue()
                assert ad.restore_undelivered_completions(restored) >= 1
                while True:
                    completed = restored.get_nowait()
                    if completed["delegation_id"] == handle["delegation_id"]:
                        break
            clear_session_vars([])
            captured = []
            async def accept(synthetic):
                captured.append(synthetic)
                synthetic._gateway_accepted = True
            adapter = SimpleNamespace(supports_async_delivery=True, handle_message=accept)
            runner = object.__new__(GatewayRunner)
            runner.config = GatewayConfig()
            runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
            runner._resolve_injection_adapter = lambda *_: adapter
            assert await runner._inject_watch_notification("opaque completion", completed) is True
            synthetic = captured[0]
            assert synthetic.message_type == MessageType.TEXT
            assert synthetic.source.message_id is None  # reply anchor intentionally absent
            resumed = await prepare(synthetic, tmp_path)
            assert resumed.session_id == entry.session_id
            assert hook_origin(monkeypatch, entry.session_id) == {**original, "internal": True}
    finally:
        ad._reset_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["metadata", "proof", "session_id", "session_key", "chat_id", "thread_id", "user_id", "profile", "scope_id", "platform", "chat_type", "parent"])
async def test_continuation_rejects_forged_or_cross_route_identity(tmp_path, monkeypatch, tamper):
    from gateway.session_context import attach_notification_origin
    event = event_for()
    entry = await prepare(event, tmp_path)
    original = hook_origin(monkeypatch, entry.session_id)
    synthetic = event_for(MessageType.TEXT)
    synthetic.internal = True
    synthetic.metadata = {"gateway_session_id": entry.session_id, "gateway_session_key": entry.session_key,
                          "request_origin": original, "original_trigger_message_id": "original",
                          "notification_origin": "process_registry_synthetic"}
    producer = {"request_origin": original, "parent_session_id": entry.session_id, "session_key": entry.session_key}
    if tamper == "parent":
        producer["parent_session_id"] = "another-session"
    if tamper != "metadata":
        attach_notification_origin(synthetic, producer)
    if tamper == "proof":
        synthetic._request_origin_proof = ("forged", original)
    elif tamper in ("session_id", "session_key"):
        synthetic.metadata["gateway_" + tamper] = "other"
    elif tamper not in ("metadata", "parent"):
        setattr(synthetic.source, tamper, Platform.DISCORD if tamper == "platform" else "other")
    resumed = await prepare(synthetic, tmp_path)
    assert hook_origin(monkeypatch, resumed.session_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["process", "delegation"])
@pytest.mark.parametrize("difference", [None, "internal", "message_id", "missing", "parent_session_id", "session_key"])
async def test_notification_batches_expose_origin_only_when_every_member_agrees(tmp_path, monkeypatch, mode, difference):
    import asyncio
    event = event_for()
    entry = await prepare(event, tmp_path)
    original = hook_origin(monkeypatch, entry.session_id)
    first = {"request_origin": original, "parent_session_id": entry.session_id, "session_key": entry.session_key,
             "type": "async_delegation", "delegation_id": "first", "status": "completed", "summary": "one"}
    second = {**first, "delegation_id": "second", "summary": "two", "request_origin": dict(original)}
    if difference == "missing":
        second.pop("request_origin")
    elif difference in ("parent_session_id", "session_key"):
        second[difference] = "other"
    elif difference == "internal":
        second["request_origin"][difference] = True
    elif difference:
        second["request_origin"][difference] = "other"
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    captured = []
    async def accept(synthetic):
        captured.append(synthetic)
        synthetic._gateway_accepted = True
    adapter = SimpleNamespace(supports_async_delivery=True, handle_message=accept)
    runner._resolve_injection_adapter = lambda *_: adapter
    async def deliver(text, evt, **kwargs):
        return await runner._inject_watch_notification(text, evt)
    runner._deliver_completion_notification = deliver
    if mode == "process":
        entries = [("one", first, asyncio.get_running_loop().create_future()),
                   ("two", second, asyncio.get_running_loop().create_future())]
        runner._completion_notification_batch_window = 0
        runner._completion_notification_batches = {("key",): entries}
        runner._completion_notification_batch_tasks = {}
        runner._record_coalesced_completion_siblings = lambda *_: None
        await runner._flush_process_completion_batch(("key",))
        assert all(future.result() is True for _, _, future in entries)
    else:
        runner._completion_identity_seen = lambda *_: False
        runner._completion_delivery_ready = AsyncMock(return_value=True)
        monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *_: "claim")
        assert await runner._deliver_async_delegation_group_scoped([first, second]) is True
    assert len(captured) == 1
    await prepare(captured[0], tmp_path)
    expected = {**original, "internal": True} if difference in (None, "internal") else None
    assert hook_origin(monkeypatch, entry.session_id) == expected


@pytest.mark.asyncio
async def test_origin_is_task_local_and_new_bindings_never_inherit_authority(tmp_path, monkeypatch):
    import asyncio
    from gateway.session_context import get_request_origin, set_session_vars
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    import hermes_state
    # The suite normally pins DEFAULT_DB_PATH to one temp home. Enable the real
    # context-scoped resolver while both scopes below remain inside tmp_path.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)

    token_a = set_hermes_home_override(tmp_path / "a")
    try:
        event_a = event_for()
        event_a.source.profile = "a"
        entry_a = await prepare(event_a, tmp_path / "a")
        original = hook_origin(monkeypatch, entry_a.session_id)
        async def other_turn():
            # create_task inherits A; the normal gateway ingress reset must remove it.
            reset_session_vars()
            assert get_request_origin(entry_a.session_id) is None
            token_b = set_hermes_home_override(tmp_path / "b")
            try:
                event_b = event_for(MessageType.TEXT)
                event_b.source.profile = "b"
                entry_b = await prepare(event_b, tmp_path / "b")
                assert entry_a.session_id != entry_b.session_id
                origin_b = get_request_origin(entry_b.session_id)
                assert origin_b is not None and origin_b["profile"] == "b"
                assert get_request_origin(entry_a.session_id) is None
                set_session_vars(session_id=entry_b.session_id)
                assert get_request_origin(entry_b.session_id) is None
            finally:
                reset_hermes_home_override(token_b)
        await asyncio.create_task(other_turn())
        assert hook_origin(monkeypatch, entry_a.session_id) == original
    finally:
        reset_hermes_home_override(token_a)
