"""Pending delegation is a delivery state, not a reason to keep a model turn open."""

import asyncio
import queue
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from tools import async_delegation as ad
from tools.delegate_tool_registry import _list_payload
from tools.process_registry import process_registry
from tools.process_registry_notifications import format_process_notification


@pytest.fixture
def ledger(monkeypatch):
    ad._reset_for_tests()
    monkeypatch.setattr(process_registry, "completion_queue", queue.Queue())
    db = SessionDB()
    db.create_session("parent", source="cli")
    yield db
    ad._reset_for_tests()
    db.close()


def _seed_unit(session_id, delegation_id, *, state="running"):
    record = {
        "delegation_id": delegation_id, "parent_session_id": session_id,
        "session_key": "same-chat", "origin_ui_session_id": "reused-tab",
        "dispatched_at": time.time(), "goal": "Review the implementation",
    }
    ad._persist_dispatch(record)
    event = {
        **record, "type": "async_delegation", "status": state,
        "completed_at": time.time(), "summary": "Review evidence",
    }
    if state != "running":
        ad._persist_completion(event, {"summary": event["summary"]})
    return event


def test_pending_delivery_is_conversation_owned_and_survives_child_exit(ledger, monkeypatch, tmp_path):
    """List and completion notices must not equate an empty live roster with received evidence."""
    _seed_unit("parent", "active")
    event = _seed_unit("parent", "ready", state="completed")
    _seed_unit("unrelated", "foreign")
    parent = SimpleNamespace(session_id="parent", _session_db=ledger)

    # RAM is empty, as after a parent/process rebuild. Read the durable unit lifecycle.
    payload = _list_payload(parent)
    expected = {"active": ["active"], "awaiting_delivery": ["ready"], "status_known": True}
    assert payload["count"] == 0
    assert payload["pending_delegations"] == expected
    assert "awaiting delivery" in payload["note"]

    # A claimed result is still pending until acceptance; formatting is a read, not an ack.
    claim = ad.claim_event_delivery(event, "test")
    assert claim
    for shape in (event, {**event, "is_batch": True, "goals": [event["goal"]],
                          "results": [{"task_index": 0, "status": "completed", "summary": "Review evidence"}]}):
        text = format_process_notification(shape)
        assert text is not None
        assert "1 active unit(s), 0 result(s) awaiting delivery" in text
        assert "when this notification was prepared" in text
    assert ad.pending_delegations("parent") == expected
    durable = ad.get_durable_delegation("ready")
    assert durable is not None and durable["delivery_state"] == "pending"

    # An interim failure shares the unit id but is not its final result.
    interim = {**event, "delegation_id": "active", "task_failure_notice": True,
               "n_tasks": 2, "results": [{"task_index": 0, "status": "error", "error": "test failure"}]}
    text = format_process_notification(interim)
    assert text is not None and "1 active unit(s), 1 result(s) awaiting delivery" in text

    ledger.end_session("parent", "compression")
    ledger.create_session("tip", source="cli", parent_session_id="parent")
    ledger.create_session("fresh", source="cli")
    assert ad.pending_delegations("tip") == expected
    assert ad.pending_delegations("fresh")["active"] == []  # /new may reuse the tab/chat
    assert ad.pending_delegations("")["status_known"] is False

    ad.complete_event_delivery(event, claim)
    # The ledger, not a stale finalizing RAM record, decides whether a result was accepted.
    monkeypatch.setitem(ad._records, "ready", {"status": "finalizing", "parent_session_id": "parent"})
    assert ad.pending_delegations("tip")["awaiting_delivery"] == []
    failed = _seed_unit("parent", "failed", state="error")
    assert ad.pending_delegations("tip")["awaiting_delivery"] == ["failed"]
    claim = ad.claim_event_delivery(failed, "test")
    assert claim
    assert ad.drop_completion_delivery("failed", claim)
    assert ad.pending_delegations("tip")["awaiting_delivery"] == []

    # Read failures must not be advertised as a verified empty roster.
    with patch.object(SessionDB, "_read_all", side_effect=sqlite3.OperationalError("database is locked")):
        assert ad.pending_delegations("parent")["status_known"] is False
    # A different profile is not allowed to inherit this ledger through a global RAM count.
    with monkeypatch.context() as m:
        m.setattr(ad, "get_hermes_home", lambda: tmp_path / "other-profile")
        assert ad.pending_delegations("parent")["active"] == []
        assert not (tmp_path / "other-profile" / "state.db").exists()

    # The TUI poller is outside the destination turn's context. It must bind that
    # profile for formatting AND the delivery claim/ack, then restore the caller.
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import server
    event = _seed_unit("parent", "profile-result", state="completed")
    session = {"session_key": "same-chat", "profile_home": str(ledger.db_path.parent),
               "history_lock": threading.Lock(), "running": False}
    delivered = []
    monkeypatch.setattr(server, "_emit", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda _rid, _sid, _s, text, **_kw: delivered.append(text))
    other_home = tmp_path / "poller-profile"
    other = SessionDB(other_home / "state.db")
    token = set_hermes_home_override(other_home)
    try:
        server._notif_handle_ready("reused-tab", session, [event], set(), process_registry,
                                   format_process_notification, [])
        assert len(delivered) == 1
        assert "1 active unit(s), 0 result(s) awaiting delivery" in delivered[0]
        assert get_hermes_home() == other_home
    finally:
        reset_hermes_home_override(token)
        other.close()
    durable = ad.get_durable_delegation("profile-result")
    assert durable is not None and durable["delivery_state"] == "delivered"


def test_text_turn_releases_parent_without_hiding_pending_results(ledger):
    """Real dispatch/SQLite/queue + model-loop test: no retry, text rewrite, or early delivery ack."""
    from run_agent import AIAgent
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from tests.gateway.relay.test_relay_live_cards import _connected_adapter
    from tests.run_agent.test_run_agent import _mock_response

    statuses = []
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://example.invalid/v1",
            model="test-model", session_id="parent", quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            status_callback=lambda kind, text: statuses.append((kind, text)),
        )
    agent._session_db = ledger
    agent.client = MagicMock()
    agent._cached_system_prompt = "Stable system prefix."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.tool_delay = 0
    answer = "Implementation done."
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content=answer, finish_reason="stop") for _ in range(3)
    ]
    gate = threading.Event()

    def child():
        assert gate.wait(30), "test did not release its child"
        return {"status": "completed", "summary": "Required review finished."}

    handle = ad.dispatch_async_delegation(
        goal="Required review", context=None, toolsets=None, role="leaf", model=None,
        session_key="", parent_session_id="parent", runner=child,
    )
    delegation_id = handle["delegation_id"]
    adapter, transport = _connected_adapter(supported_ops=("send", "edit", "typing", "draft"))
    metadata = {"thread_ts": "1700.42"}
    loop = asyncio.new_event_loop()
    ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata=metadata,
                          _run_still_current=lambda: True, _cleanup_progress=False,
                          source=SimpleNamespace(platform=Platform.SLACK))
    turn_runner = TurnRunner(None, ctx)
    turn_runner._schedule = lambda coro, _log_message: loop.run_until_complete(coro)

    def status_callback(kind, text):
        statuses.append((kind, text))
        turn_runner._status_callback_sync(kind, text)

    agent.status_callback = status_callback
    try:
        loop.run_until_complete(adapter.send_draft("C1", 5, "Implementation", metadata=metadata))
        result = agent.run_conversation("Implement and review this change.")
        assert adapter._open_draft_by_chat.get(adapter._draft_key("C1", metadata)) == 5
        assert not any(op.get("final") for op in transport.sent if op["op"] == "draft")
        loop.run_until_complete(adapter.send("C1", result["final_response"], metadata=metadata))
        seals = [op for op in transport.sent if op["op"] == "draft" and op.get("final")]
        assert len(seals) == 1 and seals[0]["content"] == answer
        assert not gate.is_set()  # The parent returned while its child was still blocked.
        assert result["pending_delegations"]["active"] == [delegation_id]
        assert any("1 active unit(s)" in text for _, text in statuses)
        assert result["completed"] is True  # Preserve the existing *turn* contract.
        assert result["final_response"] == answer
        assert result["api_calls"] == 1
        assert [m["role"] for m in result["messages"]] == ["user", "assistant"]
        prefix = agent._cached_system_prompt

        gate.set()
        event = process_registry.completion_queue.get(timeout=5)
        statuses.clear()
        ready = agent.run_conversation("Report current progress.", conversation_history=result["messages"])
        assert ready["pending_delegations"]["awaiting_delivery"] == [delegation_id]
        assert any("1 result(s) awaiting delivery" in text for _, text in statuses)
        durable = ad.get_durable_delegation(delegation_id)
        assert durable is not None and durable["delivery_state"] == "pending"

        # Match between-turn admission: accept the completion, then let the parent incorporate it.
        claim = ad.claim_event_delivery(event, "test")
        assert claim
        notification = format_process_notification(event)
        assert notification is not None
        ad.complete_event_delivery(event, claim)
        statuses.clear()
        done = agent.run_conversation(notification, conversation_history=ready["messages"])
        assert "pending_delegations" not in done
        assert not any("awaiting delivery" in text for _, text in statuses)
        assert done["completed"] is True and done["api_calls"] == 1
        assert done["final_response"] == answer
        assert agent._cached_system_prompt == prefix
        assert agent.client.chat.completions.create.call_count == 3
        assert [m["role"] for m in done["messages"]] == ["user", "assistant"] * 3
        assert ledger.get_messages("parent")[-1]["content"] == answer
    finally:
        gate.set()
        if ad._executor is not None:
            ad._executor.shutdown(wait=True)
        loop.close()
