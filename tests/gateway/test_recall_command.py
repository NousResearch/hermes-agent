"""Tests for /recall — one-message stop + deny-approvals + rewind.

Store-level tests pin the durable turn-lease gating that makes a same-session
abort safe (a live lease blocks rewind; the abort force-release fences the
killed turn's late flush). Runner tests pin the busy variant's ordering:
deny parked approvals -> hard interrupt -> lease-settled rewind.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


# --------------------------------------------------------------------------- #
# Store-level lease gating
# --------------------------------------------------------------------------- #


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    s = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    s._db = db
    return s


def _seed(store, sid="recall-1", turns=2):
    store._db.create_session(sid, source="feishu")
    for i in range(1, turns + 1):
        store._db.append_message(sid, "user", f"q{i}")
        store._db.append_message(sid, "assistant", f"a{i}")
    return sid


def test_rewind_blocked_while_durable_lease_live_then_succeeds_after_force_release(store):
    import os
    sid = _seed(store)
    holder = f"pid={os.getpid()}:turn:{sid}:platform=feishu"  # live pid -> non-stale
    assert store._db.try_acquire_session_turn_lease(sid, holder, ttl_seconds=300)
    assert store.turn_lease_holder(sid) == holder
    # A live cross-process lease refuses the same-session destructive mutation.
    assert store.rewind_session(sid, 1) is None
    # Abort backstop: force-release fences the killed holder, then rewind lands.
    assert store.force_release_turn_lease(sid) is True
    assert store.turn_lease_holder(sid) is None
    result = store.rewind_session(sid, 1)
    assert result is not None and result["target_text"] == "q2"
    assert [m["content"] for m in store.load_transcript(sid)] == ["q1", "a1"]


def test_stale_dead_pid_lease_is_reclaimed_not_treated_as_live(store):
    sid = _seed(store, "recall-stale")
    # A holder stamped with THIS pid namespace but a provably-dead pid is reclaimed (not
    # deferred to TTL). Build the pidns fragment exactly like turn_facade_lease does.
    from hermes_state_pidns import holder_namespace_token
    dead_holder = f"pid=999999{holder_namespace_token()}:turn=dead:platform=feishu"
    assert store._db.try_acquire_session_turn_lease(sid, dead_holder, ttl_seconds=300)
    # A dead-PID lease is transparent: neither read as a live holder nor blocking rewind.
    assert store.turn_lease_holder(sid) is None
    assert store.rewind_session(sid, 1) is not None


def test_force_release_without_lease_is_noop(store):
    sid = _seed(store, "recall-nolease")
    assert store.force_release_turn_lease(sid) is False


# --------------------------------------------------------------------------- #
# Runner busy / idle command paths
# --------------------------------------------------------------------------- #


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1",
                         user_name="tester", chat_type="dm")


def _event(text: str = "/recall") -> MessageEvent:
    return MessageEvent(text=text, source=_source(), message_id="m1")


def _runner():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    # The real AsyncSessionStore offloads SYNC SessionStore methods via to_thread, so the
    # underlying store is a plain MagicMock with sync return values.
    store = MagicMock()
    entry = SimpleNamespace(session_id="sid-1", last_prompt_tokens=10)
    store.get_or_create_session.return_value = entry
    store.turn_lease_holder.return_value = None
    store.force_release_turn_lease.return_value = False
    store.rewind_session.return_value = {
        "turns_undone": 1, "rewound_count": 2, "target_text": "bad prompt"}
    from gateway.session import AsyncSessionStore
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._interrupt_and_clear_session = AsyncMock()
    runner._evict_cached_agent = MagicMock()
    runner._session_key_for_source = lambda _source: "quick-key"
    return runner


def _clear_approvals():
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()


def _park_approval(session_key):
    from tools.approval import _gateway_queues
    from tools.approval_gateway_wait import _ApprovalEntry
    entry = _ApprovalEntry({"command": "rm -rf /"})
    _gateway_queues.setdefault(session_key, []).append(entry)
    return entry


@pytest.mark.asyncio
async def test_busy_recall_denies_interrupts_then_rewinds():
    _clear_approvals()
    runner = _runner()
    parked = _park_approval("quick-key")
    result = await runner._busy_recall_command(_event(), "quick-key", _source())

    # Parked approval is resolved as DENY (the recalled command must not execute).
    assert parked.event.is_set() and parked.result == "deny"
    # Hard interrupt fired with the stop reason before the rewind.
    runner._interrupt_and_clear_session.assert_awaited_once()
    kwargs = runner._interrupt_and_clear_session.await_args.kwargs
    assert kwargs["invalidation_reason"] == "recall_command"
    # Same-session rewind ran after lease settlement; reply reports stop+rewind.
    runner.session_store.rewind_session.assert_called_once_with("sid-1", 1)
    runner.session_store.force_release_turn_lease.assert_not_called()
    assert "Recalled" in result or "recall" in result.lower()


@pytest.mark.asyncio
async def test_busy_recall_passes_n_and_invalid_count_aborts_before_interrupt():
    _clear_approvals()
    runner = _runner()
    # Invalid N: localized error, no interrupt, no rewind.
    bad = await runner._busy_recall_command(_event("/recall abc"), "quick-key", _source())
    assert "Invalid count" in bad
    runner._interrupt_and_clear_session.assert_not_awaited()
    runner.session_store.rewind_session.assert_not_called()
    # Valid N parses through to the rewind call.
    await runner._busy_recall_command(_event("/recall 3"), "quick-key", _source())
    runner.session_store.rewind_session.assert_called_with("sid-1", 3)


@pytest.mark.asyncio
async def test_idle_recall_rewinds_without_interrupt_and_clears_stale_approval():
    _clear_approvals()
    runner = _runner()
    parked = _park_approval("quick-key")
    result = await runner._handle_recall_command(_event())

    # No running turn to interrupt; idle recall only rewinds (and drains stale approvals).
    runner._interrupt_and_clear_session.assert_not_awaited()
    assert parked.event.is_set() and parked.result == "deny"
    runner.session_store.rewind_session.assert_called_once_with("sid-1", 1)
    assert "Recalled" in result


@pytest.mark.asyncio
async def test_busy_recall_force_releases_wedged_lease_then_rewinds():
    _clear_approvals()
    runner = _runner()
    # Lease stays held through the grace window -> the abort backstop must fire once.
    runner.session_store.turn_lease_holder.return_value = "pid=1:turn:x"
    import gateway.slash_commands_session as scs
    old_grace = scs._RECALL_LEASE_GRACE_S
    scs._RECALL_LEASE_GRACE_S = 0.1
    try:
        await runner._busy_recall_command(_event(), "quick-key", _source())
    finally:
        scs._RECALL_LEASE_GRACE_S = old_grace
    runner.session_store.force_release_turn_lease.assert_called_once_with("sid-1")
    runner.session_store.rewind_session.assert_called_once_with("sid-1", 1)
