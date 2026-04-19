"""Tests for the resume_pending session continuity path.

Covers the behaviour introduced to fix the ``Gateway shutting down ...
task will be interrupted`` follow-up bug (spec: PR #11852, builds on
PRs #9850, #9934, #7536):

1. When a gateway restart drain times out and agents are force-interrupted,
   the affected sessions are flagged ``resume_pending=True`` — not
   ``suspended`` — so the next user message on the same session_key
   auto-resumes from the existing transcript instead of getting routed
   through ``suspend_recently_active()`` and converted into a fresh
   session.

2. ``suspended=True`` (from ``/stop`` or stuck-loop escalation) still
   wins over ``resume_pending`` — the forced-wipe path is preserved.

3. The restart-resume system note injected into the next user message is
   a superset of the existing tool-tail auto-continue note (from
   PR #9934), using session-entry metadata rather than just transcript
   shape so it fires even when the interrupted transcript does NOT end
   with a ``tool`` role.

4. The existing ``.restart_failure_counts`` stuck-loop counter from
   PR #7536 remains the single source of escalation — no parallel
   counter is added on ``SessionEntry``.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, SessionStore
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(platform=Platform.TELEGRAM, chat_id="123", user_id="u1"):
    return SessionSource(platform=platform, chat_id=chat_id, user_id=user_id)


def _make_store(tmp_path):
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _simulate_note_injection(
    agent_history: list,
    user_message: str,
    resume_entry: SessionEntry | None,
) -> str:
    """Mirror the note-injection logic in gateway/run.py _run_agent().

    Matches the production code in the ``run_sync`` closure so we can
    test the decision tree without a full gateway runner.
    """
    message = user_message
    is_resume_pending = bool(
        resume_entry is not None and getattr(resume_entry, "resume_pending", False)
    )

    if is_resume_pending:
        reason = getattr(resume_entry, "resume_reason", None) or "restart_timeout"
        reason_phrase = (
            "a gateway restart"
            if reason == "restart_timeout"
            else "a gateway shutdown"
            if reason == "shutdown_timeout"
            else "a gateway interruption"
        )
        message = (
            f"[System note: Your previous turn in this session was interrupted "
            f"by {reason_phrase}. The conversation history below is intact. "
            f"If it contains unfinished tool result(s), process them first and "
            f"summarize what was accomplished, then address the user's new "
            f"message below.]\n\n"
            + message
        )
    elif agent_history and agent_history[-1].get("role") == "tool":
        message = (
            "[System note: Your previous turn was interrupted before you could "
            "process the last tool result(s). The conversation history contains "
            "tool outputs you haven't responded to yet. Please finish processing "
            "those results and summarize what was accomplished, then address the "
            "user's new message below.]\n\n"
            + message
        )
    return message


# ---------------------------------------------------------------------------
# SessionEntry field + serialization
# ---------------------------------------------------------------------------


class TestSessionEntryResumeFields:
    def test_defaults(self):
        now = datetime.now()
        entry = SessionEntry(
            session_key="agent:main:telegram:dm:1",
            session_id="sid",
            created_at=now,
            updated_at=now,
        )
        assert entry.resume_pending is False
        assert entry.resume_reason is None
        assert entry.last_resume_marked_at is None

    def test_roundtrip_with_resume_fields(self):
        now = datetime(2026, 4, 18, 12, 0, 0)
        entry = SessionEntry(
            session_key="agent:main:telegram:dm:1",
            session_id="sid",
            created_at=now,
            updated_at=now,
            resume_pending=True,
            resume_reason="restart_timeout",
            last_resume_marked_at=now,
        )
        restored = SessionEntry.from_dict(entry.to_dict())
        assert restored.resume_pending is True
        assert restored.resume_reason == "restart_timeout"
        assert restored.last_resume_marked_at == now

    def test_from_dict_legacy_without_resume_fields(self):
        """Old sessions.json without the new fields deserialize cleanly."""
        now = datetime.now()
        legacy = {
            "session_key": "agent:main:telegram:dm:1",
            "session_id": "sid",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "chat_type": "dm",
        }
        restored = SessionEntry.from_dict(legacy)
        assert restored.resume_pending is False
        assert restored.resume_reason is None
        assert restored.last_resume_marked_at is None

    def test_malformed_timestamp_is_tolerated(self):
        now = datetime.now()
        data = {
            "session_key": "k",
            "session_id": "sid",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "resume_pending": True,
            "resume_reason": "restart_timeout",
            "last_resume_marked_at": "not-a-timestamp",
        }
        restored = SessionEntry.from_dict(data)
        # resume_pending still honoured, only the broken timestamp drops
        assert restored.resume_pending is True
        assert restored.resume_reason == "restart_timeout"
        assert restored.last_resume_marked_at is None


# ---------------------------------------------------------------------------
# SessionStore.mark_resume_pending / clear_resume_pending
# ---------------------------------------------------------------------------


class TestMarkResumePending:
    def test_marks_existing_session(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)

        assert store.mark_resume_pending(entry.session_key) is True
        refreshed = store._entries[entry.session_key]
        assert refreshed.resume_pending is True
        assert refreshed.resume_reason == "restart_timeout"
        assert refreshed.last_resume_marked_at is not None

    def test_custom_reason_persists(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)

        store.mark_resume_pending(entry.session_key, reason="shutdown_timeout")
        assert store._entries[entry.session_key].resume_reason == "shutdown_timeout"

    def test_returns_false_for_unknown_key(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.mark_resume_pending("no-such-key") is False

    def test_does_not_override_suspended(self, tmp_path):
        """suspended wins — mark_resume_pending is a no-op on a suspended entry."""
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.suspend_session(entry.session_key)

        assert store.mark_resume_pending(entry.session_key) is False
        e = store._entries[entry.session_key]
        assert e.suspended is True
        assert e.resume_pending is False

    def test_survives_roundtrip_through_json(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.mark_resume_pending(entry.session_key, reason="restart_timeout")

        # Reload from disk
        store2 = _make_store(tmp_path)
        store2._ensure_loaded()
        reloaded = store2._entries[entry.session_key]
        assert reloaded.resume_pending is True
        assert reloaded.resume_reason == "restart_timeout"


class TestClearResumePending:
    def test_clears_flag(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.mark_resume_pending(entry.session_key)

        assert store.clear_resume_pending(entry.session_key) is True
        e = store._entries[entry.session_key]
        assert e.resume_pending is False
        assert e.resume_reason is None
        assert e.last_resume_marked_at is None

    def test_returns_false_when_not_pending(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        # Not marked
        assert store.clear_resume_pending(entry.session_key) is False

    def test_returns_false_for_unknown_key(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.clear_resume_pending("no-such-key") is False


# ---------------------------------------------------------------------------
# SessionStore.get_or_create_session resume_pending behaviour
# ---------------------------------------------------------------------------


class TestGetOrCreateResumePending:
    def test_resume_pending_preserves_session_id(self, tmp_path):
        """This is THE core behavioural fix — resume_pending ≠ new session."""
        store = _make_store(tmp_path)
        source = _make_source()
        first = store.get_or_create_session(source)
        original_sid = first.session_id
        store.mark_resume_pending(first.session_key)

        second = store.get_or_create_session(source)
        assert second.session_id == original_sid
        assert second.was_auto_reset is False
        assert second.auto_reset_reason is None
        # Flag is NOT cleared on read — only on successful turn completion.
        assert second.resume_pending is True

    def test_suspended_still_creates_new_session(self, tmp_path):
        """Regression guard — suspended must still force a clean slate."""
        store = _make_store(tmp_path)
        source = _make_source()
        first = store.get_or_create_session(source)
        original_sid = first.session_id
        store.suspend_session(first.session_key)

        second = store.get_or_create_session(source)
        assert second.session_id != original_sid
        assert second.was_auto_reset is True
        assert second.auto_reset_reason == "suspended"

    def test_suspended_overrides_resume_pending(self, tmp_path):
        """Terminal escalation: a session that somehow has BOTH flags must
        behave like ``suspended`` — forced wipe + auto_reset_reason."""
        store = _make_store(tmp_path)
        source = _make_source()
        first = store.get_or_create_session(source)
        original_sid = first.session_id

        # Force the pathological state directly (normally mark_resume_pending
        # refuses to run when suspended=True, but a stuck-loop escalation
        # can set suspended=True AFTER resume_pending is set).
        with store._lock:
            e = store._entries[first.session_key]
            e.resume_pending = True
            e.resume_reason = "restart_timeout"
            e.suspended = True
            store._save()

        second = store.get_or_create_session(source)
        assert second.session_id != original_sid
        assert second.was_auto_reset is True
        assert second.auto_reset_reason == "suspended"


# ---------------------------------------------------------------------------
# SessionStore.suspend_recently_active skip behaviour
# ---------------------------------------------------------------------------


class TestSuspendRecentlyActiveSkipsResumePending:
    def test_resume_pending_entries_not_suspended(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.mark_resume_pending(entry.session_key)

        count = store.suspend_recently_active()
        assert count == 0
        e = store._entries[entry.session_key]
        assert e.suspended is False
        assert e.resume_pending is True

    def test_non_resume_pending_still_suspended(self, tmp_path):
        """Non-resume sessions still get the old crash-recovery suspension."""
        store = _make_store(tmp_path)
        source_a = _make_source(chat_id="a")
        source_b = _make_source(chat_id="b")
        entry_a = store.get_or_create_session(source_a)
        entry_b = store.get_or_create_session(source_b)
        store.mark_resume_pending(entry_a.session_key)

        count = store.suspend_recently_active()
        assert count == 1
        assert store._entries[entry_a.session_key].suspended is False
        assert store._entries[entry_b.session_key].suspended is True


# ---------------------------------------------------------------------------
# Restart-resume system-note injection
# ---------------------------------------------------------------------------


class TestResumePendingSystemNote:
    def _pending_entry(self, reason="restart_timeout") -> SessionEntry:
        now = datetime.now()
        return SessionEntry(
            session_key="agent:main:telegram:dm:1",
            session_id="sid",
            created_at=now,
            updated_at=now,
            resume_pending=True,
            resume_reason=reason,
            last_resume_marked_at=now,
        )

    def test_resume_pending_restart_note_mentions_restart(self):
        entry = self._pending_entry(reason="restart_timeout")
        result = _simulate_note_injection(
            agent_history=[{"role": "assistant", "content": "in progress"}],
            user_message="what happened?",
            resume_entry=entry,
        )
        assert "[System note:" in result
        assert "gateway restart" in result
        assert "what happened?" in result

    def test_resume_pending_shutdown_note_mentions_shutdown(self):
        entry = self._pending_entry(reason="shutdown_timeout")
        result = _simulate_note_injection(
            agent_history=[{"role": "assistant", "content": "in progress"}],
            user_message="ping",
            resume_entry=entry,
        )
        assert "gateway shutdown" in result

    def test_resume_pending_fires_without_tool_tail(self):
        """Key improvement over PR #9934: the restart-resume note fires
        even when the transcript's last role is NOT ``tool``."""
        entry = self._pending_entry()
        history = [
            {"role": "user", "content": "run a long thing"},
            {"role": "assistant", "content": "ok, starting..."},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=entry)
        assert "[System note:" in result
        assert "gateway restart" in result

    def test_resume_pending_subsumes_tool_tail_note(self):
        """When BOTH conditions are true, the restart-resume note wins —
        no duplicate notes."""
        entry = self._pending_entry()
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=entry)
        assert result.count("[System note:") == 1
        assert "gateway restart" in result
        # Old tool-tail wording absent
        assert "haven't responded to yet" not in result

    def test_no_resume_pending_preserves_tool_tail_note(self):
        """Regression: the old PR #9934 tool-tail behaviour is unchanged."""
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=None)
        assert "[System note:" in result
        assert "tool result" in result

    def test_no_note_when_nothing_to_resume(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=None)
        assert result == "ping"


# ---------------------------------------------------------------------------
# Drain-timeout path marks sessions resume_pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_timeout_marks_resume_pending():
    """End-to-end: a drain timeout during gateway stop should flag every
    active session as resume_pending BEFORE the interrupt fires, so the
    next startup's suspend_recently_active() does not destroy them."""
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    runner._restart_drain_timeout = 0.05

    running_agent = MagicMock()
    session_key_one = "agent:main:telegram:dm:A"
    session_key_two = "agent:main:telegram:dm:B"
    runner._running_agents = {
        session_key_one: running_agent,
        session_key_two: MagicMock(),
    }

    # Plug a mock session_store that records marks.
    session_store = MagicMock()
    session_store.mark_resume_pending = MagicMock(return_value=True)
    runner.session_store = session_store

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.write_runtime_status"
    ):
        await runner.stop()

    # Both active sessions were marked with the shutdown_timeout reason.
    calls = session_store.mark_resume_pending.call_args_list
    marked = {args[0][0] for args in calls}
    assert marked == {session_key_one, session_key_two}
    for args in calls:
        assert args[0][1] == "shutdown_timeout"


@pytest.mark.asyncio
async def test_drain_timeout_uses_restart_reason_when_restarting():
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    runner._restart_drain_timeout = 0.05
    runner._restart_requested = True

    running_agent = MagicMock()
    runner._running_agents = {"agent:main:telegram:dm:A": running_agent}

    session_store = MagicMock()
    session_store.mark_resume_pending = MagicMock(return_value=True)
    runner.session_store = session_store

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.write_runtime_status"
    ):
        await runner.stop(restart=True, detached_restart=False, service_restart=True)

    calls = session_store.mark_resume_pending.call_args_list
    assert calls, "expected at least one mark_resume_pending call"
    for args in calls:
        assert args[0][1] == "restart_timeout"


@pytest.mark.asyncio
async def test_clean_drain_does_not_mark_resume_pending():
    """If the drain completes within timeout (no force-interrupt), no
    sessions should be flagged — the normal shutdown path is unchanged."""
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()

    running_agent = MagicMock()
    runner._running_agents = {"agent:main:telegram:dm:A": running_agent}

    # Finish the agent before the (generous) drain deadline
    async def finish_agent():
        await asyncio.sleep(0.05)
        runner._running_agents.clear()

    asyncio.create_task(finish_agent())

    session_store = MagicMock()
    session_store.mark_resume_pending = MagicMock(return_value=True)
    runner.session_store = session_store

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.write_runtime_status"
    ):
        await runner.stop()

    session_store.mark_resume_pending.assert_not_called()
    running_agent.interrupt.assert_not_called()


# ---------------------------------------------------------------------------
# Shutdown banner wording
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_banner_uses_try_to_resume_wording():
    """The notification sent before drain should hedge the resume promise
    — the session-continuity fix is best-effort (stuck-loop counter can
    still escalate to suspended)."""
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    runner._running_agents["agent:main:telegram:dm:999"] = MagicMock()

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 1
    msg = adapter.sent[0]
    assert "restarting" in msg
    assert "try to resume" in msg


# ---------------------------------------------------------------------------
# Stuck-loop escalation integration
# ---------------------------------------------------------------------------


class TestStuckLoopEscalation:
    """The existing .restart_failure_counts counter (PR #7536) remains the
    single source of terminal escalation — no parallel counter on
    SessionEntry was added.  After the configured threshold, the startup
    path flips suspended=True which overrides resume_pending."""

    def test_escalation_via_stuck_loop_counter_overrides_resume_pending(
        self, tmp_path, monkeypatch
    ):
        """Simulate a session that keeps getting restart-interrupted and
        hits the stuck-loop threshold: next startup should force it to
        fresh-session despite resume_pending being set."""
        import json

        from gateway.run import GatewayRunner

        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.mark_resume_pending(entry.session_key, reason="restart_timeout")

        # Simulate counter already at threshold (3 consecutive interrupted
        # restarts).  _suspend_stuck_loop_sessions will flip suspended=True.
        counts_file = tmp_path / ".restart_failure_counts"
        counts_file.write_text(json.dumps({entry.session_key: 3}))

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        runner = object.__new__(GatewayRunner)
        runner.session_store = store

        suspended_count = GatewayRunner._suspend_stuck_loop_sessions(runner)
        assert suspended_count == 1
        assert store._entries[entry.session_key].suspended is True
        # resume_pending is still set on the entry, but suspended wins in
        # get_or_create_session so the next message still gets a new sid.
        second = store.get_or_create_session(source)
        assert second.session_id != entry.session_id
        assert second.auto_reset_reason == "suspended"

    def test_successful_turn_flow_clears_both_counter_and_resume_pending(
        self, tmp_path, monkeypatch
    ):
        """The gateway's post-turn cleanup should clear both signals so a
        future restart-interrupt starts with a fresh counter."""
        import json

        from gateway.run import GatewayRunner

        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.mark_resume_pending(entry.session_key, reason="restart_timeout")

        counts_file = tmp_path / ".restart_failure_counts"
        counts_file.write_text(json.dumps({entry.session_key: 2}))

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        runner = object.__new__(GatewayRunner)
        runner.session_store = store

        GatewayRunner._clear_restart_failure_count(runner, entry.session_key)
        store.clear_resume_pending(entry.session_key)

        assert store._entries[entry.session_key].resume_pending is False
        assert not counts_file.exists()
