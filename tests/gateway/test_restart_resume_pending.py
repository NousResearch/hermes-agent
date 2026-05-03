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
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import (
    _auto_continue_freshness_window,
    _coerce_gateway_timestamp,
    _is_fresh_gateway_interruption,
    _last_transcript_timestamp,
)
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


def _build_agent_history(history: list) -> list:
    """Mirror gateway/run.py's ``history → agent_history`` conversion.

    This is the transformation that strips ``timestamp`` off tool/tool_call
    rows before the agent sees them.  Tests that check the freshness gate
    must go through this conversion so they exercise the *real* data the
    note-injection code sees.
    """
    agent_history: list = []
    for msg in history:
        role = msg.get("role")
        if not role or role in ("session_meta", "system"):
            continue
        has_tool_calls = "tool_calls" in msg
        has_tool_call_id = "tool_call_id" in msg
        is_tool_message = role == "tool"
        if has_tool_calls or has_tool_call_id or is_tool_message:
            agent_history.append({k: v for k, v in msg.items() if k != "timestamp"})
        else:
            content = msg.get("content")
            if content:
                agent_history.append({"role": role, "content": content})
    return agent_history


def _simulate_note_injection(
    history: list,
    user_message: str,
    resume_entry: SessionEntry | None,
    *,
    agent_history: list | None = None,
    window_secs: float | None = None,
) -> str:
    """Mirror the note-injection logic in gateway/run.py _run_agent().

    The freshness signal reads ``history[-1].timestamp`` (the raw transcript
    row), NOT ``agent_history[-1].timestamp`` (which has been stripped).
    Tests pass the raw ``history`` — ``agent_history`` is derived from it
    via the real conversion if not supplied explicitly.
    """
    if agent_history is None:
        agent_history = _build_agent_history(history)

    window = (
        float(window_secs)
        if window_secs is not None
        else _auto_continue_freshness_window()
    )
    interruption_is_fresh = _is_fresh_gateway_interruption(
        _last_transcript_timestamp(history),
        window_secs=window,
    )

    message = user_message
    is_resume_pending = bool(
        resume_entry is not None
        and getattr(resume_entry, "resume_pending", False)
        and interruption_is_fresh
    )
    has_fresh_tool_tail = bool(
        agent_history
        and agent_history[-1].get("role") == "tool"
        and interruption_is_fresh
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
    elif has_fresh_tool_tail:
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

    def test_non_resume_pending_gets_resume_pending(self, tmp_path):
        """Non-resume sessions are now marked resume_pending (not suspended)."""
        store = _make_store(tmp_path)
        source_a = _make_source(chat_id="a")
        source_b = _make_source(chat_id="b")
        entry_a = store.get_or_create_session(source_a)
        entry_b = store.get_or_create_session(source_b)
        store.mark_resume_pending(entry_a.session_key)

        count = store.suspend_recently_active()
        # entry_a is already resume_pending → skipped. entry_b gets marked.
        assert count == 1
        assert store._entries[entry_a.session_key].suspended is False
        assert store._entries[entry_b.session_key].resume_pending is True
        assert store._entries[entry_b.session_key].suspended is False


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
            history=[
                {"role": "assistant", "content": "in progress", "timestamp": time.time()},
            ],
            user_message="what happened?",
            resume_entry=entry,
        )
        assert "[System note:" in result
        assert "gateway restart" in result
        assert "what happened?" in result

    def test_resume_pending_shutdown_note_mentions_shutdown(self):
        entry = self._pending_entry(reason="shutdown_timeout")
        result = _simulate_note_injection(
            history=[
                {"role": "assistant", "content": "in progress", "timestamp": time.time()},
            ],
            user_message="ping",
            resume_entry=entry,
        )
        assert "gateway shutdown" in result

    def test_resume_pending_fires_without_tool_tail(self):
        """Key improvement over PR #9934: the restart-resume note fires
        even when the transcript's last role is NOT ``tool``."""
        entry = self._pending_entry()
        history = [
            {"role": "user", "content": "run a long thing", "timestamp": time.time() - 10},
            {"role": "assistant", "content": "ok, starting...", "timestamp": time.time()},
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
            ], "timestamp": time.time() - 1},
            {"role": "tool", "tool_call_id": "c1", "content": "result",
             "timestamp": time.time()},
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
            ], "timestamp": time.time() - 1},
            {"role": "tool", "tool_call_id": "c1", "content": "result",
             "timestamp": time.time()},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=None)
        assert "[System note:" in result
        assert "tool result" in result

    def test_stale_resume_pending_does_not_inject_restart_note(self):
        """Old restart markers must not revive an unrelated stale task.

        The transcript's last row is from an hour ago — well outside the
        default 1h freshness window (fixture uses window=1800 to exercise
        the stale path without tying the test to the production default).
        """
        entry = self._pending_entry()
        entry.last_resume_marked_at = datetime.now() - timedelta(hours=1)

        history = [
            {"role": "assistant", "content": "old in progress",
             "timestamp": time.time() - 3600},
        ]
        result = _simulate_note_injection(
            history=history,
            user_message="start a new task",
            resume_entry=entry,
            window_secs=1800,
        )
        assert result == "start a new task"

    def test_fresh_tool_tail_preserves_auto_continue_note(self):
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 1},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "result",
                "timestamp": time.time(),
            },
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=None)
        assert "[System note:" in result
        assert "tool result" in result

    def test_stale_tool_tail_does_not_inject_auto_continue_note(self):
        """The core bug fix: stale tool-tail must not revive a dead task.

        Uses window_secs=1800 (30 min) to verify the gate fires at 1h —
        keeps the test stable regardless of the production default.
        """
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 3601},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "stale result",
                "timestamp": time.time() - 3600,
            },
        ]
        result = _simulate_note_injection(
            history,
            "start a new task",
            resume_entry=None,
            window_secs=1800,
        )
        assert result == "start a new task"

    def test_stale_tool_tail_with_production_data_shape(self):
        """Regression guard for #16802: exercise the REAL production path
        where ``agent_history`` has been stripped of timestamps.

        The original PR #16802 fix read ``agent_history[-1].get("timestamp")``
        — which is always ``None`` at runtime because the gateway strips
        ``timestamp`` off tool/tool_call rows in ``history → agent_history``.
        This test builds a stale history, runs it through the real
        ``_build_agent_history`` conversion, then asserts:

          1. The stripped ``agent_history`` carries NO timestamp (protects
             against someone "fixing" the original PR by re-adding the
             stripped field — which would break the API contract).
          2. The freshness gate still correctly classifies the transcript
             as stale because the signal is read from ``history`` BEFORE
             the strip.
          3. No auto-continue note is injected.
        """
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 7201},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "stale result",
                "timestamp": time.time() - 7200,  # 2 hours old
            },
        ]
        agent_history = _build_agent_history(history)

        # Invariant 1: strip contract preserved
        assert agent_history[-1]["role"] == "tool"
        assert "timestamp" not in agent_history[-1], (
            "agent_history tool rows must NOT carry a timestamp — the "
            "freshness gate must read from raw history, not agent_history"
        )

        # Invariant 2+3: stale classification, no note injection
        result = _simulate_note_injection(
            history,
            "start a new task",
            resume_entry=None,
            agent_history=agent_history,
        )
        assert result == "start a new task"

    def test_freshness_gate_disabled_via_zero_window(self):
        """window_secs=0 restores pre-fix behaviour (always inject)."""
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 86400},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "day-old result",
                "timestamp": time.time() - 86400,  # 24 hours old
            },
        ]
        result = _simulate_note_injection(
            history, "ping", resume_entry=None, window_secs=0,
        )
        assert "[System note:" in result
        assert "tool result" in result

    def test_legacy_history_without_timestamps_still_injects(self):
        """Transcripts predating timestamp persistence must keep the old
        behaviour — freshness unknown → treat as fresh."""
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
            {"role": "user", "content": "hello", "timestamp": time.time() - 2},
            {"role": "assistant", "content": "hi", "timestamp": time.time() - 1},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=None)
        assert result == "ping"


# ---------------------------------------------------------------------------
# Freshness helpers
# ---------------------------------------------------------------------------


class TestFreshnessHelpers:
    def test_coerce_datetime(self):
        now = datetime.now()
        assert _coerce_gateway_timestamp(now) == pytest.approx(now.timestamp(), abs=1e-3)

    def test_coerce_epoch_seconds(self):
        assert _coerce_gateway_timestamp(1_700_000_000) == 1_700_000_000.0
        assert _coerce_gateway_timestamp(1_700_000_000.5) == 1_700_000_000.5

    def test_coerce_epoch_milliseconds(self):
        # Values > 10^10 treated as ms
        assert _coerce_gateway_timestamp(1_700_000_000_000) == 1_700_000_000.0

    def test_coerce_iso_string(self):
        iso = "2026-04-18T12:00:00+00:00"
        expected = datetime.fromisoformat(iso).timestamp()
        assert _coerce_gateway_timestamp(iso) == pytest.approx(expected, abs=1e-3)

    def test_coerce_iso_string_with_z_suffix(self):
        iso_z = "2026-04-18T12:00:00Z"
        expected = datetime.fromisoformat("2026-04-18T12:00:00+00:00").timestamp()
        assert _coerce_gateway_timestamp(iso_z) == pytest.approx(expected, abs=1e-3)

    def test_coerce_numeric_string(self):
        assert _coerce_gateway_timestamp("1700000000") == 1_700_000_000.0

    def test_coerce_rejects_garbage(self):
        assert _coerce_gateway_timestamp(None) is None
        assert _coerce_gateway_timestamp("") is None
        assert _coerce_gateway_timestamp("not-a-timestamp") is None
        assert _coerce_gateway_timestamp(True) is None  # bool rejected
        assert _coerce_gateway_timestamp(False) is None
        assert _coerce_gateway_timestamp([1, 2, 3]) is None

    def test_is_fresh_unknown_is_fresh(self):
        """Legacy-compat: unknown timestamp → fresh."""
        assert _is_fresh_gateway_interruption(None) is True
        assert _is_fresh_gateway_interruption("not-a-timestamp") is True

    def test_is_fresh_window_bounds(self):
        now = 1_700_000_000.0
        # 1h window, 30min old → fresh
        assert _is_fresh_gateway_interruption(
            now - 1800, now=now, window_secs=3600,
        ) is True
        # 1h window, 2h old → stale
        assert _is_fresh_gateway_interruption(
            now - 7200, now=now, window_secs=3600,
        ) is False
        # 1h window, exactly at boundary → fresh (<=)
        assert _is_fresh_gateway_interruption(
            now - 3600, now=now, window_secs=3600,
        ) is True

    def test_is_fresh_zero_window_always_fresh(self):
        """Opt-out: window_secs=0 disables the gate entirely."""
        assert _is_fresh_gateway_interruption(
            0.0, now=1_700_000_000.0, window_secs=0,
        ) is True
        assert _is_fresh_gateway_interruption(
            -1.0, now=1_700_000_000.0, window_secs=-5,
        ) is True

    def test_last_transcript_timestamp_skips_meta(self):
        history = [
            {"role": "user", "content": "hi", "timestamp": 100.0},
            {"role": "assistant", "content": "hey", "timestamp": 200.0},
            {"role": "session_meta", "content": "tools:{}", "timestamp": 999.0},
            {"role": "system", "content": "ignore", "timestamp": 999.0},
        ]
        assert _last_transcript_timestamp(history) == 200.0

    def test_last_transcript_timestamp_empty(self):
        assert _last_transcript_timestamp([]) is None
        assert _last_transcript_timestamp(None) is None

    def test_last_transcript_timestamp_row_without_timestamp(self):
        """Legacy transcript row (no timestamp) returns None → caller
        treats as fresh."""
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        assert _last_transcript_timestamp(history) is None

    def test_auto_continue_freshness_window_reads_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "7200")
        assert _auto_continue_freshness_window() == 7200.0

    def test_auto_continue_freshness_window_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_AUTO_CONTINUE_FRESHNESS", raising=False)
        # Default is 1 hour
        assert _auto_continue_freshness_window() == 3600.0

    def test_auto_continue_freshness_window_malformed_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "not-a-number")
        assert _auto_continue_freshness_window() == 3600.0

    def test_auto_continue_freshness_window_empty_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "")
        assert _auto_continue_freshness_window() == 3600.0


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


@pytest.mark.asyncio
async def test_drain_timeout_only_marks_still_running_sessions():
    """A session that finished gracefully during the drain window must
    NOT be marked ``resume_pending`` — it completed cleanly and its
    next turn should be a normal fresh turn, not one prefixed with the
    restart-interruption system note.

    Regression guard for using ``self._running_agents`` at timeout
    rather than the ``active_agents`` drain-start snapshot.
    """
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    # Long enough for the finisher to exit, short enough to still time out
    # with the stuck session still present.
    runner._restart_drain_timeout = 0.3

    session_key_finisher = "agent:main:telegram:dm:A"
    session_key_stuck = "agent:main:telegram:dm:B"
    runner._running_agents = {
        session_key_finisher: MagicMock(),
        session_key_stuck: MagicMock(),
    }

    async def finish_one():
        await asyncio.sleep(0.05)
        runner._running_agents.pop(session_key_finisher, None)

    asyncio.create_task(finish_one())

    session_store = MagicMock()
    session_store.mark_resume_pending = MagicMock(return_value=True)
    runner.session_store = session_store

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.write_runtime_status"
    ):
        await runner.stop()

    calls = session_store.mark_resume_pending.call_args_list
    marked = {args[0][0] for args in calls}
    # Only the session still running at timeout is marked; the finisher is not.
    assert marked == {session_key_stuck}


@pytest.mark.asyncio
async def test_drain_timeout_skips_pending_sentinel_sessions():
    """Pending sentinels — sessions whose AIAgent construction hasn't
    produced a real agent yet — are skipped by
    ``_interrupt_running_agents()``.  The resume_pending marking must
    mirror that: no agent started means no turn was interrupted.
    """
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    runner._restart_drain_timeout = 0.05

    session_key_real = "agent:main:telegram:dm:A"
    session_key_sentinel = "agent:main:telegram:dm:B"
    runner._running_agents = {
        session_key_real: MagicMock(),
        session_key_sentinel: _AGENT_PENDING_SENTINEL,
    }

    session_store = MagicMock()
    session_store.mark_resume_pending = MagicMock(return_value=True)
    runner.session_store = session_store

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.write_runtime_status"
    ):
        await runner.stop()

    calls = session_store.mark_resume_pending.call_args_list
    marked = {args[0][0] for args in calls}
    assert marked == {session_key_real}


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

    def test_increment_restart_failure_counts_uses_atomic_json_write(
        self, tmp_path, monkeypatch
    ):
        from gateway.run import GatewayRunner

        source = _make_source()
        session_key = _make_store(tmp_path).get_or_create_session(source).session_key

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        calls = []

        def _fake_atomic_json_write(path, payload, **kwargs):
            calls.append((path, payload, kwargs))

        monkeypatch.setattr("gateway.run.atomic_json_write", _fake_atomic_json_write)

        runner = object.__new__(GatewayRunner)
        runner._increment_restart_failure_counts({session_key})

        assert calls == [
            (
                tmp_path / ".restart_failure_counts",
                {session_key: 1},
                {"indent": None},
            )
        ]

    def test_clear_restart_failure_count_uses_atomic_json_write_when_entries_remain(
        self, tmp_path, monkeypatch
    ):
        import json

        from gateway.run import GatewayRunner

        source = _make_source()
        session_key = _make_store(tmp_path).get_or_create_session(source).session_key
        other_key = "agent:main:telegram:dm:other"
        counts_file = tmp_path / ".restart_failure_counts"
        counts_file.write_text(
            json.dumps({session_key: 2, other_key: 1}),
            encoding="utf-8",
        )

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        calls = []

        def _fake_atomic_json_write(path, payload, **kwargs):
            calls.append((path, payload, kwargs))

        monkeypatch.setattr("gateway.run.atomic_json_write", _fake_atomic_json_write)

        runner = object.__new__(GatewayRunner)
        runner._clear_restart_failure_count(session_key)

        assert calls == [
            (
                tmp_path / ".restart_failure_counts",
                {other_key: 1},
                {"indent": None},
            )
        ]
