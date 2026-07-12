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
import inspect
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import (
    GatewayRunner,
    _AGENT_PENDING_SENTINEL,
    _auto_continue_freshness_window,
    _coerce_gateway_timestamp,
    _is_fresh_gateway_interruption,
    _last_transcript_timestamp,
    _should_clear_resume_pending_after_turn,
)
from gateway.session import SessionEntry, SessionSource, SessionStore
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_resume_pending_is_cleared_only_after_successful_turn():
    """Interrupted/failed drain results must keep the restart recovery marker.

    Regression for dogfood failure: during gateway restart the interrupted run
    returned an empty final response and was normalized into a user-facing
    fallback, but the gateway cleared ``resume_pending`` before startup could
    auto-resume it.
    """
    assert _should_clear_resume_pending_after_turn({"final_response": "done"}) is True
    assert _should_clear_resume_pending_after_turn({"completed": True}) is True
    assert _should_clear_resume_pending_after_turn({"interrupted": True}) is False
    assert _should_clear_resume_pending_after_turn({"completed": False}) is False
    assert _should_clear_resume_pending_after_turn({"failed": True}) is False
    assert _should_clear_resume_pending_after_turn({"partial": True}) is False
    assert _should_clear_resume_pending_after_turn({"error": "boom"}) is False


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
        if not role or role in {"session_meta", "system"}:
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
    resume_mark_is_fresh = False
    if resume_entry is not None and getattr(resume_entry, "resume_pending", False):
        resume_mark_is_fresh = _is_fresh_gateway_interruption(
            getattr(resume_entry, "last_resume_marked_at", None),
            window_secs=window,
        )
    is_resume_pending = bool(
        resume_entry is not None
        and getattr(resume_entry, "resume_pending", False)
        and (interruption_is_fresh or resume_mark_is_fresh)
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
        if message:
            resume_guidance = (
                "Address the user's NEW message below FIRST and focus "
                "on what the user is asking now."
            )
        else:
            resume_guidance = (
                "Report to the user that the session was restored "
                "successfully and ask what they would like to do next."
            )
        message = (
            f"[System note: The previous turn was interrupted by "
            f"{reason_phrase}; the gateway is now back online. "
            f"Any restart/shutdown command in the history has already "
            f"run — do NOT re-execute or verify it. {resume_guidance} "
            f"Do NOT re-execute old tool calls — skip any unfinished "
            f"work from the conversation history.]"
            + (f"\n\n{message}" if message else "")
        )
    elif has_fresh_tool_tail:
        message = (
            "[System note: A new message has arrived. The conversation "
            "history contains pending tool outputs from an interrupted turn. "
            "IGNORE those pending results. Address the user's NEW message "
            "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
            + message
        )

    # Empty-turn safety net: mirrors gateway/run.py — a blank
    # auto-resume turn on a resume_pending session must never reach the model.
    if (
        isinstance(message, str)
        and not message.strip()
        and resume_entry is not None
        and getattr(resume_entry, "resume_pending", False)
    ):
        sn_reason = getattr(resume_entry, "resume_reason", None) or "restart_timeout"
        sn_reason_phrase = (
            "a gateway restart"
            if sn_reason == "restart_timeout"
            else "a gateway shutdown"
            if sn_reason == "shutdown_timeout"
            else "a gateway interruption"
        )
        message = (
            f"[System note: The previous turn was interrupted by "
            f"{sn_reason_phrase}; the gateway is now back online. "
            f"Any restart/shutdown command in the history has already "
            f"run — do NOT re-execute or verify it. Report to the user "
            f"that the session was restored successfully and ask what "
            f"they would like to do next. Do NOT re-execute old tool "
            f"calls — skip any unfinished work from the conversation "
            f"history.]"
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

    def test_resume_pending_follows_compression_tip(self, tmp_path):
        """Interrupted platform mappings must not stay pinned to compressed roots."""
        store = _make_store(tmp_path)
        source = _make_source(
            platform=Platform.WEIXIN,
            chat_id="wx-chat",
            user_id="wx-user",
        )
        first = store.get_or_create_session(source)
        original_sid = first.session_id
        store.mark_resume_pending(first.session_key)

        with patch.object(
            store, "_compression_tip_for_session_id", return_value="child-session"
        ) as mock_tip:
            second = store.get_or_create_session(source)

        assert second.session_id == "child-session"
        assert second.resume_pending is True
        mock_tip.assert_called_with(original_sid)

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
        assert "NEW message" in result
        assert "Do NOT re-execute" in result
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
        assert "NEW message" in result

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
        assert "pending tool outputs" in result
        assert "Do NOT re-execute" in result

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

    def test_fresh_resume_mark_fires_despite_stale_transcript(self):
        """Regression: the recovery note must fire when the restart
        watchdog just stamped the session, even if the last persisted
        transcript row is far older than the freshness window.

        This is the exact gap that produced the blank-turn symptom: an
        active thread returned to after >1h of silence has a stale
        transcript clock, but the interruption itself (last_resume_marked_at)
        is seconds old. The two freshness signals must agree.
        """
        entry = self._pending_entry()
        entry.last_resume_marked_at = datetime.now()  # interrupted just now

        history = [
            {"role": "assistant", "content": "older context",
             "timestamp": time.time() - 3600},  # transcript clock stale
        ]
        result = _simulate_note_injection(
            history=history,
            user_message="continue",
            resume_entry=entry,
            window_secs=1800,
        )
        assert "[System note:" in result
        assert "gateway restart" in result

    def test_empty_resume_turn_never_reaches_model_blank(self):
        """Regression: a blank auto-resume turn on a resume_pending
        session must be backfilled with a recovery note, never sent empty.

        _schedule_resume_pending_sessions dispatches an empty-text internal
        event. If the resume_pending branch did not fire, the safety net
        must still produce non-blank text so the model does not reply with
        confused 'the message came through blank' noise.
        """
        entry = self._pending_entry()
        # Force the resume_pending branch to miss by making BOTH signals stale,
        # so only the empty-turn safety net can save us.
        entry.last_resume_marked_at = datetime.now() - timedelta(hours=2)
        history = [
            {"role": "assistant", "content": "old", "timestamp": time.time() - 7200},
        ]
        result = _simulate_note_injection(
            history=history,
            user_message="",  # the empty auto-resume event text
            resume_entry=entry,
            window_secs=1800,
        )
        assert result.strip(), "blank turn must never reach the model"
        assert "[System note:" in result

    def test_empty_turn_guard_only_applies_to_resume_pending(self):
        """The empty-turn backfill must NOT fire for ordinary sessions —
        a legitimately empty user turn (e.g. an uncaptioned image) on a
        non-resume_pending session is left untouched.
        """
        result = _simulate_note_injection(
            history=[
                {"role": "assistant", "content": "hi", "timestamp": time.time()},
            ],
            user_message="",
            resume_entry=None,
        )
        assert result == ""

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
        assert "pending tool outputs" in result
        assert "Do NOT re-execute" in result

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
        assert "pending tool outputs" in result
        assert "Do NOT re-execute" in result

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
        assert "pending tool outputs" in result
        assert "Do NOT re-execute" in result

    def test_no_note_when_nothing_to_resume(self):
        history = [
            {"role": "user", "content": "hello", "timestamp": time.time() - 2},
            {"role": "assistant", "content": "hi", "timestamp": time.time() - 1},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=None)
        assert result == "ping"

    def test_resume_pending_note_warns_against_reexecuting_restart(self):
        """The resume-pending note tells the model any restart/shutdown
        command in the history already ran and must not be re-executed or
        verified — the cognitive backstop to the source-level tail strip.
        """
        entry = self._pending_entry(reason="restart_timeout")
        result = _simulate_note_injection(
            history=[
                {"role": "assistant", "content": "in progress", "timestamp": time.time()},
            ],
            user_message="restarted!",
            resume_entry=entry,
        )
        assert "[System note:" in result
        assert "back online" in result
        assert "already" in result and "do NOT re-execute or verify" in result
        assert "restarted!" in result

    def test_resume_pending_empty_message_reports_recovery(self):
        """On the empty-message auto-resume startup turn there is no NEW user
        message, so the note instructs the model to report recovery and ask
        for instructions rather than 'address the user's NEW message'.
        """
        entry = self._pending_entry(reason="restart_timeout")
        result = _simulate_note_injection(
            history=[
                {"role": "assistant", "content": "in progress", "timestamp": time.time()},
            ],
            user_message="",
            resume_entry=entry,
        )
        assert "[System note:" in result
        assert "gateway restart" in result
        assert "restored successfully" in result
        assert "ask what they would like to do next" in result
        assert "do NOT re-execute or verify" in result
        # No phantom "NEW message" instruction when there is no new message.
        assert "NEW message" not in result
        # Nothing appended after the closing bracket (no empty user text).
        assert result.rstrip().endswith("]")


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
# Gateway startup auto-resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_auto_resume_schedules_fresh_pending_sessions():
    """Fresh resume_pending sessions should continue automatically after startup.

    This closes the UX gap where restart recovery only happened if the user sent
    another message after the gateway came back.
    """
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="resume-chat", thread_id="topic-1")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:group:resume-chat:topic-1",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="group",
        resume_pending=True,
        resume_reason="restart_timeout",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)

    assert scheduled == 1
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert isinstance(event, MessageEvent)
    assert event.internal is True
    assert event.message_type == MessageType.TEXT
    assert event.source == source
    # Text is empty — the existing _is_resume_pending branch in
    # _handle_message_with_agent owns the system-note injection so we don't
    # double it up.
    assert event.text == ""


@pytest.mark.asyncio
async def test_startup_auto_resume_includes_crash_recovery():
    """Crash-recovered sessions (reason=restart_interrupted) are also auto-resumed.

    suspend_recently_active() marks in-flight sessions with resume_reason
    "restart_interrupted" when the previous gateway exit was not clean
    (crash/SIGKILL/OOM).  These should get the same magic continuation as
    drain-timeout interruptions.
    """
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="crash-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:crash-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)

    assert scheduled == 1
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_auto_resume_skips_stale_entries():
    """Entries older than the freshness window must not be auto-resumed."""
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="stale-chat")
    stale_marker = datetime.now() - timedelta(
        seconds=_auto_continue_freshness_window() + 60
    )
    stale_entry = SessionEntry(
        session_key="agent:main:telegram:dm:stale-chat",
        session_id="sid",
        created_at=stale_marker,
        updated_at=stale_marker,
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
        last_resume_marked_at=stale_marker,
    )
    runner.session_store._entries = {stale_entry.session_key: stale_entry}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()

    assert scheduled == 0
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_startup_auto_resume_skips_suspended_and_originless():
    """suspended entries and entries with no origin are excluded."""
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="ok")
    suspended_entry = SessionEntry(
        session_key="agent:main:telegram:dm:suspended",
        session_id="sid-s",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
        suspended=True,
        last_resume_marked_at=datetime.now(),
    )
    originless = SessionEntry(
        session_key="agent:main:telegram:dm:originless",
        session_id="sid-o",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=None,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {
        suspended_entry.session_key: suspended_entry,
        originless.session_key: originless,
    }
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()

    assert scheduled == 0
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_startup_auto_resume_skips_disallowed_reasons():
    """Reasons outside the auto-resume set (e.g. a future custom reason) are skipped.

    These sessions still auto-resume on the next real user message via the
    existing _is_resume_pending branch — we just don't synthesize a turn
    for them at startup.
    """
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="other")
    other_entry = SessionEntry(
        session_key="agent:main:telegram:dm:other",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="manual_resume_request",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {other_entry.session_key: other_entry}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()

    assert scheduled == 0
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_startup_auto_resume_skips_unauthorized_owner():
    """A resume-pending session whose owner is no longer authorized under the
    current allowlist must not receive a synthesized agent turn on restart.

    Auto-resume dispatches a full agent turn without going through the normal
    inbound-message auth gate, so it re-checks _is_user_authorized here
    (issue #23778).  An unauthorized owner is skipped WITHOUT claiming a
    _running_agents slot or persisting one — the slot claim happens only
    after this gate passes.
    """
    runner, adapter = make_restart_runner()
    runner._is_user_authorized = lambda _source: False
    runner._persist_active_agents = MagicMock()
    source = make_restart_source(chat_id="revoked-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:revoked-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)

    assert scheduled == 0
    adapter.handle_message.assert_not_called()
    # No slot was claimed and nothing was persisted for the skipped session.
    assert pending_entry.session_key not in runner._running_agents
    runner._persist_active_agents.assert_not_called()


@pytest.mark.asyncio
async def test_startup_auto_resume_fails_closed_on_auth_error():
    """If the authorization check itself raises, the session is skipped
    (fail-closed) rather than resumed — a broken auth check must never
    default to granting a full agent turn.
    """
    runner, adapter = make_restart_runner()

    def _boom(_source):
        raise RuntimeError("allowlist backend down")

    runner._is_user_authorized = _boom
    runner._persist_active_agents = MagicMock()
    source = make_restart_source(chat_id="err-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:err-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)

    assert scheduled == 0
    adapter.handle_message.assert_not_called()
    assert pending_entry.session_key not in runner._running_agents
    runner._persist_active_agents.assert_not_called()


@pytest.mark.asyncio
async def test_startup_auto_resume_skips_when_adapter_unavailable():
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="resume-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:resume-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    runner.adapters = {}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()

    assert scheduled == 0
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_reconnect_reschedules_pending_after_late_platform_connect():
    """A platform offline at startup gets its pending sessions auto-resumed
    once it reconnects.

    Regression: the startup pass skips sessions whose adapter isn't connected
    yet (see test_startup_auto_resume_skips_when_adapter_unavailable). Before
    the fix those sessions were never rescheduled and recovered only if the
    user sent a fresh message — the documented startup auto-resume silently
    dropped. The reconnect watcher now retries the platform-scoped pass.
    """
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="late-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:late-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    # Platform was not connected at gateway startup → session skipped.
    runner.adapters = {}
    assert runner._schedule_resume_pending_sessions() == 0
    adapter.handle_message.assert_not_called()

    # Platform reconnects → its pending session is retried.
    runner.adapters = {Platform.TELEGRAM: adapter}
    scheduled = runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM)
    await asyncio.sleep(0)

    assert scheduled == 1
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert isinstance(event, MessageEvent)
    assert event.internal is True
    assert event.message_type == MessageType.TEXT
    assert event.text == ""
    assert event.source == source


@pytest.mark.asyncio
async def test_reconnect_reschedule_is_platform_scoped():
    """The platform filter limits the pass to that platform's sessions, so
    reconnecting one platform never resumes another's pending session."""
    runner, adapter = make_restart_runner()
    tg_source = make_restart_source(chat_id="tg-chat")
    discord_source = SessionSource(
        platform=Platform.DISCORD, chat_id="dc-chat", chat_type="dm", user_id="u1"
    )
    tg_entry = SessionEntry(
        session_key="agent:main:telegram:dm:tg-chat",
        session_id="sid-tg",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=tg_source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    discord_entry = SessionEntry(
        session_key="agent:main:discord:dm:dc-chat",
        session_id="sid-dc",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=discord_source,
        platform=Platform.DISCORD,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {
        tg_entry.session_key: tg_entry,
        discord_entry.session_key: discord_entry,
    }
    adapter.handle_message = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}

    scheduled = runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM)
    await asyncio.sleep(0)

    # Only the telegram session is resumed; the discord session waits for its
    # own reconnect.
    assert scheduled == 1
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source == tg_source


@pytest.mark.asyncio
async def test_auto_resume_skips_sessions_with_running_agent():
    """A session already being resumed (agent in-flight) is not scheduled
    again — guards against a double resume when a platform reconnects while a
    startup-scheduled resume is still running."""
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="inflight-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:inflight-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    runner._running_agents = {pending_entry.session_key: object()}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM)

    assert scheduled == 0
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_startup_restore_gate_queues_real_inbound_messages():
    """Real inbound messages wait while startup restore is in progress."""
    runner, _adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = []

    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
    )

    result = await runner._handle_message(inbound)

    assert result is None
    assert runner._startup_restore_queue == [inbound]


@pytest.mark.asyncio
async def test_startup_restore_waits_for_resume_before_draining_inbound():
    """Queued inbound turns replay only after startup resume tasks finish."""
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = []
    runner._startup_restore_tasks = []

    source = make_restart_source(chat_id="restore-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:restore-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}

    resume_done = asyncio.Event()
    seen: list[str] = []

    async def fake_handle_message(event: MessageEvent) -> None:
        if event.internal:
            seen.append("resume-start")
            task = asyncio.create_task(resume_done.wait())
            adapter._session_tasks[pending_entry.session_key] = task
            return
        seen.append(f"inbound:{event.text}")

    adapter.handle_message = fake_handle_message

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)

    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
    )
    assert await runner._handle_message(inbound) is None
    assert scheduled == 1
    assert seen == ["resume-start"]
    assert runner._startup_restore_queue == [inbound]

    finish_task = asyncio.create_task(runner._finish_startup_restore())
    await asyncio.sleep(0)
    assert seen == ["resume-start"]

    resume_done.set()
    await finish_task
    await asyncio.sleep(0)

    assert seen == ["resume-start", "inbound:hello"]
    assert runner._startup_restore_queue == []
    assert runner._startup_restore_in_progress is False


@pytest.mark.asyncio
async def test_startup_restore_gate_releases_when_a_resume_turn_runs_long(monkeypatch):
    """A single slow boot-resume turn must NOT hold the inbound gate shut.

    Regression for the live incident (2026-07-10): after a restart, one
    resumed session ran a 760s / 46-API-call turn.  Because
    ``_finish_startup_restore`` awaited *every* boot-resume task to completion
    before releasing ``_startup_restore_in_progress``, the whole gateway kept
    QUEUEING inbound messages for EVERY channel for ~13 min — no channel could
    be replied to (including #reply) until that one turn finished.

    The fix bounds the wait: after
    ``agent.gateway_startup_restore_drain_timeout`` seconds the gate releases,
    the inbound queue drains, and the slow resume turn keeps running in the
    background (NOT cancelled).  Duplicate-agent safety is preserved because the
    resume slot is pre-claimed synchronously in
    ``_schedule_resume_pending_sessions`` (asserted below), so a drained inbound
    queues behind that slot rather than spawning a second agent.
    """
    # Tiny bound so the test is fast; exercises the real config→env bridge name.
    monkeypatch.setenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", "0.2")

    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = []
    runner._startup_restore_tasks = []

    source = make_restart_source(chat_id="slow-restore-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:slow-restore-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}

    never_finishes = asyncio.Event()  # deliberately never set → the slow turn
    seen: list[str] = []

    async def fake_handle_message(event: MessageEvent) -> None:
        if event.internal:
            seen.append("resume-start")
            # A resume turn that runs longer than the drain bound.
            task = asyncio.create_task(never_finishes.wait())
            adapter._session_tasks[pending_entry.session_key] = task
            return
        seen.append(f"inbound:{event.text}")

    adapter.handle_message = fake_handle_message

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)
    assert scheduled == 1
    # The resume slot is claimed SYNCHRONOUSLY — this is what makes an early
    # drain safe (a drained inbound queues behind the slot, no duplicate agent).
    assert runner._running_agents.get(pending_entry.session_key) is _AGENT_PENDING_SENTINEL

    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
    )
    assert await runner._handle_message(inbound) is None
    assert seen == ["resume-start"]
    assert runner._startup_restore_queue == [inbound]

    # The gate must RELEASE despite the resume turn never finishing — bounded by
    # the drain timeout.  A generous asyncio timeout here fails loudly (rather
    # than hanging the suite) if the bound regresses back to "wait forever".
    await asyncio.wait_for(runner._finish_startup_restore(), timeout=5.0)
    await asyncio.sleep(0)

    # Gate opens before the background replay owner drains queued inbound.
    assert runner._startup_restore_in_progress is False
    assert runner._startup_restore_queue == []
    assert seen == ["resume-start", "inbound:hello"]

    # The slow resume turn was NOT cancelled — it keeps running in the
    # background (the whole point of asyncio.wait over wait_for/gather+timeout).
    slow_task = adapter._session_tasks[pending_entry.session_key]
    assert not slow_task.done()

    # Cleanup: release the background turn so the test doesn't leak a task.
    never_finishes.set()
    await asyncio.sleep(0)
    slow_task.cancel()


@pytest.mark.asyncio
async def test_startup_restore_gate_releases_before_queued_replay_finishes():
    """A blocked queued-message replay must not keep the global gate closed."""
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    runner._startup_restore_queue = [
        MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=make_restart_source(chat_id="blocked-replay-chat"),
        )
    ]

    replay_started = asyncio.Event()
    release_replay = asyncio.Event()

    async def blocked_handle_message(event: MessageEvent) -> None:
        del event
        replay_started.set()
        await release_replay.wait()

    adapter.handle_message = blocked_handle_message
    finish_task = asyncio.create_task(runner._finish_startup_restore())
    try:
        await asyncio.wait_for(replay_started.wait(), timeout=1.0)
        await asyncio.wait_for(asyncio.shield(finish_task), timeout=1.0)
        replay_task = runner._startup_restore_replay_task
        assert replay_task is not None
        assert runner._startup_restore_in_progress is False
        assert not replay_task.done()
    finally:
        release_replay.set()
        await finish_task
        replay_task = getattr(runner, "_startup_restore_replay_task", None)
        if replay_task is not None and not replay_task.done():
            await replay_task


@pytest.mark.asyncio
async def test_startup_restore_watchdog_releases_gate_if_finish_is_never_reached(
    monkeypatch,
):
    """A detached resume turn cannot bypass the gate's absolute deadline."""
    monkeypatch.setenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", "0.01")
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = [
        MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=make_restart_source(chat_id="watchdog-replay-chat"),
        )
    ]

    # Live recurrence: the tracked dispatch wrapper finished before the real
    # boot-resume turn entered the runner, so asyncio.wait saw done=1/pending=0
    # while the actual turn remained active outside _startup_restore_tasks.
    wrapper = asyncio.create_task(asyncio.sleep(0))
    await wrapper
    actual_turn = asyncio.create_task(asyncio.Event().wait())
    runner._startup_restore_tasks = [wrapper]

    replay_started = asyncio.Event()
    release_replay = asyncio.Event()

    async def blocked_handle_message(event: MessageEvent) -> None:
        del event
        replay_started.set()
        await release_replay.wait()

    adapter.handle_message = blocked_handle_message
    watchdog_task = asyncio.create_task(runner._startup_restore_gate_watchdog())
    await asyncio.wait_for(replay_started.wait(), timeout=1.0)
    await watchdog_task
    replay_task = runner._startup_restore_replay_task
    assert replay_task is not None

    try:
        assert runner._startup_restore_in_progress is False
        assert not actual_turn.done()
        assert not replay_task.done()
    finally:
        release_replay.set()
        await replay_task
        actual_turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await actual_turn


@pytest.mark.asyncio
async def test_startup_restore_zero_timeout_intentionally_disables_watchdog(monkeypatch):
    """The documented zero-value opt-out preserves the pre-fix unbounded mode."""
    monkeypatch.setenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", "0")
    runner, _adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = []

    await runner._startup_restore_gate_watchdog()

    assert runner._startup_restore_in_progress is True


@pytest.mark.asyncio
async def test_startup_restore_wait_error_fails_open(caplog):
    """Unexpected restore bookkeeping errors release the global inbound gate."""
    runner, _adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = []
    blocker = asyncio.create_task(asyncio.Event().wait())
    runner._startup_restore_tasks = [blocker]

    try:
        with patch(
            "gateway.run.asyncio.wait",
            new=AsyncMock(side_effect=RuntimeError("broken restore wait")),
        ):
            await runner._finish_startup_restore()
    finally:
        blocker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocker

    assert runner._startup_restore_in_progress is False
    assert "Startup-restore wait failed; releasing inbound gate" in caplog.text


@pytest.mark.asyncio
async def test_startup_restore_replay_error_fails_open(caplog):
    """A replay exception retains its event and cannot orphan the queue tail."""
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    runner._startup_restore_queue = [
        MessageEvent(
            text="first",
            message_type=MessageType.TEXT,
            source=make_restart_source(chat_id="same-session-chat"),
        ),
        MessageEvent(
            text="second",
            message_type=MessageType.TEXT,
            source=make_restart_source(chat_id="same-session-chat"),
        ),
        MessageEvent(
            text="other",
            message_type=MessageType.TEXT,
            source=make_restart_source(chat_id="other-session-chat"),
        ),
    ]
    adapter.handle_message = AsyncMock(
        side_effect=[RuntimeError("replay failed"), None, None, None]
    )

    await runner._finish_startup_restore()
    replay_task = runner._startup_restore_replay_task
    assert replay_task is not None
    await asyncio.wait_for(replay_task, timeout=1.0)

    assert runner._startup_restore_in_progress is False
    assert runner._startup_restore_queue == []
    assert [
        (call.args[0].source.chat_id, call.args[0].text)
        for call in adapter.handle_message.await_args_list
    ] == [
        ("same-session-chat", "first"),
        ("other-session-chat", "other"),
        ("same-session-chat", "first"),
        ("same-session-chat", "second"),
    ]
    assert "reason=dispatch raised before acceptance" in caplog.text


@pytest.mark.asyncio
async def test_startup_restore_replay_waits_for_connecting_adapter():
    """Watchdog replay retains inbound until its adapter is registered."""
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="connecting-adapter-chat"),
    )
    runner._startup_restore_queue = [event]
    runner.adapters = {}
    adapter.handle_message = AsyncMock()

    await runner._finish_startup_restore()
    replay_task = runner._startup_restore_replay_task
    assert replay_task is not None
    await asyncio.sleep(0.02)

    assert runner._startup_restore_queue == [event]
    assert not replay_task.done()

    runner.adapters[event.source.platform] = adapter
    await asyncio.wait_for(replay_task, timeout=1.0)

    adapter.handle_message.assert_awaited_once_with(event)
    assert runner._startup_restore_queue == []


@pytest.mark.asyncio
async def test_startup_restore_permanent_retry_warning_is_rate_limited(caplog):
    """A missing adapter reports depth/age once instead of warning every retry."""
    runner, _adapter = make_restart_runner()
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="missing-adapter-chat"),
    )
    runner._startup_restore_queue = [event]
    runner.adapters = {}

    with patch(
        "gateway.run.asyncio.sleep",
        new=AsyncMock(side_effect=[None, None, asyncio.CancelledError]),
    ):
        with pytest.raises(asyncio.CancelledError):
            await runner._drain_startup_restore_queue()

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= 30 and "Startup-restore replay deferred" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "queue_depth=1" in warnings[0]
    assert "age=" in warnings[0]
    assert runner._startup_restore_queue == [event]


@pytest.mark.asyncio
async def test_startup_restore_dispatch_retry_phase_warning_is_rate_limited(caplog):
    """Repeated dispatch failures do not emit a WARNING per attempt."""
    runner, adapter = make_restart_runner()
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="failing-adapter-chat"),
    )
    runner._startup_restore_queue = [event]
    adapter.handle_message = AsyncMock(side_effect=RuntimeError("not accepted"))

    with patch(
        "gateway.run.asyncio.sleep",
        new=AsyncMock(side_effect=[None, None, asyncio.CancelledError]),
    ):
        with pytest.raises(asyncio.CancelledError):
            await runner._drain_startup_restore_queue()

    warnings = [
        record.getMessage() for record in caplog.records if record.levelno >= 30
    ]
    assert sum("PHASE=startup_restore_replay_enter" in msg for msg in warnings) == 1
    assert sum("Startup-restore replay deferred" in msg for msg in warnings) == 1
    assert adapter.handle_message.await_count == 3
    assert runner._startup_restore_queue == [event]


@pytest.mark.asyncio
async def test_startup_restore_cross_session_failures_yield_and_back_off():
    """Two permanently blocked sessions cannot form a non-yielding rotation."""
    runner, _adapter = make_restart_runner()
    runner._startup_restore_queue = [
        MessageEvent(
            text="first",
            message_type=MessageType.TEXT,
            source=make_restart_source(chat_id="blocked-a"),
        ),
        MessageEvent(
            text="second",
            message_type=MessageType.TEXT,
            source=make_restart_source(chat_id="blocked-b"),
        ),
    ]
    runner.adapters = {}
    adapter_lookup = MagicMock(return_value=None)
    runner._adapter_for_source = adapter_lookup
    unrelated_ran = asyncio.Event()

    async def unrelated_work() -> None:
        await asyncio.sleep(0)
        unrelated_ran.set()

    replay_task = asyncio.create_task(runner._drain_startup_restore_queue())
    unrelated_task = asyncio.create_task(unrelated_work())
    try:
        await asyncio.wait_for(unrelated_ran.wait(), timeout=0.1)
        await asyncio.sleep(0.25)
        assert 1 <= adapter_lookup.call_count <= 3
        assert len(runner._startup_restore_queue) == 2
    finally:
        replay_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replay_task
        await unrelated_task


@pytest.mark.asyncio
async def test_startup_restore_watchdog_and_finisher_share_one_replay_owner(monkeypatch):
    """A watchdog/finisher boundary dispatches each queued event exactly once."""
    monkeypatch.setenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", "0.01")
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    runner._startup_restore_queue = [
        MessageEvent(
            text="first",
            message_type=MessageType.TEXT,
            source=make_restart_source(chat_id="race-first-chat"),
        ),
        MessageEvent(
            text="second",
            message_type=MessageType.TEXT,
            source=make_restart_source(chat_id="race-second-chat"),
        ),
    ]
    replay_started = asyncio.Event()
    release_replay = asyncio.Event()
    handled = []

    async def blocked_first_replay(event: MessageEvent) -> None:
        handled.append(event.source.chat_id)
        if event.source.chat_id == "race-first-chat":
            replay_started.set()
            await release_replay.wait()

    adapter.handle_message = blocked_first_replay
    watchdog_task = asyncio.create_task(runner._startup_restore_gate_watchdog())
    await asyncio.wait_for(replay_started.wait(), timeout=1.0)
    await watchdog_task
    replay_task = runner._startup_restore_replay_task
    assert replay_task is not None

    await runner._finish_startup_restore()

    assert runner._startup_restore_replay_task is replay_task
    assert runner._startup_restore_in_progress is False
    release_replay.set()
    await asyncio.wait_for(replay_task, timeout=1.0)

    assert handled == ["race-first-chat", "race-second-chat"]
    assert runner._startup_restore_queue == []


@pytest.mark.asyncio
async def test_startup_restore_replay_cancellation_retains_current_and_tail():
    """Cancelling replay preserves ownership of every not-yet-accepted event."""
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    events = [
        MessageEvent(
            text="first",
            message_type=MessageType.TEXT,
            source=make_restart_source(chat_id="cancel-first-chat"),
        ),
        MessageEvent(
            text="second",
            message_type=MessageType.TEXT,
            source=make_restart_source(chat_id="cancel-second-chat"),
        ),
    ]
    runner._startup_restore_queue = list(events)
    replay_started = asyncio.Event()
    release_first_owner = asyncio.Event()
    attempts = []

    async def blocked_replay(event: MessageEvent) -> None:
        attempts.append(event.text)
        if len(attempts) == 1:
            replay_started.set()
            await release_first_owner.wait()

    adapter.handle_message = blocked_replay
    finish_task = asyncio.create_task(runner._finish_startup_restore())
    try:
        await asyncio.wait_for(replay_started.wait(), timeout=1.0)
        await asyncio.wait_for(asyncio.shield(finish_task), timeout=1.0)
        replay_task = runner._startup_restore_replay_task
        assert replay_task is not None

        replay_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replay_task

        for _ in range(10):
            successor = runner._startup_restore_replay_task
            if successor is not None and successor is not replay_task:
                break
            await asyncio.sleep(0)
        else:
            pytest.fail("cancelled live replay owner was not replaced")

        await asyncio.wait_for(successor, timeout=1.0)
        assert attempts == ["first", "first", "second"]
        assert runner._startup_restore_queue == []
    finally:
        release_first_owner.set()
        await finish_task


def test_gateway_start_arms_restore_watchdog_before_platform_connects():
    """The absolute deadline is wired at the same point as the global gate."""
    source = inspect.getsource(GatewayRunner.start)
    gate_arm = source.index("self._startup_restore_in_progress = True")
    arm_log = source.index(
        "PHASE=startup_restore_gate_flip state=armed caller=start", gate_arm
    )
    watchdog_arm = source.index("self._startup_restore_gate_watchdog()", gate_arm)
    connect_loop = source.index(
        "for platform, platform_config in self.config.platforms.items()", gate_arm
    )

    assert gate_arm < arm_log < watchdog_arm < connect_loop
    assert '"timeout=%.3fs"' in source[arm_log:watchdog_arm]


def test_adapter_replay_contract_defines_exception_as_not_accepted():
    """Startup replay retries only before adapter ownership transfer."""
    assert BasePlatformAdapter.handle_message.__doc__ is not None
    contract = " ".join(BasePlatformAdapter.handle_message.__doc__.lower().split())
    assert "exception means ownership was not transferred" in contract


@pytest.mark.asyncio
async def test_background_resume_result_logs_late_failure_not_success_or_cancel():
    """A boot-resume turn that outlives the startup-restore gate and later FAILS
    must still be logged, not silently swallowed once it's discarded from
    ``_background_tasks`` (Greptile P2). Success and cancellation are quiet.
    """
    async def _boom():
        raise RuntimeError("late resume failure")

    async def _ok():
        return "fine"

    async def _blocks():
        await asyncio.Event().wait()

    # Failure => logged.
    failed = asyncio.create_task(_boom())
    try:
        await failed
    except RuntimeError:
        pass
    with patch("gateway.run.logger") as mock_logger:
        GatewayRunner._log_background_resume_result(failed)
        assert mock_logger.debug.call_count == 1

    # Success => quiet.
    ok = asyncio.create_task(_ok())
    await ok
    with patch("gateway.run.logger") as mock_logger:
        GatewayRunner._log_background_resume_result(ok)
        assert mock_logger.debug.call_count == 0

    # Cancellation (shutdown) => quiet, and must not raise on .exception().
    cancelled = asyncio.create_task(_blocks())
    cancelled.cancel()
    try:
        await cancelled
    except asyncio.CancelledError:
        pass
    with patch("gateway.run.logger") as mock_logger:
        GatewayRunner._log_background_resume_result(cancelled)
        assert mock_logger.debug.call_count == 0


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


@pytest.mark.asyncio
async def test_restart_notifies_home_channel_even_without_active_sessions():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )

    await runner._notify_active_sessions_of_shutdown()

    assert adapter.sent == [
        "⚠️ Gateway restarting — Your current task will be interrupted. "
        "Send any message after restart and I'll try to resume where you left off."
    ]


@pytest.mark.asyncio
async def test_restart_home_channel_notification_dedupes_active_chat():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    runner._running_agents["agent:main:telegram:dm:999"] = MagicMock()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="999",
        name="Ops Home",
    )

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_restart_home_channel_notification_not_deduped_across_threads():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    session_key = "agent:main:telegram:group:999"
    runner.session_store._entries[session_key] = MagicMock(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="999",
            chat_type="group",
            user_id="u1",
            thread_id="topic-7",
        )
    )
    runner._running_agents[session_key] = MagicMock()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="999",
        name="Ops Home",
    )

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 2
    assert adapter.sent_calls[0][2] == {"thread_id": "topic-7"}
    assert adapter.sent_calls[1][2] is None


@pytest.mark.asyncio
async def test_restart_home_channel_notification_ignores_false_send_result():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="network down"))

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_called_once()


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


@pytest.mark.asyncio
async def test_auto_resume_sets_sentinel_before_task_execution():
    """Auto-resume must claim the session slot before the task starts.

    Regression for #45456: between ``asyncio.create_task()`` and the task's
    first await (where ``_process_message_background`` sets the real
    sentinel), an inbound message could arrive and spin up a duplicate
    AIAgent.  The fix pre-claims the slot so the inbound path sees it as
    occupied.
    """
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="race-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:race-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}

    # Slow mock: hold the task open so we can inspect _running_agents
    # while it's in-flight.
    gate = asyncio.Event()

    async def _slow_handle(event):
        await gate.wait()

    adapter.handle_message = _slow_handle

    scheduled = runner._schedule_resume_pending_sessions()

    assert scheduled == 1
    # The sentinel must be set immediately — before the task starts executing.
    assert pending_entry.session_key in runner._running_agents
    assert runner._running_agents[pending_entry.session_key] is _AGENT_PENDING_SENTINEL
    assert pending_entry.session_key in runner._running_agents_ts

    # Release the task and let it complete.
    gate.set()
    await asyncio.sleep(0.05)

    # After the task completes, the sentinel should be cleaned up.
    assert pending_entry.session_key not in runner._running_agents


@pytest.mark.asyncio
async def test_auto_resume_sentinel_cleaned_on_task_failure():
    """If handle_message raises before _process_message_background, the
    sentinel must still be released so the session is not locked forever.
    """
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="fail-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:fail-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}

    async def _failing_handle(event):
        raise RuntimeError("adapter exploded")

    adapter.handle_message = _failing_handle

    scheduled = runner._schedule_resume_pending_sessions()
    assert scheduled == 1

    # Sentinel is set immediately.
    assert pending_entry.session_key in runner._running_agents

    # Let the task run and fail.
    await asyncio.sleep(0.05)

    # The sentinel must be cleaned up despite the failure.
    assert pending_entry.session_key not in runner._running_agents


@pytest.mark.asyncio
async def test_auto_resume_runs_agent_exactly_once_through_full_path():
    """Full-path regression: the pre-claim must NOT make auto-resume a no-op.

    The two tests above mock ``adapter.handle_message`` outright, so they
    only prove the sentinel is set/cleaned around a stub — they never
    exercise the real dispatch chain.  This drives the production path
    end to end:

        _schedule_resume_pending_sessions
          -> _guarded_handle_message
            -> adapter.handle_message            (real)
              -> _process_message_background      (real)
                -> _handle_message                (real)

    The risk the pre-claim introduces is a *self-bounce*: the resume
    turn's own ``_handle_message`` sees the sentinel it pre-claimed at
    the early running-agent guard, queues the event into
    ``_pending_messages`` and returns ``None`` without running the
    agent.  The adapter's late-arrival drain (in
    ``_process_message_background``'s ``finally``) re-dispatches the
    queued event, and because the guard wrapper's ``finally`` releases
    the pre-claim before the spawned drain task starts, the agent runs
    exactly once.  This test locks that invariant in: the resume agent
    must run once — never zero (regression) and never twice (the bug
    the fix targets).
    """
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="full-path-chat")
    session_key = runner._session_key_for_source(source)
    pending_entry = SessionEntry(
        session_key=session_key,
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {session_key: pending_entry}

    # Wire the REAL runner pipeline that _handle_message depends on.
    from gateway.run import GatewayRunner

    runner._handle_message = GatewayRunner._handle_message.__get__(
        runner, GatewayRunner
    )
    runner._release_running_agent_state = (
        GatewayRunner._release_running_agent_state.__get__(runner, GatewayRunner)
    )
    runner._check_slash_access = lambda *a, **k: None
    runner._begin_session_run_generation = lambda session_key: 1
    runner._is_session_run_current = lambda session_key, generation: True
    runner._invalidate_session_run_generation = lambda *a, **k: 0
    runner._claim_active_session_slot = lambda session_key, source: (object(), None)
    runner._active_session_leases = {}
    runner._busy_ack_ts = {}
    runner._post_turn_goal_continuation = AsyncMock()
    runner.session_store.get_or_create_session.return_value = None

    # Count how many times an actual agent run is started for this session.
    agent_runs: list[str] = []

    async def _fake_run(event, source, _quick_key, run_generation):
        agent_runs.append(_quick_key)
        return "RESUMED OK"

    runner._handle_message_with_agent = _fake_run

    # Route the adapter's real background pipeline at the real handler,
    # and stub the leaf send/typing calls so delivery is a no-op.
    adapter.set_message_handler(runner._handle_message)
    adapter.send = AsyncMock()
    adapter._keep_typing = AsyncMock()
    adapter._stop_typing_refresh = AsyncMock()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="1")
    )
    adapter._run_processing_hook = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    assert scheduled == 1
    # Pre-claim must be visible immediately.
    assert runner._running_agents.get(session_key) is _AGENT_PENDING_SENTINEL

    # Let the guarded task, the background task, and the late-arrival
    # drain task all settle.
    for _ in range(20):
        await asyncio.sleep(0.02)

    # Exactly one agent run for the resumed session — not zero (the
    # pre-claim did not swallow the resume) and not two (no duplicate).
    assert agent_runs == [session_key]
    # No leaked sentinel and no orphaned queued event.
    assert session_key not in runner._running_agents
    assert session_key not in getattr(adapter, "_pending_messages", {})
