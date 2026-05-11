"""Reproduction tests for #18370 fallout: lazy session creation regressions.

Tests cover:
1. Bug #20001 — _finalize_session() uses stale session_key after compression rotation
2. Bug #20001 — _sync_session_key_after_compress called post-run_conversation
3. Bug #19029 — pending_title ValueError leaves title wedged
4. Bug #18765 — gateway surfaces null response when agent did work
5. Prune — finalize_orphaned_compression_sessions catches ghost continuations
"""

import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Helpers
# ===========================================================================

def _make_session_db(tmp_path):
    """Create a real SessionDB for integration-style tests."""
    from hermes_state import SessionDB
    db_path = tmp_path / "test_state.db"
    return SessionDB(db_path=db_path)


def _tui_session(agent=None, session_key="session-key-old", **extra):
    """Minimal TUI gateway session dict matching server._sessions values."""
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(session_id=session_key),
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "pending_title": None,
        **extra,
    }


# ===========================================================================
# Bug #20001: _finalize_session uses stale session_key
# ===========================================================================

class TestFinalizeSessionUsesAgentSessionId:
    """After compression rotates agent.session_id, _finalize_session()
    must call end_session() on the NEW (current) session_id, not the stale
    session_key stored in the session dict."""

    def test_finalize_targets_agent_session_id_not_stale_key(self, tmp_path):
        """Reproduction: agent.session_id rotated by compression, but
        session['session_key'] still holds old value. _finalize_session()
        should end the agent's current session."""
        from tui_gateway import server

        db = _make_session_db(tmp_path)

        # Create two sessions: parent (already ended by compression) and continuation
        db.create_session(session_id="parent-session", source="tui", model="test")
        db.end_session("parent-session", "compression")

        db.create_session(
            session_id="continuation-session",
            source="tui",
            model="test",
            parent_session_id="parent-session",
        )
        # Continuation is NOT ended — this is the bug state

        # Agent has rotated to continuation session
        agent = types.SimpleNamespace(
            session_id="continuation-session",
            commit_memory_session=lambda h: None,
        )

        # Session dict still holds stale key (the bug condition)
        session = _tui_session(
            agent=agent,
            session_key="parent-session",
            history=[{"role": "user", "content": "hello"}],
        )

        # Monkeypatch _get_db to return our test DB
        with patch.object(server, "_get_db", return_value=db):
            with patch.object(server, "_notify_session_boundary", lambda *a: None):
                server._finalize_session(session, end_reason="tui_close")

        # The continuation session should be ended
        continuation = db.get_session("continuation-session")
        assert continuation["ended_at"] is not None, (
            "_finalize_session should end the agent's current session (continuation), "
            "not the already-ended parent"
        )
        assert continuation["end_reason"] == "tui_close"

    def test_finalize_fallback_to_session_key_when_agent_is_none(self, tmp_path):
        """When agent is None (e.g. session never fully initialized),
        _finalize_session falls back to session_key."""
        from tui_gateway import server

        db = _make_session_db(tmp_path)
        db.create_session(session_id="orphan-key", source="tui", model="test")

        session = _tui_session(agent=None, session_key="orphan-key")

        with patch.object(server, "_get_db", return_value=db):
            with patch.object(server, "_notify_session_boundary", lambda *a: None):
                server._finalize_session(session, end_reason="tui_close")

        row = db.get_session("orphan-key")
        assert row["ended_at"] is not None
        assert row["end_reason"] == "tui_close"


# ===========================================================================
# Bug #20001: _sync_session_key_after_compress post-run_conversation
# ===========================================================================

class TestSyncSessionKeyAfterAutoCompress:
    """When auto-compression fires inside run_conversation(), the post-turn
    code in _run_prompt_submit must call _sync_session_key_after_compress
    to update session_key for downstream consumers (title, goals, etc.)."""

    def test_session_key_synced_after_run_conversation_with_compression(self, monkeypatch):
        """Simulate: run_conversation() internally compresses and rotates
        agent.session_id. After it returns, session['session_key'] must match."""
        from tui_gateway import server

        class _CompressingAgent:
            """Agent that simulates compression-driven session_id rotation."""
            def __init__(self):
                self.session_id = "pre-compress-key"
                self._cached_system_prompt = ""

            def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
                # Simulate what _compress_context does: rotate session_id
                self.session_id = "post-compress-key"
                return {
                    "final_response": "done",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "done"},
                    ],
                }

        agent = _CompressingAgent()
        session = _tui_session(agent=agent, session_key="pre-compress-key")

        # Track if _sync_session_key_after_compress was called
        sync_calls = []
        original_sync = server._sync_session_key_after_compress

        def _tracking_sync(sid, sess, **kwargs):
            sync_calls.append((sid, sess.get("session_key")))
            # Just update the key directly (skip approval routing etc.)
            new_id = getattr(sess.get("agent"), "session_id", None) or ""
            if new_id and new_id != sess.get("session_key"):
                sess["session_key"] = new_id

        monkeypatch.setattr(server, "_sync_session_key_after_compress", _tracking_sync)
        monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
        monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
        monkeypatch.setattr(server, "render_message", lambda raw, cols: None)

        # Use _ImmediateThread pattern to run synchronously
        class _ImmediateThread:
            def __init__(self, target=None, daemon=None, **kw):
                self._target = target
            def start(self):
                self._target()

        server._sessions["test-sid"] = session
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

        try:
            server.handle_request({
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "test-sid", "text": "hello"},
            })

            # Sync should have been called
            assert len(sync_calls) > 0, (
                "_sync_session_key_after_compress must be called after run_conversation "
                "to pick up compression-driven session_id rotation"
            )

            # session_key should now match agent.session_id
            assert session["session_key"] == "post-compress-key", (
                "session_key must be updated to match agent.session_id after compression"
            )
        finally:
            server._sessions.pop("test-sid", None)


# ===========================================================================
# Bug #19029: pending_title ValueError wedge
# ===========================================================================

class TestPendingTitleValueError:
    """When set_session_title raises ValueError (duplicate/invalid title),
    pending_title must be cleared — not left wedged forever."""

    def test_valueerror_clears_pending_title(self, monkeypatch):
        """ValueError from set_session_title should drop pending_title."""
        from tui_gateway import server

        mock_db = MagicMock()
        mock_db.set_session_title.side_effect = ValueError("duplicate title")

        class _Agent:
            session_id = "test-session"
            _cached_system_prompt = ""
            def run_conversation(self, prompt, **kw):
                return {
                    "final_response": "ok",
                    "messages": [{"role": "assistant", "content": "ok"}],
                }

        session = _tui_session(
            agent=_Agent(),
            session_key="test-session",
            pending_title="My Title",
        )

        monkeypatch.setattr(server, "_get_db", lambda: mock_db)
        monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
        monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
        monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
        monkeypatch.setattr(
            server, "_sync_session_key_after_compress", lambda *a, **kw: None
        )

        class _ImmediateThread:
            def __init__(self, target=None, daemon=None, **kw):
                self._target = target
            def start(self):
                self._target()

        server._sessions["sid"] = session
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

        try:
            server.handle_request({
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hello"},
            })

            # pending_title should be cleared on ValueError, not left wedged
            assert session.get("pending_title") is None, (
                "ValueError from set_session_title must clear pending_title "
                "so auto-title can take over"
            )
        finally:
            server._sessions.pop("sid", None)

    def test_other_exception_keeps_pending_title_for_retry(self, monkeypatch):
        """Non-ValueError exceptions should keep pending_title for retry."""
        from tui_gateway import server

        mock_db = MagicMock()
        mock_db.set_session_title.side_effect = RuntimeError("transient DB lock")

        class _Agent:
            session_id = "test-session"
            _cached_system_prompt = ""
            def run_conversation(self, prompt, **kw):
                return {
                    "final_response": "ok",
                    "messages": [{"role": "assistant", "content": "ok"}],
                }

        session = _tui_session(
            agent=_Agent(),
            session_key="test-session",
            pending_title="My Title",
        )

        monkeypatch.setattr(server, "_get_db", lambda: mock_db)
        monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
        monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
        monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
        monkeypatch.setattr(
            server, "_sync_session_key_after_compress", lambda *a, **kw: None
        )

        class _ImmediateThread:
            def __init__(self, target=None, daemon=None, **kw):
                self._target = target
            def start(self):
                self._target()

        server._sessions["sid"] = session
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

        try:
            server.handle_request({
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hello"},
            })

            # Non-ValueError should keep pending_title for retry
            assert session.get("pending_title") == "My Title", (
                "Non-ValueError exceptions should keep pending_title intact "
                "for retry on next turn"
            )
        finally:
            server._sessions.pop("sid", None)


# ===========================================================================
# Bug #18765: Gateway surfaces null response
# ===========================================================================

class TestGatewaySurfacesNullResponse:
    """When the agent does work (api_calls > 0) but returns no final_response,
    the gateway must surface an error to the user instead of silently sending
    nothing. Tests exercise the production _normalize_empty_agent_response helper."""

    def test_partial_response_surfaces_error(self):
        """Agent returns partial=True with no response → user sees error."""
        from gateway.run import _normalize_empty_agent_response

        agent_result = {
            "final_response": None,
            "api_calls": 5,
            "partial": True,
            "interrupted": False,
            "error": "Model generated invalid tool call: nonexistent_tool",
        }

        response = agent_result.get("final_response") or ""
        response = _normalize_empty_agent_response(
            agent_result, response, history_len=10,
        )

        assert response != "", "Null response with api_calls>0 must be surfaced"
        assert "nonexistent_tool" in response

    def test_interrupted_response_stays_empty(self):
        """Interrupted agent → response stays empty (platform handles UX)."""
        from gateway.run import _normalize_empty_agent_response

        agent_result = {
            "final_response": None,
            "api_calls": 3,
            "partial": False,
            "interrupted": True,
        }

        response = agent_result.get("final_response") or ""
        response = _normalize_empty_agent_response(
            agent_result, response, history_len=10,
        )

        assert response == "", "Interrupted turns should not get synthetic responses"

    def test_failed_context_overflow(self):
        """Agent failed with context overflow → specific guidance message."""
        from gateway.run import _normalize_empty_agent_response

        agent_result = {
            "final_response": None,
            "api_calls": 0,
            "failed": True,
            "error": "400 Bad Request: context length exceeded",
        }

        response = agent_result.get("final_response") or ""
        response = _normalize_empty_agent_response(
            agent_result, response, history_len=60,
        )

        assert "context window" in response
        assert "/compact" in response

    def test_failed_generic_error(self):
        """Agent failed with non-context error → generic error message."""
        from gateway.run import _normalize_empty_agent_response

        agent_result = {
            "final_response": None,
            "api_calls": 0,
            "failed": True,
            "error": "500 Internal Server Error",
        }

        response = agent_result.get("final_response") or ""
        response = _normalize_empty_agent_response(
            agent_result, response, history_len=5,
        )

        assert "500 Internal Server Error" in response
        assert "/reset" in response

    def test_nonempty_response_passes_through(self):
        """Non-empty response is returned unchanged."""
        from gateway.run import _normalize_empty_agent_response

        agent_result = {"final_response": "Hello!", "api_calls": 1}
        response = "Hello!"
        result = _normalize_empty_agent_response(
            agent_result, response, history_len=5,
        )

        assert result == "Hello!"


# ===========================================================================
# Prune: finalize_orphaned_compression_sessions
# ===========================================================================

class TestFinalizeOrphanedCompressionSessions:
    """The prune migration marks ghost compression continuations as ended."""

    def test_marks_ghost_continuation_with_compression_parent(self, tmp_path):
        """Ghost session with compression-ended parent + messages → finalized."""
        db = _make_session_db(tmp_path)

        # Parent session (ended by compression — this is the key condition)
        db.create_session(session_id="parent", source="tui", model="test")
        db.end_session("parent", "compression")

        # Ghost continuation (has messages, never finalized)
        db.create_session(
            session_id="ghost-cont",
            source="tui",
            model="test",
            parent_session_id="parent",
        )
        db.append_message("ghost-cont", role="user", content="hello")
        db.append_message("ghost-cont", role="assistant", content="hi")

        # Make it old enough (fake started_at)
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (time.time() - 800000, "ghost-cont"),  # ~9 days old
            )
        )

        count = db.finalize_orphaned_compression_sessions()
        assert count == 1

        session = db.get_session("ghost-cont")
        assert session["ended_at"] is not None
        assert session["end_reason"] == "orphaned_compression"

    def test_skips_session_without_parent(self, tmp_path):
        """Ghost session without parent_session_id is NOT a compression
        continuation — should not be touched by this prune."""
        db = _make_session_db(tmp_path)

        db.create_session(session_id="ghost-notitle", source="tui", model="test")
        db.append_message("ghost-notitle", role="user", content="test")

        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (time.time() - 800000, "ghost-notitle"),
            )
        )

        count = db.finalize_orphaned_compression_sessions()
        assert count == 0

    def test_skips_recent_sessions(self, tmp_path):
        """Sessions younger than 7 days are not touched."""
        db = _make_session_db(tmp_path)

        # Create parent first to satisfy FK constraint
        db.create_session(session_id="some-parent", source="tui", model="test")
        db.create_session(
            session_id="recent",
            source="tui",
            model="test",
            parent_session_id="some-parent",
        )
        db.append_message("recent", role="user", content="hello")
        # started_at is now() — within 7 days

        count = db.finalize_orphaned_compression_sessions()
        assert count == 0

    def test_skips_sessions_with_end_reason(self, tmp_path):
        """Properly finalized sessions (even without api_call_count) are skipped."""
        db = _make_session_db(tmp_path)

        # Create parent first to satisfy FK constraint
        db.create_session(session_id="parent", source="tui", model="test")
        db.end_session("parent", "compression")

        db.create_session(
            session_id="already-ended",
            source="tui",
            model="test",
            parent_session_id="parent",
        )
        db.append_message("already-ended", role="user", content="hello")
        db.end_session("already-ended", "user_exit")

        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (time.time() - 800000, "already-ended"),
            )
        )

        count = db.finalize_orphaned_compression_sessions()
        assert count == 0

    def test_skips_session_with_non_compression_parent(self, tmp_path):
        """Child session whose parent was NOT ended by compression should
        not be touched — it's not from the compression continuation path."""
        db = _make_session_db(tmp_path)

        # Parent ended by user_exit, not compression
        db.create_session(session_id="parent", source="tui", model="test")
        db.end_session("parent", "user_exit")

        db.create_session(
            session_id="child",
            source="tui",
            model="test",
            parent_session_id="parent",
        )
        db.append_message("child", role="user", content="hello")

        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (time.time() - 800000, "child"),
            )
        )

        count = db.finalize_orphaned_compression_sessions()
        assert count == 0

    def test_skips_sessions_without_messages(self, tmp_path):
        """Empty sessions (no messages) are NOT targeted by this prune —
        those are handled by prune_empty_ghost_sessions()."""
        db = _make_session_db(tmp_path)

        # Create parent first to satisfy FK constraint
        db.create_session(session_id="parent", source="tui", model="test")
        db.end_session("parent", "compression")

        db.create_session(
            session_id="empty-ghost",
            source="tui",
            model="test",
            parent_session_id="parent",
        )
        # No messages appended

        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (time.time() - 800000, "empty-ghost"),
            )
        )

        count = db.finalize_orphaned_compression_sessions()
        assert count == 0

    def test_titled_ghost_with_parent_is_caught(self, tmp_path):
        """Ghost continuation that HAS a title (propagated from parent by
        _compress_context) is still caught via parent with end_reason='compression'."""
        db = _make_session_db(tmp_path)

        # Create parent first — ended by compression
        db.create_session(session_id="parent", source="tui", model="test")
        db.set_session_title("parent", "Chat")
        db.end_session("parent", "compression")

        db.create_session(
            session_id="titled-ghost",
            source="tui",
            model="test",
            parent_session_id="parent",
        )
        db.set_session_title("titled-ghost", "Chat (2)")
        db.append_message("titled-ghost", role="user", content="continued...")

        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (time.time() - 800000, "titled-ghost"),
            )
        )

        count = db.finalize_orphaned_compression_sessions()
        assert count == 1

        session = db.get_session("titled-ghost")
        assert session["end_reason"] == "orphaned_compression"
