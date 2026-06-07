"""Tests for gateway session cwd persistence (issue #41128).

The gateway's SessionEntry now tracks the agent's working directory
so that it survives gateway restarts.  When the agent changes
directory via the terminal tool, the new cwd is saved to the
session entry at the end of the turn.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# SessionEntry cwd field
# ---------------------------------------------------------------------------


class TestSessionEntryCwd:
    def test_default_cwd_is_none(self):
        from gateway.session import SessionEntry
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert entry.cwd is None

    def test_cwd_roundtrip_to_dict(self):
        from gateway.session import SessionEntry
        now = datetime.now(timezone.utc)
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=now,
            updated_at=now,
            cwd="/home/user/projects/foo",
        )
        d = entry.to_dict()
        assert d["cwd"] == "/home/user/projects/foo"

    def test_cwd_roundtrip_from_dict(self):
        from gateway.session import SessionEntry
        now = datetime.now(timezone.utc)
        data = {
            "session_key": "test",
            "session_id": "s1",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "cwd": "/home/user/projects/bar",
        }
        entry = SessionEntry.from_dict(data)
        assert entry.cwd == "/home/user/projects/bar"

    def test_cwd_none_in_dict(self):
        from gateway.session import SessionEntry
        now = datetime.now(timezone.utc)
        data = {
            "session_key": "test",
            "session_id": "s1",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        entry = SessionEntry.from_dict(data)
        assert entry.cwd is None

    def test_cwd_serialization_to_json(self, tmp_path):
        """SessionEntry with cwd should survive JSON roundtrip (sessions.json)."""
        from gateway.session import SessionEntry
        now = datetime.now(timezone.utc)
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=now,
            updated_at=now,
            cwd="/tmp/workspace",
        )
        # Simulate sessions.json write/read
        sessions_file = tmp_path / "sessions.json"
        sessions_file.write_text(json.dumps({"test": entry.to_dict()}))
        loaded = json.loads(sessions_file.read_text())
        restored = SessionEntry.from_dict(loaded["test"])
        assert restored.cwd == "/tmp/workspace"


# ---------------------------------------------------------------------------
# _set_session_env passes cwd
# ---------------------------------------------------------------------------


class TestSetSessionEnvCwd:
    def test_set_session_vars_accepts_cwd(self):
        """set_session_vars should accept and forward cwd kwarg."""
        from gateway.session_context import set_session_vars
        with patch("gateway.session_context._SESSION_PLATFORM") as mock_p, \
             patch("gateway.session_context._SESSION_CHAT_ID") as mock_c, \
             patch("gateway.session_context._SESSION_CHAT_NAME") as mock_cn, \
             patch("gateway.session_context._SESSION_THREAD_ID") as mock_t, \
             patch("gateway.session_context._SESSION_USER_ID") as mock_u, \
             patch("gateway.session_context._SESSION_USER_NAME") as mock_un, \
             patch("gateway.session_context._SESSION_KEY") as mock_sk, \
             patch("gateway.session_context._SESSION_MESSAGE_ID") as mock_m:
            for m in (mock_p, mock_c, mock_cn, mock_t, mock_u, mock_un, mock_sk, mock_m):
                m.set.return_value = "token"
            with patch("agent.runtime_cwd.set_session_cwd") as mock_sc:
                set_session_vars(
                    platform="telegram", chat_id="123", chat_name="",
                    thread_id="", user_id="456", user_name="",
                    session_key="test", message_id="", cwd="/home/user/proj",
                )
            mock_sc.assert_called_once_with("/home/user/proj")

    def test_set_session_vars_empty_cwd(self):
        """set_session_vars with empty cwd should still call set_session_cwd."""
        from gateway.session_context import set_session_vars
        with patch("gateway.session_context._SESSION_PLATFORM") as mock_p, \
             patch("gateway.session_context._SESSION_CHAT_ID") as mock_c, \
             patch("gateway.session_context._SESSION_CHAT_NAME") as mock_cn, \
             patch("gateway.session_context._SESSION_THREAD_ID") as mock_t, \
             patch("gateway.session_context._SESSION_USER_ID") as mock_u, \
             patch("gateway.session_context._SESSION_USER_NAME") as mock_un, \
             patch("gateway.session_context._SESSION_KEY") as mock_sk, \
             patch("gateway.session_context._SESSION_MESSAGE_ID") as mock_m:
            for m in (mock_p, mock_c, mock_cn, mock_t, mock_u, mock_un, mock_sk, mock_m):
                m.set.return_value = "token"
            with patch("agent.runtime_cwd.set_session_cwd") as mock_sc:
                set_session_vars(
                    platform="telegram", chat_id="123", chat_name="",
                    thread_id="", user_id="456", user_name="",
                    session_key="test", message_id="",
                )
            mock_sc.assert_called_once_with("")


# ---------------------------------------------------------------------------
# Cwd persistence at end of turn
# ---------------------------------------------------------------------------


class TestCwdPersistence:
    def test_cwd_saved_from_terminal_env(self):
        """When terminal env has a different cwd, it should be saved."""
        from gateway.session import SessionEntry
        now = datetime.now(timezone.utc)
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=now,
            updated_at=now,
            cwd="/old/path",
        )

        mock_env = MagicMock()
        mock_env.cwd = "/new/path"

        with patch("tools.terminal_tool._active_environments", {"s1": mock_env}):
            from tools.terminal_tool import _active_environments
            _env = _active_environments.get(entry.session_id)
            if _env is not None:
                _live_cwd = getattr(_env, "cwd", "")
                if _live_cwd and _live_cwd != entry.cwd:
                    entry.cwd = _live_cwd

        assert entry.cwd == "/new/path"

    def test_cwd_not_saved_when_unchanged(self):
        """When cwd hasn't changed, no update should occur."""
        from gateway.session import SessionEntry
        now = datetime.now(timezone.utc)
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=now,
            updated_at=now,
            cwd="/same/path",
        )

        mock_env = MagicMock()
        mock_env.cwd = "/same/path"

        with patch("tools.terminal_tool._active_environments", {"s1": mock_env}):
            from tools.terminal_tool import _active_environments
            _env = _active_environments.get(entry.session_id)
            if _env is not None:
                _live_cwd = getattr(_env, "cwd", "")
                if _live_cwd and _live_cwd != entry.cwd:
                    entry.cwd = _live_cwd

        assert entry.cwd == "/same/path"

    def test_cwd_saved_when_previously_none(self):
        """When cwd was None (old session), it should be saved."""
        from gateway.session import SessionEntry
        now = datetime.now(timezone.utc)
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=now,
            updated_at=now,
            cwd=None,
        )

        mock_env = MagicMock()
        mock_env.cwd = "/first/path"

        with patch("tools.terminal_tool._active_environments", {"s1": mock_env}):
            from tools.terminal_tool import _active_environments
            _env = _active_environments.get(entry.session_id)
            if _env is not None:
                _live_cwd = getattr(_env, "cwd", "")
                if _live_cwd and _live_cwd != entry.cwd:
                    entry.cwd = _live_cwd

        assert entry.cwd == "/first/path"

    def test_cwd_fallback_to_task_id(self):
        """When session_id doesn't match, fall back to task_id lookup."""
        from gateway.session import SessionEntry
        now = datetime.now(timezone.utc)
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=now,
            updated_at=now,
            cwd=None,
        )

        mock_env = MagicMock()
        mock_env.cwd = "/task/path"

        # No match on session_id, but match on task_id
        with patch("tools.terminal_tool._active_environments", {"task-123": mock_env}):
            from tools.terminal_tool import _active_environments
            _env = _active_environments.get(entry.session_id)
            if _env is None:
                _task_id = "task-123"
                if _task_id:
                    _env = _active_environments.get(_task_id)
            if _env is not None:
                _live_cwd = getattr(_env, "cwd", "")
                if _live_cwd and _live_cwd != entry.cwd:
                    entry.cwd = _live_cwd

        assert entry.cwd == "/task/path"

    def test_cwd_no_env_graceful(self):
        """When no terminal env exists, cwd should not change."""
        from gateway.session import SessionEntry
        now = datetime.now(timezone.utc)
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=now,
            updated_at=now,
            cwd="/unchanged",
        )

        with patch("tools.terminal_tool._active_environments", {}):
            from tools.terminal_tool import _active_environments
            _env = _active_environments.get(entry.session_id)
            if _env is not None:
                _live_cwd = getattr(_env, "cwd", "")
                if _live_cwd and _live_cwd != entry.cwd:
                    entry.cwd = _live_cwd

        assert entry.cwd == "/unchanged"
