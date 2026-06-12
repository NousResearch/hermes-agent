"""Tests for --source flag session persistence (issue #45107).

Verify that HERMES_SESSION_SOURCE env var (set by `hermes chat --source`)
takes precedence over the default platform source when creating sessions.
"""

import os
import tempfile
from pathlib import Path

import pytest


class TestSourceFlagPersistence:
    """Verify --source flag is respected in session creation."""

    def _make_agent(self, db, platform="cli"):
        """Create a minimal AIAgent with session_db wired up."""
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent._session_db = db
        agent._session_db_created = False
        agent.session_id = "test-source-session"
        agent.platform = platform
        agent.model = "test/model"
        agent._session_init_model_config = {}
        agent._cached_system_prompt = None
        agent._parent_session_id = None
        return agent

    def test_source_flag_overrides_cli_platform(self, monkeypatch):
        """When HERMES_SESSION_SOURCE is set, it should override platform='cli'."""
        from hermes_state import SessionDB

        monkeypatch.setenv("HERMES_SESSION_SOURCE", "distiller")
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            try:
                agent = self._make_agent(db, platform="cli")
                agent._ensure_db_session()

                sessions = db.list_sessions_rich(limit=1)
                assert len(sessions) == 1
                assert sessions[0]["source"] == "distiller"
            finally:
                db.close()

    def test_platform_used_when_no_env_var(self, monkeypatch):
        """Without HERMES_SESSION_SOURCE, platform should be used as source."""
        from hermes_state import SessionDB

        monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            try:
                agent = self._make_agent(db, platform="cli")
                agent._ensure_db_session()

                sessions = db.list_sessions_rich(limit=1)
                assert len(sessions) == 1
                assert sessions[0]["source"] == "cli"
            finally:
                db.close()

    def test_platform_used_when_env_var_empty(self, monkeypatch):
        """Empty HERMES_SESSION_SOURCE should fall through to platform."""
        from hermes_state import SessionDB

        monkeypatch.setenv("HERMES_SESSION_SOURCE", "")
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            try:
                agent = self._make_agent(db, platform="cli")
                agent._ensure_db_session()

                sessions = db.list_sessions_rich(limit=1)
                assert len(sessions) == 1
                assert sessions[0]["source"] == "cli"
            finally:
                db.close()

    def test_gateway_platform_preserved_when_no_env_var(self, monkeypatch):
        """Gateway platform (e.g. telegram) should be preserved when no env var."""
        from hermes_state import SessionDB

        monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            try:
                agent = self._make_agent(db, platform="telegram")
                agent._ensure_db_session()

                sessions = db.list_sessions_rich(limit=1)
                assert len(sessions) == 1
                assert sessions[0]["source"] == "telegram"
            finally:
                db.close()

    def test_source_flag_overrides_gateway_platform(self, monkeypatch):
        """HERMES_SESSION_SOURCE should also override gateway platform."""
        from hermes_state import SessionDB

        monkeypatch.setenv("HERMES_SESSION_SOURCE", "custom-tool")
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            try:
                agent = self._make_agent(db, platform="telegram")
                agent._ensure_db_session()

                sessions = db.list_sessions_rich(limit=1)
                assert len(sessions) == 1
                assert sessions[0]["source"] == "custom-tool"
            finally:
                db.close()
