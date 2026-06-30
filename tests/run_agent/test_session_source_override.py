"""Regression: hermes chat --source tool must persist source=tool in state.db."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tool_source_env(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "tool")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_ensure_db_session_honors_hermes_session_source(tool_source_env):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=db,
                session_id="graphify-worker-test",
                platform="cli",
                skip_context_files=True,
                skip_memory=True,
            )
            agent._ensure_db_session()

            rows = db._conn.execute(
                "SELECT source FROM sessions WHERE id = ?",
                (agent.session_id,),
            ).fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "tool", (
                "platform='cli' must not override HERMES_SESSION_SOURCE=tool"
            )
        finally:
            db.close()
