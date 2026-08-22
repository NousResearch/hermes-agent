"""Regression tests for local CLI launch-directory persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.mark.parametrize("platform", ["cli", None])
def test_cli_source_override_keeps_launch_cwd(monkeypatch, tmp_path, platform):
    """A durable source label must not erase the local CLI workspace."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")
    monkeypatch.setenv("TERMINAL_ENV", "local")

    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        session_db=db,
        session_id="source-override",
        platform=platform,
        skip_context_files=True,
        skip_memory=True,
    )
    try:
        agent._ensure_db_session()
        row = db.get_session(agent.session_id)

        assert row["source"] == "desktop"
        assert Path(row["cwd"]) == workspace
    finally:
        agent.close()
        db.close()


def test_non_cli_platform_does_not_borrow_cli_source_cwd(monkeypatch, tmp_path):
    """A relabeled gateway session must not inherit the process launch cwd."""
    workspace = tmp_path / "gateway-launch"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "cli")
    monkeypatch.setenv("TERMINAL_ENV", "local")

    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        session_db=db,
        session_id="gateway-source-override",
        platform="desktop",
        skip_context_files=True,
        skip_memory=True,
    )
    try:
        agent._ensure_db_session()
        row = db.get_session(agent.session_id)

        assert row["source"] == "cli"
        assert row["cwd"] is None
    finally:
        agent.close()
        db.close()
