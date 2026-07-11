"""Session transcript stores are read-only to agent file tools.

Inspired by Claude Code 2.1.205's auto-mode rule preventing transcript
manipulation. Hermes keeps canonical conversation history in state.db and may
also emit legacy JSON snapshots under sessions/; agent tools must not rewrite
or delete either store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def fake_homes(tmp_path, monkeypatch):
    import agent.file_safety as fs

    root = tmp_path / ".hermes"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    return root, profile


@pytest.mark.parametrize("relative", ["state.db", "sessions/session_abc.json"])
def test_session_state_paths_are_write_denied(fake_homes, relative):
    from agent.file_safety import is_write_denied

    _root, profile = fake_homes
    target = profile / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("existing transcript", encoding="utf-8")

    assert is_write_denied(str(target)) is True


def test_default_profile_state_db_is_write_denied_from_profile(fake_homes):
    from agent.file_safety import is_write_denied

    root, _profile = fake_homes
    target = root / "state.db"
    target.write_text("canonical transcript", encoding="utf-8")

    assert is_write_denied(str(target)) is True


def test_project_local_state_db_remains_writable(fake_homes, tmp_path):
    from agent.file_safety import is_write_denied

    target = tmp_path / "project" / "state.db"
    target.parent.mkdir()
    target.write_text("application database", encoding="utf-8")

    assert is_write_denied(str(target)) is False


def test_write_file_tool_preserves_existing_session_snapshot(fake_homes, monkeypatch):
    import tools.file_tools as ft

    _root, profile = fake_homes
    target = profile / "sessions" / "session_abc.json"
    target.parent.mkdir(parents=True)
    target.write_text("original transcript", encoding="utf-8")
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)

    result = json.loads(ft.write_file_tool(str(target), "tampered"))

    assert "error" in result
    assert target.read_text(encoding="utf-8") == "original transcript"
