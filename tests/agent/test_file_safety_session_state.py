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
    """Point HERMES_HOME at a temp profile dir.

    Uses the real env-var resolution chain (get_hermes_home /
    get_default_hermes_root) instead of monkeypatching private helpers —
    a stale monkeypatch on a since-deleted helper broke CI in July 2026
    (monkeypatch.setattr raises AttributeError on missing attributes).
    HERMES_HOME=<root>/profiles/<name> makes get_default_hermes_root()
    derive <root> via the `profiles` parent-dir rule, so both the
    profile-scoped and root-scoped deny lists resolve into tmp_path.
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
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


def test_write_file_tool_preserves_existing_session_snapshot(fake_homes):
    import tools.file_tools as ft

    _root, profile = fake_homes
    target = profile / "sessions" / "session_abc.json"
    target.parent.mkdir(parents=True)
    target.write_text("original transcript", encoding="utf-8")

    result = json.loads(ft.write_file_tool(str(target), "tampered"))

    assert "error" in result
    assert target.read_text(encoding="utf-8") == "original transcript"
