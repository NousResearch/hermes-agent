"""``hermes --resume <id>`` for a session that lives in another profile.

A bare ``hermes --resume <id>`` opens the sticky active profile. A terminal multiplexer that
restores panes by typing that command into a fresh shell (no ``-p``, no ``HERMES_HOME``) hit
"Session not found" for every pane that ran under a named profile. The profile pre-parse now
looks the exact id up read-only in the sibling profiles: one owner is launched instead, an
explicit profile choice only gets a hint, and titles / ``latest`` never leave the active profile.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SID = "20260101_120000_abc123"


def _store(home: Path, *sessions: tuple[str, str | None]) -> None:
    from hermes_state import SessionDB

    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    db = SessionDB(db_path=home / "state.db")
    try:
        for session_id, title in sessions:
            db.create_session(session_id, source="cli")
            if title:
                db.set_session_title(session_id, title)
    finally:
        db.close()


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Root with default + work (sticky active) + coder (owns SID) + a store-less ``empty``."""
    root = tmp_path / ".hermes"
    _store(root)
    _store(root / "profiles" / "work", ("20260101_120000_aaa111", "work chat"))
    _store(root / "profiles" / "coder", (SID, None))
    (root / "profiles" / "empty").mkdir(parents=True)
    (root / "profiles" / "empty" / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "active_profile").write_text("work\n", encoding="utf-8")
    monkeypatch.setattr("hermes_constants._get_platform_default_hermes_home", lambda: root)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    for var in ("HERMES_SUPERVISED_CHILD", "HERMES_S6_SUPERVISED_CHILD", "INVOCATION_ID",
                "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"):
        monkeypatch.delenv(var, raising=False)
    return root


def _launch(monkeypatch, *argv: str) -> Path:
    """Run the startup profile pre-parse for ``hermes <argv>``; the HERMES_HOME it selected."""
    monkeypatch.setattr(sys, "argv", ["hermes", *argv])
    from hermes_cli.main import _apply_profile_override

    _apply_profile_override()
    return Path(os.environ["HERMES_HOME"])


@pytest.mark.parametrize("argv", [
    ("--resume", SID),
    ("--tui", "--resume", SID),
    ("chat", "-r", SID),
    (f"--resume={SID}", "--cli"),
])
def test_sticky_profile_resumes_id_in_its_owning_profile(root, monkeypatch, capsys, argv):
    assert _launch(monkeypatch, *argv) == root / "profiles" / "coder"
    assert f"Session {SID} belongs to profile coder; resuming there." in capsys.readouterr().err


def test_explicit_profile_flag_is_kept_and_hint_names_the_owner(root, monkeypatch, capsys):
    assert _launch(monkeypatch, "-p", "work", "--resume", SID) == root / "profiles" / "work"
    assert f"hermes -p coder --resume {SID}" in capsys.readouterr().err


def test_profile_pinned_by_hermes_home_is_kept_and_hint_names_the_owner(root, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "work"))

    assert _launch(monkeypatch, "--resume", SID) == root / "profiles" / "work"
    assert f"hermes -p coder --resume {SID}" in capsys.readouterr().err


def test_id_in_several_profiles_is_not_guessed(root, monkeypatch, capsys):
    _store(root, (SID, None))

    assert _launch(monkeypatch, "--resume", SID) == root / "profiles" / "work"
    assert "exists in profiles default, coder" in capsys.readouterr().err


@pytest.mark.parametrize("target", [
    ("work chat",), ("latest",), ("20260101_120000_aaa111",), ("20260101_120000_ffffff",),
    (SID, "notes"),  # unquoted multi-word title that merely starts with an id
])
def test_titles_keywords_and_local_or_unknown_ids_stay_in_the_active_profile(root, monkeypatch, capsys, target):
    assert _launch(monkeypatch, "--resume", *target) == root / "profiles" / "work"
    assert capsys.readouterr().err == ""
    assert not (root / "profiles" / "empty" / "state.db").exists()  # the lookup never creates a store
