"""CLI resume accepts exact IDs, unambiguous prefixes, and titles."""

import pytest

from hermes_cli.main import _resolve_session_by_name_or_id
from hermes_state import SessionDB


@pytest.mark.parametrize("query", ["root-session-id", "root-session", "remembered title"])
def test_resume_resolves_to_compression_tip(tmp_path, monkeypatch, query):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("root-session-id", source="cli")
        db.set_session_title("root-session-id", "remembered title")
        db.end_session("root-session-id", "compression")
        db.create_session("current-session-id", source="cli", parent_session_id="root-session-id")
    finally:
        db.close()
    assert _resolve_session_by_name_or_id(query) == "current-session-id"


def test_ambiguous_prefix_does_not_choose_a_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("shared-prefix-one", source="cli")
        db.create_session("shared-prefix-two", source="cli")
    finally:
        db.close()
    assert _resolve_session_by_name_or_id("shared-prefix") is None
    assert _resolve_session_by_name_or_id("shared-prefix-one") == "shared-prefix-one"


def test_title_wins_over_an_id_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("notes-2026-session", source="cli")
        db.create_session("titled-session", source="cli")
        db.set_session_title("titled-session", "notes")
    finally:
        db.close()
    assert _resolve_session_by_name_or_id("notes") == "titled-session"
