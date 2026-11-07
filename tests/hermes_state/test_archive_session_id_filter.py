"""`sessions archive` reaches one specific session by `--session-id` in the shared filter chain.

Upstream's `--include-open` (#139597) lifted the ended-session guard for bulk archive; this
carries the id half of #131106: the shared filter chain accepts a session id, so archive can
target exactly one session — live ones still require `--include-open`.
"""
import argparse

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "test_state.db")
    yield session_db
    session_db.close()


def test_archive_by_session_id_targets_only_that_session(db):
    db.create_session(session_id="want", source="cli")
    db.create_session(session_id="other", source="cli")
    db.end_session("want", "completed")
    db.end_session("other", "completed")

    assert db.archive_sessions(session_id="want") == 1
    assert db.get_session("want")["archived"] == 1
    assert db.get_session("other")["archived"] == 0


def test_archive_session_id_respects_ended_guard_unless_include_open(db):
    db.create_session(session_id="live", source="desktop")

    assert db.archive_sessions(session_id="live") == 0
    assert db.get_session("live")["archived"] == 0

    assert db.archive_sessions(session_id="live", include_open=True) == 1
    assert db.get_session("live")["archived"] == 1


def test_sessions_parser_registers_session_id_once_and_archive_accepts_it():
    """Regression: `--session-id` lives in the shared filter chain only. Export's standalone
    registration had to go, or argparse raised 'conflicting option string' (#131106)."""
    from hermes_cli.subcommands.sessions import build_sessions_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    build_sessions_parser(subparsers, cmd_sessions=lambda *a, **k: None)

    args = parser.parse_args(["sessions", "archive", "--session-id", "abc", "--include-open"])
    assert args.session_id == "abc"
    assert args.include_open is True

    args = parser.parse_args(["sessions", "export", "--session-id", "abc", "out.jsonl"])
    assert args.session_id == "abc"
