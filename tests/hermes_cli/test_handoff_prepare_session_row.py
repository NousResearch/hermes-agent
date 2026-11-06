"""Regression tests for #133726 — /handoff on a session that has no DB row yet.

``_handoff_prepare_session`` used to "create" the missing row via ``set_session_title``,
which never inserts — on a missing row it is a silent no-op (it only UPDATEs). The
subsequent ``request_handoff`` UPDATE matched zero rows, returned False, and the CLI
reported the handoff as "already in flight" — a reason unrelated to the actual failure,
on the exact session state (fresh, nothing exchanged) a handoff is launched for.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return SessionDB(db_path=home / "state.db")


def _cli(db, session_id="sess-handoff-1"):
    cli = MagicMock()
    cli._agent_running = False
    cli._session_db = db
    cli.session_id = session_id
    cli.model = "test-model"
    cli.max_turns = 7
    cli.reasoning_config = None
    return cli


def test_prepare_creates_the_missing_row(db):
    """A fresh session must leave _handoff_prepare_session with a persisted row."""
    cli = _cli(db)
    title = CLICommandsMixin._handoff_prepare_session(cli)
    assert title, "prepare must return a title so the handoff can proceed"
    assert db.get_session(cli.session_id), (
        "the session row must exist after _handoff_prepare_session"
    )


def test_request_handoff_succeeds_after_prepare(db):
    """The reported failure: fresh session → prepare → request_handoff returned False."""
    cli = _cli(db)
    CLICommandsMixin._handoff_prepare_session(cli)
    assert db.request_handoff(cli.session_id, "telegram") is True
    assert db.get_handoff_state(cli.session_id)["state"] == "pending"


def test_row_created_with_cli_source_and_model(db):
    cli = _cli(db)
    CLICommandsMixin._handoff_prepare_session(cli)
    row = db.get_session(cli.session_id)
    assert row["source"] == "cli"
    assert row["model"] == "test-model"


def test_title_conflict_does_not_block_handoff(db, monkeypatch):
    """The handoff-<id> title is display-only; a conflict must not abort the handoff."""
    cli = _cli(db)

    def _conflict(*_args, **_kwargs):
        raise ValueError("Title is already in use")

    monkeypatch.setattr(db, "set_session_title", _conflict)
    title = CLICommandsMixin._handoff_prepare_session(cli)
    assert title, "falls back to session_id[:8]; the handoff still proceeds"
    assert db.get_session(cli.session_id)


def test_existing_row_is_left_untouched(db):
    """A session that already has a row keeps it (no create_session churn, no title rewrite)."""
    cli = _cli(db)
    db.create_session(session_id=cli.session_id, source="gateway")
    db.set_session_title(cli.session_id, "kept-title")
    CLICommandsMixin._handoff_prepare_session(cli)
    row = db.get_session(cli.session_id)
    assert row["title"] == "kept-title"
    assert row["source"] == "gateway"
