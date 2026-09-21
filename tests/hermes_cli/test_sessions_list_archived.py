"""``hermes sessions list --archived`` surfaces archived sessions (#118084).

Archived sessions are hidden from the default listing; before this fix there was no
way to list them from the CLI short of reading state.db directly.
"""

from argparse import Namespace

import pytest

from hermes_cli import sessions_cmd


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess_active", "cli")
    db.append_message("sess_active", "user", "still going")
    db.create_session("sess_archived", "cli")
    db.append_message("sess_archived", "user", "old conversation")
    db.set_session_archived("sess_archived", True)
    yield db
    db.close()


def _list(db, capsys, *, archived=False):
    sessions_cmd._cmd_list(db, Namespace(limit=20, source=None, workspace=None, archived=archived))
    return capsys.readouterr().out


def test_default_list_excludes_archived(db, capsys):
    out = _list(db, capsys)
    assert "sess_active" in out
    assert "sess_archived" not in out


def test_archived_flag_shows_only_archived(db, capsys):
    out = _list(db, capsys, archived=True)
    assert "sess_archived" in out
    assert "sess_active" not in out
