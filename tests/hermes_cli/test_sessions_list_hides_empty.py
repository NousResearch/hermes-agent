"""``hermes sessions list`` must not list content-less session rows (#143659).

A TUI start writes a session row before any prompt exists, so every launch that never
receives a message leaves a 0-message row behind. Clients that launch ``hermes`` as a
per-tab terminal command (the Obsidian Hermes Console does) turn that into one ghost row
per abandoned tab, and the listing rendered them as unnamed entries.

The rows are not deleted — resume, routing and ``sessions prune/archive --max-messages 0``
still see them — they are simply not conversations to list.
"""

from argparse import Namespace

import pytest

from hermes_cli import sessions_cmd


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess_real", "cli")
    db.append_message("sess_real", "user", "a real conversation")
    # A launch that never got a prompt: no messages, no title.
    db.create_session("sess_ghost", "tui")
    yield db
    db.close()


def _list(db, capsys, limit=20):
    sessions_cmd._cmd_list(db, Namespace(limit=limit, source=None, workspace=None))
    return capsys.readouterr().out


def test_content_less_session_is_not_listed(db, capsys):
    out = _list(db, capsys)
    assert "sess_real" in out
    assert "sess_ghost" not in out


def test_content_less_session_stays_reachable_for_maintenance(db):
    """Hidden from the listing, not removed: prune/archive still match it."""
    rows = db.list_sessions_rich(source=None, limit=50, min_message_count=0)
    assert any(row["id"] == "sess_ghost" for row in rows)
