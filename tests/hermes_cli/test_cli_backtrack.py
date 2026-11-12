"""Backtrack (/rewind, Esc Esc) in the classic CLI: edit & resend cuts nothing until the edit is
sent; fork branches before the picked prompt and leaves the parent session's rows active."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_agent.hermes_state import SessionDB
from tests.hermes_cli.test_cli_init import _make_cli


def _active(db, session_id):
    return [(role, content, active) for role, content, active in db._conn.execute(
        "SELECT role, content, active FROM messages WHERE session_id = ? ORDER BY id", (session_id,))]


@pytest.fixture
def cli(tmp_path):
    cli = _make_cli()
    cli._session_db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db = db
    cli.session_id = "20261104_010000_aaaaaa"
    db.create_session(cli.session_id, source="cli")
    for role, content in (("user", "ALPHA"), ("assistant", "a1"), ("user", "BETA"), ("assistant", "b1"),
                          ("user", "GAMMA"), ("assistant", "g1")):
        db.append_message(cli.session_id, role, content)
    cli.conversation_history = db.get_messages_as_conversation(cli.session_id)
    cli.agent = None
    cli._agent_running = False
    cli._prefill_input_buffer = MagicMock()
    cli._invalidate = MagicMock()
    cli._pending_input = MagicMock()
    yield cli
    db.close()


def _submit(cli, text):
    buf = MagicMock()
    buf.text = text
    cli._attached_images = []
    cli._inline_pastes = MagicMock()
    cli._tui_submit_buffer(SimpleNamespace(app=MagicMock()), buf)
    return cli._pending_input.put.call_args[0][0]


def test_picker_lists_prompts_newest_first(cli):
    assert cli._open_backtrack_picker()
    assert [text for _o, text in cli._backtrack_state["entries"]] == ["GAMMA", "BETA", "ALPHA"]
    assert cli._backtrack_state["selected"] == 0


def test_edit_cuts_nothing_until_the_edit_is_sent(cli):
    sid = cli.session_id
    cli._handle_rewind_command("/rewind 2")
    cli._prefill_input_buffer.assert_called_once_with("BETA")
    assert all(active == 1 for *_x, active in _active(cli._session_db, sid))

    item = _submit(cli, "DELTA")
    assert str(item) == "DELTA" and cli._backtrack_edit is None
    assert all(active == 1 for *_x, active in _active(cli._session_db, sid))  # still nothing cut

    assert cli._backtrack_apply_resend(item) == "DELTA"
    rows = _active(cli._session_db, sid)
    assert [(c, a) for _r, c, a in rows] == [
        ("ALPHA", 1), ("a1", 1), ("BETA", 0), ("b1", 0), ("GAMMA", 0), ("g1", 0)]
    assert [m["content"] for m in cli.conversation_history] == ["ALPHA", "a1"]


def test_cancelled_edit_cuts_nothing(cli):
    sid = cli.session_id
    cli._handle_rewind_command("/rewind 2")
    buf = MagicMock()
    buf.text = ""
    cli._attached_images = []
    cli._tui_submit_buffer(SimpleNamespace(app=MagicMock()), buf)  # empty submit cancels
    assert cli._backtrack_edit is None
    cli._pending_input.put.assert_not_called()
    assert all(active == 1 for *_x, active in _active(cli._session_db, sid))
    assert len(cli.conversation_history) == 6


def test_resend_refuses_when_the_picked_turn_changed(cli):
    sid = cli.session_id
    cli._handle_rewind_command("/rewind 2")
    item = _submit(cli, "DELTA")
    cli.undo_last(2, prefill=False)  # BETA is gone before the edit reaches the loop
    assert cli._backtrack_apply_resend(item) is None
    assert [m["content"] for m in cli.conversation_history] == ["ALPHA", "a1"]
    assert cli._prefill_input_buffer.call_args[0][0] == "DELTA"  # the edit is handed back
    assert sum(active for *_x, active in _active(cli._session_db, sid)) == 2


def test_fork_branches_before_the_prompt_and_keeps_the_parent(cli):
    parent = cli.session_id
    cli._handle_rewind_command("/rewind fork 2")
    child = cli.session_id
    assert child != parent
    db = cli._session_db
    assert db.get_session(child)["parent_session_id"] == parent
    assert [m["content"] for m in db.get_messages_as_conversation(child)] == ["ALPHA", "a1"]
    assert all(active == 1 for *_x, active in _active(db, parent))
    assert len(_active(db, parent)) == 6
    assert cli._prefill_input_buffer.call_args[0][0] == "BETA"
