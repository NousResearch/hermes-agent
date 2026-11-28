"""/rewind cut N and the browser's ``d`` key: cut the session back to just before a prompt
without sending anything. Rows are archived (active=0), never deleted."""

from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB
from tests.hermes_cli.test_cli_init import _make_cli


def _rows(db, session_id):
    return [(content, active) for content, active in db._conn.execute(
        "SELECT content, active FROM messages WHERE session_id = ? ORDER BY id", (session_id,))]


@pytest.fixture
def cli(tmp_path):
    cli = _make_cli()
    cli._session_db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db = db
    cli.session_id = "20261108_010000_cccccc"
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


def test_cut_archives_from_the_prompt_on_and_sends_nothing(cli):
    sid = cli.session_id
    cli._handle_rewind_command("/rewind cut 2")
    assert _rows(cli._session_db, sid) == [
        ("ALPHA", 1), ("a1", 1), ("BETA", 0), ("b1", 0), ("GAMMA", 0), ("g1", 0)]
    assert [m["content"] for m in cli.conversation_history] == ["ALPHA", "a1"]
    cli._prefill_input_buffer.assert_called_once_with("BETA")
    cli._pending_input.put.assert_not_called()  # nothing queued to the agent


def test_cut_first_prompt_empties_live_history_but_keeps_rows(cli):
    sid = cli.session_id
    cli._handle_rewind_command("/rewind cut 1")
    assert cli.conversation_history == []
    assert len(_rows(cli._session_db, sid)) == 6
    assert all(active == 0 for _c, active in _rows(cli._session_db, sid))


@pytest.mark.parametrize("arg", ["/rewind cut", "/rewind cut x", "/rewind cut 1 2"])
def test_cut_usage_errors_change_nothing(cli, arg):
    sid = cli.session_id
    cli._handle_rewind_command(arg)
    assert all(active == 1 for _c, active in _rows(cli._session_db, sid))


def test_cut_out_of_range_changes_nothing(cli):
    sid = cli.session_id
    cli._handle_rewind_command("/rewind cut 9")
    assert all(active == 1 for _c, active in _rows(cli._session_db, sid))


def test_cut_refused_while_agent_running(cli):
    sid = cli.session_id
    cli._agent_running = True
    cli._handle_rewind_command("/rewind cut 2")
    assert all(active == 1 for _c, active in _rows(cli._session_db, sid))


def test_browser_d_key_queues_cut_for_selected_prompt(cli):
    assert cli._open_backtrack_picker()
    cli._backtrack_step(1)  # newest is GAMMA (3); one older is BETA (2)
    cli._backtrack_choose_cut()
    assert cli._backtrack_state is None
    cli._pending_input.put.assert_called_once_with("/rewind cut 2")
    assert all(active == 1 for _c, active in _rows(cli._session_db, cli.session_id))  # not yet cut
