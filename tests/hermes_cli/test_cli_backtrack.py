"""Backtrack (/rewind, Esc Esc) in the classic CLI: edit & resend cuts nothing until the edit is
sent; fork branches before the picked prompt and leaves the parent session's rows active.

The CLI's store is its ``GatewaySessionStore`` on a connection to a real session authority in this
process; the rows the assertions read are the owner's, through its storage interface handle."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from persistence import RowVisibility
from persistence.conversation.messages import TranscriptRead, TranscriptView
from tests.fixtures.gateway_owner import GatewayOwner
from tests.fixtures.storage_seed import seed_messages
from tests.hermes_cli.test_cli_init import _make_cli


def _active(owner, session_id):
    return [(row.role, row.content, int(row.visibility is RowVisibility.ACTIVE))
            for row in owner.persistence.messages.read_transcript(session_id, TranscriptRead(view=TranscriptView.ALL))]


@pytest.fixture
def owner(tmp_path, monkeypatch):
    owner = GatewayOwner(tmp_path, monkeypatch)
    yield owner
    owner.close()


@pytest.fixture
def cli(owner):
    cli = _make_cli()
    cli._session_db.close()
    cli._session_db = owner.client()
    cli.session_id = cli._session_db.open_session(cli)
    cli._gateway_session_open = True
    seed_messages(owner.persistence, cli.session_id, *(
        {"role": role, "content": content}
        for role, content in (("user", "ALPHA"), ("assistant", "a1"), ("user", "BETA"), ("assistant", "b1"),
                              ("user", "GAMMA"), ("assistant", "g1"))))
    cli.conversation_history = cli._session_db.get_messages_as_conversation(cli.session_id)
    cli.agent = None
    cli._agent_running = False
    cli._prefill_input_buffer = MagicMock()
    cli._invalidate = MagicMock()
    cli._pending_input = MagicMock()
    yield cli


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


def test_edit_cuts_nothing_until_the_edit_is_sent(cli, owner):
    sid = cli.session_id
    cli._handle_rewind_command("/rewind 2")
    cli._prefill_input_buffer.assert_called_once_with("BETA")
    assert all(active == 1 for *_x, active in _active(owner, sid))

    item = _submit(cli, "DELTA")
    assert str(item) == "DELTA" and cli._backtrack_edit is None
    assert all(active == 1 for *_x, active in _active(owner, sid))  # still nothing cut

    assert cli._backtrack_apply_resend(item) == "DELTA"
    rows = _active(owner, sid)
    assert [(c, a) for _r, c, a in rows] == [
        ("ALPHA", 1), ("a1", 1), ("BETA", 0), ("b1", 0), ("GAMMA", 0), ("g1", 0)]
    assert [m["content"] for m in cli.conversation_history] == ["ALPHA", "a1"]


def test_cancelled_edit_cuts_nothing(cli, owner):
    sid = cli.session_id
    cli._handle_rewind_command("/rewind 2")
    buf = MagicMock()
    buf.text = ""
    cli._attached_images = []
    cli._tui_submit_buffer(SimpleNamespace(app=MagicMock()), buf)  # empty submit cancels
    assert cli._backtrack_edit is None
    cli._pending_input.put.assert_not_called()
    assert all(active == 1 for *_x, active in _active(owner, sid))
    assert len(cli.conversation_history) == 6


def test_resend_refuses_when_the_picked_turn_changed(cli, owner):
    sid = cli.session_id
    cli._handle_rewind_command("/rewind 2")
    item = _submit(cli, "DELTA")
    cli.undo_last(2, prefill=False)  # BETA is gone before the edit reaches the loop
    assert cli._backtrack_apply_resend(item) is None
    assert [m["content"] for m in cli.conversation_history] == ["ALPHA", "a1"]
    assert cli._prefill_input_buffer.call_args[0][0] == "DELTA"  # the edit is handed back
    assert sum(active for *_x, active in _active(owner, sid)) == 2


def test_fork_branches_before_the_prompt_and_keeps_the_parent(cli, owner):
    parent = cli.session_id
    cli._handle_rewind_command("/rewind fork 2")
    child = cli.session_id
    assert child != parent
    db = cli._session_db
    assert db.get_session(child)["parent_session_id"] == parent
    assert [m["content"] for m in db.get_messages_as_conversation(child)] == ["ALPHA", "a1"]
    assert all(active == 1 for *_x, active in _active(owner, parent))
    assert len(_active(owner, parent)) == 6
    assert cli._prefill_input_buffer.call_args[0][0] == "BETA"
