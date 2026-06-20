from types import SimpleNamespace

import pytest

import hermes_undo
from agent.agent_runtime_helpers import repair_message_sequence
from cli import HermesCLI
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path, monkeypatch):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(hermes_undo, "_session_db", session_db)
    hermes_undo.clear_state()
    yield session_db
    session_db.close()
    hermes_undo.clear_state()


def _cli(db, sid):
    buf = SimpleNamespace(text="", cursor_position=0)
    app = SimpleNamespace(current_buffer=buf, invalidate=lambda: None)
    cli_obj = SimpleNamespace(
        session_id=sid,
        _session_db=db,
        conversation_history=[],
        agent=None,
        _app=app,
        buffer=buf,
    )
    cli_obj._reload_active_history_after_rewind = (
        lambda rewound=False: HermesCLI._reload_active_history_after_rewind(
            cli_obj, rewound=rewound
        )
    )
    cli_obj._prefill_input_buffer = lambda text: HermesCLI._prefill_input_buffer(cli_obj, text)
    return cli_obj


def _seed(db, sid):
    u1 = db.append_message(sid, "user", "first exact surviving text")
    a1 = db.append_message(sid, "assistant", "answer one")
    u2 = db.append_message(sid, "user", "second exact surviving text")
    a2 = db.append_message(sid, "assistant", "answer two")
    return u1, a1, u2, a2


def test_d4_prefill_uses_exact_surviving_new_tail_for_n_1_2_3(db):
    for n, expected in (
        (1, "second exact surviving text"),
        (2, ""),
        (3, "first exact surviving text"),
    ):
        sid = f"cli-prefill-{n}"
        db.create_session(sid, source="cli")
        _seed(db, sid)
        cli_obj = _cli(db, sid)

        HermesCLI.undo_last(cli_obj, n)

        assert cli_obj.buffer.text == expected
        if expected:
            assert cli_obj.buffer.cursor_position == len(expected)
        active = db.get_messages(sid)
        tail = active[-1] if active else None
        if expected:
            assert tail["role"] == "user"
            assert tail["content"] == expected
        else:
            assert tail["role"] == "assistant"


def test_str_prefill_edit_resend_merges_two_user_rows_before_provider(db):
    sid = "cli-edit-resend"
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "original draft")
    db.append_message(sid, "assistant", "answer")
    cli_obj = _cli(db, sid)

    HermesCLI.undo_last(cli_obj, 1)
    assert cli_obj.buffer.text == "original draft"

    db.append_message(sid, "user", "edited draft")
    hermes_undo.on_user_message_appended(sid)
    messages = db.get_messages_as_conversation(sid)
    assert [m["role"] for m in messages] == ["user", "user"]

    repairs = repair_message_sequence(object(), messages)

    assert repairs == 1
    assert messages == [{"role": "user", "content": "original draft\n\nedited draft"}]


def test_clear_redo_on_send_leaves_undo_stack_available(db):
    sid = "cli-clear-redo"
    db.create_session(sid, source="cli")
    _seed(db, sid)
    hermes_undo.undo(sid, 1)
    hermes_undo.redo(sid, 1)
    state = hermes_undo.get_state(sid)
    assert state.redo_stack

    db.append_message(sid, "user", "new branch")
    hermes_undo.on_user_message_appended(sid)

    assert state.redo_stack == []
    assert state.undo_stack == []


@pytest.mark.parametrize(
    "command, expected_n",
    [
        ("/redo", 1),
        ("/redo 2", 2),
        ("/redo 0", 1),    # non-positive clamps to 1 (parity with /undo)
        ("/redo -3", 1),   # negative clamps to 1
    ],
)
def test_redo_command_clamps_non_positive_count_to_one(command, expected_n):
    """/redo 0 and /redo -N must clamp to 1 like /undo, not fall through to a
    misleading 'nothing to redo' (Greptile PR #49 finding)."""
    captured = {}
    cli_obj = SimpleNamespace(
        _pending_resume_sessions=None,
        redo_last=lambda n=1: captured.__setitem__("n", n),
    )
    cont = HermesCLI.process_command(cli_obj, command)
    assert cont is True
    assert captured.get("n") == expected_n


def test_redo_command_rejects_non_numeric_count():
    """A non-numeric /redo argument is rejected, not silently treated as 1."""
    captured = {}
    cli_obj = SimpleNamespace(
        _pending_resume_sessions=None,
        redo_last=lambda n=1: captured.__setitem__("n", n),
    )
    HermesCLI.process_command(cli_obj, "/redo abc")
    assert "n" not in captured  # redo_last never called on a parse error
