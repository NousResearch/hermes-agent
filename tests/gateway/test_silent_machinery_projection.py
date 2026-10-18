"""Display projection: wake plumbing and deliberate silence never reach a transcript as content."""
from gateway.response_filters import project_silent_machinery_turns

WAKE = "[kanban] Task t_3fc68611 completed.\nTitle: Fix it\nAssignee: @w\nBoard: default\n\nCheck the result."


def _kinds(rows):
    return [r.get("display_kind") for r in rows]


def test_silent_wake_turn_is_hidden_whole_including_tool_rounds():
    rows = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": WAKE, "display_kind": "internal_notification"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "{}", "tool_call_id": "c1"},
        {"role": "assistant", "content": "NO_REPLY"},
        {"role": "user", "content": "next question"},
    ]
    out = project_silent_machinery_turns(rows)
    assert _kinds(out) == [None, None, "hidden", "hidden", "hidden", "hidden", None]
    assert rows[2]["display_kind"] == "internal_notification"  # input untouched


def test_wake_turn_with_a_real_reply_becomes_a_collapsible_board_update_card():
    rows = [
        {"role": "user", "content": WAKE},  # legacy untyped wake row
        {"role": "assistant", "content": "Card t_3fc68611 shipped: the chat is quiet now."},
    ]
    out = project_silent_machinery_turns(rows)
    # The async-result card every shipped client already renders collapsed (older desktop builds too).
    assert _kinds(out) == ["async_delegation_complete", None]
    assert out[0]["display_metadata"]["display_text"] == "Board update · t_3fc68611 completed · Fix it"
    assert out[0]["content"] == WAKE  # full text stays one click away


def test_wake_turn_with_no_assistant_row_at_all_is_hidden():
    out = project_silent_machinery_turns([{"role": "user", "content": WAKE, "display_kind": "internal_notification"}])
    assert _kinds(out) == ["hidden"]


def test_human_turn_is_never_hidden_even_when_the_reply_is_silent():
    for typed in ("[kanban] is a word I typed about t_zz",
                  "[kanban] Task t_3fc68611 is stuck, can you check?",  # pasted one-line ping
                  "[kanban] Task t_3fc68611 completed.\nTitle: Fix it\nwhy did this happen?"):
        out = project_silent_machinery_turns([{"role": "user", "content": typed},
                                              {"role": "assistant", "content": "[SILENT]"}])
        # The user's words stay; only the bare marker row is hidden.
        assert _kinds(out) == [None, "hidden"], typed


def test_pasted_ping_with_a_reply_stays_the_users_message():
    rows = [{"role": "user", "content": "[kanban] Task t_3fc68611 is stuck, can you check?"},
            {"role": "assistant", "content": "Checking the board now."}]
    assert _kinds(project_silent_machinery_turns(rows)) == [None, None]


def test_prose_mentioning_no_reply_is_content():
    rows = [{"role": "user", "content": WAKE, "display_kind": "internal_notification"},
            {"role": "assistant", "content": "I chose NO_REPLY earlier, but this needs you: approve the merge."}]
    assert _kinds(project_silent_machinery_turns(rows)) == ["async_delegation_complete", None]


def test_other_typed_user_rows_are_not_machinery():
    rows = [{"role": "user", "content": WAKE, "display_kind": "steer"}, {"role": "assistant", "content": "NO_REPLY"}]
    assert _kinds(project_silent_machinery_turns(rows)) == ["steer", "hidden"]
