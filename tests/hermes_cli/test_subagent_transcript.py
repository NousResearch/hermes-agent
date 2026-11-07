"""A stored conversation as readable entries for the classic CLI's subagent viewer."""
import json


def _call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _child_run():
    return [
        {"role": "user", "content": "Run the tests and report"},
        {"role": "assistant", "content": "Running the suite now.",
         # Stored rows carry tool_calls as a JSON string.
         "tool_calls": json.dumps([_call("c1", "terminal", {"command": "pytest -q"})])},
        {"role": "tool", "tool_call_id": "c1", "tool_name": "terminal",
         "content": json.dumps({"output": "12 passed", "exit_code": 0})},
        {"role": "assistant", "content": "All 12 tests pass."},
    ]


def test_a_subagent_run_reads_as_brief_steps_and_result():
    from hermes_cli.subagent_transcript import build_blocks

    blocks = build_blocks(_child_run(), subagent=True)

    assert [b.kind for b in blocks] == ["brief", "assistant", "tool", "final"]
    assert blocks[0].body == "Run the tests and report"
    assert blocks[-1].body == "All 12 tests pass."


def test_a_tool_call_is_one_line_until_expanded():
    from hermes_cli.subagent_transcript import build_blocks, render

    blocks = build_blocks(_child_run(), subagent=True)
    tool = next(i for i, b in enumerate(blocks) if b.kind == "tool")
    assert "pytest -q" in blocks[tool].title
    assert blocks[tool].collapsible

    collapsed = _text(render(blocks, width=80, expanded=set()))
    opened = _text(render(blocks, width=80, expanded={tool}))

    assert "12 passed" not in collapsed
    assert "12 passed" in opened


def test_master_history_shows_a_delegation_with_each_task_and_outcome():
    from hermes_cli.subagent_transcript import build_blocks

    result = {"results": [
        {"task_index": 0, "status": "completed", "summary": "Docs updated"},
        {"task_index": 1, "status": "failed", "error": "HTTP 500"},
    ]}
    history = [
        {"role": "user", "content": "Split this up"},
        {"role": "assistant", "tool_calls": [_call("d1", "delegate_task", {"tasks": [
            {"goal": "Update the docs"}, {"goal": "Fix the build"}]})]},
        # The parent's stored delegate_task result is a JSON string of JSON.
        {"role": "tool", "tool_call_id": "d1", "tool_name": "delegate_task", "content": json.dumps(json.dumps(result))},
    ]

    tasks = [b for b in build_blocks(history, subagent=False) if b.kind == "delegation"]

    assert [b.goal for b in tasks] == ["Update the docs", "Fix the build"]
    assert [b.outcome for b in tasks] == ["completed", "failed"]
    assert "Docs updated" in tasks[0].body and "HTTP 500" in tasks[1].body
    assert "Update the docs" in tasks[0].title


def test_a_single_goal_delegation_is_one_task():
    from hermes_cli.subagent_transcript import build_blocks

    history = [{"role": "assistant", "tool_calls": [_call("d1", "delegate_task", {"goal": "Only task"})]}]

    tasks = [b for b in build_blocks(history, subagent=False) if b.kind == "delegation"]

    assert [b.goal for b in tasks] == ["Only task"]
    assert tasks[0].outcome == "running"


def test_a_background_dispatch_is_running_until_its_result_arrives():
    from hermes_cli.subagent_transcript import build_blocks

    handle = {"status": "dispatched", "delegation_id": "deleg_1", "goals": ["Long task"]}
    history = [
        {"role": "assistant", "tool_calls": [_call("d1", "delegate_task", {"goal": "Long task"})]},
        {"role": "tool", "tool_call_id": "d1", "tool_name": "delegate_task", "content": json.dumps(handle)},
    ]

    task = next(b for b in build_blocks(history, subagent=False) if b.kind == "delegation")

    assert task.outcome == "running"


def test_the_master_view_has_no_brief_and_its_last_reply_is_not_a_result():
    from hermes_cli.subagent_transcript import build_blocks

    kinds = [b.kind for b in build_blocks(_child_run(), subagent=False)]

    assert kinds == ["user", "assistant", "tool", "assistant"]


def test_timeline_events_are_shown_with_their_text():
    from hermes_cli.subagent_transcript import build_blocks

    history = [{"role": "user", "content": "", "display_kind": "async_delegation_complete",
                "display_metadata": {"display_text": "Subagent finished: Update the docs"}}]

    blocks = build_blocks(history, subagent=False)

    assert [(b.kind, b.title) for b in blocks] == [("event", "Subagent finished: Update the docs")]


def test_a_steer_shows_the_text_that_was_sent():
    from agent.prompt_builder import steer_user_row
    from hermes_cli.subagent_transcript import build_blocks, render

    blocks = build_blocks([{"role": "user", "content": "Do the task"}, steer_user_row("Also print the date")],
                          subagent=True)

    assert blocks[-1].kind == "event"
    text = _text(render(blocks, width=80, expanded=set(), subagent=True))
    assert "Also print the date" in text
    assert "OUT-OF-BAND" not in text


def test_a_failed_turn_shows_its_notice():
    from hermes_cli.subagent_transcript import build_blocks, render

    history = [{"role": "assistant", "content": "This turn did not complete.", "display_kind": "failed_turn"}]

    assert "This turn did not complete." in _text(render(build_blocks(history, subagent=True), width=80,
                                                         expanded=set(), subagent=True))


def test_hidden_and_system_rows_are_skipped():
    from hermes_cli.subagent_transcript import build_blocks

    history = [{"role": "system", "content": "prompt"}, {"role": "user", "content": "x", "display_kind": "hidden"}]

    assert build_blocks(history, subagent=False) == []


def test_stored_text_cannot_drive_the_terminal():
    from hermes_cli.subagent_transcript import build_blocks, render

    history = [{"role": "user", "content": "\x1b[2Jclear\x1b]0;title\x07 me"}]

    text = _text(render(build_blocks(history, subagent=True), width=80, expanded=set()))

    assert "\x1b" not in text and "\x07" not in text
    assert "clear" in text


def test_long_text_wraps_to_the_width():
    from hermes_cli.subagent_transcript import build_blocks, render

    history = [{"role": "user", "content": "word " * 60}]

    lines = _text(render(build_blocks(history, subagent=True), width=40, expanded=set())).splitlines()

    assert lines and all(len(line) <= 40 for line in lines)


def test_render_reports_where_each_entry_starts_and_marks_the_cursor():
    from hermes_cli.subagent_transcript import build_blocks, render

    blocks = build_blocks(_child_run(), subagent=True)

    rendered = render(blocks, width=80, expanded=set(), cursor=2)

    assert len(rendered.starts) == len(blocks)
    assert rendered.starts == sorted(rendered.starts)
    header = rendered.lines[rendered.starts[2]]
    assert any("cursor" in style for style, _ in header)


def _text(rendered):
    return "\n".join("".join(text for _, text in line) for line in rendered.lines)
