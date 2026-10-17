"""One reply frame per turn in the classic CLI: a turn that goes text → tools → text prints ONE
``╭─ ☤ Hermes`` top rule (at its first text) and ONE ``╰─`` bottom rule (at turn end), with the
tool rows inside; a modal prompt closes the frame first. Plus the live running-tool spinner line.
"""
import json
import re
from unittest.mock import patch

from tests.hermes_cli.test_tool_progress_scrollback import _make_cli
import tests.hermes_cli.test_tool_progress_scrollback as scrollback

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _cli():
    cli = _make_cli(tool_progress="all")
    cli.show_reasoning = False
    cli.final_response_markdown = "raw"
    cli.show_timestamps = False
    cli._reset_stream_state()
    cli._reply_frame_open = False
    cli._running_tools = []
    cli._invalidate = lambda *a, **k: None
    return cli


def _run(cli, steps):
    printed = []
    with patch.object(scrollback._cli_mod, "_cprint", lambda line: printed.append(_ANSI.sub("", str(line)))):
        for step in steps:
            step(cli)
    return [line for chunk in printed for line in chunk.split("\n")]


def _tool(name, args, result):
    def step(cli):
        cli._on_tool_progress("tool.started", name, None, args)
        cli._on_tool_progress("tool.completed", name, None, None, duration=0.1, is_error=False, result=result)
    return step


def _text(chunk):
    return lambda cli: cli._stream_delta(chunk)


_BOUNDARY = _text(None)
_OK = json.dumps({"output": "ok", "exit_code": 0, "error": None})


def _rules(lines):
    return [i for i, line in enumerate(lines) if line.startswith("╭")], [i for i, line in enumerate(lines) if line.startswith("╰")]


def test_text_tools_text_turn_has_one_top_and_one_bottom_rule_around_everything():
    cli = _cli()
    lines = _run(cli, [_text("Now I'll create app.py:\n"), _BOUNDARY, _tool("terminal", {"command": "make"}, _OK),
                       _text("Done, it builds.\n"), lambda c: c._flush_stream()])
    tops, bottoms = _rules(lines)
    assert len(tops) == 1 and len(bottoms) == 1, lines
    first = next(i for i, line in enumerate(lines) if "create app.py" in line)
    tool = next(i for i, line in enumerate(lines) if "make" in line and "✓" in line)
    last = next(i for i, line in enumerate(lines) if "Done, it builds." in line)
    assert tops[0] < first < tool < last < bottoms[0]


def test_tool_rows_after_the_last_text_still_close_the_frame_once_at_turn_end():
    cli = _cli()
    lines = _run(cli, [_text("Running it.\n"), _BOUNDARY, _tool("terminal", {"command": "ls"}, _OK),
                       lambda c: c._flush_stream()])
    tops, bottoms = _rules(lines)
    tool = next(i for i, line in enumerate(lines) if "✓" in line)
    assert len(tops) == 1 and bottoms == [len(lines) - 1] and tool < bottoms[0]


def test_a_prompt_closes_the_frame_first_and_later_text_opens_a_new_one():
    cli = _cli()

    def approve(c):
        with patch.object(type(c), "_poll_modal_queue", lambda self, *a, **k: "once"), \
             patch.object(type(c), "_paint_now", lambda self: None), \
             patch.object(type(c), "_ring_bell", lambda self, **k: None), \
             patch.object(type(c), "_persist_prompt_summary", lambda self, *a: None):
            assert c._approval_callback("rm -rf build", "delete") == "once"
    lines = _run(cli, [_text("I need to delete build/.\n"), _BOUNDARY, approve, _text("Deleted.\n"),
                       lambda c: c._flush_stream()])
    tops, bottoms = _rules(lines)
    assert len(tops) == 2 and len(bottoms) == 2, lines
    assert tops[0] < bottoms[0] < tops[1] < bottoms[1]


def test_a_turn_with_no_text_prints_no_frame():
    cli = _cli()
    lines = _run(cli, [_tool("terminal", {"command": "ls"}, _OK), lambda c: c._flush_stream()])
    assert _rules(lines) == ([], [])


def test_spinner_names_the_running_tool_then_counts_concurrent_ones():
    cli = _cli()
    _run(cli, [lambda c: c._on_tool_progress("tool.started", "terminal", None, {"command": "ls -la /etc"})])
    line = cli._render_spinner_text()
    assert "ls -la /etc" in line and re.search(r"\d+\.\ds", line)
    _run(cli, [lambda c: c._on_tool_progress("tool.started", "read_file", None, {"path": "notes.md"})])
    from agent.i18n import t
    assert t("cli.tool_render.tools_running", count=2) in cli._render_spinner_text()
    _run(cli, [lambda c: c._on_tool_progress("tool.completed", "terminal", None, None, duration=0.1, result=_OK)])
    assert "notes.md" in cli._render_spinner_text()
    _run(cli, [lambda c: c._on_tool_progress("tool.completed", "read_file", None, None, duration=0.1, result="{}")])
    assert cli._running_tool_status() is None and not re.search(r"\d+\.\ds", cli._render_spinner_text())


def test_a_tool_row_lands_in_scrollback_only_when_the_tool_completes():
    cli = _cli()
    started = _run(cli, [lambda c: c._on_tool_progress("tool.started", "terminal", None, {"command": "sleep 5"})])
    assert not any("sleep 5" in line for line in started)
    done = _run(cli, [lambda c: c._on_tool_progress("tool.completed", "terminal", None, None, duration=5.0,
                                                    result=_OK)])
    assert any("sleep 5" in line and "✓" in line for line in done)


def test_todo_tool_prints_a_checklist_not_a_progress_count():
    cli = _cli()
    todos = [{"id": "1", "content": "write tests", "status": "completed"},
             {"id": "2", "content": "fix frame", "status": "in_progress"},
             {"id": "3", "content": "ship", "status": "pending"}]
    result = json.dumps({"todos": todos, "summary": {"total": 3, "completed": 1}})
    lines = _run(cli, [_tool("todo_list", {"todos": todos}, result)])
    assert any("☒ write tests" in line for line in lines)
    assert any("◼ fix frame" in line for line in lines)
    assert any("☐ ship" in line for line in lines)


def test_spinner_status_survives_tools_finishing_while_it_renders(monkeypatch):
    """The agent thread drops running tools while the UI thread builds the spinner text."""
    import hermes_cli.cli_stream_mixin as stream
    cli = _cli()
    _run(cli, [lambda c: c._on_tool_progress("tool.started", "terminal", None, {"command": "ls"}),
               lambda c: c._on_tool_progress("tool.started", "read_file", None, {"path": "a.md"})])
    real_t = stream.t

    def finishing_t(key, **kwargs):  # both tools complete mid-render
        cli._note_tool_finished("terminal")
        cli._note_tool_finished("read_file")
        return real_t(key, **kwargs)

    monkeypatch.setattr(stream, "t", finishing_t)
    text, started = cli._running_tool_status()
    assert text == real_t("cli.tool_render.tools_running", count=2) and started > 0
    assert cli._running_tool_status() is None
