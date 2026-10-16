"""Classic-CLI tool activity in scrollback, driven through the real ``HermesCLI`` callbacks:
result previews follow ``display.tool_output_preview``, consecutive read-only calls fold into
one "explored" row without editing printed lines, collapsed reasoning prints one line, the
"preparing" line moves to the spinner while the TUI is up, and every call reaches the Ctrl+O log.
"""
import json
import re
from unittest.mock import patch

from agent.i18n import t
from tests.hermes_cli.test_tool_progress_scrollback import _make_cli
import tests.hermes_cli.test_tool_progress_scrollback as scrollback

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _run(cli, steps):
    printed = []
    with patch.object(scrollback._cli_mod, "_cprint", lambda line: printed.append(_ANSI.sub("", str(line)))):
        for step in steps:
            step(cli)
    return printed


def _tool(name, args, result, duration=0.2, is_error=False):
    def step(cli):
        cli._on_tool_progress("tool.started", name, None, args)
        cli._on_tool_progress("tool.completed", name, None, None, duration=duration, is_error=is_error, result=result)
    return step


def _out(text, code=0):
    return json.dumps({"output": text, "exit_code": code, "error": None})


def test_preview_lines_follow_the_configured_budget():
    cli = _make_cli(tool_progress="all")
    cli._tool_output_preview = 2
    printed = _run(cli, [_tool("terminal", {"command": "seq 9"}, _out("\n".join(map(str, range(9)))))])
    assert "✓" in printed[0]
    assert [line.split("⎿")[-1].strip() for line in printed[1:3]] == ["0", "1"]
    assert t("cli.tool_render.more_lines", count=7, shortcut="ctrl+o") in printed[3]
    cli._tool_output_preview = 0
    assert len(_run(cli, [_tool("terminal", {"command": "seq 9"}, _out("1\n2\n3"))])) == 1


def test_verbose_mode_previews_more_lines_than_all():
    output = _out("\n".join(map(str, range(40))))
    budgets = {}
    for mode in ("all", "verbose"):
        cli = _make_cli(tool_progress=mode)
        budgets[mode] = len(_run(cli, [_tool("terminal", {"command": "seq"}, output)]))
    assert budgets["verbose"] > budgets["all"]


def test_consecutive_reads_print_one_explored_row_when_a_non_read_tool_starts():
    cli = _make_cli(tool_progress="all")
    reads = [_tool("read_file", {"path": f"src/{n}.py"}, json.dumps({"total_lines": 3})) for n in ("a", "b")]
    held = _run(cli, [*reads, _tool("search_files", {"pattern": "foo"}, json.dumps({"total_count": 1}))])
    assert held == []  # nothing printed yet: still exploring
    printed = _run(cli, [_tool("terminal", {"command": "make"}, _out("ok"))])
    assert t("cli.tool_render.explored") in printed[0] and "a.py, b.py" in printed[0] and "foo" in printed[0]
    assert "make" in printed[1]


def test_a_single_read_flushes_as_an_ordinary_row_at_the_answer():
    cli = _make_cli(tool_progress="all")
    _run(cli, [_tool("read_file", {"path": "notes.md"}, json.dumps({"total_lines": 3}))])
    printed = _run(cli, [lambda c: c._flush_stream()])
    assert any("notes.md" in line and "✓" in line for line in printed)
    assert not any(t("cli.tool_render.explored") in line for line in printed)


def test_a_failed_read_is_never_hidden_in_a_group():
    cli = _make_cli(tool_progress="all")
    failed = json.dumps({"success": False, "error": "File not found: /x/missing.py"})
    printed = _run(cli, [_tool("read_file", {"path": "missing.py"}, failed)])
    assert printed and "✗" in printed[0]


def test_verbose_mode_never_groups():
    cli = _make_cli(tool_progress="verbose")
    printed = _run(cli, [_tool("read_file", {"path": "a.py"}, json.dumps({"total_lines": 3}))])
    assert printed and "a.py" in printed[0]


def test_off_mode_prints_nothing_but_still_logs_for_the_viewer():
    cli = _make_cli(tool_progress="off")
    assert _run(cli, [_tool("terminal", {"command": "ls"}, _out("x"))]) == []
    entry = cli._transcript().entries[-1]
    assert entry.name == "terminal" and entry.result == "x" and entry.exit_code == 0


def test_viewer_log_keeps_exit_code_duration_and_failure():
    cli = _make_cli(tool_progress="all")
    cli._transcript_begin_turn()
    _run(cli, [_tool("terminal", {"command": "false"}, _out("nope", 2), duration=1.25)])
    entry = cli._transcript().select(all_turns=False)[-1]
    assert (entry.exit_code, entry.duration, entry.failed) == (2, 1.25, True)


def test_collapsed_reasoning_prints_one_line_and_logs_the_full_text():
    cli = _make_cli(tool_progress="all")
    cli.cli_reasoning_style = "collapsed"
    cli._reset_stream_state()
    text = "**Weighing options**\n" + "detail\n" * 30
    printed = _run(cli, [lambda c: c._stream_reasoning_delta(text), lambda c: c._close_reasoning_box()])
    assert len(printed) == 1 and "✻" in printed[0] and "Weighing options" in printed[0]
    assert cli._transcript().entries[-1].kind == "reasoning"
    assert cli._transcript().entries[-1].result == text.strip()


def test_full_reasoning_style_keeps_the_streamed_box():
    cli = _make_cli(tool_progress="all")
    cli._reset_stream_state()
    printed = _run(cli, [lambda c: c._stream_reasoning_delta("a thought\n"), lambda c: c._close_reasoning_box()])
    assert any("a thought" in line for line in printed) and len(printed) >= 3
    assert cli._transcript().entries[-1].kind == "reasoning"


def test_preparing_line_goes_to_the_spinner_while_the_tui_runs():
    cli = _make_cli(tool_progress="all")
    cli._app = object()
    cli._invalidate = lambda *a, **k: None
    assert _run(cli, [lambda c: c._on_tool_gen_start("write_file")]) == []
    assert "write_file" in cli._spinner_text


def test_gateway_inline_diff_is_rendered_with_header_and_attached_to_the_call():
    from agent.display import render_edit_diff_with_delta
    cli = _make_cli(tool_progress="all")
    lines = []
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,2 @@\n-a\n+b\n+c\n"
    render_edit_diff_with_delta("patch", json.dumps({"success": True, "diff": diff}), print_fn=lines.append)
    printed = _run(cli, [_tool("patch", {"path": "x.py"}, json.dumps({"success": True})),
                         lambda c: c._on_inline_diff("patch", "\n".join(lines))])
    header = next(line for line in printed if "x.py" in line and "+2" in line)
    assert "−1" in header
    assert "+b" in cli._transcript().entries[-1].diff


def test_gateway_inline_diff_frame_reaches_the_cli_renderer_not_raw_print():
    """A ``tool.complete`` frame's ``inline_diff`` goes to ``inline_diff_callback`` (the CLI's
    gutter renderer) instead of being printed verbatim."""
    from tests.hermes_cli.test_gateway_agent import FakeClient, SID, connect_with, frame
    from hermes_cli.gateway_agent import GatewayAgent
    from hermes_cli.gateway_connection import GatewayConnection
    seen = []
    client = FakeClient([frame("tool.complete", {"tool_id": "c1", "name": "patch", "args": {"path": "x"},
                                                 "result": "{}", "duration_s": 0.1, "inline_diff": "DIFF"}),
                         frame("message.complete", {"text": "done", "outcome": "completed"})])
    conn = GatewayConnection(connect=connect_with(client))
    conn.start(5)
    conn.state(SID)
    try:
        agent = GatewayAgent(connection=conn, session_id=SID,
                             inline_diff_callback=lambda name, diff: seen.append((name, diff)))
        printed = []
        with patch.object(scrollback._cli_mod, "_cprint", printed.append):
            agent.run_conversation("edit")
    finally:
        conn.close()
    assert seen == [("patch", "DIFF")] and "DIFF" not in printed
