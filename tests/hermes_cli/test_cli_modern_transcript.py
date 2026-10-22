"""The glyph layout (``display.cli_layout: modern``) carries the readable-transcript rows: aligned
status/verb columns, ``⎿`` result previews, grouped reads, todo checklists and numbered diffs, all
width-independent so a resize never re-wraps scrollback."""
from __future__ import annotations

import json
import re
from unittest.mock import patch

from hermes_cli import cli_diff_render, cli_tool_render
from hermes_cli.cli_modern_layout import VERB_COLUMN, row_head, tool_row

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

_UNITTEST_FAILURE = """.F..
======================================================================
FAIL: test_ten_percent (test_pricing.Pricing.test_ten_percent)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/work/test_pricing.py", line 11, in test_ten_percent
    self.assertEqual(apply_discount(19.99, 10), Decimal("17.99"))
AssertionError: Decimal('17.991') != Decimal('17.99')

----------------------------------------------------------------------
Ran 4 tests in 0.000s

FAILED (failures=1)"""


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _cli():
    from tests.hermes_cli.test_cli_layout_gate import _cli as layout_cli
    return layout_cli(True)


def _run(cli, *steps):
    import tests.hermes_cli.test_tool_progress_scrollback as scrollback
    printed: list[str] = []
    with patch.object(scrollback._cli_mod, "_cprint", lambda line: printed.append(str(line))):
        for step in steps:
            step(cli)
    return [_plain(line) for chunk in printed for line in chunk.split("\n")]


def _done(name, args, result, duration=0.2):
    def step(cli):
        cli._on_tool_progress("tool.started", name, None, args)
        cli._on_tool_progress("tool.completed", name, None, None, duration=duration, is_error=False, result=result)
    return step


def test_rows_align_their_detail_in_one_column_whatever_the_verb_or_emoji():
    rows = [_plain(row_head(e, v)) for e, v in (("💻", "$"), ("🔎", "explored"), ("✍️", "write"), ("", "x"))]
    from prompt_toolkit.utils import get_cwidth
    starts = {get_cwidth(row) for row in rows}
    assert starts == {1 + 1 + 2 + 1 + VERB_COLUMN}, rows  # mark, space, 2-cell emoji, space, verb column
    assert [row.split()[0] for row in rows] == ["✓"] * 4
    assert _plain(row_head("💻", "$", "failed")).startswith("✗ 💻 $")


def test_failed_command_preview_leads_with_what_went_wrong():
    shown, hidden = cli_tool_render.preview_lines(_UNITTEST_FAILURE.splitlines(), True, 3)
    assert shown == ["FAIL: test_ten_percent (test_pricing.Pricing.test_ten_percent)",
                     "AssertionError: Decimal('17.991') != Decimal('17.99')",
                     "FAILED (failures=1)"]
    assert hidden > 0


def test_previews_skip_rule_lines_and_blank_lines():
    shown, hidden = cli_tool_render.preview_lines(["-" * 70, "Ran 4 tests in 0.000s", "", "OK"], False, 3)
    assert (shown, hidden) == (["Ran 4 tests in 0.000s", "OK"], 0)


def test_failure_without_an_error_line_shows_the_end_of_the_output():
    lines = [f"step {i}" for i in range(10)]
    assert cli_tool_render.preview_lines(lines, True, 3) == (["step 7", "step 8", "step 9"], 7)


def test_command_row_hangs_its_preview_under_the_verb_column():
    lines = _run(_cli(), _done("terminal", {"command": "python3 -m unittest"},
                               json.dumps({"output": _UNITTEST_FAILURE, "exit_code": 1})))
    row = next(i for i, line in enumerate(lines) if "unittest" in line)
    assert lines[row].startswith("✗ 💻 $")
    assert lines[row + 1].startswith("  ⎿  FAIL: test_ten_percent")
    assert lines[row + 2].startswith("     AssertionError")


def test_reads_and_searches_fold_into_one_explored_row():
    read = json.dumps({"content": "1|x", "total_lines": 1})
    search = json.dumps({"total_count": 1, "matches": []})
    lines = _run(_cli(), _done("read_file", {"path": "a.py"}, read), _done("search_files", {"pattern": "foo"}, search),
                 _done("terminal", {"command": "true"}, json.dumps({"output": "", "exit_code": 0})))
    explored = [line for line in lines if "Explored" in line]
    assert len(explored) == 1 and "a.py" in explored[0] and "foo" in explored[0], lines
    assert not any(line.lstrip().startswith("┊") for line in lines)


def test_todo_checklist_hangs_under_the_plan_row():
    todos = {"todos": [{"id": "1", "content": "Fix it", "status": "in_progress"},
                       {"id": "2", "content": "Test it", "status": "pending"}]}
    lines = _run(_cli(), _done("todo_list", todos, json.dumps({"todos": todos["todos"], "summary": {"total": 2}})))
    row = next(i for i, line in enumerate(lines) if "Plan" in line)
    assert lines[row + 1].startswith("  ⎿  ") and "Fix it" in lines[row + 1]
    assert lines[row + 2].startswith("     ") and "Test it" in lines[row + 2]


_DIFF = """a/pricing.py → b/pricing.py
@@ -5,4 +5,4 @@
 def apply_discount(price, pct):
-    price = Decimal(str(price)).quantize(CENT)
-    return price * (100 - pct) / 100
+    price = Decimal(str(price))
+    return (price * (100 - pct) / 100).quantize(CENT)
"""


def test_gutter_diff_numbers_rows_and_tints_them_as_one_block():
    raw = cli_diff_render.render_gutter_diff(_DIFF)
    plain = [_plain(line) for line in raw]
    assert [line.split()[:3] for line in plain] == [["│", "5", "def"], ["│", "6", "-"], ["│", "7", "-"],
                                                    ["│", "6", "+"], ["│", "7", "+"]], plain
    assert all(line.startswith("  │  ") for line in plain)
    # The changed rows are padded to one width (the diff's longest row), never to the window.
    widths = {len(_plain(line)) for line in raw[1:]}
    assert len(widths) == 1, plain
    assert cli_diff_render.render_gutter_diff(_DIFF, 200) == raw  # a wide window changes nothing


def test_gutter_diff_names_each_file_only_when_an_edit_spans_several():
    two = _DIFF + "a/other.py → b/other.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
    plain = [_plain(line) for line in cli_diff_render.render_gutter_diff(two)]
    headers = [line for line in plain if "+" in line and "−" in line and "py" in line]
    assert [h.split()[1] for h in headers] == ["pricing.py", "other.py"], plain
    single = [_plain(line) for line in cli_diff_render.render_gutter_diff(_DIFF)]
    assert not any("pricing.py" in line for line in single)


def test_modern_inline_diff_goes_through_the_gutter_renderer():
    from agent.display import render_edit_diff_with_delta
    rendered: list[str] = []
    diff = "--- a/x.py\n+++ b/x.py\n@@ -3,1 +3,1 @@\n-value = compute(alpha)\n+value = compute(beta)\n"
    render_edit_diff_with_delta("patch", json.dumps({"success": True, "diff": diff}), print_fn=rendered.append)
    lines = _run(_cli(), lambda c: c._on_inline_diff("patch", "\n".join(rendered)))
    assert [line.split()[:3] for line in lines if "compute" in line] == [["│", "3", "-"], ["│", "3", "+"]], lines


def test_tool_row_keeps_the_neutral_refusal_dim():
    from hermes_cli.cli_modern_layout import _RED
    refused = json.dumps({"success": False, "error": "SKILL.md content is 100,193 characters (limit: 100,000)."})
    row = tool_row("skill_manage", {"operations": []}, 0.3, refused)
    assert _plain(row).startswith("✗ ") and _RED not in row
