"""Classic-CLI tool rows, ``⎿`` result previews, explored grouping and the collapsed reasoning
line (``hermes_cli.cli_tool_render``) plus the inline-diff re-render (``cli_diff_render``)."""
import json
import re

from agent.i18n import t
from hermes_cli import cli_tool_render as render
from hermes_cli.cli_diff_render import parse_inline_diff, render_inline_diff

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(lines):
    return [_ANSI.sub("", line) for line in lines]


def terminal_result(output, exit_code=0, error=None):
    return json.dumps({"output": output, "exit_code": exit_code, "error": error})


def test_success_and_failure_rows_carry_distinct_glyphs_and_duration():
    ok_rows, ok_failed = render.render_tool_row("terminal", {"command": "ls"}, 0.42, terminal_result("x"))
    bad_rows, bad_failed = render.render_tool_row("terminal", {"command": "false"}, 1.5, terminal_result("", 1))
    assert not ok_failed and bad_failed
    assert "✓" in plain(ok_rows)[0] and "✗" not in plain(ok_rows)[0]
    assert "✗" in plain(bad_rows)[0]
    # The failure colour is the skin's error colour, distinct from the success colour.
    assert render.error_color() in bad_rows[0] and render.ok_color() in ok_rows[0]
    assert plain(ok_rows)[0].rstrip().endswith(render.format_duration(0.42))


def test_is_error_flag_marks_a_row_failed_even_without_a_detectable_result():
    rows, failed = render.render_tool_row("mcp_tool", {}, 0.1, "plain text", is_error=True)
    assert failed and "✗" in plain(rows)[0]


def test_preview_folds_to_budget_and_counts_the_hidden_lines():
    output = "\n".join(f"line{i}" for i in range(10))
    lines = plain(render.render_preview("terminal", terminal_result(output), failed=False, max_lines=3,
                                        width=80, expand_hint=True))
    assert [line.strip().lstrip("⎿").strip() for line in lines[:3]] == ["line0", "line1", "line2"]
    assert lines[0].lstrip().startswith("⎿")
    assert lines[-1].strip() == t("cli.tool_render.more_lines_hint", count=7, shortcut=render.EXPAND_KEY)


def test_a_single_hidden_line_is_shown_instead_of_a_plus_one_marker():
    lines = plain(render.render_preview("terminal", terminal_result("a\nb\nc\nd"), failed=False, max_lines=3,
                                        width=80, expand_hint=True))
    assert len(lines) == 4 and "d" in lines[-1]


def test_empty_command_output_says_no_output_and_errors_render_in_the_error_colour():
    empty = plain(render.render_preview("terminal", terminal_result(""), failed=False, max_lines=3, width=80,
                                        expand_hint=False))
    assert empty and t("cli.tool_render.no_output") in empty[0]
    failed = render.render_preview("terminal", terminal_result("boom"), failed=True, max_lines=3, width=80,
                                   expand_hint=False)
    assert render.error_color() in failed[0]


def test_preview_zero_budget_and_diff_tools_print_nothing():
    assert render.render_preview("terminal", terminal_result("x"), failed=False, max_lines=0, width=80,
                                 expand_hint=True) == []
    assert render.render_preview("patch", json.dumps({"success": True, "diff": "x"}), failed=False, max_lines=3,
                                 width=80, expand_hint=True) == []


def test_preview_strips_terminal_escapes_and_clips_to_width():
    lines = plain(render.render_preview("terminal", terminal_result("\x1b[2J\x1b[31mred\x1b[0m " + "y" * 200),
                                        failed=False, max_lines=3, width=40, expand_hint=False))
    assert "\x1b" not in "".join(lines) and "red" in lines[0]
    from prompt_toolkit.utils import get_cwidth
    assert get_cwidth(lines[0]) <= 40


def test_read_and_search_previews_summarise_structured_results():
    read = plain(render.render_preview("read_file", json.dumps({"content": "1|x", "total_lines": 120}),
                                       failed=False, max_lines=3, width=80, expand_hint=False))
    assert t("cli.tool_render.read_lines_other", count=120) in read[0]
    search = plain(render.render_preview(
        "search_files", json.dumps({"total_count": 2, "matches": [{"path": "a.py", "line": 3, "content": "foo"}]}),
        failed=False, max_lines=3, width=80, expand_hint=False))
    assert t("cli.tool_render.matches_other", count=2) in search[0] and "a.py:3: foo" in search[1]


def test_explored_row_names_every_kind_and_dedupes_subjects():
    group = render.ExploreGroup()
    group.add("read_file", {"path": "src/a.py"}, 0.1, "{}")
    group.add("read_file", {"path": "src/a.py"}, 0.1, "{}")
    group.add("read_file", {"path": "lib/b.py"}, 0.1, "{}")
    group.add("search_files", {"pattern": "foo"}, 0.2, "{}")
    row = plain([render.render_explore_row(group, 120)])[0]
    assert t("cli.tool_render.explored_read", items="a.py, b.py") in row
    assert t("cli.tool_render.explored_search", items="foo") in row
    assert row.rstrip().endswith(render.format_duration(0.5))


def test_reasoning_line_prefers_the_first_bold_heading():
    text = "Let me look.\n**Planning the fix**\nmore"
    assert render.reasoning_headline(text) == "Planning the fix"
    assert render.reasoning_headline("## Heading\nbody") == "Heading"
    assert render.reasoning_headline("first line\nsecond") == "first line"
    line = plain([render.render_reasoning_line(text, 8.0, 100)])[0]
    assert t("cli.tool_render.thought_for", duration=render.format_duration(8.0)) in line
    assert "Planning the fix" in line


def _rendered_gateway_diff(diff: str) -> str:
    from agent.display import render_edit_diff_with_delta
    lines: list[str] = []
    assert render_edit_diff_with_delta("patch", json.dumps({"success": True, "diff": diff}), print_fn=lines.append)
    return "\n".join(lines)


DIFF = "--- a/pkg/x.py\n+++ b/pkg/x.py\n@@ -10,3 +10,4 @@\n ctx\n-old\n+new\n+added\n tail\n"


def test_diff_parse_counts_lines_and_numbers_rows_from_the_hunk_header():
    files, _ = parse_inline_diff(_rendered_gateway_diff(DIFF))
    assert [f.path for f in files] == ["pkg/x.py"]
    diff = files[0]
    assert (diff.added, diff.removed) == (2, 1)
    assert [(k, n) for k, n, _ in diff.rows] == [(" ", 10), ("-", 11), ("+", 11), ("+", 12), (" ", 13)]


def test_diff_render_has_header_counts_gutter_and_full_width_change_rows():
    width = 60
    lines = render_inline_diff(_rendered_gateway_diff(DIFF), width)
    text = plain(lines)
    assert "pkg/x.py" in text[0] and "+2" in text[0] and "−1" in text[0]
    from prompt_toolkit.utils import get_cwidth
    changed = [line for line in text[1:] if re.match(r"\s+\d+ [+-] ", line)]
    assert len(changed) == 3
    assert all(get_cwidth(line) == width - 1 for line in changed)


def test_diff_render_passes_unparseable_text_through():
    assert render_inline_diff("just a note", 80) == ["just a note"]


def test_tool_rows_show_the_tool_emoji_and_honour_skin_overrides(monkeypatch):
    from agent import display
    rows, _ = render.render_tool_row("terminal", {"command": "ls"}, 0.1, terminal_result("x"))
    labelled, _, _ = display.cute_tool_rows("terminal", {"command": "ls"})
    bare, _, _ = display.cute_tool_rows("terminal", {"command": "ls"}, emoji=False)
    emoji = labelled[0].removeprefix("┊ ").split(" ", 1)[0]
    assert emoji not in bare[0]  # the token really is the emoji slot
    assert emoji in plain(rows)[0] and "ls" in plain(rows)[0]

    class Skin:
        tool_emojis = {"terminal": "⚔"}
        tool_prefix = "┊"

        def get_color(self, key, fallback=""):
            return fallback
    monkeypatch.setattr(display, "_get_skin", lambda: Skin())
    skinned, _ = render.render_tool_row("terminal", {"command": "ls"}, 0.1, terminal_result("x"))
    assert "⚔" in plain(skinned)[0]


def test_running_tool_label_is_the_completion_rows_verb_and_detail():
    label = render.running_tool_label("terminal", {"command": "ls -la /etc"})
    assert "ls -la /etc" in label and "  " not in label.strip()
    assert render.running_tool_label("terminal", {"command": "x" * 500}, width=30).endswith("…")


def test_todo_checklist_marks_each_status_and_folds_long_lists():
    items = [{"id": str(i), "content": f"task {i}", "status": "completed" if i < 2 else
              "in_progress" if i == 2 else "pending"} for i in range(12)]
    result = json.dumps({"todos": items, "summary": {"total": 12}})
    lines = render.render_todo_checklist(render.todo_items({}, result), 80)
    text = plain(lines)
    assert "☒ task 0" in text[0] and "◼ task 2" in "".join(text) and "☐ task 3" in "".join(text)
    assert len(lines) == render.TODO_MAX_ITEMS + 1
    assert t("cli.tool_render.more_items", count=12 - render.TODO_MAX_ITEMS) in text[-1]
    in_progress = next(line for line in lines if "task 2" in line)
    done = next(line for line in lines if "task 0" in line)
    assert render.DIM in done and render.DIM not in in_progress


def test_todo_checklist_window_keeps_the_active_item_visible():
    items = [{"id": str(i), "content": f"task {i}", "status": "completed"} for i in range(10)]
    items += [{"id": "10", "content": "now", "status": "in_progress"}]
    assert any("◼ now" in line for line in plain(render.render_todo_checklist(items, 80, limit=4)))


def _row_with(lines, marker):
    return next(line for line in lines if re.search(rf"\d+ {re.escape(marker)} ", _ANSI.sub("", line)))


def test_paired_diff_lines_highlight_only_the_changed_words():
    from agent.display import _diff_ansi
    colors = _diff_ansi()
    def emphasised(line, key):  # each run between the brighter background and the row colour
        return re.findall(re.escape(colors[f"{key}_em"]) + r"(.*?)" + re.escape(colors[key]), line)
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-    total = price * qty\n+    total = price * quantity\n"
    lines = render_inline_diff(_rendered_gateway_diff(diff), 80)
    minus, plus = _row_with(lines, "-"), _row_with(lines, "+")
    assert emphasised(minus, "minus") == ["qty"] and emphasised(plus, "plus") == ["quantity"]
    # A pure insertion highlights only the inserted tokens on the added line.
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-    return a + b\n+    return int(a) + int(b)\n"
    lines = render_inline_diff(_rendered_gateway_diff(diff), 80)
    assert emphasised(_row_with(lines, "+"), "plus") == ["int(", ")", "int(", ")"]
    assert emphasised(_row_with(lines, "-"), "minus") == []


def test_unpaired_or_rewritten_diff_lines_get_no_word_highlight():
    from agent.display import _diff_ansi
    colors = _diff_ansi()
    added_only = render_inline_diff(_rendered_gateway_diff("--- a/x\n+++ b/x\n@@ -1,0 +1,1 @@\n+brand new\n"), 80)
    rewritten = render_inline_diff(_rendered_gateway_diff(
        "--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-alpha beta gamma\n+completely other words here\n"), 80)
    assert not any(colors["plus_em"] in line or colors["minus_em"] in line for line in added_only + rewritten)
