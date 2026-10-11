"""Live in-TUI todo board: renderer, collapsed/expanded budgets, and caller wiring.

The board renders above the prompt from the agent's todo store (pending-first, bordered, with
an overflow footer). It opens COLLAPSED (a small row budget; Ctrl+O expands to the full
terminal-proportional cap and collapses back) so a large plan cannot own the screen. These
tests pin the pure functions AND the caller wiring (widget construction + keybinding) so a
merge cannot silently drop the widget.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import cli_tui_mixin as m
from hermes_cli.cli_render import _panel_cwidth
from tools.todo_tool import TodoStore, _STATUS_MARKERS

TUI_SRC = Path(m.__file__).read_text()


def _items(*statuses):
    return [{"id": str(n), "content": f"task {n}", "status": s} for n, s in enumerate(statuses)]


def _text(frags):
    return "".join(t for _, t in frags)


# ── expanded row cap ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "term_rows, expected",
    [
        (0, 8), (10, 8), (16, 8), (17, 8),  # floor: 16//2 = 8, 17//2 = 8
        (18, 9), (24, 12), (40, 20),         # term_rows // 2 in range
        (60, 30), (61, 30), (62, 30),       # 60//2 = 30 == ceiling
        (200, 30),                           # ceiling
    ],
)
def test_max_rows_is_half_terminal_clamped(term_rows, expected):
    assert m._TODO_BOARD_MIN_ROWS == 8 and m._TODO_BOARD_MAX_ROWS == 30
    assert m._todo_board_max_rows(term_rows) == expected


def test_max_rows_defaults_to_live_terminal(monkeypatch):
    monkeypatch.setattr(m, "_term_rows", lambda: 50)
    assert m._todo_board_max_rows() == 25


# ── selection order / truncation ──────────────────────────────────────────────────────────────


def test_select_fits_returns_all_unchanged():
    items = _items("completed", "pending", "in_progress")
    shown, hd, ha = m._select_todo_display_items(items, 3)
    assert shown is items and (hd, ha) == (0, 0)


def test_select_drops_completed_before_pending():
    # 10 completed + 4 pending, cap 12 → all 4 pending kept, 8/10 completed shown, 2 hidden.
    items = _items(*(["completed"] * 10 + ["pending"] * 4))
    shown, hd, ha = m._select_todo_display_items(items, 12)
    assert len(shown) == 12
    assert [i["status"] for i in shown].count("pending") == 4
    assert (hd, ha) == (2, 0)
    # Original list order preserved; the oldest completed are the ones kept.
    assert [i["id"] for i in shown] == [str(n) for n in list(range(8)) + [10, 11, 12, 13]]


def test_select_in_progress_first_then_pending_head():
    items = _items("pending", "pending", "in_progress", "pending", "pending", "completed")
    shown, hd, ha = m._select_todo_display_items(items, 3)
    assert [i["id"] for i in shown] == ["0", "1", "2"]  # in_progress + first 2 pending
    assert (hd, ha) == (1, 2)


def test_select_cancelled_counts_as_done():
    items = _items("cancelled", "cancelled", "pending", "pending")
    shown, hd, ha = m._select_todo_display_items(items, 2)
    assert [i["status"] for i in shown] == ["pending", "pending"]
    assert (hd, ha) == (2, 0)


def test_select_in_progress_never_hidden_even_past_cap():
    items = _items(*(["in_progress"] * 5 + ["pending"] * 2 + ["completed"]))
    shown, hd, ha = m._select_todo_display_items(items, 3)
    assert [i["status"] for i in shown] == ["in_progress"] * 5
    assert (hd, ha) == (1, 2)


# ── text / height render ──────────────────────────────────────────────────────────────────────


def test_empty_board_renders_nothing():
    assert m.get_todo_board_text([], 100, 24) == []
    assert m.get_todo_board_height([], 24) == 0


def test_board_text_border_markers_and_header():
    items = _items("completed", "in_progress", "pending", "cancelled", "weird")
    frags = m.get_todo_board_text(items, 100, 24)
    styles = {s for s, _ in frags}
    assert styles == {"class:todo-border", "class:hint"}
    lines = _text(frags).splitlines()
    assert lines[0].startswith("╭") and lines[0].endswith("╮")
    assert lines[-1].startswith("╰") and lines[-1].endswith("╯")
    body = lines[1:-1]
    assert all(l.startswith("│ ") and l.endswith(" │") for l in body)
    assert "📋 tasks 2/5" in body[0]
    # Markers come from tools.todo_tool._STATUS_MARKERS; unknown statuses fall back to "[?]".
    assert _STATUS_MARKERS["completed"] in body[1]
    assert _STATUS_MARKERS["in_progress"] in body[2]
    assert "[ ] task 2" in body[3]
    assert _STATUS_MARKERS["cancelled"] in body[4]
    assert "[?]" in body[5]
    # Every line spans the same number of terminal cells (right border aligned).
    assert len({_panel_cwidth(l) for l in lines}) == 1
    # Content column aligned regardless of marker width.
    cols = {_panel_cwidth(l.split("task")[0]) for l in body[1:]}
    assert len(cols) == 1
    assert m.get_todo_board_height(items, 24) == len(lines) == 5 + 1 + 2


def test_board_overflow_footer_and_pending_first():
    # 24 term rows → cap 12. 20 completed + 6 pending.
    items = _items(*(["completed"] * 20 + ["pending"] * 6))
    lines = _text(m.get_todo_board_text(items, 100, 24, expanded=True)).splitlines()
    body = lines[1:-1]
    assert body[-1].strip(" │").startswith("… +14 completed hidden")
    assert sum("[ ]" in l for l in body) == 6
    # header + 12 items + footer + 2 borders
    assert m.get_todo_board_height(items, 24, expanded=True) == 1 + 12 + 1 + 2 == len(lines)


def test_board_footer_reports_hidden_pending_on_all_pending_list():
    items = _items(*(["pending"] * 20))
    lines = _text(m.get_todo_board_text(items, 100, 16, expanded=True)).splitlines()  # cap = 8
    assert "… +12 pending hidden" in lines[-2]


@pytest.mark.parametrize("term_rows, cap", [(10, 8), (40, 20), (100, 30)])
def test_height_tracks_cap_boundaries(term_rows, cap):
    items = _items(*(["pending"] * 50))
    # header + cap items + footer + 2 borders
    assert m.get_todo_board_height(items, term_rows, expanded=True) == cap + 4


def test_narrow_terminal_trims_rows_keeps_border_aligned():
    items = [{"id": "1", "content": "x" * 200, "status": "pending"}]
    lines = _text(m.get_todo_board_text(items, 30, 24)).splitlines()
    assert len({_panel_cwidth(l) for l in lines}) == 1
    assert all(l.endswith(("╮", "│", "╯")) for l in lines)


def test_multiline_content_flattened():
    items = [{"id": "1", "content": "line one\nline two", "status": "pending"}]
    lines = _text(m.get_todo_board_text(items, 100, 24)).splitlines()
    assert len(lines) == 4 and "line one line two" in lines[2]


# ── collapsed default: fewer rows + Ctrl+O expand/collapse ────────────────────────────────────


def test_collapsed_default_caps_rows_and_shows_expand_hint():
    # 20 completed + 6 pending → collapsed budget 5: header + 5 pending + hidden footer + borders.
    items = _items(*(["completed"] * 20 + ["pending"] * 6))
    lines = _text(m.get_todo_board_text(items, 100, 24)).splitlines()
    body = lines[1:-1]
    assert "Ctrl+O expand" in body[0]
    assert "… +1 pending / 20 completed hidden" in body[-1]
    assert sum("[ ]" in l for l in body) == 5
    assert m.get_todo_board_height(items, 24) == 1 + 5 + 1 + 2 == len(lines)


def test_expanded_view_flips_hint_to_collapse():
    items = _items(*(["completed"] * 20 + ["pending"] * 6))
    lines = _text(m.get_todo_board_text(items, 100, 24, expanded=True)).splitlines()
    assert "Ctrl+O collapse" in lines[1]
    assert "Ctrl+O expand" not in lines[1]
    assert m.get_todo_board_height(items, 24, expanded=True) == 12 + 4 == len(lines)


def test_no_toggle_hint_when_list_fits_collapsed_budget():
    items = _items("pending", "pending", "in_progress")
    text = _text(m.get_todo_board_text(items, 100, 24))
    assert "Ctrl+O" not in text
    assert len(text.splitlines()) == 3 + 1 + 2


def test_toggle_todo_board_flips_state_and_invalidates():
    calls = []
    cli = SimpleNamespace(_invalidate=lambda: calls.append(1))
    assert getattr(cli, "_todo_board_expanded", False) is False
    m.toggle_todo_board(cli)
    assert cli._todo_board_expanded is True and len(calls) == 1
    m.toggle_todo_board(cli)
    assert cli._todo_board_expanded is False and len(calls) == 2


# ── wiring ────────────────────────────────────────────────────────────────────────────────────


def test_todo_board_items_reads_agent_store():
    store = TodoStore()
    store.write([{"id": "a", "content": "do it", "status": "pending"}])
    cli = SimpleNamespace(agent=SimpleNamespace(_todo_store=store))
    assert m.CLITuiMixin._todo_board_items(cli)[0]["content"] == "do it"
    assert m.CLITuiMixin._todo_board_items(SimpleNamespace(agent=None)) == []
    assert m.CLITuiMixin._todo_board_items(SimpleNamespace(agent=SimpleNamespace())) == []


def test_caller_builds_and_passes_widget():
    assert re.search(r"todo_board_widget = ConditionalContainer\(", TUI_SRC)
    assert "todo_board_widget=todo_board_widget," in TUI_SRC


def test_toggle_wiring_in_tui_source():
    assert "kb.add('c-o'" in TUI_SRC
    # Ctrl+E stays with prompt_toolkit's emacs map (end-of-line): app-level bindings merge last
    # and KeyProcessor fires matches[-1], so a binding here would silently shadow it again.
    assert "kb.add('c-e'" not in TUI_SRC
    assert "toggle_todo_board(self)" in TUI_SRC
    assert 'expanded=getattr(cli_ref, "_todo_board_expanded", False)' in TUI_SRC


def test_layout_children_places_board_above_spinner():
    sentinel_board, sentinel_spinner = object(), object()
    kw = {k: None for k in ("sudo_widget", "secret_widget", "approval_widget", "clarify_widget",
                            "spacer", "status_bar", "input_rule_top", "image_bar", "input_area",
                            "input_rule_bot", "voice_status_bar", "completions_menu")}
    cli = SimpleNamespace(_get_extra_tui_widgets=list)
    children = m.CLITuiMixin._build_tui_layout_children(
        cli, todo_board_widget=sentinel_board, spinner_widget=sentinel_spinner, **kw)
    assert children.index(sentinel_board) == children.index(sentinel_spinner) - 1


def test_real_layout_paints_board_end_to_end(monkeypatch):
    """Real path: HermesCLI._tui_build_layout → one prompt_toolkit frame shows the board."""
    from prompt_toolkit.key_binding import KeyBindings

    from tests.hermes_cli.test_cli_footer_split import _render_once, _Tty

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    from cli import HermesCLI

    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture",
                    base_url="http://127.0.0.1:1/v1")
    cli._tui_init_run_state()
    store = TodoStore()
    store.write([{"id": "a", "content": "ship the board", "status": "in_progress"},
                 {"id": "b", "content": "write tests", "status": "pending"}])
    cli.agent = SimpleNamespace(_todo_store=store)
    kb = KeyBindings()
    layout, style = cli._tui_build_layout(kb)
    tty = _Tty()
    _render_once(layout, kb, style, tty)
    assert "tasks 0/2" in tty.text
    assert "ship the board" in tty.text and "write tests" in tty.text
    assert "╭" in tty.text and "╰" in tty.text


def test_todo_border_style_registered():
    cli = SimpleNamespace()
    m.CLITuiMixin._tui_set_base_style(cli)
    assert cli._tui_style_base["todo-border"] == "#CD7F32"
