"""Codex-style backtrack browse in the classic CLI: Esc primes (hint), Esc again opens a transcript
pager with the latest prompt selected in reverse video; ←/→ pick prompts, ↑/↓ scroll, Enter
prefills the composer and cuts nothing, q/Esc close."""

import os
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from prompt_toolkit.utils import get_cwidth

from hermes_state import SessionDB
from tests.hermes_cli.test_cli_init import _make_cli


@pytest.fixture
def cli(tmp_path, monkeypatch):
    cli = _make_cli()
    cli._session_db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db = db
    cli.session_id = "20261105_010000_bbbbbb"
    db.create_session(cli.session_id, source="cli")
    for role, content in (("user", "ALPHA"), ("assistant", "a1"), ("user", "BETA"), ("assistant", "b1"),
                          ("user", "GAMMA"), ("assistant", "g1")):
        db.append_message(cli.session_id, role, content)
    cli.conversation_history = db.get_messages_as_conversation(cli.session_id)
    cli.agent = None
    cli._agent_running = False
    cli._attached_images = []
    cli._prefill_input_buffer = MagicMock()
    cli._invalidate = MagicMock()
    cli._pending_input = MagicMock()
    printed = []
    monkeypatch.setattr("cli._cprint", lambda text, *a, **k: printed.append(text))
    cli.printed = printed
    _terminal(monkeypatch, 100, 30)
    yield cli
    db.close()


def _terminal(monkeypatch, cols, rows):
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((cols, rows)))


def _event(text=""):
    buf = SimpleNamespace(text=text)
    return SimpleNamespace(app=SimpleNamespace(current_buffer=buf), current_buffer=buf)


def _esc(cli, text=""):
    cli._backtrack_handle_escape(_event(text))


def _hint(cli):
    return "".join(text for _s, text in cli._tui_hint_text())


def _rows(cli):
    """Rendered panel rows as [(text, is_reverse)]."""
    rows, line, rev = [], "", False
    for style, text in cli._get_backtrack_display_fragments():
        for i, part in enumerate(text.split("\n")):
            if i:
                rows.append((line, rev))
                line, rev = "", False
            line += part
            rev = rev or ("reverse" in style.split() and bool(part.strip()))
    return rows


def _selected_text(cli):
    return [text for text, rev in _rows(cli) if rev]


def test_first_esc_primes_with_the_hint_and_the_second_opens_browse_on_the_latest(cli):
    _esc(cli)
    assert cli._backtrack_state is None
    assert "esc again to edit previous message" in _hint(cli)
    _esc(cli)
    assert cli._backtrack_state is not None
    assert cli._backtrack_selected() == (2, "GAMMA")
    assert cli._backtrack_primed is None and "esc again" not in _hint(cli)
    assert any("GAMMA" in text for text in _selected_text(cli))


def test_a_non_escape_key_cancels_the_prime(cli):
    from prompt_toolkit.key_binding.key_processor import KeyPress
    from prompt_toolkit.keys import Keys
    _esc(cli)
    cli._backtrack_after_key_press(SimpleNamespace(key_buffer=[KeyPress(Keys.Escape, "\x1b")]))
    assert cli._backtrack_primed  # an Esc waiting for a possible Alt-sequence keeps it
    cli._backtrack_after_key_press(SimpleNamespace(key_buffer=[]))  # e.g. a typed letter
    assert cli._backtrack_primed is None and "esc again" not in _hint(cli)
    _esc(cli)
    assert cli._backtrack_state is None  # primes again rather than opening


def test_esc_with_a_draft_never_primes(cli):
    _esc(cli, "draft")
    assert cli._backtrack_primed is None and cli._backtrack_state is None


def test_zero_prompts_primes_silently_then_says_no_previous_message(cli):
    cli.conversation_history = []
    _esc(cli)
    assert _hint(cli) == ""
    _esc(cli)
    assert cli._backtrack_state is None
    assert any("No previous message to edit." in line for line in cli.printed)


def test_left_right_pick_prompts_and_clamp(cli):
    assert cli._open_backtrack_picker()
    cli._backtrack_step(-1)  # → at the newest clamps
    assert cli._backtrack_selected()[1] == "GAMMA"
    for expected in ("BETA", "ALPHA", "ALPHA"):  # ← older, clamps at the oldest
        cli._backtrack_step(1)
        assert cli._backtrack_selected()[1] == expected
    assert _selected_text(cli) and all("ALPHA" in text for text in _selected_text(cli))
    cli._backtrack_step(-1)
    assert cli._backtrack_selected()[1] == "BETA"


def test_up_down_scroll_without_moving_the_selection(cli, monkeypatch):
    _terminal(monkeypatch, 80, 14)  # body smaller than the transcript, so it scrolls
    assert cli._open_backtrack_picker()
    _rows(cli)
    bottom = cli._backtrack_state["_scroll_offset"]
    assert bottom > 0  # opened scrolled down to the latest prompt
    cli._backtrack_scroll(-1)
    _rows(cli)
    assert cli._backtrack_state["_scroll_offset"] == bottom - 1
    assert cli._backtrack_selected()[1] == "GAMMA"
    page = cli._backtrack_state["_page"]
    cli._backtrack_scroll(pages=-1)
    _rows(cli)
    assert cli._backtrack_state["_scroll_offset"] == max(0, bottom - 1 - (page - 1))
    assert cli._backtrack_selected()[1] == "GAMMA"


def test_enter_prefills_the_prompt_and_cuts_nothing(cli):
    sid = cli.session_id
    assert cli._open_backtrack_picker()
    cli._backtrack_step(1)
    cli._backtrack_choose_edit()
    assert cli._backtrack_state is None
    cli._prefill_input_buffer.assert_called_once_with("BETA")
    assert cli._backtrack_edit["ordinal"] == 1
    rows = cli._session_db._conn.execute("SELECT active FROM messages WHERE session_id = ?", (sid,)).fetchall()
    assert [a for (a,) in rows] == [1] * 6
    assert len(cli.conversation_history) == 6


@pytest.mark.parametrize("close", ["escape", "q", "ctrl_c"])
def test_close_keys_leave_everything_as_it_was(cli, close):
    assert cli._open_backtrack_picker()
    if close == "ctrl_c":
        cli._tui_handle_ctrl_c(_event())
    else:
        cli._close_backtrack_picker()  # the Esc and q bindings call this directly
    assert cli._backtrack_state is None and cli._backtrack_edit is None
    cli._prefill_input_buffer.assert_not_called()


def test_any_other_key_closes_browse_and_is_replayed_into_the_composer(cli):
    assert cli._open_backtrack_picker()
    processor = MagicMock()
    event = SimpleNamespace(app=SimpleNamespace(key_processor=processor), key_sequence=["x"])
    cli._backtrack_close_and_replay(event)
    assert cli._backtrack_state is None
    processor.feed_multiple.assert_called_once_with(["x"], first=True)


def test_rewind_command_opens_browse_directly(cli):
    cli._handle_rewind_command("/rewind")
    assert cli._backtrack_selected() == (2, "GAMMA")


def test_pager_shows_prompts_and_replies_with_markers(cli):
    assert cli._open_backtrack_picker()
    texts = [text for text, _rev in _rows(cli)]
    body = "\n".join(texts)
    for marker in ("› ALPHA", "• a1", "› BETA", "• b1", "› GAMMA", "• g1"):
        assert marker in body
    assert "Browsing transcript" in texts[-2]


@pytest.mark.parametrize("cols,rows", [(80, 24), (60, 16), (100, 30)])
def test_pager_fits_the_terminal_in_cells_with_a_cjk_prompt(cli, monkeypatch, cols, rows):
    from hermes_cli.cli_tui_mixin import _PANEL_RESERVED_BELOW
    _terminal(monkeypatch, cols, rows)
    prompt = "请帮我把这个函数重构成更小的部分，并解释每一步为什么这样做，同时保留原有的行为和测试。" * 3
    cli.conversation_history = [row for n in range(6) for row in (
        {"role": "user", "content": f"P{n} {prompt}"},
        {"role": "assistant", "content": "\n".join(f"回答 {n} 第 {i} 行：重构完成。" for i in range(8))})]
    assert cli._open_backtrack_picker()
    lines = _rows(cli)
    assert len(lines) + _PANEL_RESERVED_BELOW <= rows
    assert {get_cwidth(text) for text, _rev in lines} == {get_cwidth(lines[0][0])}
    assert get_cwidth(lines[0][0]) < cols
    selected = [text for text, rev in lines if rev]
    assert selected and "P5" in selected[0]  # the latest prompt, in reverse video, is in view
    cli._backtrack_step(5)  # jump to the oldest: brought into view
    selected = [text for text, rev in _rows(cli) if rev]
    assert selected and "P0" in selected[0]
