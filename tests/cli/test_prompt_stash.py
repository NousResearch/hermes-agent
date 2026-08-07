"""Tests for the Ctrl+S prompt stash state machine (hermes_cli.prompt_stash).

Covers the pure state machine directly — no prompt_toolkit, no TUI:
  - stashing an empty/whitespace buffer is a no-op
  - stash → restore round-trips exact text including newlines
  - repeated stashes never silently clobber an earlier draft
  - indicator / placeholder state
  - browse-panel cursor, delete, and restore
  - the resolve_ctrl_s decision table
"""

from __future__ import annotations

import pytest

from hermes_cli.prompt_stash import (
    ACTION_CLOSE_PANEL,
    ACTION_NOOP,
    ACTION_OPEN_PANEL,
    ACTION_RESTORED,
    ACTION_STASHED,
    MAX_STASH_ITEMS,
    PromptStash,
    StashEntry,
    build_preview,
    resolve_ctrl_s,
)


class _FakeClock:
    """Deterministic monotonic clock for age assertions."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


@pytest.fixture
def stash():
    return PromptStash(clock=_FakeClock())


# --------------------------------------------------------------- no-op cases


class TestStashNoOp:
    """An empty or whitespace-only composer must not create a stash entry."""

    @pytest.mark.parametrize("text", ["", "   ", "\n", "\t\t", "  \n \n  ", None])
    def test_stash_blank_buffer_is_noop(self, stash, text):
        assert stash.stash(text) is False
        assert len(stash) == 0
        assert stash.indicator() == ""

    def test_pop_empty_stash_returns_none(self, stash):
        assert stash.pop() is None

    def test_peek_empty_stash_returns_none(self, stash):
        assert stash.peek() is None

    def test_open_panel_on_empty_stash_refused(self, stash):
        assert stash.open_panel() is False
        assert stash.panel_open is False

    def test_delete_on_empty_stash_is_noop(self, stash):
        assert stash.delete_at_cursor() is False

    def test_restore_at_cursor_on_empty_stash(self, stash):
        assert stash.restore_at_cursor() is None

    def test_images_only_draft_is_stashable(self, stash):
        """Blank text but attached images is still worth parking."""
        assert stash.stash("", ["/tmp/a.png"]) is True
        assert len(stash) == 1
        assert stash.peek().preview == "(images only)"


# ------------------------------------------------------------- round-tripping


class TestRoundTrip:
    """Restore must return the draft byte-for-byte."""

    @pytest.mark.parametrize(
        "text",
        [
            "hello",
            "line one\nline two",
            "line one\nline two\nline three\n",
            "\n  leading blank and indented\n",
            "trailing spaces   ",
            "   leading spaces",
            "para one\n\npara two\n\n\npara three",
            "tabs\there\tand\there",
            "unicode ünïcödé 中文 🎉 mixed",
            "```python\ndef f():\n    return 1\n```",
        ],
    )
    def test_stash_then_restore_round_trips_exactly(self, stash, text):
        assert stash.stash(text) is True
        result = stash.pop()
        assert result is not None
        restored, images = result
        assert restored == text
        assert images == []
        # Popping consumed the entry.
        assert len(stash) == 0

    def test_multiline_draft_preserves_every_newline(self, stash):
        text = "a\nb\nc\nd\ne"
        stash.stash(text)
        restored, _ = stash.pop()
        assert restored.count("\n") == 4
        assert restored.splitlines() == ["a", "b", "c", "d", "e"]

    def test_round_trip_through_resolve_ctrl_s(self, stash):
        """The full gesture: Ctrl+S to park, Ctrl+S on empty to bring back."""
        draft = "a long prompt\nwith several lines\n"
        action, payload = resolve_ctrl_s(stash, draft)
        assert action == ACTION_STASHED
        assert payload is None
        assert len(stash) == 1

        action, payload = resolve_ctrl_s(stash, "")
        assert action == ACTION_RESTORED
        assert payload == (draft, [])
        assert len(stash) == 0

    def test_images_round_trip(self, stash):
        imgs = ["/tmp/one.png", "/tmp/two.png"]
        stash.stash("with pics", imgs)
        text, restored = stash.pop()
        assert text == "with pics"
        assert restored == imgs
        # The stash must hold its own copy — mutating the caller's list after
        # stashing cannot corrupt the parked entry.
        imgs.append("/tmp/three.png")
        assert restored == ["/tmp/one.png", "/tmp/two.png"]


# --------------------------------------------------------------- no clobbering


class TestNoSilentClobber:
    """A second Ctrl+S must not destroy the first draft."""

    def test_second_stash_keeps_first(self, stash):
        stash.stash("first draft")
        stash.stash("second draft")
        assert len(stash) == 2
        texts = [e.text for e in stash.items]
        assert "first draft" in texts
        assert "second draft" in texts

    def test_newest_first_ordering(self, stash):
        stash.stash("oldest")
        stash.stash("middle")
        stash.stash("newest")
        assert [e.text for e in stash.items] == ["newest", "middle", "oldest"]
        # Default pop takes the most recent — the "undo my last Ctrl+S" case.
        assert stash.pop()[0] == "newest"

    def test_two_items_opens_panel_instead_of_guessing(self, stash):
        """With 2+ drafts, Ctrl+S must not silently pick one."""
        stash.stash("first")
        stash.stash("second")
        action, payload = resolve_ctrl_s(stash, "")
        assert action == ACTION_OPEN_PANEL
        assert payload is None
        assert stash.panel_open is True
        # Nothing was consumed.
        assert len(stash) == 2

    def test_stash_cap_drops_oldest_not_newest(self):
        s = PromptStash(max_items=3, clock=_FakeClock())
        for i in range(5):
            s.stash(f"draft {i}")
        assert len(s) == 3
        assert [e.text for e in s.items] == ["draft 4", "draft 3", "draft 2"]

    def test_default_cap_is_bounded(self, stash):
        for i in range(MAX_STASH_ITEMS + 10):
            stash.stash(f"d{i}")
        assert len(stash) == MAX_STASH_ITEMS


# -------------------------------------------------------------- indicator state


class TestIndicatorState:
    def test_empty_stash_has_no_indicator(self, stash):
        assert stash.indicator() == ""
        assert stash.placeholder_hint() == ""
        assert bool(stash) is False

    def test_single_item_indicator(self, stash):
        stash.stash("draft")
        assert stash.indicator() == "📌 1"
        assert bool(stash) is True

    def test_count_grows_with_stash(self, stash):
        stash.stash("a")
        assert stash.indicator() == "📌 1"
        stash.stash("b")
        assert stash.indicator() == "📌 2"
        stash.stash("c")
        assert stash.indicator() == "📌 3"

    def test_indicator_marks_open_panel(self, stash):
        stash.stash("a")
        stash.stash("b")
        stash.open_panel()
        assert stash.indicator() == "📌 2 ▲"
        stash.close_panel()
        assert stash.indicator() == "📌 2"

    def test_indicator_clears_after_restoring_last_item(self, stash):
        stash.stash("only")
        stash.pop()
        assert stash.indicator() == ""

    def test_placeholder_hint_single_shows_preview(self, stash):
        stash.stash("write the migration guide")
        hint = stash.placeholder_hint()
        assert "Ctrl+S" in hint
        assert "write the migration guide" in hint

    def test_placeholder_hint_multi_shows_count(self, stash):
        stash.stash("a")
        stash.stash("b")
        stash.stash("c")
        assert stash.placeholder_hint() == "Ctrl+S to browse 3 stashed drafts"

    def test_clear_resets_all_state(self, stash):
        stash.stash("a")
        stash.stash("b")
        stash.open_panel()
        stash.clear()
        assert len(stash) == 0
        assert stash.panel_open is False
        assert stash.panel_cursor == 0
        assert stash.indicator() == ""


# ----------------------------------------------------------------- previewing


class TestBuildPreview:
    def test_empty_text(self):
        assert build_preview("") == ""

    def test_single_line_passthrough(self):
        assert build_preview("hello world") == "hello world"

    def test_newlines_collapse_to_marker(self):
        assert build_preview("a\nb") == "a ⏎ b"

    def test_crlf_normalized(self):
        assert build_preview("a\r\nb") == "a ⏎ b"

    def test_preview_is_always_single_line(self):
        preview = build_preview("x\n" * 30, width=200)
        assert "\n" not in preview

    def test_long_text_ellipsized_to_width(self):
        preview = build_preview("y" * 500, width=20)
        assert len(preview) == 20
        assert preview.endswith("…")

    def test_whitespace_runs_collapsed(self):
        assert build_preview("a     b\t\tc") == "a b c"


class TestStashEntry:
    def test_as_dict_shape_matches_panel_renderer(self, stash):
        stash.stash("draft text")
        row = stash.panel_rows()[0]
        # _render_stash_panel indexes these exact keys.
        assert set(row) >= {"text", "images", "stashed_at", "preview"}
        assert row["text"] == "draft text"
        assert row["preview"] == "draft text"

    def test_as_dict_copies_images(self):
        imgs = ["/tmp/a.png"]
        entry = StashEntry(text="t", images=imgs)
        entry.as_dict()["images"].append("/tmp/b.png")
        assert imgs == ["/tmp/a.png"]

    def test_stashed_at_uses_injected_clock(self):
        clock = _FakeClock(start=500.0)
        s = PromptStash(clock=clock)
        s.stash("first")
        clock.advance(60)
        s.stash("second")
        entries = s.items
        assert entries[0].stashed_at == 560.0  # newest first
        assert entries[1].stashed_at == 500.0


# --------------------------------------------------------------- panel browsing


class TestPanelBrowsing:
    @pytest.fixture
    def three(self):
        s = PromptStash(clock=_FakeClock())
        s.stash("oldest")
        s.stash("middle")
        s.stash("newest")
        s.open_panel()
        return s

    def test_open_panel_starts_at_top(self, three):
        assert three.panel_open is True
        assert three.panel_cursor == 0

    def test_cursor_moves_and_clamps(self, three):
        assert three.move_cursor(1) == 1
        assert three.move_cursor(1) == 2
        # Clamped at the bottom — no wraparound, no IndexError.
        assert three.move_cursor(1) == 2
        assert three.move_cursor(-5) == 0

    def test_restore_at_cursor_picks_highlighted_entry(self, three):
        three.move_cursor(1)  # "middle"
        result = three.restore_at_cursor()
        assert result == ("middle", [])
        assert three.panel_open is False
        assert [e.text for e in three.items] == ["newest", "oldest"]

    def test_delete_at_cursor_removes_only_that_entry(self, three):
        three.move_cursor(1)
        assert three.delete_at_cursor() is True
        assert [e.text for e in three.items] == ["newest", "oldest"]
        assert three.panel_open is True

    def test_delete_last_row_reclamps_cursor(self, three):
        three.move_cursor(2)  # bottom row
        three.delete_at_cursor()
        assert three.panel_cursor == 1  # clamped into the shortened list

    def test_deleting_everything_closes_panel(self, three):
        for _ in range(3):
            three.delete_at_cursor()
        assert len(three) == 0
        assert three.panel_open is False
        assert three.panel_cursor == 0

    def test_new_stash_closes_open_panel(self, three):
        three.stash("brand new")
        assert three.panel_open is False
        assert three.panel_cursor == 0

    def test_panel_rows_ordered_newest_first(self, three):
        assert [r["text"] for r in three.panel_rows()] == [
            "newest",
            "middle",
            "oldest",
        ]

    def test_pop_out_of_range_is_none(self, three):
        assert three.pop(99) is None
        assert three.pop(-1) is None
        assert len(three) == 3


# ------------------------------------------------------ resolve_ctrl_s table


class TestResolveCtrlS:
    def test_empty_buffer_empty_stash_is_noop(self, stash):
        assert resolve_ctrl_s(stash, "") == (ACTION_NOOP, None)

    def test_whitespace_buffer_empty_stash_is_noop(self, stash):
        assert resolve_ctrl_s(stash, "   \n  ") == (ACTION_NOOP, None)

    def test_content_stashes(self, stash):
        action, payload = resolve_ctrl_s(stash, "some draft")
        assert (action, payload) == (ACTION_STASHED, None)
        assert len(stash) == 1

    def test_open_panel_then_ctrl_s_closes_it(self, stash):
        stash.stash("a")
        stash.stash("b")
        stash.open_panel()
        action, payload = resolve_ctrl_s(stash, "")
        assert (action, payload) == (ACTION_CLOSE_PANEL, None)
        assert stash.panel_open is False

    def test_close_panel_takes_priority_over_stashing(self, stash):
        """With the panel open, Ctrl+S closes it rather than stashing text."""
        stash.stash("a")
        stash.stash("b")
        stash.open_panel()
        action, _ = resolve_ctrl_s(stash, "text the user typed")
        assert action == ACTION_CLOSE_PANEL
        assert len(stash) == 2  # nothing new pushed

    def test_images_only_buffer_stashes(self, stash):
        action, _ = resolve_ctrl_s(stash, "", ["/tmp/x.png"])
        assert action == ACTION_STASHED

    def test_whitespace_only_buffer_with_stash_restores(self, stash):
        """Whitespace-only counts as empty, so the restore half fires."""
        stash.stash("real draft")
        action, payload = resolve_ctrl_s(stash, "   ")
        assert action == ACTION_RESTORED
        assert payload == ("real draft", [])

    def test_stash_pop_stash_pop_cycle(self, stash):
        for text in ("one", "two\nlines", "three\n\nparas"):
            assert resolve_ctrl_s(stash, text)[0] == ACTION_STASHED
            action, payload = resolve_ctrl_s(stash, "")
            assert action == ACTION_RESTORED
            assert payload is not None
            assert payload[0] == text
            assert len(stash) == 0


# --------------------------------------------------------------------- ages


class TestAgeFormatting:
    def test_age_reflects_injected_clock(self):
        clock = _FakeClock()
        s = PromptStash(clock=clock)
        s.stash("draft")
        entry = s.peek()
        assert entry is not None
        clock.advance(120)
        assert clock() - entry.stashed_at == 120
