"""Tests for multiline paste collapsing and expansion in the CLI.

Covers issue #2404: preserve user input when pasting multiline text,
treat paste reference blocks as solid objects for cursor navigation,
and expand multiple paste references on submit.
"""

import os
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASTE_REF_RE = re.compile(r'\[Pasted text #\d+: \d+ lines \u2192 (.+?)\]')
PASTE_REF_NAV_RE = re.compile(r'\[Pasted text #\d+: \d+ lines \u2192 [^\]]+\]')


class FakeBuffer:
    """Minimal prompt_toolkit Buffer stand-in for paste tests."""

    def __init__(self, text="", cursor_position=None):
        self.text = text
        self.cursor_position = cursor_position if cursor_position is not None else len(text)
        self.on_text_changed = []

    def insert_text(self, data):
        before = self.text[:self.cursor_position]
        after = self.text[self.cursor_position:]
        self.text = before + data + after
        self.cursor_position += len(data)
        for cb in self.on_text_changed:
            cb(self)


class TestPasteCollapsePreservesUserText:
    """handle_paste should save only the pasted content to file and insert
    a placeholder at the cursor, keeping any existing user text intact."""

    def test_existing_text_preserved_after_paste_collapse(self, tmp_path):
        buf = FakeBuffer("Here is my error:")
        buf.cursor_position = len(buf.text)

        pasted = "\n".join(f"line {i}" for i in range(10))
        line_count = pasted.count('\n')
        assert line_count >= 5

        paste_file = tmp_path / "paste_1_120000.txt"
        paste_file.write_text(pasted, encoding="utf-8")
        placeholder = f"[Pasted text #1: {line_count + 1} lines \u2192 {paste_file}]"

        prefix = "\n" if buf.cursor_position > 0 and buf.text[buf.cursor_position - 1] != '\n' else ""
        buf.insert_text(prefix + placeholder)

        assert buf.text.startswith("Here is my error:\n")
        assert "[Pasted text #1:" in buf.text

    def test_paste_without_existing_text_no_prefix_newline(self, tmp_path):
        buf = FakeBuffer("")

        paste_file = tmp_path / "paste_1_120000.txt"
        paste_file.write_text("line\n" * 6, encoding="utf-8")
        placeholder = f"[Pasted text #1: 7 lines \u2192 {paste_file}]"

        prefix = "\n" if buf.cursor_position > 0 and buf.text[buf.cursor_position - 1] != '\n' else ""
        buf.insert_text(prefix + placeholder)

        assert buf.text == placeholder
        assert not buf.text.startswith("\n")

    def test_multiple_pastes_accumulate(self, tmp_path):
        buf = FakeBuffer("context:")
        buf.cursor_position = len(buf.text)

        for i in range(1, 3):
            pasted = "\n".join(f"block{i} line {j}" for j in range(6))
            paste_file = tmp_path / f"paste_{i}.txt"
            paste_file.write_text(pasted, encoding="utf-8")
            placeholder = f"[Pasted text #{i}: 6 lines \u2192 {paste_file}]"
            prefix = "\n" if buf.cursor_position > 0 and buf.text[buf.cursor_position - 1] != '\n' else ""
            buf.insert_text(prefix + placeholder)

        refs = PASTE_REF_NAV_RE.findall(buf.text)
        assert len(refs) == 2
        assert buf.text.startswith("context:")


class TestPasteRefExpansion:
    """Submit-time expansion must handle mixed user text + multiple paste refs."""

    def test_single_ref_expanded(self, tmp_path):
        content = "error log line 1\nerror log line 2\n"
        paste_file = tmp_path / "paste_1.txt"
        paste_file.write_text(content, encoding="utf-8")

        user_input = f"[Pasted text #1: 3 lines \u2192 {paste_file}]"
        expanded = PASTE_REF_RE.sub(
            lambda m: Path(m.group(1)).read_text(encoding="utf-8") if Path(m.group(1)).exists() else m.group(0),
            user_input,
        )
        assert expanded == content

    def test_user_text_plus_ref_expanded(self, tmp_path):
        content = "traceback line\n" * 5
        paste_file = tmp_path / "paste_1.txt"
        paste_file.write_text(content, encoding="utf-8")

        user_input = f"Fix this error:\n[Pasted text #1: 6 lines \u2192 {paste_file}]"
        expanded = PASTE_REF_RE.sub(
            lambda m: Path(m.group(1)).read_text(encoding="utf-8") if Path(m.group(1)).exists() else m.group(0),
            user_input,
        )
        assert expanded.startswith("Fix this error:\n")
        assert "traceback line" in expanded
        assert "[Pasted text" not in expanded

    def test_multiple_refs_expanded(self, tmp_path):
        for i in range(1, 3):
            (tmp_path / f"paste_{i}.txt").write_text(f"content_{i}\n", encoding="utf-8")

        user_input = (
            f"Compare these:\n"
            f"[Pasted text #1: 2 lines \u2192 {tmp_path / 'paste_1.txt'}]\n"
            f"vs\n"
            f"[Pasted text #2: 2 lines \u2192 {tmp_path / 'paste_2.txt'}]"
        )
        expanded = PASTE_REF_RE.sub(
            lambda m: Path(m.group(1)).read_text(encoding="utf-8") if Path(m.group(1)).exists() else m.group(0),
            user_input,
        )
        assert "Compare these:" in expanded
        assert "content_1" in expanded
        assert "content_2" in expanded
        assert "vs" in expanded
        assert "[Pasted text" not in expanded

    def test_missing_file_keeps_placeholder(self):
        user_input = "[Pasted text #1: 5 lines \u2192 /nonexistent/paste_1.txt]"
        expanded = PASTE_REF_RE.sub(
            lambda m: Path(m.group(1)).read_text(encoding="utf-8") if Path(m.group(1)).exists() else m.group(0),
            user_input,
        )
        assert expanded == user_input

    def test_visible_user_text_extraction(self, tmp_path):
        paste_file = tmp_path / "paste_1.txt"
        paste_file.write_text("data\n", encoding="utf-8")

        user_input = f"Here is the log:\n[Pasted text #1: 2 lines \u2192 {paste_file}]\nPlease fix it"
        split_parts = PASTE_REF_RE.split(user_input)
        visible = " ".join(
            split_parts[i].strip() for i in range(0, len(split_parts), 2) if split_parts[i].strip()
        )
        assert "Here is the log:" in visible
        assert "Please fix it" in visible
        assert "Pasted text" not in visible


class TestCursorNavigation:
    """Left/right arrow should jump over paste reference blocks as solid objects."""

    def _make_buffer_with_ref(self, tmp_path):
        paste_file = tmp_path / "paste_1.txt"
        paste_file.write_text("x\n" * 6, encoding="utf-8")
        placeholder = f"[Pasted text #1: 7 lines \u2192 {paste_file}]"
        text = f"Hello\n{placeholder}"
        return text, placeholder

    def test_right_arrow_jumps_over_ref(self, tmp_path):
        text, placeholder = self._make_buffer_with_ref(tmp_path)
        pos = text.index('[')
        m = PASTE_REF_NAV_RE.match(text, pos)
        assert m is not None
        new_pos = m.end()
        assert new_pos == len(text)

    def test_left_arrow_jumps_back_over_ref(self, tmp_path):
        text, placeholder = self._make_buffer_with_ref(tmp_path)
        pos = len(text)  # cursor at end
        text_before = text[:pos]
        jump_target = None
        for m in PASTE_REF_NAV_RE.finditer(text_before):
            if m.end() == pos:
                jump_target = m.start()
        assert jump_target is not None
        assert jump_target == text.index('[')

    def test_normal_movement_when_no_ref(self):
        text = "Hello world"
        pos = 5
        span = _find_paste_ref_around(text, pos)
        assert span is None
        new_pos = pos + 1
        assert text[new_pos] == 'w'

    def test_left_from_middle_of_ref_jumps_to_start(self, tmp_path):
        """If cursor somehow lands inside a ref (e.g. Home key), Left escapes."""
        text, placeholder = self._make_buffer_with_ref(tmp_path)
        ref_start = text.index('[')
        mid = ref_start + len(placeholder) // 2

        span = _find_paste_ref_around(text, mid - 1)
        assert span is not None
        assert span[0] == ref_start

    def test_right_from_middle_of_ref_jumps_to_end(self, tmp_path):
        """If cursor somehow lands inside a ref (e.g. Home key), Right escapes."""
        text, placeholder = self._make_buffer_with_ref(tmp_path)
        ref_start = text.index('[')
        mid = ref_start + len(placeholder) // 2

        span = _find_paste_ref_around(text, mid)
        assert span is not None
        assert span[1] == ref_start + len(placeholder)

    def test_right_at_end_of_ref_does_normal_move(self, tmp_path):
        """Cursor at m.end() (past the ref) should move normally, not get stuck."""
        text, placeholder = self._make_buffer_with_ref(tmp_path)
        ref_end = text.index('[') + len(placeholder)
        span = _find_paste_ref_around(text, ref_end)
        assert span is None


def _find_paste_ref_around(text, pos):
    """Mirrors the helper in cli.py — returns (start, end) of the paste
    reference spanning *pos*, or None."""
    for m in PASTE_REF_NAV_RE.finditer(text):
        if m.start() <= pos < m.end():
            return m.start(), m.end()
    return None


class TestBackspaceDeleteSolidObject:
    """Backspace/Delete should remove entire paste reference blocks as solid objects."""

    def _make_text_with_ref(self, tmp_path, idx=1):
        paste_file = tmp_path / f"paste_{idx}.txt"
        paste_file.write_text("x\n" * 6, encoding="utf-8")
        placeholder = f"[Pasted text #{idx}: 7 lines \u2192 {paste_file}]"
        return placeholder

    def test_backspace_at_end_of_ref_deletes_whole_block(self, tmp_path):
        placeholder = self._make_text_with_ref(tmp_path)
        text = f"Hello\n{placeholder}"
        pos = len(text)  # cursor at end, right after the ref

        span = _find_paste_ref_around(text, pos - 1)
        assert span is not None
        start, end = span
        before = text[:start]
        if start > 0 and before.endswith('\n'):
            before = before[:-1]
        result = before + text[end:]
        assert result == "Hello"
        assert "[Pasted text" not in result

    def test_delete_at_start_of_ref_deletes_whole_block(self, tmp_path):
        placeholder = self._make_text_with_ref(tmp_path)
        text = f"Hello\n{placeholder}\nWorld"
        pos = text.index('[')

        span = _find_paste_ref_around(text, pos)
        assert span is not None
        start, end = span
        after = text[end:]
        if after.startswith('\n'):
            after = after[1:]
        result = text[:start] + after
        assert result == "Hello\nWorld"
        assert "[Pasted text" not in result

    def test_backspace_on_normal_text_unaffected(self):
        text = "Hello world"
        pos = 5
        span = _find_paste_ref_around(text, pos - 1)
        assert span is None

    def test_delete_on_normal_text_unaffected(self):
        text = "Hello world"
        pos = 5
        span = _find_paste_ref_around(text, pos)
        assert span is None

    def test_backspace_between_two_refs_deletes_correct_one(self, tmp_path):
        ref1 = self._make_text_with_ref(tmp_path, 1)
        ref2 = self._make_text_with_ref(tmp_path, 2)
        text = f"Start\n{ref1}\nMiddle\n{ref2}\nEnd"
        pos = text.index(ref2) + len(ref2)  # cursor at end of ref2

        span = _find_paste_ref_around(text, pos - 1)
        assert span is not None
        start, end = span
        assert text[start:end] == ref2

    def test_delete_with_ref_preserves_surrounding_text(self, tmp_path):
        placeholder = self._make_text_with_ref(tmp_path)
        text = f"Before\n{placeholder}\nAfter"
        pos = text.index('[')

        span = _find_paste_ref_around(text, pos)
        start, end = span
        after = text[end:]
        if after.startswith('\n'):
            after = after[1:]
        result = text[:start] + after
        assert result == "Before\nAfter"


class TestOnTextChangedFallback:
    """_on_text_changed should skip when _paste_just_collapsed flag is set."""

    def test_flag_prevents_double_collapse(self):
        _paste_just_collapsed = [True]
        _prev_text_len = [0]
        collapsed = [False]

        def _on_text_changed(buf):
            text = buf.text
            chars_added = len(text) - _prev_text_len[0]
            _prev_text_len[0] = len(text)
            if _paste_just_collapsed[0]:
                _paste_just_collapsed[0] = False
                return
            line_count = text.count('\n')
            if line_count >= 5 and chars_added > 1:
                collapsed[0] = True

        buf = FakeBuffer("Hello\n[Pasted text #1: 7 lines \u2192 /tmp/f.txt]")
        _on_text_changed(buf)

        assert not collapsed[0]
        assert not _paste_just_collapsed[0]

    def test_fallback_still_collapses_without_flag(self):
        _paste_just_collapsed = [False]
        _prev_text_len = [0]
        collapsed = [False]

        def _on_text_changed(buf):
            text = buf.text
            chars_added = len(text) - _prev_text_len[0]
            _prev_text_len[0] = len(text)
            if _paste_just_collapsed[0]:
                _paste_just_collapsed[0] = False
                return
            line_count = text.count('\n')
            if line_count >= 5 and chars_added > 1:
                collapsed[0] = True

        buf = FakeBuffer("\n".join(f"line {i}" for i in range(10)))
        _on_text_changed(buf)

        assert collapsed[0]
