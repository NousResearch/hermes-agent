"""Sparse character <-> UTF-8 byte offset index for one subject.

``marks[k]`` is the byte offset of character ``k * 256``. A conversion looks up the nearest mark
and lets RE2's helper walk at most 256 characters, so it costs O(1) instead of O(offset).

The index grows on demand and only as far as the largest offset asked for, doubling each time,
so one ``search(text, pos)`` near the start of a large text never pays for the whole text.
State is one tuple replaced atomically: concurrent growth from two threads yields two valid
prefixes and either may win. Stdlib-only; ``bind`` receives the two engine helpers.
"""
from __future__ import annotations

from bisect import bisect_right

STEP_BITS = 8
STEP = 1 << STEP_BITS
_NEAR = 4 * STEP  # bytes: at most as far as one mark lookup would walk (4 bytes per character)
to_bytes = to_chars = None


def bind(char_len_to_bytes, bytes_to_char_len) -> None:
    """``CharLenToBytes(buffer, start, CHAR COUNT) -> byte count`` and
    ``BytesToCharLen(buffer, start, END OFFSET) -> char count`` from the engine."""
    global to_bytes, to_chars
    to_bytes, to_chars = char_len_to_bytes, bytes_to_char_len


class Offsets:
    __slots__ = ("_last", "_state", "data")

    def __init__(self, data):
        self.data = data
        self._state = ((0,), False)  # (marks, whether the last mark is within STEP chars of the end)
        self._last = (0, 0)  # last (byte offset, character index) converted: forward passes stay cheap

    def _grow(self, enough) -> tuple:
        """Extend the marks until ``enough(marks)`` or the end of the text; at least double."""
        marks, done = self._state
        if done:
            return marks
        data, size = self.data, len(self.data)
        new, last, floor = list(marks), marks[-1], 2 * len(marks)
        while len(new) < floor or not enough(new):
            last += to_bytes(data, last, STEP)
            if last >= size:  # fewer than STEP characters were left (or exactly STEP: no mark needed)
                done = True
                break
            new.append(last)
        marks = tuple(new)
        self._state = (marks, done)
        return marks

    def byte(self, char: int) -> int:
        """UTF-8 byte offset of character index ``char`` (``0 <= char <= len(text)``)."""
        index = char >> STEP_BITS
        marks = self._state[0]
        if index >= len(marks):
            marks = self._grow(lambda new: len(new) > index)
            if index >= len(marks):
                index = len(marks) - 1
        base = marks[index]
        rest = char - (index << STEP_BITS)
        return base + to_bytes(self.data, base, rest) if rest else base

    def char(self, offset: int) -> int:
        """Character index of UTF-8 byte ``offset`` (a character boundary)."""
        base, chars = self._last  # one tuple read: safe against a concurrent writer
        ahead = offset - base
        if not 0 <= ahead <= _NEAR:  # not just ahead of the last conversion: go through the marks
            marks = self._state[0]
            if offset > marks[-1]:
                marks = self._grow(lambda new: new[-1] >= offset)
            index = bisect_right(marks, offset) - 1
            base = marks[index]
            chars = index << STEP_BITS
            ahead = offset - base
        if ahead:
            chars += to_chars(self.data, base, offset)
            self._last = (offset, chars)
        return chars
