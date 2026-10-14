"""The tool-progress repeat counter must never corrupt a fenced code block (#138148).

A ``(xN)`` suffix appended to the LAST line of a multi-line fenced block lands on the closing
fence; trailing text invalidates a closing fence for anchored parsers (Telegram MarkdownV2,
CommonMark), which then render the backticks escaped. The writer puts the counter on its own
line for fenced blocks and keeps it inline for single-line messages.
"""

from types import SimpleNamespace

from gateway.run_turn_runner import TurnRunner, _suffix_repeat_counter


class TestSuffixRepeatCounter:

    def test_single_line_message_keeps_counter_inline(self):
        assert _suffix_repeat_counter("\U0001f4bb terminal ssh fed...", 2) == "\U0001f4bb terminal ssh fed... (\u00d72)"

    def test_fenced_block_gets_counter_on_own_line(self):
        block = "\U0001f4bb terminal\n```\nssh -F ~/.ssh/config fed...\n```"
        assert _suffix_repeat_counter(block, 4) == block + "\n(\u00d74)"

    def test_fenced_block_with_trailing_whitespace(self):
        block = "```\ncmd\n```  "
        assert _suffix_repeat_counter(block, 2) == block + "\n(\u00d72)"

    def test_multiline_non_fence_text_keeps_counter_inline(self):
        text = "line one\nline two"
        assert _suffix_repeat_counter(text, 3) == "line one\nline two (\u00d73)"


class TestProgressAbsorb:

    def _runner(self):
        return TurnRunner.__new__(TurnRunner)

    def test_dedup_tuple_on_fenced_block_moves_counter_off_fence(self):
        st = SimpleNamespace(progress_lines=["\U0001f4bb terminal\n```\nssh fed...\n```"])
        out = self._runner()._progress_absorb(st, ("__dedup__", "\U0001f4bb terminal\n```\nssh fed...\n```", 3))
        assert out == "\U0001f4bb terminal\n```\nssh fed...\n```\n(\u00d74)"
        assert st.progress_lines[-1] == out

    def test_dedup_tuple_on_single_line_keeps_counter_inline(self):
        st = SimpleNamespace(progress_lines=["\U0001f4bb terminal ssh fed..."])
        out = self._runner()._progress_absorb(st, ("__dedup__", "\U0001f4bb terminal ssh fed...", 1))
        assert out == "\U0001f4bb terminal ssh fed... (\u00d72)"

    def test_dedup_tuple_with_empty_buffer_returns_base(self):
        st = SimpleNamespace(progress_lines=[])
        assert self._runner()._progress_absorb(st, ("__dedup__", "msg", 5)) == "msg"

    def test_plain_line_appends(self):
        st = SimpleNamespace(progress_lines=["first"])
        assert self._runner()._progress_absorb(st, "second") == "second"
        assert st.progress_lines == ["first", "second"]
