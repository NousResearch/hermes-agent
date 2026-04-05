"""Tests for agent/rich_output.py — syntax highlighting, diff rendering, code block detection."""

import re
import pytest
from unittest.mock import patch

from agent.rich_output import (
    _DIFF_BG_ADD_HL,
    _DIFF_BG_DEL_HL,
    _DIFF_MAX_LINES,
    DiffRenderer,
    FilePathFormatter,
    LanguageDetector,
    StreamingBlockBuffer,
    StreamingCodeBlockHighlighter,
    SyntaxHighlighter,
    _highlight_inline_code,
    _NUM_RE,
    _SETEXT_H1_RE,
    _SETEXT_H2_RE,
    _TABLE_STRICT_ROW_RE,
    _intra_diff,
    _parse_diff_filename,
    _split_row,
    apply_block_line,
    apply_inline_markdown,
    clean_command_output,
    format_response,
    render_stateful_blocks,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _seg_color(seg) -> str:
    """Return the colour name of a Text segment as a plain string."""
    return seg.style.color.name  # rich.color.Color.name is the canonical name string


def _renderables(diff: str) -> list:
    """Return the list of Text children from DiffRenderer._style().

    Uses ``group.renderables`` which is an internal Rich attribute.  If a
    future Rich upgrade removes it, switch to rendering the Group to a Console
    buffer and inspecting the output instead.
    """
    return list(DiffRenderer()._style(diff.splitlines()).renderables)


def _strip(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---------------------------------------------------------------------------
# LanguageDetector
# ---------------------------------------------------------------------------

class TestLanguageDetector:
    def setup_method(self):
        self.det = LanguageDetector()

    def test_detect_python_from_extension(self):
        assert self.det.detect_from_filename("foo.py") == "python"

    def test_detect_typescript_from_extension(self):
        assert self.det.detect_from_filename("app.ts") == "typescript"

    def test_detect_unknown_extension_returns_none(self):
        assert self.det.detect_from_filename("file.xyz") is None

    def test_detect_dockerfile(self):
        assert self.det.detect_from_filename("Dockerfile") == "dockerfile"

    def test_detect_makefile(self):
        assert self.det.detect_from_filename("Makefile") == "makefile"

    def test_detect_python_from_content(self):
        code = "def hello():\n    return 42\n"
        assert self.det.detect_from_content(code) == "python"

    def test_detect_javascript_from_content(self):
        code = "const x = require('fs');\nmodule.exports = x;\n"
        assert self.det.detect_from_content(code) == "javascript"

    def test_detect_bash_from_content(self):
        code = "#!/bin/bash\nif [ -f foo ]; then echo hi; fi\n"
        assert self.det.detect_from_content(code) == "bash"

    def test_detect_empty_content_returns_none(self):
        assert self.det.detect_from_content("") is None
        assert self.det.detect_from_content("   \n  ") is None

    def test_detect_prefers_filename_over_content(self):
        # .js extension should win even if content looks like Python
        assert self.det.detect("def foo(): pass", filename="script.js") == "javascript"


# ---------------------------------------------------------------------------
# FilePathFormatter
# ---------------------------------------------------------------------------

class TestFilePathFormatter:
    def test_python_icon(self):
        assert FilePathFormatter.get_file_icon("main.py") == "🐍"

    def test_rust_icon(self):
        assert FilePathFormatter.get_file_icon("lib.rs") == "🦀"

    def test_unknown_extension_fallback(self):
        assert FilePathFormatter.get_file_icon("weird.xyz") == "📄"

    def test_titled_includes_icon_and_path(self):
        result = FilePathFormatter.titled("main.py", compact=False)
        assert "🐍" in result
        assert "main.py" in result

    def test_format_path_compact_returns_relative(self, tmp_path):
        file_path = str(tmp_path / "sub" / "foo.py")
        result = FilePathFormatter.format_path(file_path, compact=True, cwd=str(tmp_path))
        assert result == "sub/foo.py"

    def test_format_path_verbose_returns_full(self, tmp_path):
        file_path = str(tmp_path / "foo.py")
        result = FilePathFormatter.format_path(file_path, compact=False)
        assert result == file_path


# ---------------------------------------------------------------------------
# SyntaxHighlighter
# ---------------------------------------------------------------------------

class TestSyntaxHighlighter:
    def setup_method(self):
        self.hl = SyntaxHighlighter()

    def test_to_ansi_returns_string(self):
        result = self.hl.to_ansi("x = 1", language="python")
        assert isinstance(result, str)
        assert "x" in result

    def test_to_ansi_contains_ansi_codes(self):
        result = self.hl.to_ansi("def foo(): pass", language="python")
        # Should contain at least one ANSI escape sequence
        assert "\033[" in result

    def test_to_markup_returns_string(self):
        result = self.hl.to_markup("x = 1", language="python")
        assert isinstance(result, str)

    def test_to_ansi_empty_string(self):
        result = self.hl.to_ansi("")
        assert isinstance(result, str)

    def test_to_ansi_fallback_on_unknown_language(self):
        # Unknown language should not crash
        result = self.hl.to_ansi("some text", language="nonexistentlang123")
        assert isinstance(result, str)
        assert "some text" in result

    def test_to_ansi_no_rogue_backslash_before_bracket(self):
        # Haskell type signatures like [Integer] must render as [Integer], not
        # [Integer\] — the old code escaped ] to \] which Rich renders literally.
        import re
        _strip = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)
        result = _strip(self.hl.to_ansi("fib :: [Integer]", language="haskell"))
        assert "[Integer]" in result, f"Expected [Integer], got: {result!r}"
        assert r"\]" not in result, f"Rogue backslash-bracket: {result!r}"

    def test_to_ansi_lambda_backslash_no_leaking_close_tag(self):
        # Haskell lambda \ is tokenised by Pygments as Name.Function (bold
        # yellow). Without backslash-doubling the \ combined with the [/bold
        # yellow] close tag to form \[ (Rich's escape), leaking the tag text.
        import re
        _strip = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)
        result = _strip(self.hl.to_ansi(r"fact = foldl (\a b -> a * b) 1", language="haskell"))
        assert "[/bold" not in result, f"Leaked close tag: {result!r}"

    def test_to_ansi_brackets_not_interpreted_as_rich_markup(self):
        # Rich markup tags inside code string literals must survive as literal
        # text — brackets like [bold green] and [/bold green] should appear
        # verbatim in output, not vanish because Rich consumed them as tags.
        import re
        _strip = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)
        result = _strip(self.hl.to_ansi('printf "[bold green]hello[/bold green]\\n"', language="haskell"))
        assert "hello" in result
        assert "[bold green]" in result, f"Bracket text vanished: {result!r}"
        assert "[/bold green]" in result, f"Bracket text vanished: {result!r}"

    def test_to_markup_brackets_escaped(self):
        # to_markup output goes to Console.print() — [ must be \[-escaped so
        # Rich never interprets code content as formatting markup.
        result = self.hl.to_markup('x = "[bold]text[/bold]"', language="python")
        assert "[bold]" not in result or r"\[bold" in result


# ---------------------------------------------------------------------------
# DiffRenderer
# ---------------------------------------------------------------------------

class TestDiffRenderer:
    def setup_method(self):
        self.dr = DiffRenderer()

    def test_to_lines_returns_list(self):
        diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
        lines = self.dr.to_lines(diff)
        assert isinstance(lines, list)
        assert len(lines) > 0

    def test_to_lines_contains_content(self):
        diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
        lines = self.dr.to_lines(diff)
        combined = "\n".join(lines)
        assert "old" in combined
        assert "new" in combined

    def test_from_content_produces_renderable(self):
        from rich.console import Group
        result = self.dr.from_content("old line\n", "new line\n", file_path="test.py")
        assert isinstance(result, Group)

    def test_from_unified_empty_diff(self):
        # Empty diff should not crash
        result = self.dr.to_lines("")
        assert isinstance(result, list)

    def test_file_header_formatted(self):
        diff = "--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1 @@\n-x\n+y\n"
        lines = self.dr.to_lines(diff)
        combined = "\n".join(lines)
        assert "main.py" in combined

    def test_to_lines_does_not_crash_on_malformed_diff(self):
        result = self.dr.to_lines("not a real diff at all\njust some text\n")
        assert isinstance(result, list)

    def test_del_line_numbers_stay_in_context_scale(self):
        # Regression: when ln_old runs ahead of ln_new (e.g. a net-deletion earlier
        # in the hunk), deletion line numbers must NOT jump above the surrounding
        # context numbers.  All three of context, del, and add should use the same
        # new-file scale so paired lines share the same number.
        # Hunk @@ -59,16 +58,8 @@: after 3 context lines (58,59,60) ln_old=62 but
        # ln_new=61 — before the fix, the first del showed as "62" skipping "61".
        diff = (
            "--- a/f.md\n+++ b/f.md\n"
            "@@ -59,16 +58,8 @@\n"
            " ctx_a\n ctx_b\n ctx_c\n"          # context → last shown: 60
            "-del1\n-del2\n-del3\n"              # dels should be 61, 62, 63
            "+add1\n+add2\n"                     # adds should be 61, 62
        )
        renderables = _renderables(diff)
        import re
        texts = [re.sub(r"\s+", " ", r.plain).strip() for r in renderables]
        # First deletion must start at 61 (immediately after context line 60)
        del_lines = [t for t in texts if "- del" in t]
        assert del_lines, "expected deletion lines in output"
        first_del_num = int(del_lines[0].split()[0])
        assert first_del_num == 61, (
            f"first deletion line showed {first_del_num}, expected 61 "
            f"(must not jump to ln_old=62 when ln_new=61)"
        )


# ---------------------------------------------------------------------------
# StreamingCodeBlockHighlighter
# ---------------------------------------------------------------------------

class TestStreamingCodeBlockHighlighter:
    def setup_method(self):
        self.hl = StreamingCodeBlockHighlighter()

    def test_plain_lines_pass_through(self):
        # Lines with no inline code are returned verbatim.
        assert self.hl.process_line("Hello world") == "Hello world"
        assert self.hl.process_line("Another line") == "Another line"

    def test_plain_line_with_inline_code_styled(self):
        import re
        result = self.hl.process_line("Use `foo()` here.")
        assert result is not None
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result)
        assert "foo()" in plain
        assert "\033[" in result

    def test_opening_fence_suppressed(self):
        assert self.hl.process_line("```python") is None

    def test_code_lines_buffered(self):
        self.hl.process_line("```python")
        assert self.hl.process_line("x = 1") is None
        assert self.hl.process_line("y = 2") is None

    def test_closing_fence_flushes_highlighted(self):
        self.hl.process_line("```python")
        self.hl.process_line("x = 1")
        result = self.hl.process_line("```")
        assert result is not None
        assert "x" in result

    def test_full_code_block_sequence(self):
        lines = ["Here is code:", "```python", "def foo(): pass", "```", "Done."]
        outputs = []
        for line in lines:
            out = self.hl.process_line(line)
            if out is not None:
                outputs.append(out)
        tail = self.hl.flush()
        if tail:
            outputs.append(tail)

        combined = "\n".join(outputs)
        assert "Here is code:" in combined
        assert "foo" in combined
        assert "Done." in combined

    def test_flush_returns_none_when_no_open_block(self):
        assert self.hl.flush() is None

    def test_flush_returns_content_for_unclosed_block(self):
        self.hl.process_line("```python")
        self.hl.process_line("x = 1")
        result = self.hl.flush()
        assert result is not None
        assert "x" in result

    def test_reset_clears_state(self):
        self.hl.process_line("```python")
        self.hl.process_line("x = 1")
        self.hl.reset()
        assert self.hl.flush() is None
        # Should behave as fresh after reset
        assert self.hl.process_line("normal line") == "normal line"

    def test_multiple_blocks_in_sequence(self):
        lines = [
            "Block one:", "```python", "a = 1", "```",
            "Block two:", "```javascript", "var b = 2;", "```",
        ]
        outputs = [self.hl.process_line(l) for l in lines]
        non_none = [o for o in outputs if o is not None]
        assert len(non_none) == 4  # "Block one:", highlighted, "Block two:", highlighted

    def test_no_language_hint_still_works(self):
        self.hl.process_line("```")
        self.hl.process_line("SELECT * FROM users;")
        result = self.hl.process_line("```")
        assert result is not None
        assert "SELECT" in result

    def test_lang_hint_passed_to_highlighter(self):
        """Opening fence ```python should call to_ansi with language='python'."""
        with patch.object(self.hl._hl, "to_ansi", return_value="highlighted") as mock_ansi:
            self.hl.process_line("```python")
            self.hl.process_line("x = 1")
            self.hl.process_line("```")
        mock_ansi.assert_called_once()
        _, kwargs = mock_ansi.call_args
        assert kwargs.get("language") == "python"

    def test_no_lang_hint_calls_content_detection(self):
        """Opening fence with no hint should fall back to detect_from_content."""
        with patch.object(self.hl._det, "detect_from_content", return_value=None) as mock_det:
            self.hl.process_line("```")
            self.hl.process_line("x = 1")
            self.hl.process_line("```")
        mock_det.assert_called_once()

    def test_four_backtick_fence_opened_and_closed(self):
        """4-backtick opening fence is handled; 4-backtick closing fence closes it."""
        assert self.hl.process_line("````python") is None
        assert self.hl.process_line("x = 1") is None
        result = self.hl.process_line("````")
        assert result is not None
        assert "x" in result

    def test_four_backtick_fence_three_backtick_close_ignored(self):
        """3-backtick closing fence inside a 4-backtick block is buffered, not a close."""
        assert self.hl.process_line("````python") is None
        assert self.hl.process_line("x = 1") is None
        assert self.hl.process_line("```") is None  # still buffering
        result = self.hl.flush()
        assert result is not None
        assert "x" in result

    def test_prose_after_four_backtick_block_rendered(self):
        """Lines after a properly-closed 4-backtick block pass through as prose."""
        self.hl.process_line("````python")
        self.hl.process_line("x = 1")
        self.hl.process_line("````")
        out = self.hl.process_line("plain text")
        assert out == "plain text"


# ---------------------------------------------------------------------------
# format_response
# ---------------------------------------------------------------------------

class TestFormatResponse:
    def test_plain_text_unchanged(self):
        text = "No code here, just text."
        result = format_response(text)
        assert "No code here, just text." in result

    def test_code_block_highlighted(self):
        text = "Here:\n```python\ndef foo(): pass\n```\nDone."
        result = format_response(text)
        assert "foo" in result
        assert "Here:" in result
        assert "Done." in result

    def test_multiple_code_blocks(self):
        text = "First:\n```python\nx = 1\n```\nSecond:\n```javascript\nvar y = 2;\n```"
        result = format_response(text)
        assert "x" in result
        assert "y" in result

    def test_no_code_blocks_returns_original(self):
        text = "Just a response with no fences."
        assert format_response(text) == text

    def test_empty_string(self):
        assert format_response("") == ""

    def test_code_block_without_language(self):
        text = "```\nSELECT * FROM t;\n```"
        result = format_response(text)
        assert "SELECT" in result

    def test_no_lang_hint_calls_content_detection(self):
        """Fence with no lang tag should call LanguageDetector.detect_from_content."""
        from unittest.mock import patch, MagicMock
        with patch("agent.rich_output.LanguageDetector") as MockLD:
            mock_instance = MagicMock()
            mock_instance.detect_from_content.return_value = None
            MockLD.return_value = mock_instance
            format_response("```\nSELECT * FROM t;\n```")
        mock_instance.detect_from_content.assert_called_once()

    def test_fence_delimiters_not_in_output(self):
        """format_response must not include raw ``` in the highlighted output."""
        text = "```python\ndef foo(): pass\n```"
        result = format_response(text)
        import re as _re
        plain = _re.sub(r"\x1b\[[0-9;]*m", "", result)
        for line in plain.splitlines():
            assert not line.strip().startswith("```"), f"fence leaked: {line!r}"

    def test_four_backtick_fence_consumed(self):
        """format_response handles 4-backtick fences via backreference."""
        text = "Intro.\n````python\nx = 1\n````\nDone."
        result = format_response(text)
        assert "Intro." in result
        assert "Done." in result
        assert "x" in result
        import re as _re
        plain = _re.sub(r"\x1b\[[0-9;]*m", "", result)
        for line in plain.splitlines():
            assert not line.strip().startswith("````"), f"4-backtick fence leaked: {line!r}"

    @pytest.mark.parametrize("lang", ["c++", "objective-c", "shell-session", "f#"])
    def test_fence_info_strings_accept_common_punctuation(self, lang):
        plain = _strip(format_response(f"```{lang}\nint x;\n```\n"))
        assert "```" not in plain
        assert "int x;" in plain

    def test_inline_code_in_prose_styled(self):
        """Inline code spans in prose get ANSI styling."""
        text = "Use `foo()` to call it."
        result = format_response(text)
        assert "\033[" in result
        import re as _re
        plain = _re.sub(r"\x1b\[[0-9;]*m", "", result)
        assert "foo()" in plain

    def test_inline_code_not_applied_inside_fenced_block(self):
        """Backtick spans inside fenced code blocks are not double-styled."""
        text = "```python\nx = `foo`\n```"
        result = format_response(text)
        # The fenced block content should not contain the inline-code ANSI prefix
        # (48;5;237 is the inline code background index)
        assert "48;5;237" not in result

    def test_inline_code_preserved_in_plain_text(self):
        """Inline code content survives styling."""
        import re as _re
        text = "The `fmap` function maps over a functor."
        result = format_response(text)
        plain = _re.sub(r"\x1b\[[0-9;]*m", "", result)
        assert "fmap" in plain

    def test_prose_without_backticks_unchanged(self):
        """Plain prose with no backticks is returned verbatim."""
        text = "Just a response with no fences."
        assert format_response(text) == text


# ---------------------------------------------------------------------------
# _highlight_inline_code unit tests
# ---------------------------------------------------------------------------

class TestHighlightInlineCode:
    def _strip(self, s: str) -> str:
        import re
        return re.sub(r"\x1b\[[0-9;]*m", "", s)

    def test_single_span_styled(self):
        result = _highlight_inline_code("Use `foo()` here.")
        assert "\033[" in result
        assert "foo()" in self._strip(result)

    def test_multiple_spans_styled(self):
        result = _highlight_inline_code("`a` and `b`")
        plain = self._strip(result)
        assert "a" in plain and "b" in plain
        assert result.count("\033[48;5;237m") == 2

    def test_no_backticks_unchanged(self):
        text = "No inline code here."
        assert _highlight_inline_code(text) == text

    def test_backtick_content_preserved(self):
        result = _highlight_inline_code("`m :: f (a -> Either e a)`")
        assert "m :: f (a -> Either e a)" in self._strip(result)

    def test_multiline_span_not_matched(self):
        """A backtick span crossing a newline must NOT be treated as inline code."""
        text = "`line one\nline two`"
        result = _highlight_inline_code(text)
        assert result == text  # no ANSI injected across a newline


# ---------------------------------------------------------------------------
# clean_command_output
# ---------------------------------------------------------------------------

class TestCleanCommandOutput:
    def test_strips_venv_paths(self):
        noisy = "/home/user/venv/lib/python3.11/site-packages/foo.py\nActual output"
        result = clean_command_output(noisy)
        assert "site-packages" not in result
        assert "Actual output" in result

    def test_keeps_meaningful_lines(self):
        output = "Build succeeded\n3 tests passed\nDone."
        result = clean_command_output(output)
        assert "Build succeeded" in result
        assert "3 tests passed" in result

    def test_empty_string(self):
        assert clean_command_output("") == ""

    def test_removes_excessive_blank_lines(self):
        output = "line1\n\n\n\n\nline2"
        result = clean_command_output(output)
        assert result.count("\n") < 3


# ---------------------------------------------------------------------------
# _intra_diff unit tests
# ---------------------------------------------------------------------------

class TestIntraDiff:
    # _intra_diff now returns ([del_text], [add_text]) — single-element lists
    # where each element is a Rich Text with spans rather than a list of
    # per-segment Text objects.  Changed regions are marked with a brighter
    # background (bold) instead of a bright foreground colour.

    def test_equal_spans_use_base_colour(self):
        # Identical lines → no changed region → no bright-highlight spans.
        del_segs, add_segs = _intra_diff("abc", "abc")
        del_text, add_text = del_segs[0], add_segs[0]
        assert del_text.plain == "abc"
        assert add_text.plain == "abc"
        # No span should carry the brighter highlight background.
        del_bgs = {sp.style.bgcolor for sp in del_text._spans if sp.style.bgcolor}
        add_bgs = {sp.style.bgcolor for sp in add_text._spans if sp.style.bgcolor}
        from rich.color import Color
        hl_del = Color.parse(_DIFF_BG_DEL_HL)
        hl_add = Color.parse(_DIFF_BG_ADD_HL)
        assert hl_del not in del_bgs
        assert hl_add not in add_bgs

    def test_changed_span_highlighted(self):
        # "foo bar" → "foo baz": only the last char differs.
        del_segs, add_segs = _intra_diff("foo bar", "foo baz")
        del_text, add_text = del_segs[0], add_segs[0]
        from rich.color import Color
        hl_del = Color.parse(_DIFF_BG_DEL_HL)
        hl_add = Color.parse(_DIFF_BG_ADD_HL)
        del_bgs = {sp.style.bgcolor for sp in del_text._spans if sp.style.bgcolor}
        add_bgs = {sp.style.bgcolor for sp in add_text._spans if sp.style.bgcolor}
        assert hl_del in del_bgs, "expected bright del bg on changed span"
        assert hl_add in add_bgs, "expected bright add bg on changed span"
        del_highlighted = any(sp.style.bgcolor == hl_del for sp in del_text._spans)
        assert del_highlighted, "changed del span must be highlighted"

    def test_delete_opcode_no_add_seg(self):
        del_segs, add_segs = _intra_diff("abcXYZ", "abc")
        assert any("XYZ" in s.plain for s in del_segs)
        assert any("abc" in s.plain for s in add_segs)

    def test_insert_opcode_no_del_seg(self):
        del_segs, add_segs = _intra_diff("abc", "abcXYZ")
        assert any("XYZ" in s.plain for s in add_segs)
        assert any("abc" in s.plain for s in del_segs)


# ---------------------------------------------------------------------------
# _parse_diff_filename unit tests
# ---------------------------------------------------------------------------

class TestParseDiffFilename:
    def test_strips_b_prefix(self):
        assert _parse_diff_filename("b/src/foo.py") == "src/foo.py"

    def test_strips_a_prefix(self):
        assert _parse_diff_filename("a/src/foo.py") == "src/foo.py"

    def test_bare_path(self):
        assert _parse_diff_filename("path/bar.py") == "path/bar.py"

    def test_devnull_falls_back_to_from(self):
        assert _parse_diff_filename("/dev/null", "a/old.py") == "old.py"

    def test_devnull_no_fallback_returns_question(self):
        assert _parse_diff_filename("/dev/null") == "?"


# ---------------------------------------------------------------------------
# DiffRenderer rendering tests
# ---------------------------------------------------------------------------

_SIMPLE_DIFF = (
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ -1,2 +1,2 @@\n"
    "-foo bar\n"
    "+foo baz\n"
    " context\n"
)

_LOW_RATIO_DIFF = (
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ -1 +1 @@\n"
    "-aaaa\n"
    "+zzzz\n"
)


class TestDiffRendererV2:
    def test_intra_diff_skipped_below_ratio(self):
        import re
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        Console(file=buf, force_terminal=True, highlight=False, width=220).print(
            DiffRenderer()._style(_LOW_RATIO_DIFF.splitlines())
        )
        lines = buf.getvalue().splitlines()
        del_line = next(l for l in lines if "aaaa" in re.sub(r"\x1b\[[0-9;]*m", "", l))
        # bright_red bold is encoded as \x1b[1;91; — must not appear on a flat-colour line
        assert "\x1b[1;91;" not in del_line

    def test_pairing_per_run_not_per_hunk(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        # Use pairs with ratio > 0.5 so intra-diff triggers.
        # "return foo_value" vs "return bar_value": share "return " + "_value" = 13 chars,
        # total = 32, ratio = 26/32 ≈ 0.81.
        diff = (
            "--- a/f.py\n+++ b/f.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-return foo_value\n"
            "+return bar_value\n"
            " context\n"
            "-return foo_result\n"
            "+return bar_result\n"
        )
        import re
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        Console(
            file=buf,
            force_terminal=True,
            highlight=False,
            no_color=False,
            color_system="truecolor",
            width=220,
        ).print(
            DiffRenderer()._style(diff.splitlines())
        )
        output = buf.getvalue()
        plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
        assert "return foo_value" in plain
        assert "return bar_value" in plain
        assert "return foo_result" in plain
        assert "return bar_result" in plain
        assert output.count("48;2;180;48;48") >= 2
        assert output.count("48;2;40;148;40") >= 2

    def test_alternating_run_flush(self):
        # -A +B -C +D with no context between — should pair (-A,+B) and (-C,+D)
        diff = (
            "--- a/f.py\n+++ b/f.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-alpha\n"
            "+ALPHA\n"
            "-beta\n"
            "+BETA\n"
        )
        renderables = _renderables(diff)
        all_plain = " ".join(r.plain for r in renderables)
        assert "alpha" in all_plain
        assert "ALPHA" in all_plain
        assert "beta" in all_plain
        assert "BETA" in all_plain

    def test_unpaired_lines_flat_colour(self):
        diff = (
            "--- a/f.py\n+++ b/f.py\n"
            "@@ -1,2 +1,1 @@\n"
            "-first del\n"
            "-second del\n"
            "+one add\n"
        )
        import re
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        Console(file=buf, force_terminal=True, highlight=False, width=220).print(
            DiffRenderer()._style(diff.splitlines())
        )
        output = buf.getvalue()
        lines = output.splitlines()
        second_del = next(
            l for l in lines
            if "second del" in re.sub(r"\x1b\[[0-9;]*m", "", l)
        )
        assert "\x1b[91m" not in second_del  # no bright_red on unpaired line

    def test_summary_header_add_only(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n+new\n"
        renderables = _renderables(diff)
        plain = renderables[0].plain
        assert "Added" in plain
        assert "removed" not in plain

    def test_summary_header_remove_only(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n"
        plain = _renderables(diff)[0].plain
        assert "Removed" in plain
        assert "Added" not in plain

    def test_summary_header_mixed(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
        plain = _renderables(diff)[0].plain
        assert "Added" in plain
        assert "removed" in plain

    def test_summary_header_plural(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +2 @@\n+line1\n+line2\n"
        plain = _renderables(diff)[0].plain
        assert "Added 2 lines" in plain

    def test_summary_header_singular(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n+line1\n"
        plain = _renderables(diff)[0].plain
        assert "Added 1 line" in plain
        assert "lines" not in plain

    def test_summary_header_contains_filename(self):
        diff = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n+x\n"
        plain = _renderables(diff)[0].plain
        assert "src/foo.py" in plain

    def test_summary_header_strips_b_prefix(self):
        diff = "--- a/path/bar.py\n+++ b/path/bar.py\n@@ -1 +1 @@\n+x\n"
        plain = _renderables(diff)[0].plain
        assert "path/bar.py" in plain
        assert "b/path/bar.py" not in plain

    def test_summary_header_bare_path(self):
        diff = "--- path/bar.py\n+++ path/bar.py\n@@ -1 +1 @@\n+x\n"
        plain = _renderables(diff)[0].plain
        assert "path/bar.py" in plain

    def test_summary_header_devnull_fallback(self):
        diff = "--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
        plain = _renderables(diff)[0].plain
        assert "old.py" in plain

    def test_summary_header_keeps_distinct_relative_paths(self):
        diff = (
            "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n+x\n"
            "--- a/tests/foo.py\n+++ b/tests/foo.py\n@@ -1 +1 @@\n+y\n"
        )
        header_plains = [r.plain for r in _renderables(diff) if "●" in r.plain]
        assert any("src/foo.py" in p for p in header_plains)
        assert any("tests/foo.py" in p for p in header_plains)

    def test_multi_file_diff_two_headers(self):
        diff = (
            "--- a/one.py\n+++ b/one.py\n@@ -1 +1 @@\n+x\n"
            "--- a/two.py\n+++ b/two.py\n@@ -1 +1 @@\n+y\n"
        )
        renderables = _renderables(diff)
        header_plains = [r.plain for r in renderables if "●" in r.plain]
        assert len(header_plains) == 2
        assert any("one.py" in p for p in header_plains)
        assert any("two.py" in p for p in header_plains)

    def test_separator_width_matches_header(self):
        diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n+x\n"
        renderables = _renderables(diff)
        header = renderables[0]
        separator = renderables[1]
        assert len(separator.plain) == len(header.plain)


# ---------------------------------------------------------------------------
# DiffRenderer truncation tests
# ---------------------------------------------------------------------------

import re as _re
_ANSI_RE = _re.compile(r"\x1b\[[0-9;]*m")


def _make_long_diff(n_lines: int) -> str:
    """Generate a unified diff with n_lines changed lines."""
    old = "\n".join(f"line {i}" for i in range(n_lines))
    new = "\n".join(f"line {i} changed" for i in range(n_lines))
    import difflib
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="a/f.py", tofile="b/f.py",
    ))


class TestDiffRendererTruncation:
    def setup_method(self):
        self.dr = DiffRenderer()

    def test_short_diff_not_truncated(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
        lines = self.dr.to_lines(diff)
        assert not any("omitted" in _ANSI_RE.sub("", l) for l in lines)
        assert any("old" in _ANSI_RE.sub("", l) for l in lines)
        assert any("new" in _ANSI_RE.sub("", l) for l in lines)

    def test_long_diff_truncated(self):
        diff = _make_long_diff(_DIFF_MAX_LINES + 10)
        lines = self.dr.to_lines(diff)
        plain_last = _ANSI_RE.sub("", lines[-1])
        assert "omitted" in plain_last
        assert not any(f"line {_DIFF_MAX_LINES + 1} changed" in _ANSI_RE.sub("", l) for l in lines)

    def test_footer_singular(self):
        # Render a diff fully, then re-render capped at total-1 → exactly 1 omitted
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n-a\n-b\n-c\n+a2\n+b2\n+c2\n"
        full = self.dr.to_lines(diff, max_lines=0)
        lines = self.dr.to_lines(diff, max_lines=len(full) - 1)
        footer = _ANSI_RE.sub("", lines[-1])
        assert "1 more line omitted" in footer
        assert "lines" not in footer

    def test_footer_plural(self):
        diff = _make_long_diff(_DIFF_MAX_LINES + 5)
        lines = self.dr.to_lines(diff)
        footer = _ANSI_RE.sub("", lines[-1])
        assert "more lines omitted" in footer

    def test_max_lines_zero_disables_cap(self):
        diff = _make_long_diff(_DIFF_MAX_LINES + 20)
        lines = self.dr.to_lines(diff, max_lines=0)
        assert not any("omitted" in _ANSI_RE.sub("", l) for l in lines)
        assert len(lines) > _DIFF_MAX_LINES

    def test_custom_max_lines_respected(self):
        diff = _make_long_diff(20)
        lines = self.dr.to_lines(diff, max_lines=10)
        assert len(lines) == 11  # 10 content + footer
        assert "omitted" in _ANSI_RE.sub("", lines[-1])


# ---------------------------------------------------------------------------
# apply_inline_markdown


class TestApplyInlineMarkdown:
    def test_bold_double_asterisk(self):
        result = apply_inline_markdown("**foo**")
        assert "\033[1m" in result
        assert "foo" in result
        assert "**" not in result

    def test_bold_double_underscore(self):
        result = apply_inline_markdown("__foo__")
        assert "\033[1m" in result
        assert "foo" in result
        assert "__" not in result

    def test_italic_single_asterisk(self):
        result = apply_inline_markdown("*foo*")
        assert "\033[3m" in result
        assert "foo" in result
        assert result.count("*") == 0

    def test_italic_single_underscore(self):
        result = apply_inline_markdown("_foo_")
        assert "\033[3m" in result
        assert "foo" in result
        assert "_" not in result

    def test_italic_underscore_multi_word(self):
        result = apply_inline_markdown("_underline - kinda works_")
        assert "\033[3m" in result
        assert "underline - kinda works" in result
        assert "_" not in result

    def test_italic_single_underscore_with_spaces(self):
        result = apply_inline_markdown("This is _super bold and italic_ text.")
        assert "\033[3m" in result
        assert "super bold and italic" in result
        assert "_" not in result

    def test_underscore_inside_word_ignored(self):
        result = apply_inline_markdown("snake_case_var")
        assert result == "snake_case_var"

    def test_trailing_underscore_ignored(self):
        assert apply_inline_markdown("value_") == "value_"

    def test_leading_underscore_ignored(self):
        assert apply_inline_markdown("_private") == "_private"

    def test_backtick_code_span(self):
        result = apply_inline_markdown("`foo`")
        assert "\033[97m" in result
        assert "\033[48;5;237m" in result  # dark background applied
        assert "foo" in result
        assert "`" in result  # backticks preserved inside the styled span

    def test_strikethrough(self):
        result = apply_inline_markdown("~~foo~~")
        assert "\033[9m" in result
        assert "foo" in result
        assert "~~" not in result

    def test_mixed_bold_and_code(self):
        result = apply_inline_markdown("**Line 88**: `cdOffset`")
        assert "\033[1m" in result           # bold applied
        assert "\033[97m" in result          # code span applied
        assert "\033[48;5;237m" in result    # code span background applied
        assert "**" not in result
        assert "`" in result  # backticks preserved inside the styled span

    def test_mixed_strikethrough_and_code(self):
        result = apply_inline_markdown("~~deprecated~~ use `new_api()` instead")
        assert "\033[9m" in result
        assert "\033[48;5;237m" in result
        assert "~~" not in result
        assert "`" in result

    def test_mixed_underline_and_code(self):
        result = apply_inline_markdown("<u>important</u>: call `init()` first")
        assert "\033[4m" in result
        assert "\033[48;5;237m" in result
        assert "<u>" not in result
        assert "`" in result

    def test_mixed_bold_italic_and_code(self):
        result = apply_inline_markdown("***critical***: run `setup()` now")
        assert "\033[1;3m" in result
        assert "\033[48;5;237m" in result
        assert "***" not in result
        assert "`" in result

    def test_mixed_mark_and_code(self):
        result = apply_inline_markdown("<mark>highlight</mark> then call `fn()`")
        assert "\033[7m" in result
        assert "\033[48;5;237m" in result
        assert "<mark>" not in result
        assert "`" in result

    def test_mixed_ins_and_code(self):
        result = apply_inline_markdown("<ins>added</ins> via `patch()`")
        assert "\033[4m" in result
        assert "\033[48;5;237m" in result
        assert "<ins>" not in result
        assert "`" in result

    def test_multiple_code_spans_with_bold(self):
        result = apply_inline_markdown("**bold** uses `foo()` and `bar()`")
        assert "\033[1m" in result
        assert result.count("\033[48;5;237m") == 2
        assert "**" not in result

    def test_bold_italic_strikethrough_and_code(self):
        result = apply_inline_markdown("**bold** *italic* ~~strike~~ `code`")
        assert "\033[1m" in result
        assert "\033[3m" in result
        assert "\033[9m" in result
        assert "\033[48;5;237m" in result
        assert "**" not in result
        assert "~~" not in result

    def test_asterisks_inside_backtick_untouched(self):
        result = apply_inline_markdown("`**not bold**`")
        # Content inside code span must not be bold-rendered
        assert "\033[1m" not in result
        assert "**not bold**" in result

    def test_already_ansi_returned_unchanged(self):
        ansi_line = "\033[32mgreen\033[0m"
        assert apply_inline_markdown(ansi_line) is ansi_line

    def test_empty_string(self):
        assert apply_inline_markdown("") == ""

    def test_plain_text_unchanged(self):
        assert apply_inline_markdown("plain text") == "plain text"

    def test_reset_suffix_restored_between_spans(self):
        colour = "\033[32m"
        result = apply_inline_markdown("**a** and *b*", reset_suffix=colour)
        # Each closing reset should be followed by the colour suffix
        assert f"\033[0m{colour}" in result

    def test_no_markdown(self):
        assert apply_inline_markdown("no markdown here") == "no markdown here"

    def test_em_tag_italic(self):
        result = apply_inline_markdown("<em>foo</em>")
        assert "\033[3m" in result
        assert "foo" in result
        assert "<em>" not in result
        assert "</em>" not in result

    def test_strong_tag_bold(self):
        result = apply_inline_markdown("<strong>foo</strong>")
        assert "\033[1m" in result
        assert "foo" in result
        assert "<strong>" not in result

    def test_u_tag_underline(self):
        result = apply_inline_markdown("<u>foo</u>")
        assert "\033[4m" in result
        assert "foo" in result
        assert "<u>" not in result

    def test_mark_tag_highlight(self):
        result = apply_inline_markdown("<mark>foo</mark>")
        assert "\033[7m" in result
        assert "foo" in result
        assert "<mark>" not in result

    def test_u_tag_with_inner_bold_restores_underline(self):
        # Inner bold reset must restore underline, not drop it.
        result = apply_inline_markdown("<u>**bold** normal</u>")
        # Underline code appears before bold
        assert result.index("\033[4m") < result.index("\033[1m")
        # reset_suffix "\033[4m" appears after the bold reset, restoring underline
        ansi_codes = [result[i:i+4] for i in range(len(result)) if result[i:i+2] == "\033["]
        assert result.count("\033[4m") >= 2  # outer open + reset_suffix restore

    def test_bold_italic_triple_star(self):
        result = apply_inline_markdown("***foo***")
        assert "\033[1;3m" in result
        assert "foo" in result
        assert "*" not in _strip(result)

    def test_bold_italic_triple_underscore(self):
        result = apply_inline_markdown("___foo___")
        assert "\033[1;3m" in result
        assert "foo" in result

    def test_i_tag_italic(self):
        result = apply_inline_markdown("<i>foo</i>")
        assert "\033[3m" in result
        assert "<i>" not in result

    def test_b_tag_bold(self):
        result = apply_inline_markdown("<b>foo</b>")
        assert "\033[1m" in result
        assert "<b>" not in result

    def test_s_tag_strikethrough(self):
        result = apply_inline_markdown("<s>foo</s>")
        assert "\033[9m" in result
        assert "<s>" not in result

    def test_strike_tag_strikethrough(self):
        result = apply_inline_markdown("<strike>foo</strike>")
        assert "\033[9m" in result
        assert "<strike>" not in result

    def test_del_tag_strikethrough(self):
        result = apply_inline_markdown("<del>foo</del>")
        assert "\033[9m" in result
        assert "<del>" not in result

    def test_code_tag_inline(self):
        result = apply_inline_markdown("<code>foo</code>")
        assert "\033[97m" in result
        assert "<code>" not in result

    def test_kbd_tag_code_style(self):
        result = apply_inline_markdown("<kbd>Ctrl+C</kbd>")
        assert "\033[97m" in result
        assert "<kbd>" not in result

    def test_ins_tag_underline(self):
        result = apply_inline_markdown("<ins>foo</ins>")
        assert "\033[4m" in result
        assert "<ins>" not in result

    def test_sup_tag_stripped(self):
        result = apply_inline_markdown("x<sup>2</sup>")
        assert "<sup>" not in result
        assert "2" in result

    def test_sub_tag_stripped(self):
        result = apply_inline_markdown("H<sub>2</sub>O")
        assert "<sub>" not in result
        assert "2" in result
        assert "H" in result
        assert "O" in result

    def test_link_underlined(self):
        result = apply_inline_markdown("[click here](https://x.com)")
        assert "\033[4m" in result  # underline (part of link style)
        assert "click here" in result
        assert "https://x.com" in result  # URL preserved for copy/ctrl+click
        assert "[click here]" not in _strip(result)

    def test_image_placeholder(self):
        result = apply_inline_markdown("![logo](img.png)")
        assert "[img: logo]" in result
        assert "\033[2m" in result
        assert "img.png" not in result

    def test_image_before_link(self):
        result = apply_inline_markdown("![a](u) [b](v)")
        assert "[img: a]" in result
        assert "\033[4m" in result  # underline (part of link style)
        assert "b" in result

    def test_image_then_link_no_ansi_corruption(self):
        # Regression: image step emits \033[0m; the link regex must not match
        # the "[0m ... [linktext](url)" span and leave orphaned ESC bytes that
        # cause subsequent ANSI sequences to print as literal text in the terminal.
        result = apply_inline_markdown("![logo](img.png) and [click](https://x.com)")
        plain = _strip(result)
        assert "logo" in plain
        assert "click" in plain
        assert "https://x.com" in plain
        # No raw ANSI fragments may appear as visible text
        assert "0m" not in plain
        assert "38;2" not in plain
        # The link must be styled (underline present)
        assert "\033[4m" in result

    def test_bare_url_styled(self):
        result = apply_inline_markdown("1. https://www.google.com")
        assert "\033[4m" in result  # underline applied
        assert "https://www.google.com" in result

    def test_bare_url_trailing_period_stripped(self):
        result = apply_inline_markdown("See https://example.com.")
        assert "https://example.com" in result
        # The period must NOT be inside the styled span
        stripped = _strip(result)
        assert stripped.endswith(".")
        url_end = stripped.index("https://example.com") + len("https://example.com")
        assert stripped[url_end] == "."

    def test_bare_file_url_styled(self):
        result = apply_inline_markdown("file:///home/user/tmp")
        assert "\033[4m" in result
        assert "file:///home/user/tmp" in result

    def test_bare_www_domain_styled(self):
        result = apply_inline_markdown("Check www.example.com for info")
        assert "\033[4m" in result
        assert "www.example.com" in result

    def test_bare_www_not_matched_mid_word(self):
        result = apply_inline_markdown("xwww.example.com")
        assert "\033[4m" not in result

    def test_bare_url_does_not_double_process_markdown_link(self):
        result = apply_inline_markdown("[text](https://x.com) and https://y.com")
        # markdown link: text shown, not the raw [text](url)
        assert "[text]" not in _strip(result)
        assert "text" in _strip(result)
        # bare URL also styled (appears once)
        assert result.count("https://y.com") == 1

    def test_bare_url_inside_bold_no_orphan_ansi(self):
        # Regression: bold/italic wrapping a bare URL caused the ESC byte from
        # the inner apply_inline_markdown's reset to be captured by the outer
        # _MD_BARE_URL_RE (ESC is not excluded from [^\s<>\[\]()\"] by default),
        # leaving a literal "[0m[0m" in the rendered output.
        for wrapper in ("**{url}** rest", "*{url}* rest"):
            line = wrapper.format(url="https://example.com/path")
            result = apply_inline_markdown(line, reset_suffix="\033[38;2;200;200;200m")
            plain = _strip(result)
            assert "[0m" not in plain, f"orphan '[0m' in output of {wrapper!r}: {plain!r}"
            assert "https://example.com/path" in plain


class TestApplyBlockLine:
    def test_h1_stripped_and_bold(self):
        result = apply_block_line("# Foo")
        assert "\033[1;97m" in result
        assert "Foo" in result
        assert "#" not in result

    def test_h2_dimmer_than_h1(self):
        result = apply_block_line("## Foo")
        assert "\033[1;37m" in result
        assert "97m" not in result

    def test_h4_bold_dim(self):
        result = apply_block_line("#### Foo")
        assert "\033[1;2m" in result

    def test_h1_with_inline_span(self):
        result = apply_block_line("# **Foo**")
        assert "\033[1;97m" in result
        assert "\033[1m" in result
        assert "Foo" in result
        assert "**" not in result

    def test_hr_dashes_replaced(self):
        result = apply_block_line("---")
        assert "─" in result
        assert "-" not in _strip(result)

    def test_hr_stars_replaced(self):
        result = apply_block_line("***")
        assert "─" in result

    def test_hr_underscores_replaced(self):
        result = apply_block_line("___")
        assert "─" in result

    def test_non_hr_dashes_unchanged(self):
        result = apply_block_line("some --- text")
        assert result == "some --- text"

    def test_blockquote_gutter(self):
        result = apply_block_line("> hello")
        assert "▌" in result
        assert "hello" in result
        assert ">" not in result

    def test_blockquote_nested_collapsed(self):
        result = apply_block_line(">> deep")
        assert result.count("▌") == 1

    def test_blockquote_inline_span(self):
        result = apply_block_line("> **bold**")
        assert "▌" in result
        assert "\033[1m" in result
        assert "**" not in result

    def test_blockquote_inline_span_restores_dim(self):
        # Bold span inside a blockquote must restore the dim gutter style on close,
        # not reset to terminal default — fixes missing reset_suffix on blockquote branch.
        result = apply_block_line("> **bold** plain")
        # Dim style (\033[2m) must appear after the bold close (\033[0m)
        assert "\033[0m\033[2m" in result

    def test_blockquote_with_inline_code(self):
        result = apply_block_line("> see `foo()` for details")
        assert "▌" in result
        assert "\033[48;5;237m" in result
        assert "foo()" in result
        assert ">" not in result

    def test_blockquote_with_bold_and_code(self):
        result = apply_block_line("> **important**: call `init()`")
        assert "▌" in result
        assert "\033[1m" in result
        assert "\033[48;5;237m" in result
        assert "**" not in result

    def test_heading_with_inline_code(self):
        result = apply_block_line("# Use `setup()` first")
        assert "\033[1;97m" in result
        assert "\033[48;5;237m" in result
        assert "setup()" in result
        assert "#" not in result

    def test_heading_with_bold_and_code(self):
        result = apply_block_line("## **Required**: run `init()`")
        assert "\033[1;37m" in result
        assert "\033[1m" in result
        assert "\033[48;5;237m" in result
        assert "**" not in result

    def test_list_bullet_dot(self):
        result = apply_block_line("- item")
        assert "•" in result
        assert "item" in result
        assert result.startswith("•")

    def test_list_bullet_circle_nested(self):
        result = apply_block_line("  - item")
        assert "◦" in result

    def test_list_bullet_triangle_double_nested(self):
        result = apply_block_line("    - item")
        assert "▸" in result

    def test_list_star_and_plus(self):
        assert "•" in apply_block_line("* item")
        assert "•" in apply_block_line("+ item")

    def test_ordered_list_rendered(self):
        result = apply_block_line("1. item")
        # OL items are now rendered with dim numeral
        assert "\033[2m1.\033[0m" in result
        assert "item" in result

    def test_reference_link_suppressed(self):
        result = apply_block_line("[ref]: https://x.com")
        assert result == ""

    def test_reference_link_with_quoted_title_suppressed(self):
        assert apply_block_line('[ref]: https://x.com "Page Title"') == ""

    def test_reference_link_with_paren_title_suppressed(self):
        assert apply_block_line("[ref]: https://x.com (Page Title)") == ""

    def test_ansi_lines_skipped(self):
        ansi_line = "\033[32mgreen\033[0m"
        assert apply_block_line(ansi_line) is ansi_line

    def test_multiline_skipped(self):
        multi = "line1\nline2"
        assert apply_block_line(multi) is multi

    def test_plain_line_unchanged(self):
        assert apply_block_line("just text") == "just text"


class TestFormatResponseInlineMarkdown:
    """Integration: format_response applies inline markdown to prose, not code."""

    def test_bold_in_prose_rendered(self):
        text = "This is **important** text."
        result = format_response(text)
        assert "\033[1m" in result
        assert "important" in result
        assert "**" not in result

    def test_bold_and_inline_code_on_same_line(self):
        # Regression: bold markers must survive alongside inline code spans.
        text = "**Line 230**: `eocd.writeUInt32LE(cdSize, 8)` - EOCD should have **CD total size**"
        result = format_response(text)
        assert "\033[1m" in result            # bold ANSI applied
        assert "\033[48;5;237m" in result     # code span background applied
        assert "**" not in result             # no raw bold markers in output

    def test_italic_and_inline_code_on_same_line(self):
        text = "*note*: see `foo()` for details"
        result = format_response(text)
        assert "\033[3m" in result            # italic ANSI applied
        assert "\033[48;5;237m" in result     # code span background applied
        assert "*note*" not in result

    def test_strikethrough_and_inline_code_on_same_line(self):
        text = "~~deprecated~~ use `new_api()` instead"
        result = format_response(text)
        assert "\033[9m" in result            # strikethrough ANSI applied
        assert "\033[48;5;237m" in result     # code span background applied
        assert "~~" not in result

    def test_underline_and_inline_code_on_same_line(self):
        text = "<u>important</u>: call `init()` first"
        result = format_response(text)
        assert "\033[4m" in result            # underline ANSI applied
        assert "\033[48;5;237m" in result     # code span background applied
        assert "<u>" not in result
        assert "</u>" not in result

    def test_heading_followed_by_paragraph_preserves_newline(self):
        # apply_block_line drops the trailing \n from matched lines; format_response
        # must compensate so the paragraph starts on its own line.
        text = "# Title\nParagraph text"
        result = format_response(text)
        plain = _strip(result)
        # Heading and paragraph must be on separate lines
        assert plain.index("Title") < plain.index("\n")
        assert "Paragraph text" in plain

    def test_list_followed_by_paragraph_preserves_newline(self):
        text = "- item one\nnext line"
        result = format_response(text)
        plain = _strip(result)
        assert "item one" in plain
        assert plain.index("item one") < plain.index("\n")
        assert "next line" in plain

    def test_code_block_not_double_escaped(self):
        text = "Note **this**:\n```python\nx = **1**\n```\nEnd **here**."
        result = format_response(text)
        # Prose bold rendered
        assert "\033[1m" in result
        # The Python code block was syntax-highlighted; the ** inside are code
        # content — they appear as plain chars inside the highlighted block,
        # not as ANSI bold markers.  Verify no double-escape by checking that
        # the result does not contain literal \033[1m immediately followed by
        # content that was already inside an ANSI span.
        # Simpler: strip all ANSI and confirm code content intact
        plain = _strip(result)
        assert "x = **1**" in plain

    def test_backslash_escape_stripped(self):
        r"""CommonMark backslash escapes like \] and \* are stripped from output."""
        result = apply_inline_markdown(r"- [ \] unchecked")
        assert r"\]" not in result
        assert "]" in result

    def test_backslash_escape_checkbox(self):
        r"""[x\] renders as [x] — backslash before ] removed."""
        result = apply_inline_markdown(r"- [x\] checked item")
        assert r"\]" not in result
        assert "[x]" in _strip(result)


# ---------------------------------------------------------------------------
# render_stateful_blocks — regex smoke tests
# ---------------------------------------------------------------------------

class TestStatefulBlockRegexes:
    def test_setext_h1_re_matches(self):
        assert _SETEXT_H1_RE.match("==")
        assert _SETEXT_H1_RE.match("===")
        assert _SETEXT_H1_RE.match("===  ")
        assert not _SETEXT_H1_RE.match("=")
        assert not _SETEXT_H1_RE.match("=== text")

    def test_setext_h2_re_matches(self):
        assert _SETEXT_H2_RE.match("--")
        assert _SETEXT_H2_RE.match("---")
        assert _SETEXT_H2_RE.match("---  ")
        assert not _SETEXT_H2_RE.match("-")
        assert not _SETEXT_H2_RE.match("--- text")

    def test_table_row_re(self):
        assert _TABLE_STRICT_ROW_RE.match("| a | b |")
        assert _TABLE_STRICT_ROW_RE.match("|---|---|")
        assert not _TABLE_STRICT_ROW_RE.match("a | b")
        assert not _TABLE_STRICT_ROW_RE.match("| no trailing")

    def test_num_re(self):
        assert _NUM_RE.match("42")
        assert _NUM_RE.match("1,000")
        assert _NUM_RE.match("3.14")
        assert _NUM_RE.match("-7")
        assert not _NUM_RE.match("abc")
        assert not _NUM_RE.match("1a")

    def test_split_row(self):
        assert _split_row("| a | b |") == [" a ", " b "]
        assert _split_row("|---|---|") == ["---", "---"]
        # Loose format — no boundary pipes
        assert _split_row("a | b | c") == ["a ", " b ", " c"]
        assert _split_row("---|---|---") == ["---", "---", "---"]
        # Mixed — trailing pipe only
        assert _split_row("a | b |") == ["a ", " b "]


# ---------------------------------------------------------------------------
# render_stateful_blocks — setext headings
# ---------------------------------------------------------------------------

class TestRenderStatefulBlocksSetext:
    def test_setext_h1(self):
        result = render_stateful_blocks("Foo\n===")
        assert "\033[1;97m" in result
        assert "Foo" in result
        assert "===" not in result

    def test_setext_h2(self):
        result = render_stateful_blocks("Bar\n---")
        assert "\033[1;37m" in result
        assert "Bar" in result
        assert "---" not in result

    def test_blank_line_dash_is_hr_not_h2(self):
        result = format_response("\n---")
        plain = _strip(result)
        assert "─" in plain
        assert "\033[1;37m" not in result

    def test_list_item_dash_is_hr(self):
        result = format_response("- x\n---")
        assert "\033[1;37m" not in result
        plain = _strip(result)
        assert "─" in plain

    def test_setext_with_inline_span(self):
        result = render_stateful_blocks("**Foo**\n===")
        assert "\033[1;97m" in result
        assert "\033[1m" in result
        assert "Foo" in result
        assert "**" not in result

    def test_setext_at_end_of_string_no_newline(self):
        result = render_stateful_blocks("Title\n===")
        assert "\033[1;97m" in result
        assert "===" not in result

    def test_ansi_pending_not_heading(self):
        result = render_stateful_blocks("\033[1mcode\033[0m\n===")
        assert "===" in result
        assert "\033[1;97m" not in result

    def test_trailing_whitespace_marker(self):
        result = render_stateful_blocks("Foo\n===  ")
        assert "\033[1;97m" in result
        assert "===" not in result

    def test_marker_at_document_start(self):
        # --- at document start renders as hr, not setext h2
        result = format_response("---\ntext")
        plain = _strip(result)
        assert "─" in plain
        assert "text" in plain
        assert "\033[1;37m" not in result


# ---------------------------------------------------------------------------
# render_stateful_blocks — multi-line blockquote continuation
# ---------------------------------------------------------------------------

class TestRenderStatefulBlocksBlockquote:
    def test_continuation_has_gutter(self):
        result = render_stateful_blocks("> q\ncontinuation")
        assert result.count("▌") == 2

    def test_blank_line_ends_continuation(self):
        result = render_stateful_blocks("> q\n\nnormal")
        lines = result.splitlines()
        normal_line = [l for l in lines if "normal" in l][0]
        assert "▌" not in normal_line

    def test_explicit_bq_resets(self):
        result = render_stateful_blocks("> q\n\n> new")
        assert result.count("▌") == 2


# ---------------------------------------------------------------------------
# render_stateful_blocks — tables
# ---------------------------------------------------------------------------

class TestRenderStatefulBlocksTables:
    _TABLE = "| Name | Age |\n|------|-----|\n| Alice | 28 |\n| Bob | 32 |"

    def test_basic_table_rendered(self):
        result = render_stateful_blocks(self._TABLE)
        plain = _strip(result)
        assert "─" in plain
        assert "Alice" in plain
        assert "Bob" in plain
        assert "|" not in plain

    def test_right_aligned_column(self):
        t = "| Name | Age |\n|------|----:|\n| Alice | 28 |"
        result = render_stateful_blocks(t)
        lines = _strip(result).splitlines()
        data = [l for l in lines if "Alice" in l][0]
        # "28" should appear right-justified (preceded by spaces)
        assert "28" in data
        idx_28 = data.index("28")
        assert data[idx_28 - 1] == " "

    def test_centre_aligned_column(self):
        t = "| Name |\n|:----:|\n| Hi |"
        result = render_stateful_blocks(t)
        plain = _strip(result)
        assert "Hi" in plain

    def test_number_auto_right(self):
        t = "| Item | Count |\n|------|-------|\n| foo | 42 |"
        result = render_stateful_blocks(t)
        plain = _strip(result)
        assert "42" in plain

    def test_ragged_row_padded(self):
        t = "| A | B | C |\n|---|---|---|\n| x |"
        result = render_stateful_blocks(t)
        assert "x" in _strip(result)

    def test_ragged_align_no_error(self):
        t = "| A | B | C |\n|---|---|\n| x | y | z |"
        result = render_stateful_blocks(t)
        assert "x" in _strip(result)

    def test_table_at_end_no_newline(self):
        t = "| A |\n|---|\n| x |"
        result = render_stateful_blocks(t)
        assert "x" in _strip(result)
        assert "|" not in _strip(result)

    def test_table_no_separator(self):
        # Strict table with no separator row: still renders framed (no sep_idx,
        # so all rows are treated as content with inter-row dividers).
        t = "| A | B |\n| x | y |\n| z | w |"
        result = render_stateful_blocks(t)
        plain = _strip(result)
        assert "x" in plain
        assert "┌" in plain  # box frame present even without separator


    def test_emoji_cells_do_not_misalign_columns(self):
        # Wide emoji (✅ = 2 cols, ❌ = 2 cols) must be counted correctly.
        from agent.rich_output import _visual_len
        assert _visual_len("✅") == 2
        assert _visual_len("❌") == 2
        assert _visual_len("⚠️") == 2
        assert _visual_len("ok") == 2
        md = "| A | B |\n|---|---|\n| ✅ | yes |\n| ❌ | no |"
        out = format_response(md)
        lines = [l for l in out.splitlines() if l.strip() and "─" not in l]
        import re as _re
        ansi = _re.compile(r"\x1b\[[0-9;]*m")
        widths = [_visual_len(ansi.sub("", l)) for l in lines]
        assert len(set(widths)) == 1, f"Column widths diverged: {widths}"

    def test_inline_markdown_in_cells_does_not_misalign_columns(self):
        # Cells with **bold** markup: rendered visual width must match padding.
        md = "| A | B |\n|---|---|\n| **hi** | x |\n| bye | y |"
        out = format_response(md)
        lines = [l for l in out.splitlines() if l.strip() and "─" not in l]
        # All data lines must have the same visual length (consistent column widths).
        import re
        ansi = re.compile(r"\x1b\[[0-9;]*m")
        visual_lens = [len(ansi.sub("", l)) for l in lines]
        assert len(set(visual_lens)) == 1, f"Column widths diverged: {visual_lens}"


# ---------------------------------------------------------------------------
# StreamingBlockBuffer
# ---------------------------------------------------------------------------

class TestStreamingBlockBuffer:
    def setup_method(self):
        self.buf = StreamingBlockBuffer()

    def test_setext_h1_on_marker(self):
        assert self.buf.process_line("Foo") is None
        result = self.buf.process_line("===")
        assert result is not None
        assert "\033[1;97m" in result
        assert "Foo" in result

    def test_setext_non_marker_releases_pending(self):
        assert self.buf.process_line("Foo") is None
        result = self.buf.process_line("bar")
        assert result == "Foo"
        # "bar" is now pending
        flushed = self.buf.flush()
        assert flushed == "bar"

    def test_setext_flush_emits_pending(self):
        assert self.buf.process_line("Foo") is None
        result = self.buf.flush()
        assert result == "Foo"

    def test_setext_ansi_line_not_held(self):
        ansi = "\033[1mx\033[0m"
        result = self.buf.process_line(ansi)
        assert result is ansi  # returned immediately

    def test_table_rows_none_until_done(self):
        assert self.buf.process_line("| A | B |") is None
        assert self.buf.process_line("|---|---|") is None
        assert self.buf.process_line("| x | y |") is None
        non_table = "done"
        result = self.buf.process_line(non_table)
        assert result is not None
        assert "x" in _strip(result)
        # Next call returns the non-table line
        next_result = self.buf.process_line("anything")
        assert next_result == "done"

    def test_table_flush_emits_partial(self):
        self.buf.process_line("| A |")
        self.buf.process_line("|---|")
        self.buf.process_line("| x |")
        result = self.buf.flush()
        assert result is not None
        assert "x" in _strip(result)

    def test_table_non_table_line_identity(self):
        self.buf.process_line("| A |")
        self.buf.process_line("|---|")
        self.buf.process_line("| x |")
        non_table = "plain line"
        self.buf.process_line(non_table)  # returns rendered table
        # Next call should return the non-table line with same identity
        result = self.buf.process_line("next")
        assert result is non_table

    def test_blockquote_continuation_stateful(self):
        self.buf.process_line("some")  # goes to pending
        self.buf.flush()
        self.buf.reset()
        # Fresh: enter blockquote.
        # The first BQ line is buffered for setext-in-blockquote lookahead (returns None).
        r1 = self.buf.process_line("> quote")
        # Continuation flushes the buffered BQ line (returns the rendered BQ line)
        r2 = self.buf.process_line("continuation")
        # Between r1 and r2 at least one should have the gutter
        assert r2 is not None
        assert "▌" in r2
        # The continuation itself is also in blockquote — next call has it via emit_next
        r3 = self.buf.process_line("more")
        assert r3 is not None
        assert "▌" in r3

    def test_blockquote_ansi_gets_gutter(self):
        # ANSI line inside blockquote keeps the gutter.
        # First BQ line is buffered (returns None); subsequent ANSI line
        # flushes the pending BQ line and defers the ANSI line.
        self.buf.process_line("> start")
        ansi = "\033[1mx\033[0m"
        r1 = self.buf.process_line(ansi)
        # r1 is the rendered "> start" line (pending flushed)
        assert r1 is not None
        assert "▌" in r1
        # ansi is deferred in _emit_next; flush it to get the ANSI+gutter line
        flushed = self.buf.flush()
        assert flushed is not None
        assert ansi in flushed
        assert "▌" in flushed

    def test_blockquote_fence_exits_state(self):
        # Code fence line exits blockquote so the code highlighter can handle it.
        # First BQ line is buffered; fence flushes pending and defers itself.
        self.buf.process_line("> start")
        r1 = self.buf.process_line("```python")
        # r1 is the flushed pending BQ line; "```python" is deferred
        assert r1 is not None
        assert "▌" in r1
        # Blockquote exits when fence is encountered
        assert self.buf._bq_depth == 0
        # Flush gives the fence line
        flushed = self.buf.flush()
        assert flushed is not None
        assert "```python" in flushed

    def test_mode_transition_pending_plus_blockquote(self):
        assert self.buf.process_line("pending_line") is None
        result = self.buf.process_line("> blockquote")
        assert result == "pending_line"
        # Next call should return rendered blockquote
        result2 = self.buf.process_line("next")
        assert result2 is not None
        assert "▌" in result2

    def test_mode_transition_pending_plus_table(self):
        assert self.buf.process_line("pending_line") is None
        result = self.buf.process_line("| A |")
        assert result == "pending_line"
        # Next call processes "| A |" (buffered), returns None
        result2 = self.buf.process_line("| B |")
        assert result2 is None

    def test_reset_clears_all_state(self):
        self.buf.process_line("pending")
        self.buf._bq_depth = 2
        self.buf._table_buf.append("| x |")
        self.buf._emit_next = "something"
        self.buf.reset()
        assert self.buf._pending is None
        assert self.buf._bq_depth == 0
        assert self.buf._table_buf == []
        assert self.buf._emit_next is None

    def test_setext_marker_as_emit_next_via_flush(self):
        """Deferred line stored in _emit_next is a setext marker: flush renders heading."""
        # Turn 1: "Title" → pending
        assert self.buf.process_line("Title") is None
        # Turn 2: ">" line arrives while pending → returns "Title", stores ">" in _emit_next
        result = self.buf.process_line("> quote")
        assert result == "Title"
        # flush: _emit_next = "> quote", _pending = None
        flushed = self.buf.flush()
        assert flushed is not None
        assert "▌" in flushed

    def test_pending_flushed_before_table(self):
        """Prose line pending before a table must be emitted before table rows."""
        result = render_stateful_blocks("prose\n| A | B |\n|---|---|\n| x | y |")
        lines = _strip(result).splitlines()
        prose_idx = next(i for i, l in enumerate(lines) if "prose" in l)
        table_idx = next(i for i, l in enumerate(lines) if "x" in l)
        assert prose_idx < table_idx

    def test_ansi_line_in_table_flushes_table(self):
        """An ANSI line mid-table must flush the accumulated rows before emitting the ANSI line."""
        ansi = "\033[32mcode\033[0m"
        text = "| A | B |\n|---|---|\n| x | y |\n" + ansi + "\nnormal"
        result = render_stateful_blocks(text)
        lines = result.splitlines()
        # Table content must appear before the ANSI line
        table_idx = next(i for i, l in enumerate(lines) if "x" in _strip(l))
        ansi_idx = next(i for i, l in enumerate(lines) if ansi in l)
        assert table_idx < ansi_idx

    def test_blockquote_pending_prose_flushes_before_ansi_code(self):
        result = render_stateful_blocks("> quote\n\033[2m1 │\033[0m x=1\n")
        lines = _strip(result).splitlines()
        assert lines[0].startswith("▌ quote")
        assert "1 │ x=1" in lines[1]

    def test_ol_item_not_setext_candidate_with_hr(self):
        """OL item followed by '---' must NOT become a setext heading."""
        buf = StreamingBlockBuffer()
        assert buf.process_line("1. item one") is None
        result = buf.process_line("---")
        # '1. item one' must be emitted as plain text, not a heading
        assert result is not None
        assert "\033[1;37m" not in result  # no H2 heading style
        assert "1. item one" in result
        # '---' should be buffered now (pending for next setext check)
        assert buf._pending == "---"

    def test_ol_item_followed_by_setext_underline(self):
        """OL item followed by '===' must NOT become a setext heading."""
        buf = StreamingBlockBuffer()
        assert buf.process_line("3. another item") is None
        result = buf.process_line("===")
        assert result is not None
        assert "\033[1;97m" not in result  # no H1 heading style
        assert "3. another item" in result

    def test_loose_table_strict_separator(self):
        """GFM optional-boundary pipes: header/data rows have no leading pipe."""
        t = "Lang | Type\n|---|---|\nPython | Dynamic\nRust | Static"
        result = render_stateful_blocks(t)
        plain = _strip(result)
        assert "Lang" in plain
        assert "Python" in plain
        assert "Rust" in plain
        # Must not contain raw pipe-separator row
        assert "|---|---|" not in plain

    def test_loose_table_fully_loose(self):
        """Fully-loose GFM table: no boundary pipes anywhere."""
        t = "A | B | C\n---|---|---\nx | y | z"
        result = render_stateful_blocks(t)
        plain = _strip(result)
        assert "A" in plain
        assert "x" in plain
        # separator row must be replaced by dashes
        assert "---|" not in plain

    def test_loose_table_separator_shape_must_match_header(self):
        result = render_stateful_blocks("foo | bar\n---\n")
        plain = _strip(result)
        assert "foo | bar" in plain
        assert "foo  bar" not in plain

    def test_streaming_loose_table_strict_separator(self):
        """StreamingBlockBuffer handles loose header + strict separator."""
        buf = StreamingBlockBuffer()
        assert buf.process_line("Lang | Type") is None      # pending
        assert buf.process_line("|---|---|") is None        # rescues header, buffers sep
        assert buf.process_line("Python | Dynamic") is None # loose data row
        rendered = buf.flush()
        assert rendered is not None
        plain = _strip(rendered)
        assert "Lang" in plain
        assert "Python" in plain

    def test_streaming_loose_table_fully_loose(self):
        """StreamingBlockBuffer handles fully-loose table (no boundary pipes)."""
        buf = StreamingBlockBuffer()
        assert buf.process_line("A | B") is None
        assert buf.process_line("---|---") is None
        assert buf.process_line("x | y") is None
        rendered = buf.flush()
        assert rendered is not None
        plain = _strip(rendered)
        assert "A" in plain
        assert "x" in plain


# ---------------------------------------------------------------------------
# Feature 1: Task lists
# ---------------------------------------------------------------------------

class TestTaskLists:
    """apply_block_line renders task list items with checkbox symbols."""

    def test_unchecked_box_gets_circle_symbol(self):
        result = apply_block_line("- [ ] do something")
        assert "○" in result

    def test_checked_box_gets_checkmark_symbol(self):
        result = apply_block_line("- [x] done")
        assert "✓" in result

    def test_checked_uppercase_x(self):
        result = apply_block_line("- [X] also done")
        assert "✓" in result

    def test_unchecked_has_dim_style(self):
        result = apply_block_line("- [ ] pending task")
        # dim style for unchecked checkbox
        assert "\033[2m" in result
        assert "○" in result

    def test_checked_has_green_style(self):
        result = apply_block_line("- [x] completed task")
        # green bold style for checked
        assert "\033[1;32m" in result
        assert "✓" in result

    def test_task_content_is_rendered_inline(self):
        result = apply_block_line("- [x] **bold** item")
        assert "✓" in result
        assert "\033[1m" in result  # bold applied to content

    def test_task_unchecked_contains_content(self):
        result = apply_block_line("- [ ] buy groceries")
        assert "buy groceries" in result

    def test_task_bullet_present(self):
        result = apply_block_line("- [ ] task")
        assert "•" in result

    def test_nested_task_indented(self):
        result = apply_block_line("  - [x] sub-task")
        # indented task list item
        assert "✓" in result
        assert result.startswith("  ")

    def test_non_task_ul_not_affected(self):
        result = apply_block_line("- regular item")
        assert "○" not in result
        assert "✓" not in result
        assert "•" in result

    def test_task_via_format_response(self):
        text = "- [ ] unchecked\n- [x] checked\n"
        result = format_response(text)
        assert "○" in result
        assert "✓" in result


# ---------------------------------------------------------------------------
# Feature 2: Ordered lists
# ---------------------------------------------------------------------------

class TestOrderedLists:
    """apply_block_line renders OL items with dim numeral."""

    def test_simple_ol_item(self):
        result = apply_block_line("1. first item")
        assert "\033[2m1.\033[0m" in result
        assert "first item" in result

    def test_ol_with_paren_delimiter(self):
        result = apply_block_line("2) second item")
        assert "\033[2m2.\033[0m" in result
        assert "second item" in result

    def test_ol_preserves_source_number(self):
        result = apply_block_line("42. forty-two")
        assert "\033[2m42.\033[0m" in result
        assert "forty-two" in result

    def test_ol_content_inline_rendered(self):
        result = apply_block_line("3. **bold content**")
        assert "\033[1m" in result  # bold
        assert "bold content" in result

    def test_ol_indented(self):
        result = apply_block_line("  1. nested")
        assert result.startswith("  ")
        assert "\033[2m1.\033[0m" in result

    def test_ol_not_setext_candidate(self):
        # "1. text" followed by "---" should not be treated as a heading
        result = render_stateful_blocks("1. item\n---\n")
        # Should not contain h2 heading style
        assert "\033[1;37m" not in result
        # Should contain the OL rendering
        assert "item" in result

    def test_ol_via_format_response(self):
        text = "1. first\n2. second\n3. third\n"
        result = format_response(text)
        assert "\033[2m1.\033[0m" in result
        assert "\033[2m2.\033[0m" in result
        assert "\033[2m3.\033[0m" in result

    def test_ol_stateful_multiple_items(self):
        text = "1. alpha\n2. beta\n3. gamma\n"
        result = render_stateful_blocks(text)
        # All items pass through for apply_block_line in pass 3
        # render_stateful_blocks just passes them; apply_block_line does the work
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result


# ---------------------------------------------------------------------------
# Feature 3: Nested blockquotes
# ---------------------------------------------------------------------------

class TestNestedBlockquotes:
    """Blockquote depth is tracked and rendered with additional indentation/dimming."""

    def test_depth_1_basic(self):
        result = apply_block_line("> hello")
        assert "▌" in result
        assert "hello" in result

    def test_depth_2_has_indent(self):
        result = apply_block_line("> > nested")
        assert "▌" in result
        assert "nested" in result
        # depth-2 should have 2 spaces of indent before the gutter
        assert result.startswith("  ")

    def test_depth_3_deeper_indent(self):
        result = apply_block_line("> > > deep")
        assert "▌" in result
        # depth-3: 4 spaces of indent
        assert result.startswith("    ")

    def test_depth_2_has_extra_dim(self):
        result = apply_block_line("> > nested")
        # depth-2 uses dim prefix on top of base blockquote ANSI
        # Base _BLOCKQUOTE_ANSI = "\033[2m", depth-2 adds one more dim
        assert result.count("\033[2m") >= 2

    def test_depth_1_no_extra_indent(self):
        result = apply_block_line("> single")
        assert not result.startswith("  ")

    def test_render_stateful_depth1(self):
        text = "> quote line\n"
        result = render_stateful_blocks(text)
        assert "▌" in result
        assert "quote line" in result

    def test_render_stateful_depth2(self):
        text = "> > nested\n"
        result = render_stateful_blocks(text)
        assert "▌" in result
        assert "nested" in result
        assert result.startswith("  ")

    def test_bq_depth_reset_on_blank(self):
        result = render_stateful_blocks("> q\n\n> new")
        assert result.count("▌") == 2

    def test_streaming_depth1(self):
        buf = StreamingBlockBuffer()
        # First BQ line is buffered for setext lookahead
        r = buf.process_line("> depth1")
        assert r is None
        flushed = buf.flush()
        assert flushed is not None
        assert "▌" in flushed
        assert "depth1" in flushed

    def test_streaming_depth2(self):
        buf = StreamingBlockBuffer()
        # First BQ line buffered; flush to get it
        buf.process_line("> > depth2")
        flushed = buf.flush()
        assert flushed is not None
        assert "▌" in flushed
        assert flushed.startswith("  ")

    def test_streaming_depth_continuation(self):
        buf = StreamingBlockBuffer()
        buf.process_line("> > level2")
        result = buf.process_line("continuation line")
        # Continuation is rendered at current depth
        assert result is not None
        assert "▌" in result

    def test_format_response_nested(self):
        text = "> > double nested\n"
        result = format_response(text)
        assert "▌" in result
        assert "double nested" in result


# ---------------------------------------------------------------------------
# Feature 4: Setext headings inside blockquotes
# ---------------------------------------------------------------------------

class TestSetextInBlockquote:
    """Setext markers inside blockquotes produce styled headings with gutter."""

    def test_setext_h1_in_blockquote(self):
        text = "> Heading\n> ========\n"
        result = render_stateful_blocks(text)
        # Should contain the h1 heading style inside a gutter
        assert "▌" in result
        assert "Heading" in result
        # h1 style
        assert "\033[1;97m" in result
        # The setext underline itself should NOT appear as a rendered BQ line
        assert "=======" not in _strip(result)

    def test_setext_h2_in_blockquote(self):
        text = "> Subheading\n> ----------\n"
        result = render_stateful_blocks(text)
        assert "▌" in result
        assert "Subheading" in result
        # h2 style
        assert "\033[1;37m" in result
        # The setext underline should not appear in plain output
        assert "----------" not in _strip(result)

    def test_non_setext_two_bq_lines(self):
        text = "> first\n> second\n"
        result = render_stateful_blocks(text)
        # Both lines should appear as normal blockquote lines
        assert result.count("▌") == 2
        assert "first" in result
        assert "second" in result

    def test_streaming_setext_h1_in_blockquote(self):
        buf = StreamingBlockBuffer()
        r1 = buf.process_line("> Heading")  # buffered → None
        r2 = buf.process_line("> ========")  # setext detected → returns heading in gutter
        flushed = buf.flush()
        combined = "\n".join(x for x in [r1, r2, flushed] if x)
        assert "▌" in combined
        assert "Heading" in combined
        assert "\033[1;97m" in combined

    def test_streaming_setext_h2_in_blockquote(self):
        buf = StreamingBlockBuffer()
        r1 = buf.process_line("> Sub")   # buffered → None
        r2 = buf.process_line("> ---")   # setext detected → returns h2 heading in gutter
        flushed = buf.flush()
        combined = "\n".join(x for x in [r1, r2, flushed] if x)
        assert "Sub" in combined
        assert "\033[1;37m" in combined

    def test_blank_line_not_setext(self):
        # Blank inner content is not a heading candidate
        text = "> \n> ====\n"
        result = render_stateful_blocks(text)
        # Should not apply heading style
        assert "\033[1;97m" not in result

    def test_format_response_setext_in_bq(self):
        text = "> Title\n> =====\n"
        result = format_response(text)
        assert "▌" in result
        assert "Title" in result
        assert "\033[1;97m" in result


# ---------------------------------------------------------------------------
# Feature 5: Link reference definitions → resolved links
# ---------------------------------------------------------------------------

class TestRefLinkResolution:
    """Reference link definitions are collected and resolved in inline text."""

    def test_ref_link_def_suppressed(self):
        # [ref]: url lines produce empty output
        result = apply_block_line("[myref]: https://example.com")
        assert result == ""

    def test_ref_link_use_resolved(self):
        ref_map = {"myref": "https://example.com"}
        result = apply_inline_markdown("[click here][myref]", ref_map=ref_map)
        assert "click here" in result
        assert "https://example.com" in result
        # Should use link ANSI style
        assert "\033[38;2;88;166;255m" in result

    def test_ref_link_collapsed_resolved(self):
        ref_map = {"myref": "https://example.com"}
        result = apply_inline_markdown("[myref][]", ref_map=ref_map)
        assert "myref" in result
        assert "https://example.com" in result

    def test_ref_link_case_insensitive_key(self):
        ref_map = {"myref": "https://example.com"}
        result = apply_inline_markdown("[text][MyRef]", ref_map=ref_map)
        assert "https://example.com" in result

    def test_ref_link_unknown_leaves_as_is(self):
        ref_map = {"other": "https://other.com"}
        result = apply_inline_markdown("[text][unknown]", ref_map=ref_map)
        # Unknown ref should be left unchanged
        assert "[text][unknown]" in result

    def test_ref_link_no_map_leaves_as_is(self):
        result = apply_inline_markdown("[text][ref]")
        assert "[text][ref]" in result

    def test_format_response_resolves_refs(self):
        text = "[ref]: https://example.com\n\nSee [ref][] for details.\n"
        result = format_response(text)
        assert "https://example.com" in result
        assert "ref" in result
        # The ref def line itself should not appear as raw text
        lines = _strip(result).splitlines()
        assert not any(l.strip() == "[ref]: https://example.com" for l in lines)

    def test_format_response_text_ref_resolved(self):
        text = "[docs]: https://docs.example.com\n\nRead the [documentation][docs].\n"
        result = format_response(text)
        assert "https://docs.example.com" in result
        assert "documentation" in result

    def test_streaming_ref_map_accumulated(self):
        # StreamingBlockBuffer collects ref defs into _ref_map as lines arrive.
        # Inline rendering of plain text happens downstream (not inside the buffer);
        # the buffer passes ref_map to apply_inline_markdown only for BQ/heading content.
        # Verify that the ref_map is populated after processing a ref def line.
        buf = StreamingBlockBuffer()
        buf.process_line("[myref]: https://example.com")
        assert "myref" in buf._ref_map
        assert buf._ref_map["myref"] == "https://example.com"

    def test_fenced_ref_def_does_not_leak_into_batch_resolution(self):
        result = format_response("```\n[ref]: https://example.com\n```\nUse [x][ref].\n")
        plain = _strip(result)
        assert "[x][ref]" in plain

    def test_fenced_ref_def_does_not_populate_streaming_ref_map(self):
        buf = StreamingBlockBuffer()
        buf.process_line("```")
        buf.process_line("[ref]: https://example.com")
        buf.process_line("```")
        buf.flush()
        assert "ref" not in buf._ref_map

    def test_streaming_bq_line_uses_ref_map(self):
        # BQ continuation content IS rendered via apply_inline_markdown with ref_map.
        buf = StreamingBlockBuffer()
        buf.process_line("[link]: https://example.com")
        # Enter blockquote with a BQ line containing the ref link
        buf.process_line("> First line")  # buffered for setext lookahead
        # Second BQ line flushes the first one (rendered with ref_map via _render_bq_depth)
        result = buf.process_line("> See [link][] for info")
        # result is the rendered first BQ line "First line"
        # The second line is buffered in pending
        flushed = buf.flush()
        combined = "\n".join(x for x in [result, flushed] if x)
        # The second BQ line "See [link][] for info" should have the URL resolved
        assert "https://example.com" in combined

    def test_ref_map_passed_through_bold(self):
        # ref_map should be propagated through bold/italic recursive calls
        ref_map = {"r": "https://r.com"}
        result = apply_inline_markdown("**see [r][]**", ref_map=ref_map)
        assert "https://r.com" in result

    def test_ref_link_with_quoted_title_in_def(self):
        text = '[myref]: https://example.com "Example Site"\n\n[click][myref]\n'
        result = format_response(text)
        assert "https://example.com" in result

    def test_ref_link_with_paren_title_resolves(self):
        # Bug fix: parenthesized title in ref def must be collected into ref_map
        text = '[myref]: https://example.com (Example Site)\n\n[click][myref]\n'
        result = format_response(text)
        assert "https://example.com" in result
        assert "click" in result

    def test_ref_link_with_single_quote_title_resolves(self):
        # Bug fix: single-quoted title in ref def must be collected into ref_map
        text = "[myref]: https://example.com 'Example Site'\n\n[click][myref]\n"
        result = format_response(text)
        assert "https://example.com" in result
        assert "click" in result

    def test_multiple_refs_in_document(self):
        text = (
            "[a]: https://a.com\n"
            "[b]: https://b.com\n"
            "\n"
            "See [link a][a] and [link b][b].\n"
        )
        result = format_response(text)
        assert "https://a.com" in result
        assert "https://b.com" in result
        assert "link a" in result
        assert "link b" in result

    def test_ref_collapsed_label_equals_text(self):
        # [myref][] collapsed form uses text ('myref') as the lookup key
        ref_map = {"myref": "https://example.com"}
        result = apply_inline_markdown("[myref][]", ref_map=ref_map)
        assert "myref" in result
        assert "https://example.com" in result

    def test_ref_unknown_label_left_as_is(self):
        ref_map = {"other": "https://other.com"}
        result = apply_inline_markdown("[text][unknown]", ref_map=ref_map)
        assert "[text][unknown]" in result

    def test_ref_no_map_use_syntax_left_as_is(self):
        # Without ref_map, [text][ref] is not touched
        result = apply_inline_markdown("[text][ref]")
        assert "[text][ref]" in result

    def test_ref_def_line_suppressed_in_format_response(self):
        text = "[ref]: https://example.com\n\nHello world.\n"
        result = format_response(text)
        plain = _strip(result)
        assert not any(l.strip().startswith("[ref]:") for l in plain.splitlines())

    def test_streaming_ref_before_use_in_bq_resolves(self):
        # Ref defined before BQ line — resolved when BQ content is rendered
        buf = StreamingBlockBuffer()
        buf.process_line("[link]: https://example.com")
        buf.process_line("> See [link][] here")  # buffered
        result = buf.process_line("> next line")  # flushes buffered line
        flushed = buf.flush()
        combined = "\n".join(x for x in [result, flushed] if x)
        assert "https://example.com" in combined

    def test_streaming_ref_after_use_does_not_resolve(self):
        # Ref defined AFTER the usage line — acceptable: streaming can't look ahead.
        # The buffer uses a one-tick delay: "See [myref][] for info." is held as
        # pending and emitted (as-is) when the next line arrives (the ref def line).
        # apply_inline_markdown is NOT called inside StreamingBlockBuffer for plain
        # lines, so the ref cannot be resolved even if ref_map were populated.
        buf = StreamingBlockBuffer()
        r1 = buf.process_line("See [myref][] for info.")  # buffered → None
        r2 = buf.process_line("[myref]: https://example.com")  # emits usage, buffers ref def
        flushed = buf.flush()  # emits ref def line
        all_parts = [x for x in [r1, r2, flushed] if x]
        # The usage line ("for info") is emitted as plain text with literal brackets
        usage_part = next((p for p in all_parts if "for info" in p), None)
        assert usage_part is not None
        assert "[myref][]" in usage_part

    def test_streaming_paren_title_ref_collected(self):
        # Streaming collector must also handle paren-titled ref defs
        buf = StreamingBlockBuffer()
        buf.process_line("[myref]: https://example.com (Title)")
        assert "myref" in buf._ref_map
        assert buf._ref_map["myref"] == "https://example.com"

    def test_ref_in_bold_propagates_ref_map(self):
        # ref_map must propagate into bold recursive call
        ref_map = {"r": "https://r.com"}
        result = apply_inline_markdown("**see [text][r] here**", ref_map=ref_map)
        assert "https://r.com" in result
        assert "text" in result


# ---------------------------------------------------------------------------
# Feature 1 (Ordered lists) — additional edge cases
# ---------------------------------------------------------------------------

class TestOrderedListsEdgeCases:
    """Edge cases for ordered list rendering."""

    def test_ol_paren_delimiter_in_format_response(self):
        # 1) item should render same as 1. item
        result = format_response("1) first\n2) second\n")
        assert "\033[2m1.\033[0m" in result
        assert "\033[2m2.\033[0m" in result

    def test_ol_blank_line_between_items(self):
        # Blank line between OL items — both still rendered
        result = format_response("1. alpha\n\n2. beta\n")
        assert "\033[2m1.\033[0m" in result
        assert "\033[2m2.\033[0m" in result

    def test_ol_mixed_with_ul(self):
        # OL followed by UL — both render correctly
        result = format_response("1. ordered\n- unordered\n")
        assert "\033[2m1.\033[0m" in result
        assert "•" in result

    def test_ol_not_setext_with_dash_marker(self):
        # "1. foo\n---" must NOT become an h2 setext heading
        result = render_stateful_blocks("1. foo\n---\n")
        assert "\033[1;37m" not in result
        assert "foo" in result

    def test_ol_not_setext_with_paren_delimiter(self):
        # "1) foo\n---" must NOT become an h2 setext heading
        result = render_stateful_blocks("1) foo\n---\n")
        assert "\033[1;37m" not in result

    def test_ol_inline_markdown_bold_content(self):
        result = apply_block_line("1. **important**")
        assert "\033[1m" in result
        assert "important" in result

    def test_ol_inline_markdown_code_content(self):
        result = apply_block_line("2. Use `code` here")
        assert "code" in result

    def test_ol_indented_nested(self):
        # Indented OL item at level 1
        result = apply_block_line("  1. nested item")
        assert result.startswith("  ")
        assert "\033[2m1.\033[0m" in result

    def test_ol_large_number(self):
        result = apply_block_line("99. ninety-nine")
        assert "\033[2m99.\033[0m" in result

    def test_ol_via_streaming(self):
        buf = StreamingBlockBuffer()
        r1 = buf.process_line("1. first")
        r2 = buf.process_line("2. second")
        flushed = buf.flush()
        # OL lines pass through streaming as plain lines
        combined = "\n".join(x for x in [r1, r2, flushed] if x is not None)
        assert "first" in combined
        assert "second" in combined


# ---------------------------------------------------------------------------
# Feature 2 (Task lists) — additional edge cases
# ---------------------------------------------------------------------------

class TestTaskListsEdgeCases:
    """Edge cases for task list rendering."""

    def test_task_no_content_after_checkbox_checked(self):
        # "- [x]" with nothing after — should render checkbox, no crash
        result = apply_block_line("- [x]")
        assert "✓" in result

    def test_task_no_content_after_checkbox_unchecked(self):
        result = apply_block_line("- [ ]")
        assert "○" in result

    def test_task_nested_in_ul(self):
        # "  - [x] nested" — indented task list with circle bullet
        result = apply_block_line("  - [x] nested task")
        assert "✓" in result
        assert result.startswith("  ")
        # Level-1 bullet is ◦
        assert "◦" in result

    def test_task_double_nested(self):
        result = apply_block_line("    - [ ] deep task")
        assert "○" in result
        assert result.startswith("    ")

    def test_task_content_inline_code(self):
        result = apply_block_line("- [x] run `pytest`")
        assert "✓" in result
        assert "pytest" in result

    def test_task_content_bold(self):
        result = apply_block_line("- [ ] **urgent** item")
        assert "○" in result
        assert "\033[1m" in result
        assert "urgent" in result

    def test_task_star_marker(self):
        # Task with * list marker
        result = apply_block_line("* [x] done with star")
        assert "✓" in result

    def test_task_plus_marker(self):
        # Task with + list marker
        result = apply_block_line("+ [ ] pending with plus")
        assert "○" in result

    def test_task_via_render_stateful(self):
        text = "- [x] done\n- [ ] pending\n"
        result = render_stateful_blocks(text)
        # render_stateful_blocks doesn't apply block-level rendering, but items pass through
        # as plain text (apply_block_line is called in format_response pass 3)
        assert "done" in result
        assert "pending" in result

    def test_task_via_format_response_inline_bold(self):
        text = "- [x] **bold task**\n"
        result = format_response(text)
        assert "✓" in result
        assert "\033[1m" in result


# ---------------------------------------------------------------------------
# Feature 3 (Nested blockquotes) — additional edge cases
# ---------------------------------------------------------------------------

class TestNestedBlockquotesEdgeCases:
    """Edge cases for nested blockquote depth rendering."""

    def test_depth_3_cap_at_double_dim(self):
        # Depth 3 adds min(2, 2) = 2 extra dim codes (capped)
        result = apply_block_line("> > > triple")
        # 4-space indent for depth-3
        assert result.startswith("    ")
        assert "▌" in result
        # dim_prefix = "\033[2m" * min(2, 2) = 2 dims + base dim = 3 total
        assert result.count("\033[2m") >= 3

    def test_depth_2_indent_is_two_spaces(self):
        result = apply_block_line("> > nested")
        assert result.startswith("  ")
        assert not result.startswith("    ")

    def test_depth_3_indent_is_four_spaces(self):
        result = apply_block_line("> > > triple")
        assert result.startswith("    ")

    def test_depth_reset_on_blank_in_stateful(self):
        text = "> > deep\n\n> shallow\n"
        result = render_stateful_blocks(text)
        assert result.count("▌") == 2
        # After blank, shallow is depth-1, no extra indent
        lines = result.splitlines()
        shallow_line = next((l for l in lines if "shallow" in l), None)
        assert shallow_line is not None
        assert not shallow_line.startswith("  ")

    def test_lazy_continuation_at_depth2_stateful(self):
        # Lazy continuation (no >) while in depth-2 BQ
        text = "> > first line\nlazy cont\n"
        result = render_stateful_blocks(text)
        # Lazy cont rendered at current depth (2)
        assert result.count("▌") == 2
        assert "lazy cont" in result

    def test_streaming_depth2_then_depth1(self):
        buf = StreamingBlockBuffer()
        buf.process_line("> > deep")  # buffered
        result = buf.process_line("> shallow")  # emits deep, buffers shallow
        flushed = buf.flush()
        assert result is not None
        assert "deep" in result
        assert result.startswith("  ")
        assert flushed is not None
        assert "shallow" in flushed

    def test_streaming_depth_reset_on_blank(self):
        buf = StreamingBlockBuffer()
        buf.process_line("> > deep")  # buffered
        r_deep = buf.process_line("")  # blank exits BQ, emits pending
        r_shallow = buf.process_line("> shallow")
        flushed = buf.flush()
        # deep should have been emitted
        assert r_deep is not None
        assert "deep" in r_deep
        # shallow is a new BQ
        assert flushed is not None
        assert "shallow" in flushed

    def test_bq_ansi_line_adjacent(self):
        # ANSI line (pre-highlighted code) inside BQ context still has gutter
        text = "> before\n\x1b[32mcode\x1b[0m\n> after\n"
        result = render_stateful_blocks(text)
        # The ANSI line should have a gutter since it's adjacent/inside BQ
        assert "▌" in result

    def test_depth1_no_extra_dim(self):
        result = apply_block_line("> solo")
        # depth-1: no extra dim beyond _BLOCKQUOTE_ANSI itself
        # _BLOCKQUOTE_ANSI = "\033[2m", dim_prefix = "" for depth 1
        # So exactly 1 leading \033[2m
        # Split on ▌ to check prefix
        before_gutter = result.split("▌")[0]
        assert before_gutter.count("\033[2m") == 1


# ---------------------------------------------------------------------------
# Feature 4 (Setext in blockquotes) — additional edge cases
# ---------------------------------------------------------------------------

class TestSetextInBlockquoteEdgeCases:
    """Edge cases for setext headings rendered inside blockquotes."""

    def test_blank_inner_does_not_trigger_setext(self):
        # "> \n> ===" — blank content is not a heading candidate
        text = "> \n> ===\n"
        result = render_stateful_blocks(text)
        assert "\033[1;97m" not in result

    def test_ol_inner_does_not_trigger_setext(self):
        # "> 1. list\n> ---" — OL item is not a setext heading candidate
        text = "> 1. list\n> ---\n"
        result = render_stateful_blocks(text)
        assert "\033[1;37m" not in result
        assert "list" in result

    def test_setext_h1_single_eq_does_not_trigger(self):
        # Single '=' is not a setext h1 marker (needs 2+)
        text = "> Heading\n> =\n"
        result = render_stateful_blocks(text)
        assert "\033[1;97m" not in result

    def test_two_normal_bq_lines_both_rendered(self):
        text = "> first\n> second\n"
        result = render_stateful_blocks(text)
        assert result.count("▌") == 2
        assert "first" in result
        assert "second" in result

    def test_setext_h2_in_blockquote_stateful(self):
        text = "> Subtitle\n> ---\n"
        result = render_stateful_blocks(text)
        assert "▌" in result
        assert "Subtitle" in result
        assert "\033[1;37m" in result
        assert "---" not in _strip(result)

    def test_setext_h1_in_blockquote_stateful(self):
        text = "> Title\n> ===\n"
        result = render_stateful_blocks(text)
        assert "▌" in result
        assert "Title" in result
        assert "\033[1;97m" in result

    def test_streaming_setext_h2_in_bq(self):
        buf = StreamingBlockBuffer()
        r1 = buf.process_line("> Sub")
        r2 = buf.process_line("> ---")
        flushed = buf.flush()
        combined = "\n".join(x for x in [r1, r2, flushed] if x)
        assert "Sub" in combined
        assert "\033[1;37m" in combined

    def test_format_response_setext_h2_in_bq(self):
        text = "> Chapter\n> --------\n"
        result = format_response(text)
        assert "▌" in result
        assert "Chapter" in result
        assert "\033[1;37m" in result

    def test_setext_in_depth2_bq(self):
        # Setext heading inside depth-2 blockquote
        text = "> > Heading\n> > ===\n"
        result = render_stateful_blocks(text)
        assert "\033[1;97m" in result
        assert "Heading" in result
        # Depth-2 indent
        assert result.startswith("  ")


def test_diff_renderer_marker_sigils_have_distinct_colours():
    diff = (
        "--- a/f.py\n+++ b/f.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    lines = DiffRenderer().to_lines(diff)
    all_ansi = "\n".join(lines)
    assert "38;2;255;123;114" in all_ansi
    assert "38;2;86;211;100" in all_ansi
