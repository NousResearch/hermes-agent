"""Tests for agent/rich_output.py — syntax highlighting, diff rendering, code block detection."""

import pytest
from unittest.mock import patch

from agent.rich_output import (
    _DIFF_BG_ADD_HL,
    _DIFF_BG_DEL_HL,
    _DIFF_MAX_LINES,
    DiffRenderer,
    FilePathFormatter,
    LanguageDetector,
    StreamingCodeBlockHighlighter,
    SyntaxHighlighter,
    _highlight_inline_code,
    _intra_diff,
    _parse_diff_filename,
    clean_command_output,
    format_response,
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

    def test_to_ansi_brackets_and_backslashes_render_literally(self):
        import re
        plain = re.sub(r"\x1b\[[0-9;]*m", "", self.hl.to_ansi(r'x = r"\[abc]"', language="python"))
        assert r'x = r"\[abc]"' in plain
        assert r"\]" not in plain


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
        import re
        plain = re.sub(r"\x1b\[[0-9;]*m", "", format_response(f"```{lang}\nint x;\n```\n"))
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
