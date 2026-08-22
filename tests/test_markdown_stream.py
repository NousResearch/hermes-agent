"""Tests for hermes_cli.markdown_stream — streaming & non-streaming markdown rendering."""

import re
import pytest

from hermes_cli.markdown_stream import MarkdownStreamProcessor

# Helpers
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)

def _has_ansi(text: str, code: str) -> bool:
    """Check if a specific ANSI escape code appears in text."""
    return f"\033[{code}m" in text


# =========================================================================
# Inline markdown transforms (NORMAL mode)
# =========================================================================

class TestInlineBold:
    def test_bold_double_star(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("Hello **world**")
        assert _has_ansi(out, "1")  # bold
        assert "world" in _strip_ansi(out)

    def test_bold_double_under(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("Hello __world__")
        assert _has_ansi(out, "1")
        assert "world" in _strip_ansi(out)

    def test_bold_mid_sentence(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("some **bold** text here")
        plain = _strip_ansi(out)
        assert "bold" in plain
        assert "some" in plain
        assert "text here" in plain


class TestInlineItalic:
    def test_italic_star(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("Hello *world*")
        assert _has_ansi(out, "3")  # italic
        assert "world" in _strip_ansi(out)

    def test_italic_under(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("Hello _world_")
        assert _has_ansi(out, "3")


class TestInlineBoldItalic:
    def test_bold_italic(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("***both***")
        assert _has_ansi(out, "1;3")  # bold+italic
        assert "both" in _strip_ansi(out)


class TestInlineStrikethrough:
    def test_strikethrough(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("~~removed~~")
        assert _has_ansi(out, "9")  # strikethrough
        assert "removed" in _strip_ansi(out)


class TestInlineCode:
    def test_inline_code(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("Use `foo()` here")
        # Default processor uses plain bold (no skin-supplied fg/bg).
        assert _has_ansi(out, "1")  # bold
        assert "foo()" in _strip_ansi(out)

    def test_inline_code_no_match_triple_backtick(self):
        """Triple backticks should NOT be treated as inline code."""
        p = MarkdownStreamProcessor()
        out = p.feed_line("```python")
        plain = _strip_ansi(out)
        # Should be a fence header, not inline code formatting
        assert "python" in plain


class TestHeaders:
    def test_h1(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("# Title")
        assert _has_ansi(out, "1")  # bold
        assert _has_ansi(out, "4")  # underline (h1 specific)
        assert "Title" in _strip_ansi(out)

    def test_h2(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("## Subtitle")
        assert _has_ansi(out, "1")
        assert "Subtitle" in _strip_ansi(out)

    def test_h3(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("### Section")
        assert _has_ansi(out, "1")
        assert "Section" in _strip_ansi(out)

    def test_h4(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("#### Subsection")
        assert "Subsection" in _strip_ansi(out)

    def test_h5(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("##### Deep Section")
        assert _has_ansi(out, "1")
        assert "Deep Section" in _strip_ansi(out)
        assert "#" not in _strip_ansi(out)

    def test_h6(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("###### Deepest")
        assert "Deepest" in _strip_ansi(out)
        assert "#" not in _strip_ansi(out)

    def test_header_strips_inline_bold(self):
        """Bold markers inside headers should be stripped, not shown literally."""
        p = MarkdownStreamProcessor()
        out = p.feed_line("### 1. **set_context.sh - Bug**")
        plain = _strip_ansi(out)
        assert "set_context.sh - Bug" in plain
        assert "**" not in plain

    def test_header_strips_inline_code(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("## Using `foo()` in production")
        plain = _strip_ansi(out)
        assert "foo()" in plain
        assert "`" not in plain

    def test_header_strips_mixed_inline(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("# **Bold** and *italic* and `code`")
        plain = _strip_ansi(out)
        assert "Bold and italic and code" in plain
        assert "**" not in plain
        assert "*" not in plain
        assert "`" not in plain


class TestBlockquote:
    def test_blockquote(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("> quoted text")
        assert _has_ansi(out, "3")  # italic
        assert "│" in _strip_ansi(out)
        assert "quoted text" in _strip_ansi(out)


class TestListMarkers:
    def test_unordered_dash(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("- item one")
        assert "•" in _strip_ansi(out)
        assert "item one" in _strip_ansi(out)

    def test_unordered_star(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("* item two")
        assert "•" in _strip_ansi(out)

    def test_unordered_plus(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("+ item three")
        assert "•" in _strip_ansi(out)

    def test_ordered_list(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("1. first")
        plain = _strip_ansi(out)
        assert "1." in plain
        assert "first" in plain

    def test_indented_list(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("  - nested")
        assert "•" in _strip_ansi(out)
        assert "nested" in _strip_ansi(out)


class TestLinks:
    def test_link(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("See [docs](https://example.com) for info")
        plain = _strip_ansi(out)
        assert "docs" in plain
        assert "https://example.com" in plain
        assert _has_ansi(out, "4")  # underline


class TestHorizontalRule:
    def test_dashes(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("---")
        assert "─" in _strip_ansi(out)

    def test_stars(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("***")
        # Could be hr or bold-italic marker; with nothing between, hr wins
        # because the full-line regex matches first
        assert "─" in _strip_ansi(out)


class TestRobustness:
    """Edge cases that caused corruption in adversarial markdown."""

    def test_unmatched_bold_passthrough(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("some **unmatched bold")
        assert "**unmatched" in _strip_ansi(out)

    def test_unmatched_italic_passthrough(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("some *unmatched italic")
        assert "*unmatched" in _strip_ansi(out)

    def test_triple_backticks_mid_text_not_fence(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("Use ```python to start code")
        plain = _strip_ansi(out)
        assert "```python" in plain
        assert "┌" not in plain  # NOT a fence header

    def test_inline_code_protected_from_bold(self):
        """Bold markers inside inline code should not be interpreted."""
        p = MarkdownStreamProcessor()
        out = p.feed_line("Run `**not bold**` here")
        plain = _strip_ansi(out)
        assert "**not bold**" in plain

    def test_mixed_unmatched_markers(self):
        """Multiple unmatched markers shouldn't cascade."""
        p = MarkdownStreamProcessor()
        out = p.feed_line("text ** more _ stuff ~~ end")
        plain = _strip_ansi(out)
        assert "**" in plain
        assert "_" in plain
        assert "~~" in plain


class TestEscapedCharacters:
    def test_escaped_star(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("\\*not bold\\*")
        plain = _strip_ansi(out)
        assert "*not bold*" in plain
        # Should NOT have bold ANSI (only the base color, if any)
        assert "\033[1m" not in out or "\033[1;" not in out

    def test_escaped_backtick(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("Use \\`backtick\\` literally")
        plain = _strip_ansi(out)
        assert "`backtick`" in plain


class TestEmptyAndPlainLines:
    def test_empty_line(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("")
        assert _strip_ansi(out) == ""

    def test_plain_text(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("Just plain text, no markdown.")
        assert "Just plain text, no markdown." in _strip_ansi(out)


class TestBaseAnsi:
    def test_base_color_applied(self):
        base = "\033[38;2;255;248;220m"
        p = MarkdownStreamProcessor(base_ansi=base)
        out = p.feed_line("Hello")
        assert out.startswith(base)

    def test_no_base_color(self):
        p = MarkdownStreamProcessor(base_ansi="")
        out = p.feed_line("Hello")
        assert "Hello" in _strip_ansi(out)


# =========================================================================
# Code block handling (CODE_BLOCK mode)
# =========================================================================

class TestCodeBlockBasic:
    def test_open_close_cycle(self):
        p = MarkdownStreamProcessor()
        header = p.feed_line("```python")
        assert "python" in _strip_ansi(header)
        assert "┌" in _strip_ansi(header)

        code = p.feed_line("print('hello')")
        # Should have Pygments highlighting (some ANSI in there)
        assert "print" in _strip_ansi(code)

        footer = p.feed_line("```")
        assert "└" in _strip_ansi(footer)

    def test_code_not_inline_formatted(self):
        """Lines inside code block should NOT get inline markdown transforms."""
        p = MarkdownStreamProcessor()
        p.feed_line("```python")
        code_out = p.feed_line("x = **kwargs")
        plain = _strip_ansi(code_out)
        # **kwargs should be literal, not bolded
        assert "**kwargs" in plain
        p.feed_line("```")

    def test_no_language_fence(self):
        p = MarkdownStreamProcessor()
        header = p.feed_line("```")
        assert "┌" in _strip_ansi(header)
        # No language label, just the fence line
        code = p.feed_line("some text")
        assert "some text" in _strip_ansi(code)
        footer = p.feed_line("```")
        assert "└" in _strip_ansi(footer)

    def test_unknown_language_no_crash(self):
        p = MarkdownStreamProcessor()
        header = p.feed_line("```brainfuck")
        assert "brainfuck" in _strip_ansi(header)
        code = p.feed_line("++++++++[>++++[>++>+++>+++>+<<<<-]")
        assert _strip_ansi(code).strip()  # Something was emitted
        p.feed_line("```")

    def test_code_lines_indented(self):
        """Code lines should be indented with 2 spaces."""
        p = MarkdownStreamProcessor()
        p.feed_line("```python")
        code = p.feed_line("x = 1")
        assert _strip_ansi(code).startswith("  ")
        p.feed_line("```")

    def test_labelled_fence_respects_caller_width(self):
        p = MarkdownStreamProcessor(max_width=32)

        header = _strip_ansi(p.feed_line("```python"))
        footer = _strip_ansi(p.feed_line("```"))

        assert len(header) <= 32
        assert len(footer) <= 32
        assert " python " in header

    def test_bare_fence_respects_caller_width(self):
        p = MarkdownStreamProcessor(max_width=20)

        header = _strip_ansi(p.feed_line("```"))
        footer = _strip_ansi(p.feed_line("```"))

        assert len(header) <= 20
        assert len(footer) <= 20


class TestMultipleCodeBlocks:
    def test_two_blocks_with_normal_text(self):
        p = MarkdownStreamProcessor()
        # First block
        h1 = p.feed_line("```python")
        assert "┌" in _strip_ansi(h1)
        p.feed_line("x = 1")
        f1 = p.feed_line("```")
        assert "└" in _strip_ansi(f1)

        # Normal text between
        normal = p.feed_line("Some **bold** text")
        assert _has_ansi(normal, "1")  # bold applied

        # Second block
        h2 = p.feed_line("```javascript")
        assert "javascript" in _strip_ansi(h2)
        p.feed_line("const x = 1;")
        f2 = p.feed_line("```")
        assert "└" in _strip_ansi(f2)


class TestTableRendering:
    """Pipe tables should be buffered and rendered with box-drawing borders."""

    def test_basic_table(self):
        p = MarkdownStreamProcessor()
        # Table rows return None while buffering (suppressed from output)
        assert p.feed_line("| Name | Role |") is None
        assert p.feed_line("|------|------|") is None
        assert p.feed_line("| Alice | Dev |") is None
        # Non-table line flushes the table
        out = p.feed_line("After table")
        plain = _strip_ansi(out)
        assert "┌" in plain
        assert "┘" in plain
        assert "Alice" in plain
        assert "Dev" in plain
        assert "After table" in plain

    def test_table_has_header_bold(self):
        p = MarkdownStreamProcessor()
        p.feed_line("| H1 | H2 |")
        p.feed_line("|-----|-----|")
        p.feed_line("| a | b |")
        out = p.feed_line("")
        # Header row should be bold
        assert _has_ansi(out, "1")

    def test_table_flush(self):
        """Table at end of stream is flushed properly."""
        p = MarkdownStreamProcessor()
        p.feed_line("| X | Y |")
        p.feed_line("|---|---|")
        p.feed_line("| 1 | 2 |")
        tail = p.flush()
        assert tail is not None
        plain = _strip_ansi(tail)
        assert "┌" in plain
        assert "┘" in plain
        assert "1" in plain

    def test_table_separator_not_hr(self):
        """Table separator rows should NOT become horizontal rules."""
        p = MarkdownStreamProcessor()
        p.feed_line("| A | B |")
        p.feed_line("|-----|-----|")
        p.feed_line("| x | y |")
        out = p.feed_line("done")
        plain = _strip_ansi(out)
        # Should have table box chars, and ├ for the separator
        assert "├" in plain
        assert "┼" in plain

    def test_table_column_alignment(self):
        """Columns should be padded to uniform width."""
        p = MarkdownStreamProcessor()
        p.feed_line("| Short | LongerColumn |")
        p.feed_line("|-------|--------------|")
        p.feed_line("| a | b |")
        out = p.feed_line("")
        plain = _strip_ansi(out)
        # "Short" and "a" should be padded to same width
        lines = plain.split("\n")
        # Find data rows (contain │)
        data_lines = [l for l in lines if "│" in l and "─" not in l]
        assert len(data_lines) >= 2

    def test_wide_table_falls_back_within_caller_width(self):
        p = MarkdownStreamProcessor(max_width=32)
        p.feed_line("| Name | Description |")
        p.feed_line("| --- | --- |")
        p.feed_line("| Hermes | A deliberately long description that cannot fit |")

        tail = p.flush()
        assert tail is not None
        plain = _strip_ansi(tail)

        assert "Name:" in plain
        assert "Description:" in plain
        assert "┌" not in plain
        assert all(len(line) <= 32 for line in plain.splitlines())

    def test_table_strips_inline_markers(self):
        """Bold/italic/code markers in cells should be stripped."""
        p = MarkdownStreamProcessor()
        p.feed_line("| **Feature** | **Winner** |")
        p.feed_line("|---|---|")
        p.feed_line("| **Model Class** | **Qwen** |")
        out = p.feed_line("done")
        plain = _strip_ansi(out)
        assert "Feature" in plain
        assert "Winner" in plain
        assert "Model Class" in plain
        assert "Qwen" in plain
        assert "**" not in plain

    def test_reset_clears_table(self):
        p = MarkdownStreamProcessor()
        p.feed_line("| A | B |")
        assert len(p._table_rows) == 1
        p.reset()
        assert len(p._table_rows) == 0


class TestMarkdownFenceSkip:
    """LLMs often wrap responses in ```markdown ... ```. These fences should
    be transparent — content inside renders as normal markdown."""

    def test_markdown_fence_skipped(self):
        p = MarkdownStreamProcessor()
        out = p.feed_line("```markdown")
        assert out == ""  # fence line suppressed
        # Content should be rendered as normal markdown, not code
        header = p.feed_line("# Hello")
        assert _has_ansi(header, "1")  # bold header
        assert "Hello" in _strip_ansi(header)
        # Closing fence also suppressed
        close = p.feed_line("```")
        assert close == ""

    def test_md_fence_skipped(self):
        p = MarkdownStreamProcessor()
        assert p.feed_line("```md") == ""
        out = p.feed_line("**bold**")
        assert _has_ansi(out, "1")
        assert p.feed_line("```") == ""

    def test_nested_code_inside_markdown_fence(self):
        """Inner ```python blocks should still be highlighted as code."""
        p = MarkdownStreamProcessor()
        p.feed_line("```markdown")  # skipped

        # Normal markdown
        h = p.feed_line("## Title")
        assert _has_ansi(h, "1")

        # Inner code block works normally
        header = p.feed_line("```python")
        assert "┌" in _strip_ansi(header)
        assert "python" in _strip_ansi(header)
        code = p.feed_line("x = 1")
        assert "x" in _strip_ansi(code)
        footer = p.feed_line("```")
        assert "└" in _strip_ansi(footer)

        # Back to normal markdown
        bold = p.feed_line("**done**")
        assert _has_ansi(bold, "1")

        # Outer closing fence suppressed
        assert p.feed_line("```") == ""
        assert p.flush() is None

    def test_full_llm_markdown_response(self):
        """Simulate an LLM wrapping its entire response in ```markdown."""
        p = MarkdownStreamProcessor()
        lines = [
            "```markdown",
            "# Welcome",
            "",
            "---",
            "",
            "- Item one",
            "- Item two",
            "",
            "```python",
            "print('hello')",
            "```",
            "",
            "*End of demo*",
            "```",
        ]
        outputs = [p.feed_line(l) for l in lines]

        # ```markdown skipped
        assert outputs[0] == ""
        # # Welcome → rendered as header
        assert _has_ansi(outputs[1], "1")
        assert "Welcome" in _strip_ansi(outputs[1])
        # --- → HR
        assert "─" in _strip_ansi(outputs[3])
        # - Item one → bullet
        assert "•" in _strip_ansi(outputs[5])
        # ```python → code fence header
        assert "┌" in _strip_ansi(outputs[8])
        # print('hello') → code
        assert "print" in _strip_ansi(outputs[9])
        # ``` → code fence footer
        assert "└" in _strip_ansi(outputs[10])
        # *End of demo* → italic
        assert _has_ansi(outputs[12], "3")
        # Closing ``` → skipped
        assert outputs[13] == ""
        assert p.flush() is None

    def test_reset_clears_md_fence_depth(self):
        p = MarkdownStreamProcessor()
        p.feed_line("```markdown")
        assert p._md_fence_depth == 1
        p.reset()
        assert p._md_fence_depth == 0


class TestUnclosedCodeBlock:
    def test_flush_closes_block(self):
        p = MarkdownStreamProcessor()
        p.feed_line("```python")
        p.feed_line("x = 1")
        p.feed_line("y = 2")
        # Stream ends without closing fence
        tail = p.flush()
        assert tail is not None
        assert "└" in _strip_ansi(tail)

    def test_flush_when_no_open_block(self):
        p = MarkdownStreamProcessor()
        p.feed_line("Normal text")
        tail = p.flush()
        assert tail is None


class TestReset:
    def test_reset_mid_code_block(self):
        p = MarkdownStreamProcessor()
        p.feed_line("```python")
        p.feed_line("x = 1")
        p.reset()
        # After reset, should be back in NORMAL mode
        out = p.feed_line("**bold again**")
        assert _has_ansi(out, "1")  # bold
        # flush should have nothing
        assert p.flush() is None

    def test_reset_clears_all_state(self):
        p = MarkdownStreamProcessor()
        p.feed_line("```python")
        p.feed_line("code")
        p.reset()
        assert not p._in_code_block
        assert p._code_lang == ""
        assert len(p._code_lines) == 0
        assert p._lexer is None
        assert p._formatter is None


# =========================================================================
# Streaming simulation (multi-line sequences)
# =========================================================================

class TestStreamingSequence:
    def test_typical_response(self):
        """Simulate a typical LLM response with mixed content."""
        p = MarkdownStreamProcessor()
        lines = [
            "Here's an example:",
            "",
            "```python",
            "def hello():",
            '    print("Hello, world!")',
            "```",
            "",
            "This uses the **print** function.",
        ]
        outputs = [p.feed_line(line) for line in lines]

        # Line 0: normal text
        assert "Here's an example:" in _strip_ansi(outputs[0])
        # Line 2: fence header
        assert "┌" in _strip_ansi(outputs[2])
        # Lines 3-4: code (indented)
        assert "def" in _strip_ansi(outputs[3])
        assert "print" in _strip_ansi(outputs[4])
        # Line 5: fence footer
        assert "└" in _strip_ansi(outputs[5])
        # Line 7: bold
        assert _has_ansi(outputs[7], "1")
        assert "print" in _strip_ansi(outputs[7])

        # No open block at end
        assert p.flush() is None


# =========================================================================
# Inline code ANSI is resolved from the skin (not hardcoded)
# =========================================================================

class TestInlineCodeAnsi:
    """The inline code escape must be injected via the ``inline_code_ansi``
    constructor kwarg — never hardcoded cyan-bold-on-black. This keeps the
    streaming renderer skinnable (see plan: skin-inline-code)."""

    # A handful of representative inline-code inputs to exercise the code
    # path across: vanilla, with-bold-around-it, with-brackets, multiple
    # spans, and inside a list.
    _INLINE_CODE_INPUTS = [
        "Use `foo()` here",
        "See `bar` and `baz` on the line",
        "- list item with `code`",
        "**bold** then `code` then *italic*",
        "Call `obj.method(arg)` now",
    ]

    def test_default_inline_code_is_bold_only(self):
        """With no inline_code_ansi passed, inline code is wrapped in plain
        bold — no foreground color, no background color."""
        p = MarkdownStreamProcessor()
        out = p.feed_line("Use `foo()` here")
        # Plain bold present
        assert "\033[1m" in out
        # No 38;2;R;G;B foreground sequence around the code span
        assert "38;2;" not in out
        # No background sequence of any form (256-color 48;5 or true-color
        # 48;2 or indexed 40-47)
        assert "48;2;" not in out
        assert "\033[40m" not in out
        # Content still present
        assert "foo()" in _strip_ansi(out)

    def test_custom_inline_code_ansi_applied(self):
        """A sentinel escape passed via inline_code_ansi must appear verbatim
        around each inline code span."""
        sentinel = "\033[1;38;2;255;191;0m"  # bold + amber fg
        p = MarkdownStreamProcessor(inline_code_ansi=sentinel)
        out = p.feed_line("Use `foo()` here")
        assert sentinel in out
        assert "foo()" in _strip_ansi(out)

    def test_inline_code_no_hardcoded_black_bg(self):
        """Regression guard: \\033[40m (indexed black bg) must never appear
        in default-processor output across a representative set of inline
        code inputs. The entire bug the plan fixes is 'hardcoded black bg'
        — locking this in prevents regression."""
        p = MarkdownStreamProcessor()
        for text in self._INLINE_CODE_INPUTS:
            out = p.feed_line(text)
            assert "\033[40m" not in out, (
                f"Unexpected hardcoded black bg in output for {text!r}: {out!r}"
            )
            # Belt-and-braces: the legacy combined escape must also be gone.
            assert "\033[1;36;40m" not in out

    def test_inline_code_bg_from_skin(self):
        """When the skin supplies bg + fg (bold + 38;2;R;G;B + 48;2;R;G;B),
        the whole escape must land verbatim around each code span."""
        # bold + banner_accent-ish fg + panel-tint bg
        sentinel = "\033[1;38;2;44;94;160;48;2;232;237;243m"
        p = MarkdownStreamProcessor(inline_code_ansi=sentinel)
        out = p.feed_line("Use `foo()` here")
        assert sentinel in out
        assert "foo()" in _strip_ansi(out)