"""Tests for LaTeX math ($...$ / $$...$$) -> Element data-mx-maths markup."""

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    return MatrixAdapter(config)


# ---------------------------------------------------------------------------
# _latex_to_tokens / _tokens_to_mx_maths unit tests
# ---------------------------------------------------------------------------


class TestLatexToTokens:
    def setup_method(self):
        from plugins.platforms.matrix.adapter import _latex_to_tokens

        self.convert = _latex_to_tokens

    def test_inline_math_becomes_span_token(self):
        text, store = self.convert("Energy: $E = mc^2$ indeed.")
        assert store == [("span", "E = mc^2")]
        assert "E = mc^2" not in text
        assert "$" not in text

    def test_display_math_becomes_div_token(self):
        text, store = self.convert("$$\\hat{H}\\Psi = E\\Psi$$")
        assert store == [("div", "\\hat{H}\\Psi = E\\Psi")]

    def test_mixed_math_all_captured(self):
        # Display ($$...$$) is tokenized before inline ($...$), so store order
        # follows regex pass order, not text order. Index-token correspondence
        # is what matters.
        text, store = self.convert("$a$ then $$b$$ then $c$")
        assert sorted((tex, tag) for tag, tex in store) == [
            ("a", "span"),
            ("b", "div"),
            ("c", "span"),
        ]

    def test_unpaired_dollars_untouched(self):
        text, store = self.convert("Costs $5 or $10 today.")
        assert store == []
        assert text == "Costs $5 or $10 today."

    def test_no_math_no_change(self):
        text, store = self.convert("No math here.")
        assert store == []
        assert text == "No math here."

    def test_backslash_dollar_not_math(self):
        _, store = self.convert(r"Escaped \\$5 not math.")
        assert store == []


class TestTokensToMxMaths:
    def setup_method(self):
        from plugins.platforms.matrix.adapter import (
            _latex_to_tokens,
            _tokens_to_mx_maths,
        )

        self.tokenize = _latex_to_tokens
        self.expand = _tokens_to_mx_maths

    def test_inline_expansion(self):
        tex = r"\psi(x)"
        text, store = self.tokenize(f"Wave: ${tex}$")
        html = self.expand(text, store)
        assert html == f'Wave: <span data-mx-maths="{tex}">{tex}</span>'

    def test_display_expansion(self):
        text, store = self.tokenize("$$x^2$$")
        html = self.expand(text, store)
        assert 'data-mx-maths="x^2"' in html

    def test_tex_is_html_escaped(self):
        text, store = self.tokenize('$a<b & c>"d$')
        html = self.expand(text, store)
        assert "<b &" not in html
        assert "data-mx-maths=" in html

    def test_no_token_left_behind(self):
        text, store = self.tokenize("$a$ $$b$$")
        html = self.expand(text, store)
        assert "HERMESTEX" not in html

    def test_user_text_colliding_with_sentinel_is_safe(self):
        # Adversarial literal that matches the sentinel format but has no
        # matching store entry must pass through unchanged, not raise.
        html = self.expand("HERMESTEXDISPLAY99HERMESTEXEND", [])
        assert html == "HERMESTEXDISPLAY99HERMESTEXEND"

    def test_mismatched_store_is_safe(self):
        # Expansion with an empty store leaves unknown tokens alone
        html = self.expand("HERMESTEXDISPLAY99HERMESTEXEND", [])
        assert "data-mx-maths" not in html


# ---------------------------------------------------------------------------
# Integration through _markdown_to_html (the outbound pipeline)
# ---------------------------------------------------------------------------


class TestMarkdownToHtmlLatex:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_inline_math_reaches_formatted_html(self):
        html = self.adapter._markdown_to_html("Wave: $\\psi(x) = e^{ikx}$ ok")
        assert 'data-mx-maths="\\psi(x) = e^{ikx}"' in html
        assert "HERMESTEX" not in html

    def test_display_math_reaches_formatted_html(self):
        html = self.adapter._markdown_to_html("$$\\hat{H}\\Psi = E\\Psi$$")
        assert 'data-mx-maths="\\hat{H}\\Psi = E\\Psi"' in html

    def test_markdown_formatting_still_applied(self):
        html = self.adapter._markdown_to_html("**bold** with $x^2$")
        assert "<strong>" in html
        assert 'data-mx-maths="x^2"' in html

    def test_dollar_amounts_not_converted(self):
        html = self.adapter._markdown_to_html("Costs $5 or $10 today.")
        assert "data-mx-maths" not in html

    def test_math_inside_code_block_still_converted(self):
        # Note: sentinel conversion happens before Markdown sees the text,
        # so even code-span math becomes markup. Documented trade-off: the
        # composer-style consumers (chat) rarely put $...$ in code spans.
        html = self.adapter._markdown_to_html("Use `$x$` here.")
        assert 'data-mx-maths="x"' in html
