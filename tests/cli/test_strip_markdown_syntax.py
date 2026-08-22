"""Unit tests for _strip_markdown_syntax code preservation.

Regression coverage for #84377: in ``final_response_markdown: strip`` mode,
code contents share their characters with Markdown emphasis, so dunder
identifiers (``__name__``) and ``**`` operators inside code were eaten as
emphasis.  The function must protect fenced blocks and inline code spans
while still stripping prose markers and unwrapping code delimiters.

Note on the combined fix: #84379 protected code spans verbatim (keeping
fences/backticks); #84502 unwraps delimiters but protects contents.  This
suite follows #84502's contract — strip mode is "markdown marker removal",
so fences and backticks are dropped, only the code *contents* survive
intact — and additionally covers fenced blocks nested inside blockquotes,
which neither PR handled.
"""

from cli import _strip_markdown_syntax


def test_strip_markdown_syntax_preserves_inline_code():
    # Backticks are unwrapped but the contents (dunders, **) survive verbatim.
    text = "Use `__name__ == \"__main__\"` and `a**2 + b**2` here."
    assert _strip_markdown_syntax(text) == 'Use __name__ == "__main__" and a**2 + b**2 here.'


def test_strip_markdown_syntax_preserves_fenced_code():
    text = (
        "Before **bold**\n"
        "```python\n"
        'if __name__ == "__main__":\n'
        "    value = a**2\n"
        "```\n"
        "After __prose__."
    )
    expected = (
        "Before bold\n"
        "python\n"
        'if __name__ == "__main__":\n'
        "    value = a**2\n"
        "After prose."
    )
    assert _strip_markdown_syntax(text) == expected


def test_strip_markdown_syntax_preserves_unterminated_fence():
    # A fence without a closing marker still marks the intent as code.
    text = "```\nvalue = __all__[0]\n"
    assert _strip_markdown_syntax(text) == "value = __all__[0]"


def test_strip_markdown_syntax_preserves_blockquote_fenced_code():
    # Fenced blocks nested inside blockquotes (models quote docs this way)
    # must keep their code verbatim too.  Neither #84379 nor #84502 covered
    # the ``> ``` `` prefix; the fence regex accepts it now.
    text = (
        "> ```python\n"
        '> if __name__ == "__main__":\n'
        ">     value = a**2\n"
        "> ```\n"
        "After __prose__."
    )
    expected = (
        "> python\n"
        '> if __name__ == "__main__":\n'
        ">     value = a**2\n"
        "After prose."
    )
    assert _strip_markdown_syntax(text) == expected


def test_strip_markdown_syntax_preserves_other_code_symbols():
    # ~~strikethrough~~, triple-asterisk and underscore runs inside code are
    # protected by the same shelving.
    text = "```\nfoo ~~bar~~ baz\ntriple = a***b\n```"
    assert _strip_markdown_syntax(text) == "foo ~~bar~~ baz\ntriple = a***b"


def test_strip_markdown_syntax_still_strips_prose_emphasis_outside_code():
    text = "**bold** prose and `**not bold**` code"
    assert _strip_markdown_syntax(text) == "bold prose and **not bold** code"
