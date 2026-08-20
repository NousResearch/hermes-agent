from cli import _strip_markdown_syntax


def test_strip_markdown_syntax_preserves_inline_code():
    text = "Use `__name__ == \"__main__\"` and `a**2 + b**2` here."
    assert _strip_markdown_syntax(text) == text


def test_strip_markdown_syntax_preserves_fenced_code():
    text = "Before **bold**\n```python\nif __name__ == \"__main__\":\n    value = a**2\n```\nAfter __prose__."
    expected = "Before bold\n```python\nif __name__ == \"__main__\":\n    value = a**2\n```\nAfter prose."
    assert _strip_markdown_syntax(text) == expected
