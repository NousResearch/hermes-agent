"""Tests for gateway.platforms.helpers.strip_markdown."""
from gateway.platforms.helpers import strip_markdown


def test_image_stripped_to_alt_text():
    """Images ![alt](url) should become alt text, not !alt."""
    assert strip_markdown("![photo](https://example.com/img.png)") == "photo"


def test_image_with_empty_alt():
    """Images with empty alt text should collapse cleanly."""
    assert strip_markdown("![](https://example.com/img.png)") == ""


def test_link_stripped_to_display_text():
    """Links [text](url) should become text."""
    assert strip_markdown("[click here](https://example.com)") == "click here"


def test_image_before_link_no_interference():
    """Image and link regexes should not interfere."""
    text = "See ![diagram](https://img.png) and [docs](https://docs.com)"
    result = strip_markdown(text)
    assert result == "See diagram and docs"


def test_bold_and_italic():
    assert strip_markdown("**bold** and *italic*") == "bold and italic"


def test_strikethrough():
    assert strip_markdown("~~deleted~~ kept") == "deleted kept"


def test_blockquote():
    assert strip_markdown("> quoted text") == "quoted text"


def test_heading():
    assert strip_markdown("## Section Title") == "Section Title"


def test_inline_code_preserves_content():
    assert strip_markdown("Run `pip install foo`") == "Run pip install foo"


def test_code_block_removed():
    text = "```python\nprint('hello')\n```"
    result = strip_markdown(text)
    assert "python" not in result
    assert "print" in result


def test_multi_newline_collapsed():
    text = "a\n\n\n\nb"
    assert strip_markdown(text) == "a\n\nb"


def test_combined_markdown():
    """Complex mixed markdown should be stripped cleanly."""
    text = (
        "## Title\n\n"
        "Hello **world**, see ![img](http://x.png) and [link](http://x.com).\n\n"
        "> A quote with *emphasis*.\n\n"
        "~~old~~ new and `code`."
    )
    result = strip_markdown(text)
    assert "##" not in result
    assert "**" not in result
    assert "![" not in result
    assert "![img]" not in result
    assert "img" in result
    assert "link" in result
    assert ">" not in result.split("\n")[0]
    assert "~~" not in result
    assert "old" in result
    assert "new" in result
    assert "code" in result
