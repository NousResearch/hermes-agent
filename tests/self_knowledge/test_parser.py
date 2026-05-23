import pytest

from hermes_cli.self_knowledge.parser import parse_auto_blocks, replace_auto_blocks


def test_parse_auto_blocks_returns_names_and_bodies():
    text = """Intro
<!-- AUTO-START: capabilities -->
old body
<!-- AUTO-END: capabilities -->
Outro
"""

    blocks = parse_auto_blocks(text)

    assert list(blocks) == ["capabilities"]
    assert blocks["capabilities"].body == "old body\n"


def test_replace_auto_blocks_preserves_handwritten_text():
    text = """# Title

Hand written before.

<!-- AUTO-START: capabilities -->
old body
<!-- AUTO-END: capabilities -->

Hand written after.
"""

    rendered = replace_auto_blocks(text, {"capabilities": "new body"})

    assert rendered == """# Title

Hand written before.

<!-- AUTO-START: capabilities -->
new body
<!-- AUTO-END: capabilities -->

Hand written after.
"""


def test_round_trip_without_replacements_is_noop_lf():
    text = """A
<!-- AUTO-START: one -->
body
<!-- AUTO-END: one -->
B
"""

    assert replace_auto_blocks(text, {}) == text


def test_round_trip_without_replacements_is_noop_crlf():
    text = "A\r\n<!-- AUTO-START: one -->\r\nbody\r\n<!-- AUTO-END: one -->\r\nB\r\n"

    assert replace_auto_blocks(text, {}) == text


def test_duplicate_auto_block_names_raise_value_error():
    text = """<!-- AUTO-START: one -->
a
<!-- AUTO-END: one -->
<!-- AUTO-START: one -->
b
<!-- AUTO-END: one -->
"""

    with pytest.raises(ValueError, match="Duplicate AUTO block"):
        parse_auto_blocks(text)


def test_missing_end_marker_raises_value_error():
    text = """<!-- AUTO-START: one -->
body
"""

    with pytest.raises(ValueError, match="Missing AUTO-END"):
        parse_auto_blocks(text)


def test_mismatched_end_marker_raises_value_error():
    text = """<!-- AUTO-START: one -->
body
<!-- AUTO-END: two -->
"""

    with pytest.raises(ValueError, match="Mismatched AUTO block"):
        parse_auto_blocks(text)
