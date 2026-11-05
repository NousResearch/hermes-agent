"""Regressions for ``strip_markdown``'s star rules.

The bold and italic star rules were unguarded, so their delimiters paired up
unrelated ``*`` characters across a run of text. A bullet list and a literal
asterisk both look like an italic span to an unguarded ``\\*(.+?)\\*``, and the
plain-text surfaces (SMS, iMessage) showed the markers eaten.

The underscore rules already carried inside-edge guards; these tests pin the
same guards onto the star rules without loosening real emphasis stripping.
"""

import pytest

from gateway.platforms.helpers import strip_markdown


@pytest.mark.parametrize("text", [
    # Bullet list: the leading "*" of one item paired with the next item's.
    "* first item\n* second item",
    # Three items: main returned 'Here are steps:\n Install deps\n Run tests\n* Ship it'.
    "Here are steps:\n* Install deps\n* Run tests\n* Ship it",
    # Literal asterisks in prose (multiplication, wildcards).
    "Compute a * b * c for the area",
    # A single star and a double star in one line.
    "Use the * wildcard and the ** glob",
    # Bullet list using the "-" marker must also be untouched (control).
    "- first item\n- second item",
])
def test_non_emphasis_stars_survive(text):
    """Text whose ``*`` characters are not emphasis must pass through verbatim."""
    assert strip_markdown(text) == text


@pytest.mark.parametrize("text,expected", [
    ("this is **bold** text", "this is bold text"),
    ("this is *italic* text", "this is italic text"),
    ("mix **b** and _i_ here", "mix b and i here"),
    ("**bold** at the start", "bold at the start"),
    ("ends with **bold**", "ends with bold"),
])
def test_real_emphasis_is_still_stripped(text, expected):
    """The guards must not stop genuine emphasis from being stripped."""
    assert strip_markdown(text) == expected


def test_bold_inside_a_star_bullet_keeps_the_bullet():
    """The realistic LLM-output shape: bold runs inside a star bullet list.

    Main returned 'Step 1: do this\\n Step 2: do that', losing both markers.
    This is what an SMS or iMessage user actually saw.
    """
    text = "* **Step 1**: do this\n* **Step 2**: do that"
    assert strip_markdown(text) == "* Step 1: do this\n* Step 2: do that"


def test_emphasis_spanning_a_newline_still_stripped():
    """re.DOTALL is preserved: a span may cross a line break."""
    assert strip_markdown("a *multi\nline* span") == "a multi\nline span"
