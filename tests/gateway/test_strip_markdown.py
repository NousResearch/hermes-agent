"""Regression tests for ``strip_markdown`` star-emphasis handling.

The star (``*``) emphasis regexes used to lack the inside-edge guards that the
underscore variants already had, so a run of list bullets or a couple of
literal asterisks was mistaken for an italic/bold span and silently removed.
See the bullet-list / literal-asterisk cases below — they corrupt on the
unguarded regexes and pass once the ``(?![\\s*])`` / ``(?<![\\s*])`` guards are
added.
"""

from gateway.platforms.helpers import strip_markdown


def test_bullet_list_markers_preserved():
    text = "Here are steps:\n* Install deps\n* Run tests\n* Ship it"
    assert strip_markdown(text) == text


def test_two_bullet_list_preserved():
    assert strip_markdown("* first item\n* second item") == "* first item\n* second item"


def test_literal_asterisks_preserved():
    assert strip_markdown("Compute a * b * c for the area") == "Compute a * b * c for the area"


def test_wildcard_and_glob_literals_preserved():
    assert strip_markdown("Use the * wildcard and the ** glob") == "Use the * wildcard and the ** glob"


def test_real_italic_still_unwrapped():
    assert strip_markdown("*italic*") == "italic"


def test_real_bold_still_unwrapped():
    assert strip_markdown("**bold**") == "bold"


def test_mixed_emphasis_and_bullets():
    text = "Summary:\n* uses **bold** here\n* and *italic* there"
    assert strip_markdown(text) == "Summary:\n* uses bold here\n* and italic there"
