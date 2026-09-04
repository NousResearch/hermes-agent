"""Test for rotating task-oriented composer placeholders (C-09)."""

from hermes_cli.tips import COMPOSER_PLACEHOLDERS, get_random_composer_placeholder


def test_composer_placeholders_nonempty():
    assert COMPOSER_PLACEHOLDERS
    assert all(isinstance(p, str) and p.strip() for p in COMPOSER_PLACEHOLDERS)


def test_get_random_composer_placeholder_returns_a_member():
    for _ in range(20):
        assert get_random_composer_placeholder() in COMPOSER_PLACEHOLDERS
