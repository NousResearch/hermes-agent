"""Tests for hermes_cli/personality.py — personality module import and defaults."""


def test_personality_enum_has_cli():
    from hermes_cli.personality import Personality
    assert hasattr(Personality, "DEFAULT")


def test_personality_name_non_empty():
    from hermes_cli.personality import Personality
    for p in Personality:
        assert p.name
        assert isinstance(p.name, str)
