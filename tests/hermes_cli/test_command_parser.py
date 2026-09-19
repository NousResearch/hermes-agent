"""Tests for hermes_cli/command_parser.py — command parsing helpers."""


def test_parse_bare_slash():
    from hermes_cli.command_parser import parse_command
    result = parse_command("/help")
    assert result.name == "help"
