"""Top-level command discovery is ordered without changing dispatch."""

import argparse

import pytest

from hermes_cli._parser import build_top_level_parser


def test_late_registered_commands_are_alphabetical_in_help():
    parser, subparsers, _ = build_top_level_parser()
    subparsers.add_parser('zebra', help='Last command')
    subparsers.add_parser('alpha', aliases=['a'], help='First command')
    text = parser.format_help()
    assert text.index('    alpha (a)') < text.index('    chat ')
    assert text.index('    chat ') < text.index('    zebra ')
    assert parser.parse_args(['a']).command == 'a'
    assert parser.parse_args(['zebra']).command == 'zebra'


def test_chat_options_still_parse_after_help():
    parser, _, _ = build_top_level_parser()
    assert parser.format_help()
    args = parser.parse_args(['chat', '--model', 'chosen'])
    assert args.model == 'chosen'


def test_empty_command_group_and_authored_epilogue():
    parser, _, _ = build_top_level_parser()
    formatter = parser.formatter_class
    empty = argparse.ArgumentParser(formatter_class=formatter, epilog='one\n  two')
    empty.add_subparsers()
    assert 'one\n  two' in empty.format_help()


def test_formatting_does_not_change_nested_help_or_unknown_command_errors():
    parser, subparsers, _ = build_top_level_parser()
    nested = subparsers.add_parser('nested', help='Nested commands')
    children = nested.add_subparsers(dest='child')
    children.add_parser('zebra', help='Last')
    children.add_parser('alpha', help='First')
    parser.format_help()
    text = nested.format_help()
    assert text.index('    zebra ') < text.index('    alpha ')
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['not-a-command'])
    assert exc.value.code == 2
