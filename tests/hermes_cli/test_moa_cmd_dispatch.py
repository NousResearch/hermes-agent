"""``hermes moa`` dispatch accepts every subcommand spelling its parser advertises.

Regression for #137579: ``hermes moa rm <name>`` was rejected with "Unknown moa
subcommand: rm" because the hand-written dispatch table had no entry for the
``rm`` alias of ``delete``. argparse stores the literal the user typed, so the
table must cover each alias, not only the canonical name.
"""

from __future__ import annotations

import argparse

import pytest

from hermes_cli import moa_cmd
from hermes_cli.subcommands.moa import build_moa_parser


def _moa_parser() -> tuple[argparse.ArgumentParser, list[str]]:
    parser = argparse.ArgumentParser(prog="hermes")
    build_moa_parser(parser.add_subparsers(dest="command"))
    moa = parser._subparsers._group_actions[0].choices["moa"]
    choices = next(a for a in moa._actions if isinstance(a, argparse._SubParsersAction)).choices
    return parser, list(choices)


def test_every_parser_choice_has_a_dispatch_handler() -> None:
    _parser, choices = _moa_parser()
    assert choices, "parser registers subcommands"
    assert set(choices) <= set(moa_cmd._SUBCOMMANDS)


@pytest.mark.parametrize("spelling", ["delete", "rm"])
def test_delete_and_its_alias_reach_the_delete_handler(monkeypatch, spelling: str) -> None:
    """Both spellings run the real delete handler: on an empty config it refuses the unknown
    preset by name, which is the delete handler's answer, not the dispatcher's."""
    parser, _choices = _moa_parser()
    monkeypatch.setattr(moa_cmd, "load_config", dict)

    args = parser.parse_args(["moa", spelling, "preset-x"])
    with pytest.raises(SystemExit) as exc:
        args.func(args)

    assert str(exc.value) == "Unknown MoA preset: preset-x"
