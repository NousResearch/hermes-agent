"""`hermes moa rm` must dispatch like `hermes moa delete` (#137579).

argparse keeps the literal alias the user typed in the subcommand dest, so
`moa rm` reaches ``cmd_moa`` with ``moa_command="rm"``. The hand-written
dispatch table must map the alias to the delete handler, the same way it
already maps the ``ls``/``config`` aliases.
"""

import argparse
from unittest.mock import patch

from hermes_cli.moa_cmd import cmd_moa
from hermes_cli.subcommands.moa import build_moa_parser


def _preset():
    return {
        "reference_models": [
            {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}
        ],
        "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
    }


def _config_with_extra_preset():
    return {
        "moa": {
            "default_preset": "default",
            "presets": {"default": _preset(), "extra": _preset()},
        }
    }


def _parse_moa(argv):
    parser = argparse.ArgumentParser(prog="hermes")
    build_moa_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def _delete_via_cmd_moa(argv):
    args = _parse_moa(argv)
    cfg = _config_with_extra_preset()
    saved = []
    with (
        patch("hermes_cli.moa_cmd.load_config", return_value=cfg),
        patch("hermes_cli.moa_cmd.save_config", side_effect=saved.append),
    ):
        cmd_moa(args)
    return args, saved


def test_rm_alias_deletes_preset():
    args, saved = _delete_via_cmd_moa(["moa", "rm", "extra"])
    assert args.moa_command == "rm"
    assert len(saved) == 1
    assert "extra" not in saved[0]["moa"]["presets"]
    assert "default" in saved[0]["moa"]["presets"]


def test_delete_spelling_still_deletes_preset():
    args, saved = _delete_via_cmd_moa(["moa", "delete", "extra"])
    assert args.moa_command == "delete"
    assert len(saved) == 1
    assert "extra" not in saved[0]["moa"]["presets"]
