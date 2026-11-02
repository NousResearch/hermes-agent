"""The public parser's removal aliases must reach the real command handler."""

import argparse
from copy import deepcopy

import pytest


@pytest.mark.parametrize("verb", ["delete", "rm"])
def test_moa_delete_alias_removes_named_preset(monkeypatch, verb):
    from hermes_cli import moa_cmd
    from hermes_cli.subcommands.moa import build_moa_parser

    config = {
        "moa": {
            "default_preset": "keep",
            "presets": {"keep": {}, "remove-me": {}},
        }
    }
    saved = []
    monkeypatch.setattr(moa_cmd, "load_config", lambda: deepcopy(config))
    monkeypatch.setattr(moa_cmd, "save_config", lambda value: saved.append(deepcopy(value)))
    parser = argparse.ArgumentParser()
    build_moa_parser(parser.add_subparsers(dest="command"))

    args = parser.parse_args(["moa", verb, "remove-me"])
    args.func(args)

    assert len(saved) == 1
    assert set(saved[0]["moa"]["presets"]) == {"keep"}
    assert saved[0]["moa"]["default_preset"] == "keep"
