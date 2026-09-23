"""Category search through the real plugin parser, dispatcher and catalog loader."""
import argparse
import json

import pytest
import yaml

from hermes_cli import plugin_catalog
from hermes_cli.plugins_cmd import plugins_command
from hermes_cli.subcommands.plugins import build_plugins_parser


def _parser():
    parser = argparse.ArgumentParser()
    build_plugins_parser(parser.add_subparsers(dest="command"), cmd_plugins=plugins_command)
    return parser


def test_category_search_intersects_text_and_preserves_default(tmp_path, monkeypatch, capsys):
    entries = {
        "notes": ("memory", "store research notes"),
        "finder": ("web", "find research papers"),
        "browser": ("web", "browse pages"),
    }
    for name, (category, description) in entries.items():
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump({
            "name": name, "category": category, "description": description,
            "repo": f"https://github.com/example/{name}", "sha": "a" * 40,
            "maintainer": "example", "capabilities": {"provides_tools": [f"{name}_lookup"]},
        }))
    monkeypatch.setattr(plugin_catalog, "get_catalog_dir", lambda: tmp_path)
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **_: None)
    parser = _parser()
    for options, expected in [
        (["--category", "web"], {"finder", "browser"}),
        (["RESEARCH", "--category", "web"], {"finder"}),
        (["finder_lookup", "--category", "web"], {"finder"}),
        (["research"], {"notes", "finder"}),
        ([], set(entries)),
        (["--category", "voice"], set()),
    ]:
        args = parser.parse_args(["plugins", "search", *options, "--json"])
        plugins_command(args)
        payload = json.loads(capsys.readouterr().out)
        assert {entry["name"] for entry in payload["results"]} == expected

    for options in (["--category", "voice"], ["missing", "--category", "web"]):
        plugins_command(parser.parse_args(["plugins", "search", *options]))
        output = capsys.readouterr().out
        assert "No catalog entries" in output
        assert options[-1] in output
    plugins_command(parser.parse_args(["plugins", "search", "--category", "memory"]))
    output = capsys.readouterr().out
    assert "notes" in output
    assert "finder" not in output
    assert "browser" not in output


def test_category_search_rejects_unknown_category_before_loading(monkeypatch, capsys):
    def unexpected_load():
        pytest.fail("Invalid CLI input must not load the catalog")
    monkeypatch.setattr(plugin_catalog, "load_catalog_live", unexpected_load)
    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(["plugins", "search", "--category", "not-a-category"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
