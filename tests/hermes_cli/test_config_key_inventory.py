"""The config key inventory is discoverable without reading user values (#127305)."""

import argparse
import json

from hermes_cli import config
from hermes_cli.subcommands.config import build_config_parser


def _invoke(*options):
    parser = argparse.ArgumentParser()
    build_config_parser(parser.add_subparsers(), cmd_config=config.config_command)
    try:
        args = parser.parse_args(["config", "keys", *options])
    except SystemExit as exc:
        assert exc.code == 0, "config keys must be a supported discovery command"
        return
    args.func(args)


def test_inventory_covers_registered_paths_without_loading_user_config(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("key discovery must not read or create user configuration")

    monkeypatch.setattr(config, "load_config", forbidden)
    monkeypatch.setattr(config, "get_config_path", forbidden)
    _invoke("--json")
    keys = json.loads(capsys.readouterr().out)
    assert keys == sorted(set(keys))
    assert set(config._known_top_level_keys()) - {"_config_version"} <= set(keys)

    def check(mapping, prefix=""):
        for name, value in mapping.items():
            if name.startswith("_"):
                continue
            segment = name.replace(".", "\\.")
            key = f"{prefix}.{segment}" if prefix else segment
            assert key in keys
            if isinstance(value, dict):
                check(value, key)

    check(config.DEFAULT_CONFIG)
    _invoke()
    assert capsys.readouterr().out.splitlines() == keys


def test_inventory_handles_nested_empty_and_scalar_defaults(monkeypatch, capsys):
    monkeypatch.setattr(config, "DEFAULT_CONFIG", {
        "nested": {"enabled": False, "empty": {}, "names": ["not-a-key"], "value": None,
                   "attrs": {"service.name": "gw"}},
        "_private": "not-a-key",
    })
    monkeypatch.setattr(config, "_known_top_level_keys", lambda: {"nested", "unseeded"})
    _invoke("--json")
    keys = json.loads(capsys.readouterr().out)
    assert keys == [
        "nested", "nested.attrs", "nested.attrs.service\\.name", "nested.empty", "nested.enabled",
        "nested.names", "nested.value", "unseeded",
    ]
    # A literal dotted key prints escaped, so the path round-trips to the real leaf.
    assert config._get_nested(config.DEFAULT_CONFIG, keys[2]) == "gw"
    monkeypatch.setattr(config, "DEFAULT_CONFIG", {})
    monkeypatch.setattr(config, "_known_top_level_keys", lambda: set())
    _invoke("--json")
    assert json.loads(capsys.readouterr().out) == []
    _invoke()
    assert capsys.readouterr().out == ""
