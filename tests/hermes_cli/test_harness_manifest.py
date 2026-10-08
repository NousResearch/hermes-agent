"""Behavior contracts for typed, reversible harness overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from hermes_cli import config as config_module
from hermes_cli import config_effective
from hermes_cli.harness_manifest import (
    HarnessError,
    harness_identity,
    load_manifest,
    parse_value,
    revert_overlay,
    set_overlay,
)
from hermes_cli.subcommands.harness import build_harness_parser


def _clear_config_caches() -> None:
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()
    config_effective._EFFECTIVE_CACHE.clear()


def test_ordered_typed_overlays_are_effective_fingerprinted_and_reversible(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("compression:\n  threshold: 0.4\n", encoding="utf-8")
    _clear_config_caches()

    baseline = harness_identity(config_module.load_config())
    first = set_overlay("experiment", "compression.threshold", 0.65, "Measure later compaction")
    set_overlay("override", "compression.threshold", 0.75, "Compare a stronger setting")

    assert first["authored_against"] == baseline["stock_revision"]
    assert config_module.load_config()["compression"]["threshold"] == 0.75
    assert config_effective.load_user_config_effective()["compression"]["threshold"] == 0.75
    changed = harness_identity(config_module.load_config())
    assert [item["name"] for item in changed["overlay_lineage"]] == ["experiment", "override"]
    assert changed["effective_config_fingerprint"] != baseline["effective_config_fingerprint"]

    removed = revert_overlay("override", "The comparison is complete")
    assert removed["name"] == "override"
    assert config_module.load_config()["compression"]["threshold"] == 0.65
    revert_overlay("experiment", "Restore the profile setting")
    assert config_module.load_config()["compression"]["threshold"] == 0.4
    assert [event["action"] for event in load_manifest()["history"]] == ["set", "set", "revert", "revert"]

    with pytest.raises(HarnessError, match="not a tunable harness option"):
        set_overlay("unsafe", "approvals.mode", "off", "Widen permissions")
    with pytest.raises(HarnessError, match="not a tunable harness option"):
        set_overlay("secret", "auxiliary.vision.api_key", "not-a-real-key", "Store a credential")
    with pytest.raises(HarnessError, match="expects number"):
        parse_value("compression.threshold", "not-a-number")
    with pytest.raises(HarnessError, match=">= 0.0"):
        parse_value("compression.threshold", "-0.1")


def test_harness_cli_parser_and_commands_round_trip(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _clear_config_caches()

    from hermes_cli.main import cmd_harness

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_harness_parser(subparsers, cmd_harness=cmd_harness)

    args = parser.parse_args([
        "harness", "set", "focused", "compression.threshold", "0.6",
        "--reason", "Keep more recent context",
    ])
    args.func(args)
    assert "Updated harness overlay focused" in capsys.readouterr().out

    args = parser.parse_args(["harness", "show", "--json"])
    args.func(args)
    shown = json.loads(capsys.readouterr().out)
    assert shown["manifest"]["overlays"][0]["values"] == {"compression.threshold": 0.6}
    assert shown["identity"]["overlay_lineage"][0]["name"] == "focused"

    args = parser.parse_args([
        "harness", "revert", "focused", "--reason", "Return to stock",
    ])
    args.func(args)
    assert "Reverted harness overlay focused" in capsys.readouterr().out
    assert yaml.safe_load((home / "harness.yaml").read_text(encoding="utf-8"))["overlays"] == []
