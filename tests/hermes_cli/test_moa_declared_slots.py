"""Declarative ``hermes moa configure --slots`` / ``--slots-file`` (issue #102265).

A MoA preset is deployment state: a fleet setup must be able to converge it from flags or a
file instead of clicking through the interactive picker. Both tests drive the REAL argparse
tree and the real command against the per-test ``HERMES_HOME``, then assert the preset that
lands on disk — a declared slot set either converges the preset or aborts loudly.
"""

from __future__ import annotations

import argparse
import json

import pytest
import yaml

from hermes_cli.moa_cmd import cmd_moa
from hermes_cli.subcommands.moa import build_moa_parser

_DEFAULT_PRESET = {
    "reference_models": [{"provider": "openai-codex", "model": "gpt-5.5"}],
    "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"}}


def _parse(argv):
    """Parse through the subparser the CLI builds, so the flags are proven wired."""
    parser = argparse.ArgumentParser()
    build_moa_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def _seed_config():
    from hermes_cli.config import get_config_path

    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({
            "model": {"default": "glm-5.08", "provider": "openrouter"},
            "moa": {"default_preset": "default", "presets": {"default": _DEFAULT_PRESET}}}),
        encoding="utf-8")
    return path


def _refs_on_disk(path, preset_name):
    preset = yaml.safe_load(path.read_text(encoding="utf-8"))["moa"]["presets"][preset_name]
    return [(slot["provider"], slot["model"]) for slot in preset["reference_models"]], preset["aggregator"]


def _forbid_picker(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("a declared slot set must not open the interactive picker")

    monkeypatch.setattr("hermes_cli.moa_cmd._pick_slot", _boom)


def test_declared_slots_converge_a_preset_without_the_picker(monkeypatch, tmp_path):
    """``--slots`` and ``--slots-file`` both write what was declared, picker untouched."""
    _forbid_picker(monkeypatch)
    config_path = _seed_config()

    args = _parse(["moa", "configure", "review", "--slots",
                   "openrouter/moonshotai/kimi-k2.5,openai-codex/gpt-5.6",
                   "--aggregator", "openrouter/anthropic/claude-sonnet-4.9"])
    assert args.func is cmd_moa
    args.func(args)

    refs, aggregator = _refs_on_disk(config_path, "review")
    assert refs == [("openrouter", "moonshotai/kimi-k2.5"), ("openai-codex", "gpt-5.6")]
    assert aggregator == {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.9"}

    # The file form carries the same declaration (aggregator included) for versioned presets.
    slots_file = tmp_path / "fleet-slots.json"
    slots_file.write_text(json.dumps({
        "reference_models": ["openrouter/moonshotai/kimi-k2.5"],
        "aggregator": "openrouter/anthropic/claude-sonnet-4.9"}), encoding="utf-8")
    fleet_args = _parse(["moa", "configure", "fleet", "--slots-file", str(slots_file)])
    fleet_args.func(fleet_args)

    refs, aggregator = _refs_on_disk(config_path, "fleet")
    assert refs == [("openrouter", "moonshotai/kimi-k2.5")]
    assert aggregator["model"] == "anthropic/claude-sonnet-4.9"


def test_malformed_declared_slots_abort_without_writing(monkeypatch, tmp_path):
    """A half-specified slot, a recursive MoA slot, or a bad file fails closed."""
    _forbid_picker(monkeypatch)
    config_path = _seed_config()
    before = config_path.read_text(encoding="utf-8")
    broken_json = tmp_path / "broken.json"
    broken_json.write_text("{not json", encoding="utf-8")

    bad_argv = [
        # provider without a model — normalize_moa_config would silently drop the slot
        ["moa", "configure", "review", "--slots", "openrouter", "--aggregator", "openrouter/x"],
        # model without a provider
        ["moa", "configure", "review", "--slots", "/gpt-5.5", "--aggregator", "openrouter/x"],
        # the virtual MoA provider cannot be a slot (recursive preset)
        ["moa", "configure", "review", "--slots", "moa/gpt-5.5", "--aggregator", "openrouter/x"],
        # a new preset has no aggregator to inherit
        ["moa", "configure", "review", "--slots", "openrouter/moonshotai/kimi-k2.5"],
        # unreadable / malformed file
        ["moa", "configure", "review", "--slots-file", str(tmp_path / "missing.json"),
         "--aggregator", "openrouter/x"],
        ["moa", "configure", "review", "--slots-file", str(broken_json), "--aggregator", "openrouter/x"]]

    for argv in bad_argv:
        args = _parse(argv)  # the flags themselves parse; the command must reject the values
        with pytest.raises(SystemExit):
            args.func(args)

    assert config_path.read_text(encoding="utf-8") == before
    assert "review" not in yaml.safe_load(before)["moa"]["presets"]
