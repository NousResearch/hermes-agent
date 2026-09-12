"""Worker configuration through the real parser, loader, validator and save path."""

import argparse
import json

import pytest
import yaml

from hermes_cli.workers import build_workers_parser


@pytest.fixture
def worker_home(tmp_path, monkeypatch):
    from hermes_cli import config

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(config, "get_config_path", lambda: home / "config.yaml")
    monkeypatch.setattr(config, "ensure_hermes_home", lambda: home)
    monkeypatch.setattr(config, "is_managed", lambda: False)
    return home


def invoke(*argv):
    parser = argparse.ArgumentParser()
    build_workers_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["workers", *map(str, argv)])
    args.func(args)


def test_profile_roundtrip_is_scoped_and_preserves_unrelated_config(worker_home, tmp_path, capsys):
    config_path = worker_home / "config.yaml"
    config_path.write_text("model:\n  default: existing-model\ndisplay:\n  theme: custom\n")
    profile_path = tmp_path / "worker.yaml"
    profile_path.write_text("provider: openai\nmodel: test-model\nreasoning_effort: high\n")
    invoke("set", "research", "--file", profile_path)
    invoke("default", "research")
    raw = yaml.safe_load(config_path.read_text())
    assert raw["model"]["default"] == "existing-model"
    assert raw["display"]["theme"] == "custom"
    assert raw["delegation"]["default_profile"] == "research"
    assert raw["delegation"]["profiles"]["research"]["model"] == "test-model"
    invoke("validate")
    capsys.readouterr()
    invoke("inspect", "research", "--json")
    profile = json.loads(capsys.readouterr().out)
    assert profile["name"] == "research"
    assert profile["model"] == "test-model"
    invoke("default", "-")
    assert not yaml.safe_load(config_path.read_text())["delegation"].get("default_profile")


@pytest.mark.parametrize("contents", ["model: [", "model: missing-provider\napi_key: forbidden\n", "- not-a-mapping\n"])
def test_rejected_profile_leaves_config_unchanged(worker_home, tmp_path, contents):
    config_path = worker_home / "config.yaml"
    original = "model:\n  default: preserve-me\n"
    config_path.write_text(original)
    path = tmp_path / "invalid.yaml"
    path.write_text(contents)
    with pytest.raises(SystemExit) as error:
        invoke("set", "bad", "--file", path)
    assert error.value.code == 1
    assert config_path.read_text() == original


def test_unknown_default_and_managed_write_fail_without_mutation(worker_home, capsys, monkeypatch):
    from hermes_cli import config

    path = worker_home / "config.yaml"
    path.write_text("model:\n  default: preserve-me\n")
    original = path.read_bytes()
    with pytest.raises(SystemExit):
        invoke("default", "unknown")
    monkeypatch.setattr(config, "is_managed", lambda: True)
    with pytest.raises(SystemExit):
        invoke("default", "-")
    assert path.read_bytes() == original
    assert "managed" in capsys.readouterr().out


def test_list_alias_has_same_catalog_and_does_not_launch(worker_home, capsys, monkeypatch):
    from hermes_cli import runtime_provider

    def forbidden(**kwargs):
        pytest.fail("Discovery must not resolve credentials or launch a provider")

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", forbidden)
    invoke("list", "--json")
    first = json.loads(capsys.readouterr().out)
    invoke("ls", "--json")
    assert json.loads(capsys.readouterr().out) == first


def test_top_level_parser_registers_worker_command(worker_home, monkeypatch, capsys):
    import sys
    from hermes_cli.main import _build_cli_parser

    monkeypatch.setattr(sys, "argv", ["hermes", "workers", "validate"])
    parser, _ = _build_cli_parser()
    args = parser.parse_args(["workers", "validate"])
    args.func(args)
    assert "valid" in capsys.readouterr().out
