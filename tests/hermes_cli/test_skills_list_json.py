"""Machine-readable installed-skill inventory for CLI consumers (#71274)."""

import argparse
import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from hermes_cli import skills_hub
from hermes_cli.subcommands.skills import build_skills_parser
from hermes_constants import get_hermes_home
from tools.skills_hub import HubLockFile, ensure_hub_dirs


def _write_skill(root, name, category=""):
    directory = root / category / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Example inventory skill.\n---\n"
        "# Example\nPrivate fixture body is not inventory metadata.\n",
        encoding="utf-8",
    )


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    """Use the actual catalog, config, manifest and hub lock, all in a temp home."""
    from tools import skills_tool

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})
    home = get_hermes_home()
    root = home / "skills"
    long_name = "python-datetime-parsing-with-complete-untruncated-identifiers"
    _write_skill(root, long_name)
    _write_skill(root, "bundled-example", "development")
    _write_skill(root, "hub-example", "research")
    _write_skill(root, "paused-example", "research")
    _write_skill(root, "café-analysis", "research")
    (root / ".bundled_manifest").write_text("bundled-example:example-hash\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "skills:\n  disabled: [paused-example]\n", encoding="utf-8",
    )
    ensure_hub_dirs()
    HubLockFile().record_install(
        "hub-example", "github", "example/skills/hub-example", "community", "safe",
        "example-hash", "research/hub-example", ["SKILL.md"],
        metadata={"internal_marker": "not-for-export"},
    )
    return [
        {"name": long_name, "category": "", "source": "local", "trust": "local", "enabled": True},
        {"name": "bundled-example", "category": "development", "source": "builtin", "trust": "builtin", "enabled": True},
        {"name": "café-analysis", "category": "research", "source": "local", "trust": "local", "enabled": True},
        {"name": "hub-example", "category": "research", "source": "github", "trust": "community", "enabled": True},
        {"name": "paused-example", "category": "research", "source": "local", "trust": "local", "enabled": False},
    ]


def _run_cli(*flags):
    parser = argparse.ArgumentParser(prog="hermes")
    build_skills_parser(parser.add_subparsers(dest="command"), cmd_skills=skills_hub.skills_command)
    args = parser.parse_args(["skills", "list", *flags])
    assert args.func(args) is None


@pytest.mark.parametrize("source", ["all", "hub", "builtin", "local"])
@pytest.mark.parametrize("enabled_only", [False, True])
def test_cli_json_preserves_inventory_and_filters(inventory, capsys, source, enabled_only):
    flags = ["--json", "--source", source]
    if enabled_only:
        flags.append("--enabled-only")
    _run_cli(*flags)
    captured = capsys.readouterr()
    expected = [row for row in inventory
                if (source == "all" or row["source"] == ("github" if source == "hub" else source))
                and (not enabled_only or row["enabled"])]
    assert json.loads(captured.out) == expected
    assert captured.err == ""
    assert "not-for-export" not in captured.out
    assert "Private fixture body" not in captured.out
    assert str(get_hermes_home()) not in captured.out


def test_json_is_not_rich_wrapped_or_styled(inventory, capsys, monkeypatch):
    monkeypatch.setattr(skills_hub, "_console", Console(
        file=StringIO(), width=18, force_terminal=True, color_system="truecolor",
    ))
    _run_cli("--json")
    text = capsys.readouterr().out
    assert json.loads(text) == inventory
    assert inventory[0]["name"] in text
    assert "café-analysis" in text
    assert "\x1b" not in text


def test_json_uses_only_the_active_profile(inventory, capsys, tmp_path, monkeypatch):
    other = tmp_path / "other-profile"
    _write_skill(other / "skills", "other-only")
    monkeypatch.setenv("HERMES_HOME", str(other))
    _run_cli("--json")
    assert json.loads(capsys.readouterr().out) == [
        {"name": "other-only", "category": "", "source": "local", "trust": "local", "enabled": True},
    ]


@pytest.mark.parametrize("flags", [[], ["--enabled-only"]])
def test_empty_inventory_is_a_json_array(tmp_path, monkeypatch, capsys, flags):
    monkeypatch.chdir(tmp_path)
    _run_cli("--json", *flags)
    assert json.loads(capsys.readouterr().out) == []


def test_slash_json_uses_the_same_inventory(inventory, capsys):
    skills_hub.handle_skills_slash("list --json --source hub --enabled-only")
    assert json.loads(capsys.readouterr().out) == [row for row in inventory if row["source"] == "github"]


@pytest.mark.parametrize("profile", [None, "work"])
def test_launcher_json_respects_profile_selection(inventory, tmp_path, profile):
    root = get_hermes_home()
    expected = [row for row in inventory if row["enabled"]]
    profile_flags = []
    if profile:
        home = root / "profiles" / profile
        _write_skill(home / "skills", "work-only")
        (home / "config.yaml").write_text("skills: {}\n", encoding="utf-8")
        profile_flags = ["-p", profile]
        expected = [{"name": "work-only", "category": "", "source": "local", "trust": "local", "enabled": True}]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[2] / "hermes"),
         *profile_flags, "skills", "list", "--json", "--enabled-only"],
        cwd=tmp_path, env={**os.environ, "HERMES_HOME": str(root)},
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


def test_default_table_is_unchanged(inventory, capsys, monkeypatch):
    monkeypatch.setattr(skills_hub, "_console", Console(width=180, force_terminal=False))
    _run_cli()
    text = capsys.readouterr().out
    assert "Installed Skills" in text
    assert "1 hub-installed, 1 builtin, 3 local" in text
    assert "4 enabled, 1 disabled" in text
    assert "paused-example" in text
