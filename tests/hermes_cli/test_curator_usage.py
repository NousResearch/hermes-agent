"""Tests for `hermes curator usage` — the all-skills usage view.

Covers:
- Lists every skill regardless of provenance (agent / user / bundled / hub / external), unlike
  `status` which is scoped to curator-managed candidates.
- --provenance filter, --sort ordering, and --json output.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

def _fake_rows():
    return [
        {
            "name": "agent-skill", "provenance": "agent", "state": "active",
            "use_count": 2, "view_count": 1, "patch_count": 0,
            "activity_count": 3, "last_activity_at": "2026-05-01T10:00:00+00:00",
            "created_at": "2026-01-01T00:00:00+00:00", "_persisted": True,
        },
        {
            "name": "bundled-skill", "provenance": "bundled", "state": "active",
            "use_count": 9, "view_count": 4, "patch_count": 0,
            "activity_count": 13, "last_activity_at": "2026-05-10T10:00:00+00:00",
            "created_at": "2026-01-01T00:00:00+00:00", "_persisted": True,
        },
        {
            "name": "hand-authored-skill", "provenance": "user", "state": "active",
            "use_count": 1, "view_count": 0, "patch_count": 0,
            "activity_count": 1, "last_activity_at": "2026-05-05T10:00:00+00:00",
            "created_at": "2026-01-01T00:00:00+00:00", "_persisted": True,
        },
        {
            "name": "external-skill", "provenance": "external", "state": "active",
            "use_count": 0, "view_count": 0, "patch_count": 0,
            "activity_count": 0, "last_activity_at": None,
            "created_at": "2026-01-01T00:00:00+00:00", "_persisted": False,
        },
        {
            "name": "hub-skill", "provenance": "hub", "state": "active",
            "use_count": 0, "view_count": 0, "patch_count": 0,
            "activity_count": 0, "last_activity_at": None,
            "created_at": "2026-01-01T00:00:00+00:00", "_persisted": False,
        },
    ]

def test_usage_lists_all_provenances(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    from tools import skill_usage

    monkeypatch.setattr(skill_usage, "usage_report", _fake_rows)
    args = SimpleNamespace(sort="activity", provenance=None, json=False)
    assert curator_cli._cmd_usage(args) == 0
    out = capsys.readouterr().out
    # Header tally and all managed/unmanaged origins are present.
    assert "agent=1" in out and "user=1" in out and "bundled=1" in out and "hub=1" in out
    assert "agent-skill" in out
    assert "bundled-skill" in out
    assert "hand-authored-skill" in out
    assert "hub-skill" in out
    assert "external=1" in out
    assert "external-skill" in out


def test_usage_filters_hand_authored_skills(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    from tools import skill_usage

    monkeypatch.setattr(skill_usage, "usage_report", _fake_rows)
    args = SimpleNamespace(sort="name", provenance="user", json=False)
    assert curator_cli._cmd_usage(args) == 0
    out = capsys.readouterr().out
    assert "hand-authored-skill" in out
    assert "agent-skill" not in out


def test_usage_provenance_tracks_curator_management(monkeypatch):
    from tools import skill_usage

    monkeypatch.setattr(skill_usage, "is_hub_installed", lambda _name: False)
    monkeypatch.setattr(skill_usage, "is_bundled", lambda _name: False)
    assert skill_usage.provenance("managed", {"created_by": "agent"}) == "agent"
    assert skill_usage.provenance("hand-authored", {"created_by": "learn"}) == "user"
    assert skill_usage.provenance("unrecorded") == "user"


def test_usage_empty(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    from tools import skill_usage

    monkeypatch.setattr(skill_usage, "usage_report", list)
    args = SimpleNamespace(sort="activity", provenance=None, json=False)
    assert curator_cli._cmd_usage(args) == 0
    assert "no skills found" in capsys.readouterr().out


@pytest.mark.parametrize("origin", ["user", "external"])
def test_usage_command_is_registered(origin):
    """The `usage` subcommand must be wired into the curator argparse tree."""
    import argparse
    import hermes_cli.curator as curator_cli

    parser = argparse.ArgumentParser(prog="hermes curator")
    curator_cli.register_cli(parser)
    args = parser.parse_args(["usage", "--sort", "recent", "--provenance", origin, "--json"])
    assert args.func is curator_cli._cmd_usage
    assert args.sort == "recent"
    assert args.provenance == origin
    assert args.json is True


def test_usage_cli_preserves_external_and_local_management(tmp_path, monkeypatch, capsys):
    """Exercise argparse -> CLI -> disk-backed report with both conflict behaviors."""
    import argparse
    import json

    from agent import skill_utils
    from hermes_cli import curator
    from tools import skill_usage

    home = tmp_path / "home"
    skills = home / "skills"
    external = tmp_path / "external"
    for base, names in ((skills, ("managed", "legacy", "foreground", "unrecorded", "shared", "bundled", "hub")),
                        (external, ("mounted", "shared"))):
        for name in names:
            directory = base / name
            directory.mkdir(parents=True)
            (directory / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: Test skill\n---\n", encoding="utf-8")
    (skills / "mounted").mkdir()
    (skills / "mounted" / "SKILL.md").symlink_to(external / "mounted" / "SKILL.md")
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external}\n", encoding="utf-8")
    (skills / ".bundled_manifest").write_text("bundled:hash\n", encoding="utf-8")
    (skills / ".hub").mkdir()
    (skills / ".hub" / "lock.json").write_text(
        json.dumps({"installed": {"hub": {}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    skill_utils._external_dirs_cache_clear()
    try:
        assert skill_usage.save_usage({
            "managed": {"created_by": "agent"},
            "legacy": {"agent_created": True},
            "foreground": {"created_by": "learn"},
            "mounted": {"created_by": "agent"},
            "bundled": {"created_by": "agent"},
            "hub": {"created_by": "agent"},
        })
        parser = argparse.ArgumentParser()
        curator.register_cli(parser)
        args = parser.parse_args(["usage", "--json"])
        assert args.func(args) == 0
        rows = json.loads(capsys.readouterr().out)
        assert {row["name"]: row["provenance"] for row in rows} == {
            "managed": "agent", "legacy": "agent", "foreground": "user",
            "unrecorded": "user", "mounted": "external", "shared": "user",
            "bundled": "bundled", "hub": "hub",
        }
        for origin, expected in (("user", {"foreground", "unrecorded", "shared"}),
                                 ("external", {"mounted"})):
            args = parser.parse_args(["usage", "--provenance", origin, "--json"])
            assert args.func(args) == 0
            assert {row["name"] for row in json.loads(capsys.readouterr().out)} == expected
    finally:
        skill_utils._external_dirs_cache_clear()
