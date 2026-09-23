"""Tests for `hermes curator usage` — the all-skills usage view.

Covers:
- Lists every skill regardless of provenance (agent / user / bundled / hub), unlike
  `status` which is scoped to curator-managed candidates.
- --provenance filter, --sort ordering, and --json output.
"""

from __future__ import annotations

import json
from types import SimpleNamespace


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
            "name": "hub-skill", "provenance": "hub", "state": "active",
            "use_count": 0, "view_count": 0, "patch_count": 0,
            "activity_count": 0, "last_activity_at": None,
            "created_at": "2026-01-01T00:00:00+00:00", "_persisted": False,
        },
    ]


def test_usage_lists_all_provenances(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

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


def test_usage_filters_hand_authored_skills(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(skill_usage, "usage_report", _fake_rows)
    args = SimpleNamespace(sort="name", provenance="user", json=False)
    assert curator_cli._cmd_usage(args) == 0
    out = capsys.readouterr().out
    assert "hand-authored-skill" in out
    assert "agent-skill" not in out


def test_usage_provenance_tracks_curator_management(monkeypatch):
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(skill_usage, "is_hub_installed", lambda _name: False)
    monkeypatch.setattr(skill_usage, "is_bundled", lambda _name: False)
    assert skill_usage.provenance("managed", {"created_by": "agent"}) == "agent"
    assert skill_usage.provenance("hand-authored", {"created_by": "learn"}) == "user"
    assert skill_usage.provenance("unrecorded") == "user"


def test_usage_empty(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(skill_usage, "usage_report", lambda: [])
    args = SimpleNamespace(sort="activity", provenance=None, json=False)
    assert curator_cli._cmd_usage(args) == 0
    assert "no skills found" in capsys.readouterr().out


def test_usage_command_is_registered():
    """The `usage` subcommand must be wired into the curator argparse tree."""
    import argparse
    import hermes_cli.curator as curator_cli

    parser = argparse.ArgumentParser(prog="hermes curator")
    curator_cli.register_cli(parser)
    args = parser.parse_args(["usage", "--sort", "recent", "--provenance", "user", "--json"])
    assert args.func is curator_cli._cmd_usage
    assert args.sort == "recent"
    assert args.provenance == "user"
    assert args.json is True
