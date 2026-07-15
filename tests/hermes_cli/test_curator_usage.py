"""Tests for `hermes curator usage` — the all-skills usage view.

Covers:
- Lists every skill regardless of provenance (agent / bundled / hub), unlike
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
    # Header tally and all three skills present.
    assert "agent=1" in out and "bundled=1" in out and "hub=1" in out
    assert "agent-skill" in out
    assert "bundled-skill" in out
    assert "hub-skill" in out


def test_usage_sort_activity_orders_most_used_first(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(skill_usage, "usage_report", _fake_rows)
    args = SimpleNamespace(sort="activity", provenance=None, json=False)
    assert curator_cli._cmd_usage(args) == 0
    out = capsys.readouterr().out
    # bundled-skill (act=13) must appear before agent-skill (act=3).
    assert out.index("bundled-skill") < out.index("agent-skill")


def test_usage_provenance_filter(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(skill_usage, "usage_report", _fake_rows)
    args = SimpleNamespace(sort="activity", provenance="bundled", json=False)
    assert curator_cli._cmd_usage(args) == 0
    out = capsys.readouterr().out
    assert "bundled-skill" in out
    assert "agent-skill" not in out
    assert "hub-skill" not in out


def test_usage_json_output(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(skill_usage, "usage_report", _fake_rows)
    args = SimpleNamespace(sort="name", provenance=None, json=True)
    assert curator_cli._cmd_usage(args) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert {r["name"] for r in data} == {"agent-skill", "bundled-skill", "hub-skill"}
    assert {r["provenance"] for r in data} == {"agent", "bundled", "hub"}


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
    args = parser.parse_args(["usage", "--sort", "recent", "--provenance", "hub", "--json"])
    assert args.func is curator_cli._cmd_usage
    assert args.sort == "recent"
    assert args.provenance == "hub"
    assert args.json is True
