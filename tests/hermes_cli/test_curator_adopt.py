"""Tests for `hermes curator adopt` — explicit provenance handover.

Covers the --dry-run decision path: it must consult the usage registry
(.usage.json) and report already curator-managed skills as such, instead of
listing them under "would adopt" like never-adopted skills.
"""

from __future__ import annotations

from types import SimpleNamespace


def test_adopt_dry_run_reports_already_managed(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(skill_usage, "is_curator_managed", lambda name: True)
    args = SimpleNamespace(
        skill=["my-skill"], all_unmanaged=False, dry_run=True, yes=False
    )
    assert curator_cli._cmd_adopt(args) == 0
    out = capsys.readouterr().out
    assert "'my-skill' is already curator-managed" in out
    assert "would adopt" not in out
    assert "+ my-skill" not in out


def test_adopt_dry_run_lists_unmanaged_only(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(
        skill_usage, "is_curator_managed", lambda name: name == "managed-skill"
    )
    args = SimpleNamespace(
        skill=["managed-skill", "fresh-skill"],
        all_unmanaged=False,
        dry_run=True,
        yes=False,
    )
    assert curator_cli._cmd_adopt(args) == 0
    out = capsys.readouterr().out
    assert "'managed-skill' is already curator-managed" in out
    assert "would adopt 1 skill(s) (dry run):" in out
    assert "+ fresh-skill" in out
    assert "+ managed-skill" not in out


def test_adopt_dry_run_unmanaged(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(skill_usage, "is_curator_managed", lambda name: False)
    args = SimpleNamespace(
        skill=["fresh-skill"], all_unmanaged=False, dry_run=True, yes=False
    )
    assert curator_cli._cmd_adopt(args) == 0
    out = capsys.readouterr().out
    assert "would adopt 1 skill(s) (dry run):" in out
    assert "+ fresh-skill" in out
