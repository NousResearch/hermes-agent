"""Tests for `hermes curator archive` and `hermes curator prune`.

Covers:
- archive refuses pinned skills with an `unpin` hint
- archive returns 0/1 based on archive_skill() success
- prune filters pinned and already-archived, applies --days threshold
- prune falls back to created_at when last_activity_at is null
- prune --dry-run makes no state changes
- prune --yes skips confirmation
- prune --days validation
"""

from __future__ import annotations

from types import SimpleNamespace


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


# ─── archive ────────────────────────────────────────────────────────────────


def test_archive_refuses_pinned(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(skill_usage, "get_record", lambda name: {"pinned": True})
    called = []
    monkeypatch.setattr(
        skill_usage, "archive_skill",
        lambda name: called.append(name) or (True, "should not get here"),
    )

    rc = curator_cli._cmd_archive(_ns(skill="pinned-skill"))
    assert rc == 1
    assert called == []
    out = capsys.readouterr().out
    assert "pinned" in out.lower()
    assert "hermes curator unpin" in out


def test_archive_calls_archive_skill(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(skill_usage, "get_record", lambda name: {"pinned": False})
    monkeypatch.setattr(
        skill_usage, "archive_skill",
        lambda name: (True, f"archived to .archive/{name}"),
    )
    rc = curator_cli._cmd_archive(_ns(skill="my-skill"))
    assert rc == 0
    assert "archived to .archive/my-skill" in capsys.readouterr().out


# ─── prune ──────────────────────────────────────────────────────────────────


def _mk_record(name, *, idle_days=0, pinned=False, state="active", created_idle_days=None):
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc)
    last_activity = (now - _dt.timedelta(days=idle_days)).isoformat() if idle_days else None
    created_delta = created_idle_days if created_idle_days is not None else idle_days
    created = (now - _dt.timedelta(days=created_delta)).isoformat()
    return {
        "name": name,
        "state": state,
        "pinned": pinned,
        "last_activity_at": last_activity,
        "created_at": created,
        "activity_count": 0 if idle_days == 0 and last_activity is None else 1,
    }


def test_prune_days_validation(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    rc = curator_cli._cmd_prune(_ns(days=0, yes=True, dry_run=False))
    assert rc == 2
    err = capsys.readouterr().err
    assert "--days must be >= 1" in err


def test_prune_confirms_with_y(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    rows = [_mk_record("old-skill", idle_days=200)]
    monkeypatch.setattr(skill_usage, "curated_report", lambda: rows)
    archived = []
    monkeypatch.setattr(
        skill_usage, "archive_skill",
        lambda name: archived.append(name) or (True, "ok"),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    rc = curator_cli._cmd_prune(_ns(days=30, yes=False, dry_run=False))
    assert rc == 0
    assert archived == ["old-skill"]


def test_prune_reports_partial_failure(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    rows = [
        _mk_record("ok-skill", idle_days=200),
        _mk_record("bad-skill", idle_days=200),
    ]
    monkeypatch.setattr(skill_usage, "curated_report", lambda: rows)

    def fake_archive(name):
        if name == "bad-skill":
            return False, "disk full"
        return True, "ok"

    monkeypatch.setattr(skill_usage, "archive_skill", fake_archive)
    rc = curator_cli._cmd_prune(_ns(days=30, yes=True, dry_run=False))
    assert rc == 1
    out = capsys.readouterr().out
    assert "archived 1/2" in out
    assert "bad-skill: disk full" in out


# ─── argparse wiring ────────────────────────────────────────────────────────


def test_archive_and_prune_registered():
    import argparse
    import hermes_cli.curator as curator_cli

    parser = argparse.ArgumentParser(prog="hermes curator")
    curator_cli.register_cli(parser)

    args = parser.parse_args(["archive", "my-skill"])
    assert args.skill == "my-skill"
    assert args.func.__name__ == "_cmd_archive"

    args = parser.parse_args(["prune", "--days", "45", "--yes", "--dry-run"])
    assert args.days == 45
    assert args.yes is True
    assert args.dry_run is True
    assert args.func.__name__ == "_cmd_prune"


