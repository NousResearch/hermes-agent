"""``hermes curator ledger`` and ledger-entry rollback show the write origin and review route."""

import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    from tools import skill_ledger

    home = tmp_path / "home"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(skill_ledger, "get_hermes_home", lambda: home)
    return skill_ledger


def _write_rows(skill_ledger, rows):
    path = skill_ledger.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {"ts": "2026-11-15T00:00:00+00:00", "evidence": {}, "before": [], "after": [], "skill": "s"}
    with open(path, "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps({**base, **row}) + "\n" for row in rows)


def _table(capsys):
    from hermes_cli.curator import _cmd_ledger

    assert _cmd_ledger(SimpleNamespace(compact=False, skill=None, limit=20)) == 0
    return {line.split()[0]: line for line in capsys.readouterr().out.splitlines() if line[:3] in {"id-"}}


def test_ledger_table_shows_origin_and_route(ledger, capsys):
    _write_rows(ledger, [
        {"id": "id-routed", "actor": "curator", "action": "patch", "origin": "background_review",
         "route": {"model": "m", "provider": "p", "routed": True}},
        {"id": "id-same", "actor": "curator", "action": "patch", "origin": "background_review",
         "route": {"model": "m2", "provider": "p2", "routed": False}},
        {"id": "id-bgonly", "actor": "curator", "action": "patch", "origin": "background_review"},
        {"id": "id-curator", "actor": "curator", "action": "archive", "origin": "curator"},
        {"id": "id-fg", "actor": "agent", "action": "patch", "origin": "foreground"},
        {"id": "id-legacy", "actor": "curator", "action": "patch"},
    ])
    rows = _table(capsys)
    assert rows["id-routed"].endswith("s  [background_review p/m, routed]")
    assert rows["id-same"].endswith("s  [background_review p2/m2, same model]")
    assert rows["id-bgonly"].endswith("s  [background_review]")
    assert rows["id-curator"].endswith("s  [curator]")
    assert rows["id-fg"].endswith(" s")
    assert rows["id-legacy"].endswith(" s")


def test_route_suffix_follows_existing_evidence_suffix(ledger, capsys):
    _write_rows(ledger, [
        {"id": "id-absorbed", "actor": "curator", "action": "delete", "origin": "curator",
         "evidence": {"absorbed_into": "umbrella"}},
    ])
    line = _table(capsys)["id-absorbed"]
    assert line.endswith("s  → absorbed into 'umbrella'  [curator]")


def test_rollback_target_prints_origin_and_route(ledger, capsys, monkeypatch):
    import hermes_cli.curator as cli

    _write_rows(ledger, [
        {"id": "id-routed", "actor": "curator", "action": "patch", "origin": "background_review",
         "route": {"model": "m", "provider": "p", "routed": True}},
        {"id": "id-legacy", "actor": "curator", "action": "patch"},
    ])
    # Declining the confirmation prompt returns before any restore.
    monkeypatch.setattr(cli, "_confirm", lambda prompt: False)
    assert cli._rollback_ledger_entry(SimpleNamespace(yes=False), "id-routed") == 1
    out = capsys.readouterr().out
    assert "  origin: background_review\n" in out
    assert "  route:  p/m, routed\n" in out
    assert cli._rollback_ledger_entry(SimpleNamespace(yes=False), "id-legacy") == 1
    out = capsys.readouterr().out
    assert "origin:" not in out and "route:" not in out
