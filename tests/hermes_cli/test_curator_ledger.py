"""Behavior contracts for ``hermes curator ledger`` output."""

from __future__ import annotations

import json
from datetime import datetime


def _entry() -> dict:
    return {
        "id": "5070487792ca",
        "ts": "2026-09-13T14:05:06+00:00",
        "actor": "agent",
        "action": "patch",
        "skill": "obsidian-vault",
        "evidence": {},
        "before": [],
        "after": [],
    }


def test_ledger_json_is_machine_readable_and_preserves_filters(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_ledger as skill_ledger

    calls = []
    monkeypatch.setattr(
        skill_ledger,
        "list_entries",
        lambda *, skill, limit: calls.append((skill, limit)) or [_entry()],
    )

    assert curator_cli.cli_main(
        ["ledger", "--skill", "obsidian-vault", "--limit", "7", "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert calls == [("obsidian-vault", 7)]
    assert payload == [_entry()]
    assert datetime.fromisoformat(payload[0]["ts"]).utcoffset() is not None

    monkeypatch.setattr(skill_ledger, "list_entries", lambda **_kwargs: [])
    assert curator_cli.cli_main(["ledger", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_ledger_table_uses_absolute_timestamp(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_ledger as skill_ledger

    monkeypatch.setattr(skill_ledger, "list_entries", lambda **_kwargs: [_entry()])

    assert curator_cli.cli_main(["ledger"]) == 0
    output = capsys.readouterr().out

    assert "timestamp" in output
    assert _entry()["ts"] in output
    assert "ago" not in output
