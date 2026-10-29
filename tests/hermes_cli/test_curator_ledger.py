"""Tests for `hermes curator ledger` — the per-mutation skill audit view.

Covers the JSON contract a script depends on: a parseable array in every case
(including an empty ledger), entries verbatim with absolute timestamps, and the
human table left untouched as the default.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _fake_rows():
    return [
        {
            "id": "5070487792ca",
            "ts": "2026-09-13T09:41:02.512345+00:00",
            "actor": "agent",
            "action": "patch",
            "skill": "obsidian-vault",
            "evidence": {"absorbed_into": None},
        },
        {
            "id": "0f13c9a4b201",
            "ts": "2026-09-12T18:05:44.001000+00:00",
            "actor": "curator",
            "action": "archive",
            "skill": "stale-helper",
        },
    ]


def _args(**overrides):
    base = dict(skill=None, limit=20, json=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_ledger_json_is_parseable_and_verbatim(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_ledger as skill_ledger

    monkeypatch.setattr(skill_ledger, "list_entries", lambda **kw: _fake_rows())
    assert curator_cli._cmd_ledger(_args(json=True)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == _fake_rows()


def test_ledger_json_timestamps_stay_absolute(monkeypatch, capsys):
    """The table renders "21m ago"; that is a rendering, not data. A consumer
    filtering by an exact window must not have to re-derive time from the clock."""
    import hermes_cli.curator as curator_cli
    import tools.skill_ledger as skill_ledger

    monkeypatch.setattr(skill_ledger, "list_entries", lambda **kw: _fake_rows())
    curator_cli._cmd_ledger(_args(json=True))

    out = capsys.readouterr().out
    assert "2026-09-13T09:41:02.512345+00:00" in out
    assert "ago" not in out


def test_ledger_json_on_empty_ledger_is_an_empty_array(monkeypatch, capsys):
    """The human path prints a sentence when there is nothing to show. Under
    --json that sentence would be a parse error, so the empty case is `[]`."""
    import hermes_cli.curator as curator_cli
    import tools.skill_ledger as skill_ledger

    monkeypatch.setattr(skill_ledger, "list_entries", lambda **kw: [])
    assert curator_cli._cmd_ledger(_args(json=True)) == 0

    assert json.loads(capsys.readouterr().out) == []


def test_ledger_json_passes_the_filters_through(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_ledger as skill_ledger

    seen = {}

    def _capture(**kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(skill_ledger, "list_entries", _capture)
    curator_cli._cmd_ledger(_args(json=True, skill="obsidian-vault", limit=5))
    capsys.readouterr()

    assert seen == {"skill": "obsidian-vault", "limit": 5}


def test_ledger_table_is_still_the_default(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_ledger as skill_ledger

    monkeypatch.setattr(skill_ledger, "list_entries", lambda **kw: _fake_rows())
    assert curator_cli._cmd_ledger(_args()) == 0

    out = capsys.readouterr().out
    assert "5070487792ca" in out
    assert "obsidian-vault" in out
    assert "hermes curator rollback" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_ledger_parser_exposes_json(monkeypatch):
    import argparse

    import hermes_cli.curator as curator_cli

    parser = argparse.ArgumentParser()
    curator_cli.register_cli(parser)
    parsed = parser.parse_args(["ledger", "--json", "--limit", "3"])

    assert parsed.json is True
    assert parsed.limit == 3
    # Default stays the table so existing invocations are unchanged.
    assert parser.parse_args(["ledger"]).json is False
