"""kanban.dispatch_profiles: per-home claim allowlist for shared boards (#110995).

On a board shared across Hermes homes (one kanban.db mounted in several
containers) every home's ``profile_exists("default")`` is True, so any home's
dispatcher could claim cards assigned to ``default``. The allowlist wraps the
same predicate consumed by the spawn gate and the spawnable telemetry, so a
foreign assignee lands in ``skipped_nonspawnable`` and does not keep the
gateway wake-up loop hot.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_allowlist_without_default_skips_default_card(kanban_home, all_assignees_spawnable):
    """Fail-closed: the allowlist is set and ``default`` is not on it, so the
    card is neither spawnable (wake-up telemetry) nor claimed (spawn gate),
    even though every profile exists locally."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  dispatch_profiles:\n    - sage\n", encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="foreign card", assignee="default")
        assert kbd.has_spawnable_ready(conn) is False
        res = kbd.dispatch_once(conn, dry_run=True)
    assert res.spawned == []
    assert res.skipped_nonspawnable == [tid]


def test_default_assignee_outside_allowlist_leaves_card_unassigned(
    kanban_home, all_assignees_spawnable,
):
    """Invariant: a ``default_assignee`` this home may not claim must not be
    persisted onto an unassigned shared-board card (no row write, no
    ``assigned`` event) — another home owns that card."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  dispatch_profiles:\n    - sage\n  default_assignee: default\n",
        encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unassigned card")
        res = kbd.dispatch_once(
            conn, dry_run=False, spawn_fn=lambda _t, _w: 4242,
            default_assignee="default",
        )
        task = kb.get_task(conn, tid)
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert res.spawned == []
    assert res.auto_assigned_default == []
    assert task.assignee is None
    assert "assigned" not in kinds


def test_unset_allowlist_keeps_default_claimable(kanban_home, all_assignees_spawnable):
    """No key = upstream behaviour: any existing profile, ``default`` included."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="local card", assignee="default")
        assert kbd.has_spawnable_ready(conn) is True
        res = kbd.dispatch_once(conn, dry_run=True)
    assert [t for t, _a, _w in res.spawned] == [tid]
    assert res.skipped_nonspawnable == []


@pytest.mark.parametrize("value", ["", None], ids=["empty_string", "bare_key"])
def test_present_blank_or_null_allowlist_skips_all_cards(
    kanban_home, all_assignees_spawnable, value, caplog,
):
    """A present key with no names must not fall back to unrestricted claims,
    and must say so once in the log so an upgraded home that copied the old
    ``dispatch_profiles: null`` example is not silently idle (#113620)."""
    rendered = "" if value is None else ' ""'
    (kanban_home / "config.yaml").write_text(
        f"kanban:\n  dispatch_profiles:{rendered}\n", encoding="utf-8",
    )
    with kbc.connect() as conn, caplog.at_level("WARNING", logger="hermes_cli.kanban_db"):
        tid = kb.create_task(conn, title="foreign card", assignee="default")
        res = kbd.dispatch_once(conn, dry_run=True)
    assert res.spawned == []
    assert res.skipped_nonspawnable == [tid]
    assert any("present but empty" in r.getMessage() and "omit the key" in r.getMessage()
               for r in caplog.records)


def test_allowlist_config_read_failure_skips_all_cards(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A broken config read must not make this shared home claim every profile."""
    from hermes_cli import config_effective

    def raise_read_error(**_kwargs):
        raise OSError("config unavailable")

    monkeypatch.setattr(config_effective, "load_user_config_effective", raise_read_error)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="foreign card", assignee="default")
        res = kbd.dispatch_once(conn, dry_run=True)
    assert res.spawned == []
    assert res.skipped_nonspawnable == [tid]


@pytest.mark.parametrize("config, expected", [
    ("", "any"),
    ("kanban:\n  dispatch_profiles: [researcher, sage]\n", "researcher, sage"),
    ("kanban:\n  dispatch_profiles: []\n", "none (fail-closed"),
], ids=["absent", "listed", "empty_list"])
def test_diagnostics_reports_resolved_allowlist(kanban_home, capsys, config, expected):
    """`hermes kanban diagnostics` (text and --json) shows what this home may claim."""
    import argparse
    import json

    from hermes_cli import kanban as kanban_cli

    if config:
        (kanban_home / "config.yaml").write_text(config, encoding="utf-8")
    assert kanban_cli._cmd_diagnostics(argparse.Namespace(task=None, severity=None, json=False)) == 0
    text = capsys.readouterr().out
    assert f"kanban.dispatch_profiles: {expected}" in text
    assert kanban_cli._cmd_diagnostics(argparse.Namespace(task=None, severity=None, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [{"task_id": None, "dispatch_profiles": payload[-1]["dispatch_profiles"], "diagnostics": []}]
    assert payload[-1]["dispatch_profiles"].startswith(expected)


@pytest.mark.parametrize("body", ["kanban: null", "kanban: []", "kanban: invalid"])
@pytest.mark.parametrize("warm", [False, True])
def test_malformed_managed_section_never_widens_claim_scope(
    kanban_home, all_assignees_spawnable, monkeypatch, body, warm
):
    from hermes_cli.config_effective import load_user_config_effective

    (kanban_home / "config.yaml").write_text(
        "kanban: {dispatch_profiles: []}", encoding="utf-8"
    )
    managed = kanban_home / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (managed / "config.yaml").write_text(body, encoding="utf-8")
    if warm:
        load_user_config_effective()

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="synthetic foreign card", assignee="default")
        result = kbd.dispatch_once(conn, dry_run=True)
    assert result.spawned == [] and result.skipped_nonspawnable == [tid]
