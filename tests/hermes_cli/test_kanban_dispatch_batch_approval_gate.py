"""Tests for the kanban dispatcher's Decision HUD batch-approval gate (F1).

DEFAULT-GATED: an unset ``batch_approval_gate`` on a board blocks spawning —
see ``_check_batch_approval_gate`` in ``hermes_cli/kanban_db_dispatch.py`` and
decision-hub-first-work/plans/02-minimal-bridge-alternative.md. The gate
itself is config-gated off by default (``kanban.batch_approval_gate_enabled``
in config.yaml) so these tests explicitly opt in by monkeypatching
``load_config``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _enable_batch_approval_gate(monkeypatch):
    """Monkeypatch load_config so kanban.batch_approval_gate_enabled reads True."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"batch_approval_gate_enabled": True}},
    )


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _enable_batch_approval_gate(monkeypatch)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def test_dispatch_once_blocks_when_gate_unset(conn, all_assignees_spawnable):
    """No batch_approval_gate configured on the board -> blocked, nothing
    promoted/spawned, spawn_fn never called."""
    kb.create_task(conn, title="t", assignee="alice")

    spawn_calls: list = []

    def spy_spawn(task, workspace_path, board=None):
        spawn_calls.append(getattr(task, "id", task))
        return 42

    result = kbd.dispatch_once(conn, spawn_fn=spy_spawn)

    assert result.batch_approval_blocked is not None
    assert result.spawned == []
    assert spawn_calls == []


def test_dispatch_once_blocks_when_gate_not_approved(
    conn, all_assignees_spawnable, monkeypatch,
):
    kb.write_board_metadata(None, project_id="proj")
    import json

    meta_path = kb.board_metadata_path(kb._slug_or_default(None))
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["batch_approval_gate"] = "batch-1"
    meta_path.write_text(json.dumps(raw), encoding="utf-8")

    monkeypatch.setattr(
        "hermes_cli.plugin_bridges.decision_hud.check_batch_approval",
        lambda **kw: (False, "pending"),
    )

    kb.create_task(conn, title="t", assignee="alice")

    spawn_calls: list = []

    def spy_spawn(task, workspace_path, board=None):
        spawn_calls.append(getattr(task, "id", task))
        return 42

    result = kbd.dispatch_once(conn, spawn_fn=spy_spawn)

    assert result.batch_approval_blocked is not None
    assert "pending" in result.batch_approval_blocked
    assert result.spawned == []
    assert spawn_calls == []


def test_dispatch_once_proceeds_when_gate_approved(
    conn, all_assignees_spawnable, monkeypatch,
):
    import json

    meta_path = kb.board_metadata_path(kb._slug_or_default(None))
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"batch_approval_gate": "batch-1"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        "hermes_cli.plugin_bridges.decision_hud.check_batch_approval",
        lambda **kw: (True, ""),
    )

    tid = kb.create_task(conn, title="t", assignee="alice")

    def fake_spawn(task, workspace_path, board=None):
        return 42

    result = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert result.batch_approval_blocked is None
    assert any(row[0] == tid for row in result.spawned)


def test_dispatch_once_still_reclaims_when_gate_blocks(
    conn, all_assignees_spawnable,
):
    """Reclaim bookkeeping runs before the gate check, so a blocked board
    still reclaims a stale claim."""
    tid = kb.create_task(conn, title="t", assignee="worker")
    kb.claim_task(conn, tid)
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?",
        (int(time.time()) - 100, tid),
    )
    conn.commit()

    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)

    assert result.batch_approval_blocked is not None
    assert result.reclaimed >= 1


def test_check_batch_approval_gate_uses_project_id_when_set(
    conn, all_assignees_spawnable, monkeypatch,
):
    import json

    meta_path = kb.board_metadata_path(kb._slug_or_default(None))
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {"batch_approval_gate": "batch-1", "project_id": "real-project"}
        ),
        encoding="utf-8",
    )

    seen = {}

    def fake_check(*, project, batch_id):
        seen["project"] = project
        seen["batch_id"] = batch_id
        return True, ""

    monkeypatch.setattr(
        "hermes_cli.plugin_bridges.decision_hud.check_batch_approval",
        fake_check,
    )

    kb.create_task(conn, title="t", assignee="alice")
    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)

    assert seen["project"] == "real-project"
    assert seen["batch_id"] == "batch-1"


def test_gate_is_a_noop_when_feature_flag_disabled(
    conn, all_assignees_spawnable, monkeypatch,
):
    """Default-disabled config flag: with no override (real DEFAULT_CONFIG
    default of False), dispatch must behave exactly as before this change —
    no gate configured, gate not checked at all, tasks spawn normally."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"kanban": {}},
    )

    tid = kb.create_task(conn, title="t", assignee="alice")

    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)

    assert result.batch_approval_blocked is None


def test_batch_approval_gate_enabled_defaults_to_false_in_real_config():
    """Proves the real DEFAULT_CONFIG (not a test double) ships the gate
    off — regression guard for the AGENTS.md config-surface requirement:
    this must be a config.yaml key with a real default, not a bare env var
    with no config-side presence at all."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["kanban"]["batch_approval_gate_enabled"] is False
