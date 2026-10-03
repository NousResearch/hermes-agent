"""Per-provider concurrency budget for the kanban dispatcher (#123654).

The run's *resolved* provider (task ``provider_override`` first, then the
assignee profile's configured provider) is budgeted by
``kanban.provider_concurrency``. Over budget defers to the next tick —
never a kill — exactly like the existing host / per-profile caps.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


def _make_ready(conn: sqlite3.Connection, title: str, assignee: str, **overrides) -> str:
    return kb.create_task(conn, title=title, assignee=assignee, **overrides)


def _make_running(conn: sqlite3.Connection, title: str, assignee: str, **overrides) -> str:
    tid = kb.create_task(conn, title=title, assignee=assignee, **overrides)
    assert kb.claim_task(conn, tid) is not None
    return tid


def test_provider_budget_defers_over_budget_task(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Second same-provider spawn defers when the budget is exhausted."""
    monkeypatch.setattr(kbd, "_assignee_provider_config", lambda name: ("anthropic", None))
    spawns: list = []
    with kbc.connect() as conn:
        _make_running(conn, "busy", "alice")
        _make_ready(conn, "waiting", "bob")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            provider_concurrency={"anthropic": 1},
        )
    assert not spawns
    assert not res.spawned
    assert len(res.skipped_provider_capped) == 1
    tid, provider, used, budget = res.skipped_provider_capped[0]
    assert provider == "anthropic"
    assert (used, budget) == (1, 1)


def test_provider_override_keys_on_resolved_provider_not_profile(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A task override on another provider is not charged to the profile's budget."""
    # Both profiles resolve to anthropic; the waiting card pins openai per run.
    monkeypatch.setattr(kbd, "_assignee_provider_config", lambda name: ("anthropic", None))
    spawns: list = []
    with kbc.connect() as conn:
        _make_running(conn, "busy", "alice")
        _make_ready(
            conn, "other-lane", "bob",
            model_override="gpt-x", provider_override="openai",
        )
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            provider_concurrency={"anthropic": 1, "openai": 1},
        )
    assert len(spawns) == 1
    assert not res.skipped_provider_capped


def test_unlisted_provider_has_no_budget_but_default_applies(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Unlisted providers fail open; a `default` entry budgets them."""
    monkeypatch.setattr(kbd, "_assignee_provider_config", lambda name: ("anthropic", None))
    spawns: list = []
    with kbc.connect() as conn:
        _make_running(conn, "busy", "alice")
        _make_ready(conn, "waiting", "bob")
        # No budget mentions anthropic -> spawns freely.
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            provider_concurrency={"openai": 1},
        )
    assert len(spawns) == 1
    assert not res.skipped_provider_capped

    spawns.clear()
    with kbc.connect() as conn:
        _make_ready(conn, "waiting-2", "bob")
        # `default` covers the unlisted anthropic lane (1 already running).
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            provider_concurrency={"default": 1},
        )
    assert not spawns
    assert len(res.skipped_provider_capped) == 1


def test_describe_suppression_reports_provider_budget(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The tick summary names the exhausted provider budget."""
    monkeypatch.setattr(kbd, "_assignee_provider_config", lambda name: ("anthropic", None))
    with kbc.connect() as conn:
        _make_running(conn, "busy", "alice")
        _make_ready(conn, "waiting", "bob")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory([]),
            provider_concurrency={"anthropic": 1},
        )
    held = kbd.describe_suppression([res])
    assert "provider_budget[anthropic]=1/1" in held


def test_provider_budget_counts_running_work_on_other_boards(
    kanban_home,
    all_assignees_spawnable,
    monkeypatch,
):
    """Other boards' workers consume the same provider budget.

    Each board's tick only sees its own DB, so without the fleet-wide fold a
    budget of N is effectively N x active boards.
    """
    monkeypatch.setattr(
        kbd, "_assignee_provider_config", lambda name: ("anthropic", None)
    )
    kb.create_board("second")
    with kbc.connect(board="second") as conn:
        _make_running(conn, "busy-elsewhere", "alice")

    spawns: list = []
    with kbc.connect() as conn:
        _make_ready(conn, "wants-to-run", "bob")
        res = kbd.dispatch_once(
            conn,
            spawn_fn=_fake_spawn_factory(spawns),
            provider_concurrency={"anthropic": 1},
        )

    assert not spawns
    assert len(res.skipped_provider_capped) == 1
    assert kbd.count_running_tasks_other_boards_by_provider() == {"anthropic": 1}


def test_provider_budget_keeps_partial_room_across_boards(
    kanban_home,
    all_assignees_spawnable,
    monkeypatch,
):
    """One worker elsewhere + budget 2 leaves exactly one slot on this board."""
    monkeypatch.setattr(
        kbd, "_assignee_provider_config", lambda name: ("anthropic", None)
    )
    kb.create_board("second")
    with kbc.connect(board="second") as conn:
        _make_running(conn, "busy-elsewhere", "alice")

    spawns: list = []
    with kbc.connect() as conn:
        for title in ("wants-1", "wants-2"):
            _make_ready(conn, title, "bob")
        res = kbd.dispatch_once(
            conn,
            spawn_fn=_fake_spawn_factory(spawns),
            provider_concurrency={"anthropic": 2},
        )

    assert len(spawns) == 1
    # 1 elsewhere + the local spawn = budget 2, so the second ready task waits.
    assert [entry[1:] for entry in res.skipped_provider_capped] == [
        ("anthropic", 2, 2)
    ]


def test_per_provider_other_board_counts_fail_open(kanban_home, monkeypatch):
    """A broken board enumeration must not brick the tick (empty counts)."""
    monkeypatch.setattr(
        kb,
        "list_boards",
        lambda **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert kbd.count_running_tasks_other_boards_by_provider() == {}


def test_normalize_provider_budgets_parsing():
    assert kbd.normalize_provider_budgets(None) is None
    assert kbd.normalize_provider_budgets("lots") is None
    assert kbd.normalize_provider_budgets({}) is None
    assert kbd.normalize_provider_budgets({"anthropic": 2, "x": 0, "y": "nope", "z": -1}) == {
        "anthropic": 2
    }
    assert kbd.normalize_provider_budgets({"anthropic": "3", " default ": 1}) == {
        "anthropic": 3, "default": 1,
    }
