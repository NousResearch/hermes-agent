"""Regression tests for #78122 — gateway-wide kanban.max_in_progress.

The gateway enumerates boards and calls dispatch_once per board. Each board
DB only counts its own running rows, so the gateway must pass a decreasing
effective cap or N boards each receive the full allowance.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

from gateway.kanban_watchers import (
    _count_board_running,
    _effective_board_max_in_progress,
)


@pytest.fixture()
def isolated_multi_board(monkeypatch):
    """Fresh HERMES_HOME with two boards and a default profile."""
    test_home = tempfile.mkdtemp(prefix="kanban_gateway_cap_")
    os.makedirs(os.path.join(test_home, "profiles", "default"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("hermes_cli")
            or mod.startswith("hermes_state")
            or mod == "hermes_constants"
        ):
            del sys.modules[mod]
    from hermes_cli import kanban_db

    yield kanban_db


def _fake_spawn(*_args, **_kwargs):
    return 12345


def _seed_ready(kb, slug: str, n: int, *, assignee: str = "default") -> None:
    kb.create_board(slug=slug, name=slug)
    with kb.connect_closing(board=slug) as conn:
        for i in range(n):
            kb.create_task(conn, title=f"{slug}-{i}", assignee=assignee)


def _gateway_style_tick(kb, slugs: list[str], max_in_progress: int):
    """Mirror the gateway remaining-capacity loop from kanban_watchers._tick_once."""
    results = []
    for slug in slugs:
        total_running = sum(_count_board_running(kb, other) for other in slugs)
        remaining = max(0, max_in_progress - total_running)
        board_running = _count_board_running(kb, slug)
        board_cap = _effective_board_max_in_progress(board_running, remaining)
        with kb.connect_closing(board=slug) as conn:
            res = kb.dispatch_once(
                conn,
                board=slug,
                spawn_fn=_fake_spawn,
                max_in_progress=board_cap,
            )
        results.append((slug, res))
    return results


def test_effective_board_cap_leaves_exact_remaining_slots():
    assert _effective_board_max_in_progress(3, 2) == 5
    assert _effective_board_max_in_progress(0, 2) == 2
    assert _effective_board_max_in_progress(5, 0) == 5
    assert _effective_board_max_in_progress(-1, -2) == 0


def test_two_boards_cannot_exceed_gateway_cap(isolated_multi_board):
    kb = isolated_multi_board
    _seed_ready(kb, "alpha", 9)
    _seed_ready(kb, "beta", 9)
    # list_boards order: default first, then alpha, beta — use explicit order.
    slugs = ["alpha", "beta"]

    results = _gateway_style_tick(kb, slugs, max_in_progress=9)
    total_spawned = sum(len(res.spawned) for _, res in results)
    total_running = sum(_count_board_running(kb, slug) for slug in slugs)

    assert total_spawned == 9
    assert total_running == 9
    # Without the gateway-wide budget, each board would have spawned 9 (18).
    assert all(len(res.spawned) <= 9 for _, res in results)


def test_existing_workers_consume_gateway_budget(isolated_multi_board):
    kb = isolated_multi_board
    _seed_ready(kb, "alpha", 5)
    _seed_ready(kb, "beta", 5)

    # Pretend alpha already has 2 running workers from a prior tick.
    with kb.connect_closing(board="alpha") as conn:
        ids = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM tasks WHERE status = 'ready' ORDER BY created_at LIMIT 2"
            )
        ]
        with kb.write_txn(conn):
            for tid in ids:
                conn.execute(
                    "UPDATE tasks SET status = 'running' WHERE id = ?",
                    (tid,),
                )

    results = _gateway_style_tick(kb, ["alpha", "beta"], max_in_progress=3)
    total_spawned = sum(len(res.spawned) for _, res in results)
    total_running = sum(
        _count_board_running(kb, slug) for slug in ("alpha", "beta")
    )

    assert total_spawned == 1
    assert total_running == 3


def test_reversed_board_order_still_respects_cap(isolated_multi_board):
    kb = isolated_multi_board
    _seed_ready(kb, "alpha", 5)
    _seed_ready(kb, "beta", 5)

    for slugs in (["alpha", "beta"], ["beta", "alpha"]):
        # Reset both boards between orderings.
        for slug in ("alpha", "beta"):
            with kb.connect_closing(board=slug) as conn:
                with kb.write_txn(conn):
                    conn.execute(
                        "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                        "worker_pid = NULL"
                    )
        results = _gateway_style_tick(kb, slugs, max_in_progress=3)
        total_running = sum(_count_board_running(kb, slug) for slug in slugs)
        assert total_running == 3, f"order={slugs} running={total_running}"
        assert sum(len(res.spawned) for _, res in results) == 3


def test_failed_spawn_does_not_permanently_consume_budget(isolated_multi_board):
    """A spawn that raises after claim must not starve later boards forever.

    The gateway re-counts running rows before each board, so capacity follows
    live DB state rather than a stale spawned counter.
    """
    kb = isolated_multi_board
    _seed_ready(kb, "alpha", 2)
    _seed_ready(kb, "beta", 2)

    calls = {"n": 0}

    def flaky_spawn(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("spawn boom")
        return 99999

    slugs = ["alpha", "beta"]
    for slug in slugs:
        total_running = sum(_count_board_running(kb, other) for other in slugs)
        remaining = max(0, 2 - total_running)
        board_running = _count_board_running(kb, slug)
        board_cap = _effective_board_max_in_progress(board_running, remaining)
        with kb.connect_closing(board=slug) as conn:
            kb.dispatch_once(
                conn,
                board=slug,
                spawn_fn=flaky_spawn,
                max_in_progress=board_cap,
                failure_limit=10,
            )

    total_running = sum(_count_board_running(kb, slug) for slug in slugs)
    assert total_running <= 2
