"""Health classification for the embedded Kanban dispatcher."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.kanban_watchers_dispatcher import (
    _dispatcher_capacity_saturated,
    _dispatcher_tick_is_unhealthy,
)


def test_full_worker_capacity_is_deferred_not_stuck() -> None:
    """A ready queue behind a full host cap must not count as a bad tick."""
    assert not _dispatcher_tick_is_unhealthy(
        ready_pending=True,
        any_spawned=False,
        all_capacity_saturated=True,
    )


def test_spawnable_ready_work_without_progress_is_unhealthy() -> None:
    assert _dispatcher_tick_is_unhealthy(
        ready_pending=True,
        any_spawned=False,
        all_capacity_saturated=False,
    )


@pytest.mark.parametrize("all_capacity_saturated", [False, True])
def test_empty_ready_queue_is_healthy_regardless_of_capacity(
    all_capacity_saturated: bool,
) -> None:
    assert not _dispatcher_tick_is_unhealthy(
        ready_pending=False,
        any_spawned=False,
        all_capacity_saturated=all_capacity_saturated,
    )


@pytest.mark.parametrize("all_capacity_saturated", [False, True])
def test_spawned_worker_is_progress_regardless_of_capacity(
    all_capacity_saturated: bool,
) -> None:
    assert not _dispatcher_tick_is_unhealthy(
        ready_pending=True,
        any_spawned=True,
        all_capacity_saturated=all_capacity_saturated,
    )


def test_one_saturated_board_does_not_mask_an_unsaturated_board() -> None:
    results = [
        ("full", SimpleNamespace(host_capacity_saturated=True)),
        ("stuck", SimpleNamespace(host_capacity_saturated=False)),
    ]

    assert not _dispatcher_capacity_saturated(results)


def test_all_boards_saturated_defers_aggregate_health_warning() -> None:
    results = [
        ("first", SimpleNamespace(host_capacity_saturated=True)),
        ("second", SimpleNamespace(host_capacity_saturated=True)),
    ]

    assert _dispatcher_capacity_saturated(results)


def test_priority_capacity_is_rechecked_between_gateway_ticks(tmp_path, monkeypatch):
    from gateway.kanban_watchers_dispatcher import _KanbanDispatcher, _resolve_dispatcher_settings
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as dispatcher
    from hermes_cli import kanban_runtime_priority as priority
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    root = tmp_path / "project"
    root.mkdir()
    active = [False]
    monkeypatch.setattr(priority, "_process_scan", lambda: priority.ProcessScan(
        (priority.ProcessSnapshot(123, ("python3", "main.py"), str(root)),) if active[0] else (), True))
    settings = _resolve_dispatcher_settings({"max_in_progress": 8, "priority_runtime_guard": {
        "enabled": True, "project_roots": [str(root)], "entrypoints": ["main.py"], "max_in_progress": 2,
    }}, kb)
    observed = []
    monkeypatch.setattr(dispatcher, "dispatch_once", lambda conn, **kwargs: observed.append(kwargs["max_in_progress"]))
    worker = _KanbanDispatcher(kb, settings)
    for state in (False, True, False):
        active[0] = state
        worker.tick_once_for_board("default")
    assert observed == [8, 2, 8]
