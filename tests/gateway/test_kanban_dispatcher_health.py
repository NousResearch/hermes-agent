"""Health classification for the embedded Kanban dispatcher."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.kanban_watchers import (
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
