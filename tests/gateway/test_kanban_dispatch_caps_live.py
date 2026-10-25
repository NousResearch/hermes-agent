"""Tests for live kanban dispatcher WIP-cap re-read.

The gateway dispatcher used to capture ``kanban.max_in_progress`` and
``kanban.max_in_progress_per_profile`` once at boot, so an operator who set a
cap after the gateway was already up kept dispatching under the old (often
uncapped) numbers until restart. ``_reread_dispatcher_settings`` is now called
every tick. A config-read failure keeps the previous caps — it must not fall
open to uncapped.
"""

from __future__ import annotations

from gateway.kanban_watchers_dispatcher import (
    _resolve_dispatcher_settings,
    _reread_dispatcher_settings,
)


class _Kb:
    """Only ``DEFAULT_FAILURE_LIMIT`` is read off the kanban_db module here."""

    DEFAULT_FAILURE_LIMIT = 2


def test_cap_takes_effect_on_next_tick_without_restart():
    kb = _Kb()
    boot = _resolve_dispatcher_settings({}, kb)
    assert boot.max_in_progress_per_profile is None

    live = _reread_dispatcher_settings(
        lambda: {"kanban": {"max_in_progress_per_profile": 3}},
        kb,
        boot,
    )
    assert live.max_in_progress_per_profile == 3
    assert boot.max_in_progress_per_profile is None


def test_config_read_failure_keeps_previous_cap():
    kb = _Kb()
    previous = _resolve_dispatcher_settings(
        {"max_in_progress_per_profile": 3},
        kb,
    )
    assert previous.max_in_progress_per_profile == 3

    def boom():
        raise RuntimeError("yaml mid-write")

    kept = _reread_dispatcher_settings(boom, kb, previous)
    assert kept is previous
    assert kept.max_in_progress_per_profile == 3


def test_interval_stays_boot_resolved():
    kb = _Kb()
    boot = _resolve_dispatcher_settings(
        {"dispatch_interval_seconds": 60},
        kb,
    )
    assert boot.interval == 60.0

    live = _reread_dispatcher_settings(
        lambda: {
            "kanban": {
                "dispatch_interval_seconds": 5,
                "max_in_progress_per_profile": 3,
            }
        },
        kb,
        boot,
    )
    assert live.interval == 60.0
    assert live.max_in_progress_per_profile == 3
