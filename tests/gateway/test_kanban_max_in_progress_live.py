"""Tests for live ``kanban.max_in_progress`` resolution (t_62f24f45).

The gateway dispatcher used to capture ``kanban.max_in_progress`` once at
boot (``_DispatcherSettings`` docstring: "restart to apply"), so an operator
lowering the cap to relieve live memory pressure saw no effect until a full
gateway restart -- the same class of bug #49638 fixed for the auto-decompose
toggle. ``_resolve_live_max_in_progress`` is now called every tick, reading
the current config, the same way ``_resolve_auto_decompose_settings`` does.
"""

from __future__ import annotations

from gateway.kanban_watchers_common import _resolve_live_max_in_progress


def test_unset_falls_back_to_derived_default(monkeypatch):
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "derive_default_max_in_progress", lambda: 7)
    assert _resolve_live_max_in_progress(lambda: {"kanban": {}}) == 7


def test_explicit_value_wins_over_derived_default(monkeypatch):
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "derive_default_max_in_progress", lambda: 7)
    assert _resolve_live_max_in_progress(lambda: {"kanban": {"max_in_progress": 3}}) == 3


def test_lowering_the_cap_is_observed_on_the_very_next_call():
    """The whole point of the fix: no restart, no boot-captured staleness."""
    live_cfg = {"kanban": {"max_in_progress": 11}}
    assert _resolve_live_max_in_progress(lambda: live_cfg) == 11
    live_cfg["kanban"]["max_in_progress"] = 7
    assert _resolve_live_max_in_progress(lambda: live_cfg) == 7


def test_config_read_failure_fails_safe_to_none():
    def _boom():
        raise RuntimeError("config unavailable")

    assert _resolve_live_max_in_progress(_boom) is None


def test_invalid_value_is_ignored_not_crashed_on():
    assert _resolve_live_max_in_progress(lambda: {"kanban": {"max_in_progress": "oops"}}) is None
    assert _resolve_live_max_in_progress(lambda: {"kanban": {"max_in_progress": 0}}) is None
