"""Tests for live auto-decompose settings resolution (issue #49638).

The gateway dispatcher used to capture ``kanban.auto_decompose`` once at boot,
so a user who flipped it to ``false`` to STOP runaway auto-decompose (which had
created and launched tasks they didn't intend) found the flag had no effect
without a full gateway restart. ``_resolve_auto_decompose_settings`` is now
called every tick, reading the current config.
"""

from __future__ import annotations

import pytest

from gateway.kanban_watchers import _resolve_auto_decompose_settings


def test_enabled_by_default_when_key_absent():
    enabled, per_tick = _resolve_auto_decompose_settings(lambda: {"kanban": {}})
    assert enabled is True
    assert per_tick == 3


def test_disabled_when_flag_false():
    enabled, per_tick = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose": False}}
    )
    assert enabled is False


# ---------------------------------------------------------------------------
# Shared tick wiring (#87283)
#
# The auto-decompose pass now lives in
# ``hermes_cli.kanban_decompose.run_auto_decompose_tick`` and is called by
# every dispatch entry point. The gateway's module-level resolver stays a
# verbatim twin (its call site must not depend on a hermes_cli import), so
# these tests pin the twin and the canonical resolver together, and pin
# the watcher closure's delegation.
# ---------------------------------------------------------------------------


def _shared_resolver(cfg):
    from hermes_cli.kanban_decompose import resolve_auto_decompose_settings

    return resolve_auto_decompose_settings(lambda: cfg)


def test_gateway_resolver_matches_shared_resolver():
    """Drift-guard: the import-free gateway twin and the canonical shared
    resolver must agree on every representative config shape."""
    cases = [
        {},
        {"kanban": {}},
        {"kanban": {"auto_decompose": True}},
        {"kanban": {"auto_decompose": False}},
        {"kanban": {"auto_decompose_per_tick": 7}},
        {"kanban": {"auto_decompose_per_tick": 0}},
        {"kanban": {"auto_decompose_per_tick": -2}},
        {"kanban": {"auto_decompose_per_tick": "5"}},
        {"kanban": {"auto_decompose_per_tick": "junk"}},
    ]
    for cfg in cases:
        gateway = _resolve_auto_decompose_settings(lambda c=cfg: c)
        shared = _shared_resolver(cfg)
        assert gateway == shared, f"drift on cfg={cfg!r}: {gateway} != {shared}"


def test_gateway_resolver_fails_safe_on_config_error():
    def _raises():
        raise RuntimeError("config unreadable")

    # Fails CLOSED: a transient read error must not re-enable a feature
    # the user may have turned off.
    assert _resolve_auto_decompose_settings(_raises) == (False, 3)


def test_watcher_closure_delegates_to_shared_tick(monkeypatch):
    """The dispatcher-watcher closure must route through
    ``run_auto_decompose_tick`` so the four entry points cannot drift."""
    import gateway.kanban_watchers as kw_mod

    calls: list[int] = []
    import hermes_cli.kanban_decompose as decomp_mod

    monkeypatch.setattr(
        decomp_mod, "run_auto_decompose_tick",
        lambda *, per_tick=None: calls.append(per_tick) or 3,
    )
    tick = kw_mod._make_auto_decompose_tick()
    assert tick(2) == 3
    assert calls == [2]


