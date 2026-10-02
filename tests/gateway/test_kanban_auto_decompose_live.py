"""Tests for live auto-decompose settings resolution (issue #49638).

The gateway dispatcher used to capture ``kanban.auto_decompose`` once at boot,
so a user who flipped it to ``false`` to STOP runaway auto-decompose (which had
created and launched tasks they didn't intend) found the flag had no effect
without a full gateway restart. ``_resolve_auto_decompose_settings`` is now
called every tick, reading the current config.
"""

from __future__ import annotations


from gateway.kanban_watchers_common import _resolve_auto_decompose_settings
from gateway import kanban_watchers_common as kwc




def test_disabled_when_flag_false():
    enabled, per_tick = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose": False}}
    )
    assert enabled is False


def test_max_age_days_defaults_to_bound(monkeypatch):
    """#124397: no config value still bounds the ambient sweep (fail safe)."""
    from hermes_cli import config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    assert kwc._auto_decompose_max_age_days() == 7


def test_max_age_days_zero_disables(monkeypatch):
    """#124397: 0 opts back into the unbounded sweep."""
    from hermes_cli import config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config",
        lambda: {"kanban": {"auto_decompose_max_age_days": 0}},
    )
    assert kwc._auto_decompose_max_age_days() is None
