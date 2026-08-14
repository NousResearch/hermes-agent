"""Idle cron ticks must not load config (#33612 salvage).

The gateway's built-in ticker calls tick(verbose=False) every 60s. Before
the fix, idle ticks (no due jobs) fell through the verbose-only early
return and paid a full load_config() + worker-pool resolution per tick.
The fix returns early on ANY idle tick while preserving the post-tick MCP
orphan sweep that main intentionally runs even when nothing is due.
"""

from __future__ import annotations

from unittest.mock import patch

import cron.scheduler as scheduler_mod


def _run_idle_tick(**kwargs):
    """Run tick() with no due jobs; return (load_config_called, sweep_called)."""
    calls = {"load_config": 0, "sweep": 0}

    def _fake_load_config(*a, **k):
        calls["load_config"] += 1
        return {}

    def _fake_sweep():
        calls["sweep"] += 1

    with (
        patch.object(scheduler_mod, "get_due_jobs", return_value=[]),
        patch.object(scheduler_mod, "load_config", side_effect=_fake_load_config),
        patch(
            "tools.mcp_tool._kill_orphaned_mcp_children",
            side_effect=_fake_sweep,
        ),
    ):
        rc = scheduler_mod.tick(verbose=kwargs.get("verbose", False))
    return rc, calls


class TestIdleTickSkipsConfigLoad:
    def test_idle_nonverbose_tick_skips_load_config(self):
        """Gateway-style tick(verbose=False) with no due jobs: no config load."""
        rc, calls = _run_idle_tick(verbose=False)
        assert rc == 0
        assert calls["load_config"] == 0, (
            "idle tick must not load config (was loading every 60s in the gateway ticker)"
        )

    def test_idle_verbose_tick_skips_load_config(self):
        rc, calls = _run_idle_tick(verbose=True)
        assert rc == 0
        assert calls["load_config"] == 0

    def test_idle_tick_still_sweeps_mcp_orphans(self):
        """The idle-tick orphan sweep is intentional on main — must survive."""
        rc, calls = _run_idle_tick(verbose=False)
        assert rc == 0
        assert calls["sweep"] == 1, "idle tick must still reap orphaned MCP children"
