"""The gateway re-sweeps idle ``cache/scratch`` entries; boot-time-only pruning is not enough.

``get_scratch_dir(prune=True)`` sweeps at most once per process, and across processes at most
hourly via a stamp file. Both triggers need process churn: on a host whose only long-lived
Hermes process is the gateway, nothing re-sweeps after boot, and entries that go idle later
outlive the retention window. The sweep rides the housekeeping tick instead — hourly, the same
cadence (and the same ``SCRATCH_MAX_IDLE_HOURS`` rule) the platform media caches use.
"""

import gateway.run as gateway_run


class _TicksStopEvent:
    """Stop the housekeeping loop after *ticks* iterations (``_OneTickStopEvent``, widened)."""

    def __init__(self, ticks):
        self.ticks = ticks
        self.waited = 0

    def is_set(self):
        return self.waited >= self.ticks

    def wait(self, timeout=None):
        self.waited += 1
        return True


# Every other chore is stubbed out: this file is about the scratch sweep's cadence, and the
# remaining chores own real state (state.db, caches, plugin update check).
_OTHER_CHORES = (
    "_write_runtime_status_quiet",
    "_housekeeping_channel_directory",
    "_housekeeping_media_caches",
    "_housekeeping_paste_sweep",
    "_housekeeping_curator",
    "_housekeeping_state_db_maintenance",
    "_housekeeping_plugin_update_check",
    "_housekeeping_deferred_fts_retry",
    "_housekeeping_memory_trim",
    "_housekeeping_checkpoint_prune",
)


def _sweeps_over(monkeypatch, ticks):
    """Run *ticks* housekeeping iterations; return how many times the scratch sweep ran."""
    calls = []
    monkeypatch.setattr("hermes_constants.prune_scratch_dir", lambda *a, **k: calls.append(1) or 0)
    for name in _OTHER_CHORES:
        monkeypatch.setattr(gateway_run, name, lambda *a, **k: None)
    monkeypatch.setattr(
        "gateway.run_profile_reconcile.profile_scoped_chore", lambda runner, fn: (lambda *a, **k: None))
    monkeypatch.setattr(
        "gateway.run_profile_reconcile._mcp_config_reconciler", lambda runner: (lambda: None))

    gateway_run._start_gateway_housekeeping(_TicksStopEvent(ticks), interval=0)
    return len(calls)


def test_scratch_sweep_rides_the_hourly_tick(monkeypatch):
    """Not every tick (59 → 0), but on the 60th (1): the sweep is a cheap no-op when idle."""
    assert _sweeps_over(monkeypatch, 59) == 0
    assert _sweeps_over(monkeypatch, 60) == 1


def test_scratch_sweep_delegates_to_the_retention_helper(monkeypatch):
    """The chore is a thin caller: the retention rule stays in ``hermes_constants``."""
    seen = []
    monkeypatch.setattr("hermes_constants.prune_scratch_dir", lambda *a, **k: seen.append((a, k)) or 3)

    gateway_run._housekeeping_scratch_prune()

    assert seen == [((), {})]
