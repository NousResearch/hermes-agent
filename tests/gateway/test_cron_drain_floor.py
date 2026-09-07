"""Regression tests for #82161.

``restart_drain_timeout`` defaults to ``0``, and the drain applied that single
budget to every class of in-flight work. That default is deliberate for chat
turns — the user is told the gateway is restarting and the session is
pre-marked resume_pending, so interrupting one is cheap and recoverable — but
a cron run has neither property: it is written to jobs.json as a permanent
failure that nobody is waiting on, and a recurring job just skips to its next
schedule.

With the shared budget the drain short-circuited on ``timeout <= 0`` before
the wait loop, producing the reported log line: ``drain took 0.00s,
timed_out=True, cron_at_start=1, cron_now=1`` — it detected the job and killed
it anyway. Cron work now drains on its own floor (``cron_drain_timeout``),
clamped to the shutdown-watchdog leash so the extra wait can never eat the
post-drain cleanup window.
"""

import asyncio

import pytest

from gateway.restart import (
    CRON_DRAIN_CLEANUP_RESERVE_S,
    DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
    parse_cron_drain_timeout,
    resolve_cron_drain_budget,
    resolve_systemd_timeout_stop_sec,
)
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.fixture(autouse=True)
def _reset_cron_running_set():
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()
    yield
    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()


class TestDrainWaitsForCronOnDefaultConfig:
    """The reported repro: default config, cron-only workload."""

    @pytest.mark.asyncio
    async def test_zero_drain_timeout_still_waits_for_cron(self):
        import cron.scheduler as sched

        runner, _adapter = make_restart_runner()
        sched._running_job_ids.add("be62d36a9914")

        async def finish_job():
            await asyncio.sleep(0.12)
            sched._running_job_ids.discard("be62d36a9914")

        task = asyncio.create_task(finish_job())
        # restart_drain_timeout=0 (the shipped default) with a 2s cron floor.
        _snapshot, timed_out = await runner._drain_active_agents(0.0, 2.0)
        await task

        assert timed_out is False, (
            "drain returned timed_out=True with a cron job in flight — this is "
            "the 0.00s drain from #82161"
        )
        assert runner._active_cron_job_count() == 0

    @pytest.mark.asyncio
    async def test_cron_floor_is_bounded_not_indefinite(self):
        """A job that never finishes must still lose, or a cron-triggered
        restart (the reporter's `hermes update` job) would deadlock: the job
        waits for the gateway to exit while the gateway waits for the job."""
        import cron.scheduler as sched

        runner, _adapter = make_restart_runner()
        sched._running_job_ids.add("never-finishes")

        _snapshot, timed_out = await runner._drain_active_agents(0.0, 0.2)

        assert timed_out is True
        assert runner._active_cron_job_count() == 1

    @pytest.mark.asyncio
    async def test_chat_only_workload_keeps_the_zero_second_drain(self):
        """The cron floor must not silently become a chat-turn grace window —
        `restart_drain_timeout: 0` still means "interrupt chat immediately"."""
        runner, _adapter = make_restart_runner()
        runner._running_agents = {"sess-1": object()}

        loop = asyncio.get_running_loop()
        before = loop.time()
        _snapshot, timed_out = await runner._drain_active_agents(0.0, 30.0)
        elapsed = loop.time() - before

        assert timed_out is True
        assert elapsed < 1.0, f"chat-only drain waited {elapsed:.2f}s on a 0s budget"

    @pytest.mark.asyncio
    async def test_cron_timeout_defaults_to_the_shared_budget(self):
        """Callers that pass one argument keep the pre-#82161 semantics."""
        import cron.scheduler as sched

        runner, _adapter = make_restart_runner()
        sched._running_job_ids.add("job-1")

        _snapshot, timed_out = await runner._drain_active_agents(0.0)

        assert timed_out is True


class TestParseCronDrainTimeout:
    def test_missing_and_blank_fall_back_to_default(self):
        assert parse_cron_drain_timeout(None) == DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
        assert parse_cron_drain_timeout("") == DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
        assert parse_cron_drain_timeout("   ") == DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT

    def test_zero_is_a_deliberate_opt_out_not_a_missing_value(self):
        assert parse_cron_drain_timeout(0) == 0.0
        assert parse_cron_drain_timeout("0") == 0.0

    def test_garbage_falls_back_and_negatives_clamp(self):
        assert parse_cron_drain_timeout("soon") == DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
        assert parse_cron_drain_timeout(-5) == 0.0


class TestResolveCronDrainBudget:
    def test_extends_a_zero_drain_up_to_the_configured_floor(self):
        assert resolve_cron_drain_budget(
            0.0, 30.0, watchdog_delay=60.0, elapsed=1.0
        ) == 30.0

    def test_clamped_to_the_watchdog_leash_minus_cleanup_reserve(self):
        # Watchdog hard-exits at 60s; waiting 300s would guarantee a SIGKILL
        # mid-cleanup, leaving the job wedged at last_status=running.
        budget = resolve_cron_drain_budget(
            0.0, 300.0, watchdog_delay=60.0, elapsed=5.0
        )
        assert budget == pytest.approx(60.0 - 5.0 - CRON_DRAIN_CLEANUP_RESERVE_S)

    def test_never_shortens_an_explicitly_configured_drain_timeout(self):
        assert resolve_cron_drain_budget(
            120.0, 30.0, watchdog_delay=180.0, elapsed=0.0
        ) == 120.0

    def test_no_headroom_left_falls_back_to_the_drain_timeout(self):
        assert resolve_cron_drain_budget(
            0.0, 30.0, watchdog_delay=60.0, elapsed=59.0
        ) == 0.0

    def test_zero_floor_opts_out_entirely(self):
        assert resolve_cron_drain_budget(
            0.0, 0.0, watchdog_delay=60.0, elapsed=0.0
        ) == 0.0

    def test_non_numeric_inputs_degrade_instead_of_raising(self):
        assert resolve_cron_drain_budget(
            None, "30", watchdog_delay=60.0, elapsed=None
        ) == 30.0


class TestResolveSystemdTimeoutStopSec:
    """#94759: TimeoutStopSec must cover cron drain, not just chat drain."""

    def test_default_cron_floor_beats_the_old_drain_only_formula(self):
        # Old unit: max(60, 0+30) = 60. Stop path may wait 30+10=40s, then
        # still needs the 30s teardown headroom — 70s, not 60s.
        timeout = resolve_systemd_timeout_stop_sec(
            0.0, DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
        )
        assert timeout == int(
            max(
                60,
                DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT + CRON_DRAIN_CLEANUP_RESERVE_S + 30,
            )
        )
        assert timeout > 60

    def test_configured_drain_still_extends_the_deadline_directly(self):
        assert resolve_systemd_timeout_stop_sec(60.0, 30.0) == 90
        assert resolve_systemd_timeout_stop_sec(180.0, 30.0) == 210

    def test_larger_cron_floor_raises_timeout_stop_sec(self):
        # 60s cron + 10s reserve + 30s headroom = 100s
        assert resolve_systemd_timeout_stop_sec(0.0, 60.0) == 100

    def test_zero_cron_floor_is_an_opt_out_not_a_hidden_default(self):
        assert resolve_systemd_timeout_stop_sec(0.0, 0.0) == 60

    def test_cron_floor_never_shortens_a_long_drain(self):
        assert resolve_systemd_timeout_stop_sec(180.0, 30.0) == resolve_systemd_timeout_stop_sec(
            180.0, 0.0
        )

    def test_garbage_inputs_degrade_to_the_floor(self):
        assert resolve_systemd_timeout_stop_sec("soon", None) == 60
