import asyncio
import json
from datetime import datetime, timedelta

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner, _restart_loop_threshold, _restart_loop_window_secs
from gateway.session import SessionSource, SessionStore
from gateway.session_context import clear_session_vars, set_session_vars
from tests.gateway.restart_test_helpers import make_restart_runner


def _source(chat_id="123"):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, user_id="u1")


def _store(tmp_path):
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _runner(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.session_store = _store(tmp_path)
    return runner, adapter


def _entry(runner, chat_id="123"):
    return runner.session_store.get_or_create_session(_source(chat_id))


@pytest.mark.asyncio
async def test_restart_consumed_not_in_auto_resume(tmp_path, monkeypatch):
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)

    runner._session_initiated_restart[entry.session_key] = True
    marked, reason, _alert = runner._mark_resume_pending_for_shutdown(entry.session_key)

    assert marked is True
    assert reason == "restart_consumed"
    assert runner.session_store._entries[entry.session_key].resume_reason == "restart_consumed"
    assert "restart_consumed" not in runner._AUTO_RESUME_REASONS
    assert runner._schedule_resume_pending_sessions() == 0


@pytest.mark.asyncio
async def test_cross_boot_relapse_stays_consumed(tmp_path, monkeypatch):
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)

    runner._session_initiated_restart[entry.session_key] = True
    runner._mark_resume_pending_for_shutdown(entry.session_key)
    assert runner._schedule_resume_pending_sessions() == 0

    rebooted, _adapter = _runner(tmp_path, monkeypatch)
    rebooted.session_store._ensure_loaded()
    rebooted._session_initiated_restart[entry.session_key] = True
    rebooted._mark_resume_pending_for_shutdown(entry.session_key)

    assert rebooted.session_store._entries[entry.session_key].resume_reason == "restart_consumed"
    assert rebooted._schedule_resume_pending_sessions() == 0


@pytest.mark.asyncio
async def test_real_interrupted_turn_still_resumes(tmp_path, monkeypatch):
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)

    marked, reason, _alert = runner._mark_resume_pending_for_shutdown(entry.session_key)

    assert marked is True
    assert reason == "shutdown_timeout"
    assert runner._schedule_resume_pending_sessions() == 1
    await asyncio.gather(*runner._background_tasks)
    runner._clear_restart_replay_marks(entry.session_key)
    runner.session_store.clear_resume_pending(entry.session_key)
    assert runner.session_store._entries[entry.session_key].resume_pending is False


@pytest.mark.asyncio
async def test_all_restart_paths_set_flag(tmp_path, monkeypatch):
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)

    async def _stop(**_kwargs):
        return None

    runner.stop = _stop
    tokens = set_session_vars(session_key=entry.session_key)
    try:
        assert runner.request_restart(detached=False, via_service=False) is True
        await asyncio.gather(*runner._background_tasks)
    finally:
        clear_session_vars(tokens)

    assert runner._session_initiated_restart[entry.session_key] is True
    assert runner.session_store._entries[entry.session_key].resume_pending is False


def test_loop_breaker_counts_initiations_not_interruptions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    victim = _entry(runner, "victim")

    for _ in range(3):
        runner._replay_marked_during_stop = set()
        runner._mark_resume_pending_for_shutdown(victim.session_key)

    assert runner.session_store._entries[victim.session_key].suspended is False

    initiator = _entry(runner, "initiator")
    runner._resumed_this_boot.add(initiator.session_key)
    alerts = 0
    for _ in range(3):
        runner._replay_marked_during_stop = set()
        _marked, _reason, alert = runner._mark_resume_pending_for_shutdown(initiator.session_key)
        alerts += int(alert)

    assert runner.session_store._entries[initiator.session_key].suspended is True
    assert alerts == 1


def test_breaker_window_slides_and_resets(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    monkeypatch.setenv("HERMES_RESTART_LOOP_WINDOW_SECS", "10")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)

    assert runner._record_restart_replay_mark(entry.session_key, now=100.0) is False
    assert runner._record_restart_replay_mark(entry.session_key, now=120.0) is False
    assert runner._record_restart_replay_mark(entry.session_key, now=121.0) is False
    assert runner.session_store._entries[entry.session_key].suspended is False
    assert runner._record_restart_replay_mark(entry.session_key, now=122.0) is True
    assert runner._record_restart_replay_mark(entry.session_key, now=123.0) is False

    runner._clear_restart_replay_marks(entry.session_key)
    assert not (tmp_path / runner._STUCK_LOOP_FILE).exists()


@pytest.mark.asyncio
async def test_restart_consumed_e2e_schedule_skips_after_reboot(tmp_path, monkeypatch):
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    runner._session_initiated_restart[entry.session_key] = True
    runner._mark_resume_pending_for_shutdown(entry.session_key)

    rebooted, _adapter = _runner(tmp_path, monkeypatch)
    rebooted.session_store._ensure_loaded()
    assert rebooted.session_store._entries[entry.session_key].resume_reason == "restart_consumed"
    assert rebooted._schedule_resume_pending_sessions() == 0


def test_two_threshold3_counters_independent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)

    runner._increment_restart_failure_counts({entry.session_key})
    runner._increment_restart_failure_counts({entry.session_key})
    for stamp in (1.0, 2.0, 3.0):
        runner._record_restart_replay_mark(entry.session_key, now=stamp)

    runner._clear_restart_replay_marks(entry.session_key)
    data = json.loads((tmp_path / runner._STUCK_LOOP_FILE).read_text())
    assert data[entry.session_key] == 2

    runner._increment_restart_failure_counts({entry.session_key})
    data = json.loads((tmp_path / runner._STUCK_LOOP_FILE).read_text())
    assert data[entry.session_key] == 3


def test_restart_loop_config_defaults_and_clamps(monkeypatch):
    monkeypatch.delenv("HERMES_RESTART_LOOP_THRESHOLD", raising=False)
    monkeypatch.delenv("HERMES_RESTART_LOOP_WINDOW_SECS", raising=False)
    assert _restart_loop_threshold() == 3
    assert _restart_loop_window_secs() == 300.0

    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "0")
    monkeypatch.setenv("HERMES_RESTART_LOOP_WINDOW_SECS", "0")
    assert _restart_loop_threshold() == 1
    assert _restart_loop_window_secs() == 1.0

    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "bad")
    monkeypatch.setenv("HERMES_RESTART_LOOP_WINDOW_SECS", "bad")
    assert _restart_loop_threshold() == 3
    assert _restart_loop_window_secs() == 300.0


@pytest.mark.asyncio
async def test_e2e_restart_interrupted_replay_loop_is_broken(tmp_path, monkeypatch):
    """END-TO-END regression for the cascade the user actually hit.

    This drives the FULL loop on the DOMINANT path — a session marked
    ``restart_interrupted`` (what the safe-restart watcher/skill writes; NOT an
    F1-tagged ``restart_consumed`` session). That reason IS in
    _AUTO_RESUME_REASONS, so each boot re-schedules it; the resumed turn is then
    interrupted by the next restart before it can clear the flag → resume →
    re-interrupt → resume … the exact amplifier that produced the 11-alert
    cascade.

    The invariant under test: F2 (the replay-outcome circuit-breaker) observes
    resume→re-interrupt within the window and SUSPENDS the session at the
    threshold, ending the loop — even though F1 never tagged it. Without F2 the
    loop schedules forever; with it, the session stops being auto-resumed.
    """
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    monkeypatch.setenv("HERMES_RESTART_LOOP_WINDOW_SECS", "300")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner, "loop")
    sk = entry.session_key

    # Boot 0: the watcher marks the session restart_interrupted (dominant path).
    runner.session_store.mark_resume_pending(sk, "restart_interrupted")
    runner.session_store._entries[sk].last_resume_marked_at = datetime.now()
    runner.session_store._save()

    scheduled_each_boot = []
    suspended_at = None
    # Simulate up to 6 reboot cycles; the breaker must stop the loop by cycle 3-4.
    for cycle in range(6):
        booted, _ = _runner(tmp_path, monkeypatch)
        booted.session_store._ensure_loaded()

        n = booted._schedule_resume_pending_sessions()
        scheduled_each_boot.append(n)
        if n:
            # let the synthesized resume task run (it will be 'interrupted' below)
            await asyncio.gather(*booted._background_tasks)

        # The resumed turn gets interrupted by the NEXT restart before it can
        # clear the flag: re-mark during shutdown. _resumed_this_boot was set by
        # the schedule call above, so this counts as a replay (resume→re-interrupt).
        booted._replay_marked_during_stop = set()
        _m, _r, _alert = booted._mark_resume_pending_for_shutdown(sk)

        if booted.session_store._entries[sk].suspended:
            suspended_at = cycle
            break

    # The loop MUST terminate: the session is suspended, and a fresh boot after
    # suspension schedules ZERO (no more auto-resume → cascade broken).
    assert suspended_at is not None, (
        f"replay loop never broke; scheduled per boot = {scheduled_each_boot}"
    )
    final, _ = _runner(tmp_path, monkeypatch)
    final.session_store._ensure_loaded()
    assert final.session_store._entries[sk].suspended is True
    assert final._schedule_resume_pending_sessions() == 0


@pytest.mark.asyncio
async def test_e2e_single_deliberate_restart_resumes_once_no_loop(tmp_path, monkeypatch):
    """Feature-preservation counterpart: ONE deliberate safe-restart (the normal
    case) must auto-resume exactly once and then STOP — not get suspended, not
    loop. Guards that the breaker does not false-trip on healthy single restarts.
    """
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner, "single")
    sk = entry.session_key

    runner.session_store.mark_resume_pending(sk, "restart_interrupted")
    runner.session_store._entries[sk].last_resume_marked_at = datetime.now()
    runner.session_store._save()

    booted, _ = _runner(tmp_path, monkeypatch)
    booted.session_store._ensure_loaded()
    assert booted._schedule_resume_pending_sessions() == 1  # resumes once
    await asyncio.gather(*booted._background_tasks)

    # The resumed turn COMPLETES cleanly: clear the flag (what a finished turn does).
    booted._clear_restart_replay_marks(sk)
    booted.session_store.clear_resume_pending(sk)

    assert booted.session_store._entries[sk].suspended is False
    # Next boot: nothing pending → no resume, no loop.
    after, _ = _runner(tmp_path, monkeypatch)
    after.session_store._ensure_loaded()
    assert after._schedule_resume_pending_sessions() == 0


@pytest.mark.asyncio
async def test_rapid_legit_deploys_with_progress_do_not_suspend(tmp_path, monkeypatch):
    """F2 false-trip guard (Pass-3 required change): three+ rapid LEGITIMATE
    deploys of the SAME session inside the window must NOT suspend it — because
    each resumed turn makes FORWARD PROGRESS (completes a real turn) before the
    next deploy interrupts. The forward-progress gate (_resumed_this_boot.discard
    on a clean turn) is the discriminator vs. a true no-progress loop.

    Structural twin of test_e2e_restart_interrupted_replay_loop_is_broken (which
    suspends): the ONLY difference is that here the resumed turn finishes. That is
    exactly the false-trip boundary the reviewer asked to be made explicit.
    """
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    monkeypatch.setenv("HERMES_RESTART_LOOP_WINDOW_SECS", "300")

    # Two structurally identical sessions, differing ONLY in whether each resumed
    # turn made forward progress (a clean turn). The PROGRESS session must NOT be
    # suspended; the NO-PROGRESS session MUST be — that contrast is the gate.
    runner, _adapter = _runner(tmp_path, monkeypatch)
    progress = _entry(runner, "progress").session_key
    stuck = _entry(runner, "stuck").session_key
    for sk in (progress, stuck):
        runner._resumed_this_boot.add(sk)  # both were auto-resumed this boot

    # 3 deploy-interrupt cycles for each, in the same window.
    for _ in range(3):
        # PROGRESS: the resumed turn COMPLETED. The real clean-turn path (run.py
        # ~9327) both CLEARS replay marks and DISCARDS _resumed_this_boot — together
        # these are the forward-progress gate. Model exactly that path.
        runner._clear_restart_replay_marks(progress)
        runner._resumed_this_boot.discard(progress)
        runner._replay_marked_during_stop = set()
        runner._mark_resume_pending_for_shutdown(progress)
        # The next boot re-resumes it (progress session keeps being legitimately used).
        runner._resumed_this_boot.add(progress)

        # NO-PROGRESS: the resumed turn never completed — no clean-turn clear, stays
        # in _resumed_this_boot, marks accrue. This is the true loop.
        runner._replay_marked_during_stop = set()
        runner._mark_resume_pending_for_shutdown(stuck)

    # The discriminator: progress session healthy, stuck session suspended.
    assert runner.session_store._entries[progress].suspended is False, (
        "healthy rapid deploys (each resume completed) wrongly suspended — false-trip"
    )
    assert runner.session_store._entries[stuck].suspended is True, (
        "a true no-progress replay loop was NOT suspended — breaker failed"
    )


@pytest.mark.asyncio
async def test_stale_marker_after_crash_bounded_by_freshness(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "300")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    runner.session_store.mark_resume_pending(entry.session_key, "shutdown_timeout")
    runner.session_store._entries[entry.session_key].last_resume_marked_at = (
        datetime.now() - timedelta(seconds=301)
    )
    runner.session_store._save()

    rebooted, _adapter = _runner(tmp_path, monkeypatch)
    rebooted.session_store._ensure_loaded()
    assert rebooted._schedule_resume_pending_sessions() == 0

    rebooted.session_store._entries[entry.session_key].last_resume_marked_at = datetime.now()
    rebooted.session_store._save()
    assert rebooted._schedule_resume_pending_sessions() == 1
    await asyncio.gather(*rebooted._background_tasks)
