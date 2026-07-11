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


# ───────────── F2 self-completing-loop guard (A′ record-mark-at-gate + C1 skill detection) ─────────────


def test_self_completing_restart_loop_is_suspended(tmp_path, monkeypatch):
    """The bug the parent fix's WONTFIX missed: a turn that completes cleanly AND
    initiated a restart, every cycle, must be SUSPENDED — it's a loop, not progress.
    Drives the real post-turn gate (_apply_post_turn_resume_gate) per cycle."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    monkeypatch.setenv("HERMES_RESTART_LOOP_WINDOW_SECS", "300")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "selfloop").session_key

    for _ in range(3):
        # each cycle: auto-resumed, the resumed turn fires a restart, then "completes"
        runner._resumed_this_boot.add(sk)
        runner._session_initiated_restart[sk] = True
        runner._apply_post_turn_resume_gate(sk)

    assert runner.session_store._entries[sk].suspended is True


def test_single_self_restart_then_real_work_not_suspended(tmp_path, monkeypatch):
    """INV-2: one self-restart, then the next resumed turn does real work (no
    restart) → the mark clears → never reaches threshold → not suspended."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "onerestart").session_key

    # cycle 1: a self-restart turn → records one mark
    runner._resumed_this_boot.add(sk)
    runner._session_initiated_restart[sk] = True
    runner._apply_post_turn_resume_gate(sk)
    # cycle 2: real work, no restart → clears
    runner._resumed_this_boot.add(sk)
    runner._apply_post_turn_resume_gate(sk)
    # cycle 3: real work again
    runner._apply_post_turn_resume_gate(sk)

    assert runner.session_store._entries[sk].suspended is False


def test_real_work_turns_never_accrue_marks(tmp_path, monkeypatch):
    """INV-3 (parent anti-false-trip preserved): a session that only ever does
    real work (never initiates a restart) is never suspended, no matter how many
    clean turns — the gate clears every time."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "realwork").session_key
    for _ in range(6):
        runner._resumed_this_boot.add(sk)
        runner._apply_post_turn_resume_gate(sk)
    assert runner.session_store._entries[sk].suspended is False
    # and no replay marks accrued
    counts = runner._load_restart_failure_counts().get(sk, {})
    assert counts.get("replay_marks", []) == []


def test_initiated_flag_is_one_shot_per_turn(tmp_path, monkeypatch):
    """The initiated flag must be popped per turn — a single restart turn followed
    by a real-work turn must NOT keep marking (else a one-shot restart would loop
    the breaker forever)."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "oneshot").session_key

    runner._session_initiated_restart[sk] = True
    runner._apply_post_turn_resume_gate(sk)          # consumes the flag (1 mark)
    assert sk not in runner._session_initiated_restart
    runner._apply_post_turn_resume_gate(sk)          # no flag → clears
    runner._apply_post_turn_resume_gate(sk)
    assert runner.session_store._entries[sk].suspended is False


def test_skill_path_progress_callback_sets_initiated_flag(tmp_path, monkeypatch):
    """C1: the gateway must set _session_initiated_restart when it observes a
    `terminal` tool start INVOKING safe-restart.py — the dominant skill path that
    F1 (request_restart-only) is blind to — and must NOT trip on a turn that merely
    READS the script. Drives the REAL matcher (_command_invokes_safe_restart)."""
    from gateway.run import _command_invokes_safe_restart

    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "skillpath").session_key
    cb = _make_progress_callback(runner, sk)

    # a normal terminal command must NOT trip it
    cb("tool.started", tool_name="terminal", args={"command": "ls -la /tmp"})
    assert sk not in runner._session_initiated_restart

    # INSPECTION of the script must NOT trip it (the tightened false-positive guard)
    for inspect_cmd in (
        "cat ~/.hermes/skills-shared/general/safe-gateway-restart/scripts/safe-restart.py",
        "grep handoff safe-restart.py",
        "vim scripts/safe-restart.py",
    ):
        assert _command_invokes_safe_restart(inspect_cmd) is False, inspect_cmd

    # the safe-restart skill INVOCATION MUST trip it
    invoke = "python3 ~/.hermes/skills-shared/general/safe-gateway-restart/scripts/safe-restart.py --handoff x"
    assert _command_invokes_safe_restart(invoke) is True
    cb("tool.started", tool_name="terminal", args={"command": invoke})
    assert runner._session_initiated_restart.get(sk) is True

    # and feeding the gate now records a mark (skill path → loop detectable)
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner._apply_post_turn_resume_gate(sk)
    counts = runner._load_restart_failure_counts().get(sk, {})
    assert len(counts.get("replay_marks", [])) == 1


def test_command_invokes_safe_restart_matcher(tmp_path, monkeypatch):
    """Direct unit coverage of the REAL C1 matcher — execution vs inspection vs
    pipeline. Drives gateway.run._command_invokes_safe_restart, not a mirror."""
    from gateway.run import _command_invokes_safe_restart as m

    # executions → True
    assert m("python3 a/b/safe-restart.py --handoff 'x'") is True
    assert m("/usr/bin/python safe-restart.py") is True
    assert m("cd /tmp && python3 safe-restart.py --full-reload") is True  # &&-chain execution
    # inspections → False
    assert m("cat safe-restart.py") is False
    assert m("grep -n handoff scripts/safe-restart.py") is False
    assert m("less safe-restart.py") is False
    assert m("echo safe-restart.py") is False
    # unrelated / absent → False
    assert m("ls -la /tmp") is False
    assert m("") is False
    # pipeline where the script is executed in a segment → True
    assert m("echo hi | python3 safe-restart.py") is True


def test_c1_detection_present_in_real_progress_callback():
    """Guard the REAL code, not just the mirror: the progress_callback in
    gateway/run.py must contain the safe-restart.py → _session_initiated_restart
    detection. The live callback is a closure inside _run_agent (not unit-callable),
    so the behavioral test above uses a faithful mirror; this asserts the real
    branch exists so deleting it fails CI. RED: remove the C1 block → this fails."""
    import inspect
    import gateway.run as gr

    src = inspect.getsource(gr)
    # C1 is uniquely identified by the safe-restart matcher guarding the
    # initiated-restart flag in the tool-progress callback.
    assert "_command_invokes_safe_restart" in src, "C1 matcher missing from gateway/run.py"
    assert "if _command_invokes_safe_restart(_cmd):" in src, (
        "C1 conditional missing — the skill-path detection branch was removed"
    )


def _make_progress_callback(runner, session_key):
    """Mirror of the C1 callback branch in _run_agent (the real callback is a
    closure, not unit-callable). Delegates the match decision to the REAL
    _command_invokes_safe_restart so the matching logic can't drift from prod."""
    from gateway.run import _command_invokes_safe_restart

    def cb(event_type, tool_name=None, args=None, **kwargs):
        if event_type != "tool.started":
            return
        if session_key and tool_name == "terminal":
            cmd = ""
            if isinstance(args, dict):
                cmd = str(args.get("command") or args.get("cmd") or args.get("script") or "")
            if not cmd:
                cmd = str(args or "")
            if _command_invokes_safe_restart(cmd):
                runner._session_initiated_restart[session_key] = True
    return cb


# ───────── F2 initiator-detection AUTHORITATIVE BREADCRUMB (spec 2026-06-22) ─────────
#
# The breadcrumb is the authoritative restart-initiator signal: safe-restart.py
# drops a per-session, per-boot FILE that the clean-turn gate consumes. These
# tests drive the REAL gate + real on-disk files (tmp_path == _hermes_home via
# the _runner monkeypatch). They cover I-1..I-9 + D-6/D-8.

import os as _os
import time as _time

from gateway.run import (
    _restart_initiated_filename,
    _restart_initiated_ttl_secs,
)


def _write_breadcrumb(runner, session_key, *, boot_id=None, ts=None, key_override=None):
    """Write a real breadcrumb file exactly as safe-restart.py would."""
    d = runner._restart_initiated_dir()
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = d / _restart_initiated_filename(session_key)
    payload = {
        "session_key": key_override if key_override is not None else session_key,
        "ts": _time.time() if ts is None else ts,
        "boot_id": runner._current_boot_id() if boot_id is None else boot_id,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_breadcrumb_marks_when_current_boot(tmp_path, monkeypatch):
    """A fresh current-boot breadcrumb makes the gate treat the turn as a
    restart-initiator → records a replay-mark (the authoritative signal works
    with NO C1/F1 flag set)."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "crumb").session_key
    runner._resumed_this_boot.add(sk)
    _write_breadcrumb(runner, sk)
    runner._apply_post_turn_resume_gate(sk)
    counts = runner._load_restart_failure_counts().get(sk, {})
    assert len(counts.get("replay_marks", [])) == 1


def test_breadcrumb_from_prior_boot_discarded_not_marked(tmp_path, monkeypatch):
    """I-4 (the false-trip kill): a breadcrumb with a DIFFERENT boot_id (an
    interrupted initiator's crumb that survived a reboot) must NOT mark the next
    real-work resumed turn, even with a fresh ts. RED without the boot_id check."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "priorboot").session_key
    runner._resumed_this_boot.add(sk)
    # fresh ts, but a prior boot's id
    _write_breadcrumb(runner, sk, boot_id="99999:1.0", ts=_time.time())
    runner._apply_post_turn_resume_gate(sk)
    # no mark recorded, and forward-progress clear happened
    counts = runner._load_restart_failure_counts().get(sk, {})
    assert counts.get("replay_marks", []) == []
    assert sk not in runner._resumed_this_boot


def test_stale_breadcrumb_discarded_not_marked(tmp_path, monkeypatch):
    """I-5: a current-boot breadcrumb older than the TTL backstop is discarded
    unmarked."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "stale").session_key
    old_ts = _time.time() - (_restart_initiated_ttl_secs() + 60)
    _write_breadcrumb(runner, sk, ts=old_ts)
    runner._apply_post_turn_resume_gate(sk)
    counts = runner._load_restart_failure_counts().get(sk, {})
    assert counts.get("replay_marks", []) == []


def test_breadcrumb_consumed_after_gate(tmp_path, monkeypatch):
    """I-3: the breadcrumb file is unlinked on consume — a second gate call sees
    nothing (no double-mark, no lingering crumb)."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "5")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "consumed").session_key
    path = _write_breadcrumb(runner, sk)
    runner._resumed_this_boot.add(sk)
    runner._apply_post_turn_resume_gate(sk)
    assert not path.exists()
    # second gate: no breadcrumb → clear branch, marks unchanged at 1
    runner._apply_post_turn_resume_gate(sk)
    counts = runner._load_restart_failure_counts().get(sk, {})
    assert len(counts.get("replay_marks", [])) <= 1


def test_breadcrumb_and_c1_flag_same_turn_records_one_mark(tmp_path, monkeypatch):
    """I-1: with BOTH the C1/F1 flag and a breadcrumb present for one turn, the
    gate records exactly ONE mark (not two → would trip the breaker early)."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "both").session_key
    runner._resumed_this_boot.add(sk)
    runner._session_initiated_restart[sk] = True
    _write_breadcrumb(runner, sk)
    runner._apply_post_turn_resume_gate(sk)
    counts = runner._load_restart_failure_counts().get(sk, {})
    assert len(counts.get("replay_marks", [])) == 1


def test_c1_flag_true_still_consumes_breadcrumb(tmp_path, monkeypatch):
    """Pass-2 B-3: a C1/F1-flagged turn must STILL unlink the breadcrumb (no
    `flag or consume()` short-circuit), else it leaks to a later turn."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "5")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "noshort").session_key
    runner._session_initiated_restart[sk] = True
    path = _write_breadcrumb(runner, sk)
    runner._apply_post_turn_resume_gate(sk)
    assert not path.exists(), "breadcrumb leaked: short-circuit skipped the unlink"


def test_real_work_turn_consumes_no_breadcrumb(tmp_path, monkeypatch):
    """I-2: a real-work turn (no flag, no breadcrumb) clears the breaker and
    accrues no mark — anti-false-trip preserved."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "realwork2").session_key
    for _ in range(6):
        runner._resumed_this_boot.add(sk)
        runner._apply_post_turn_resume_gate(sk)
    assert runner.session_store._entries[sk].suspended is False
    counts = runner._load_restart_failure_counts().get(sk, {})
    assert counts.get("replay_marks", []) == []


def test_malformed_breadcrumb_file_is_ignored(tmp_path, monkeypatch):
    """I-6: a garbage breadcrumb file → no mark, no exception, file consumed."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "garbage").session_key
    d = runner._restart_initiated_dir()
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = d / _restart_initiated_filename(sk)
    path.write_text("}{ not json", encoding="utf-8")
    assert runner._consume_restart_initiated_breadcrumb(sk) is False
    assert not path.exists()


def test_breadcrumb_key_filename_mismatch_ignored(tmp_path, monkeypatch):
    """I-8: a breadcrumb whose stored key doesn't hash to its filename (forgery)
    is ignored."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "forge").session_key
    # write a crumb at sk's filename but with a DIFFERENT key inside
    _write_breadcrumb(runner, sk, key_override="agent:main:telegram:evil:1")
    assert runner._consume_restart_initiated_breadcrumb(sk) is False


def test_startup_sweep_prunes_wrong_boot_and_stale(tmp_path, monkeypatch):
    """D-8: startup sweep removes prior-boot + stale crumbs."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    a = _entry(runner, "swA").session_key
    b = _entry(runner, "swB").session_key
    _write_breadcrumb(runner, a, boot_id="11111:1.0")  # wrong boot
    _write_breadcrumb(runner, b, ts=_time.time() - (_restart_initiated_ttl_secs() + 99))  # stale
    removed = runner._sweep_restart_initiated_breadcrumbs()
    assert removed == 2
    assert not (runner._restart_initiated_dir() / _restart_initiated_filename(a)).exists()


def test_current_boot_breadcrumb_survives_sweep(tmp_path, monkeypatch):
    """Pass-2 RC-4: a fresh current-boot crumb (an in-flight initiator) must
    survive the startup sweep."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "survive").session_key
    path = _write_breadcrumb(runner, sk)  # current boot, fresh
    removed = runner._sweep_restart_initiated_breadcrumbs()
    assert removed == 0
    assert path.exists()


def test_two_sessions_independent_at_the_gate(tmp_path, monkeypatch):
    """Pass-2 RC-5: two sessions with independent breadcrumb files — one marks,
    the other (no crumb, real work) clears. Proves per-session isolation E2E."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    s_restart = _entry(runner, "sA").session_key
    s_work = _entry(runner, "sB").session_key
    runner._resumed_this_boot.add(s_restart)
    runner._resumed_this_boot.add(s_work)
    _write_breadcrumb(runner, s_restart)  # only A initiated a restart
    runner._apply_post_turn_resume_gate(s_restart)
    runner._apply_post_turn_resume_gate(s_work)
    a_marks = runner._load_restart_failure_counts().get(s_restart, {}).get("replay_marks", [])
    b_marks = runner._load_restart_failure_counts().get(s_work, {}).get("replay_marks", [])
    assert len(a_marks) == 1
    assert b_marks == []
    assert s_work not in runner._resumed_this_boot


def test_self_completing_loop_via_breadcrumb_is_suspended(tmp_path, monkeypatch):
    """End-to-end: a session that drops a current-boot breadcrumb every cycle
    (the alias/wrapper case C1 would miss) is SUSPENDED at threshold — the whole
    point of the authoritative signal."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    monkeypatch.setenv("HERMES_RESTART_LOOP_WINDOW_SECS", "300")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "selfloopcrumb").session_key
    for _ in range(3):
        runner._resumed_this_boot.add(sk)
        _write_breadcrumb(runner, sk)  # NO C1 flag — breadcrumb only
        runner._apply_post_turn_resume_gate(sk)
    assert runner.session_store._entries[sk].suspended is True


def test_boot_id_present_and_not_pid_only_on_this_host(tmp_path, monkeypatch):
    """I-9 regression guard: the gateway's boot_id must carry a non-empty
    create_time component — catches a return to the macOS `/proc`-None collapse
    (pid-only boot_id)."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    bid = runner._current_boot_id()
    assert ":" in bid
    pid_part, _, ct_part = bid.partition(":")
    assert pid_part.isdigit()
    assert ct_part != "", "boot_id collapsed to pid-only (create_time missing)"


def test_single_gate_call_site():
    """D-6: _apply_post_turn_resume_gate has exactly one call site (the clean-turn
    gate). A second site would need its own breadcrumb-consume reasoning."""
    import inspect
    import gateway.run as gr

    src = inspect.getsource(gr)
    n = src.count("self._apply_post_turn_resume_gate(session_key)")
    assert n == 1, f"expected 1 gate call site, found {n}"


def test_finally_consume_present_in_handler():
    """D-6 defense: the handler's finally block must consume the breadcrumb so a
    gate-skip (exception/early-return) can't leak it within-boot."""
    import inspect
    import gateway.run as gr

    src = inspect.getsource(gr)
    assert "_consume_restart_initiated_breadcrumb(_sk_cleanup)" in src, (
        "finally-block defensive consume missing"
    )


def _seed_gateway_state_via_real_producer(tmp_path, monkeypatch, *, stale_first=False):
    """Write gateway_state.json via the REAL producer (write_runtime_status),
    NOT by hand — so the test exercises the actual code path the script depends
    on (BLOCKER-2: hand-seeding masks the stale-boot_id bug). HERMES_HOME is
    pointed at tmp_path so the status file lands where the script reads it.

    When stale_first=True, first drop a prior-boot status file on disk (with a
    DIFFERENT boot_id) to simulate the file surviving a reboot, then call the
    real producer — proving the producer REFRESHES boot_id rather than leaving
    the stale value.
    """
    import gateway.status as status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # status.py resolves the status path from get_hermes_home(); make sure it
    # points at tmp_path for this call.
    monkeypatch.setattr(status, "get_hermes_home", lambda: tmp_path, raising=False)
    state_path = tmp_path / "gateway_state.json"
    if stale_first:
        state_path.write_text(json.dumps({
            "pid": 999999, "boot_id": "999999:1.0", "gateway_state": "running",
        }), encoding="utf-8")
    status.write_runtime_status(gateway_state="running")
    return state_path


def test_real_producer_refreshes_boot_id_across_restart(tmp_path, monkeypatch):
    """BLOCKER-1/2 regression: write_runtime_status MUST refresh boot_id on every
    write. gateway_state.json survives reboots, so a stale prior-boot file must
    not leave its old boot_id behind (which would make every breadcrumb mismatch
    → feature inert after restart #1). RED against the pre-fix producer."""
    import gateway.status as status

    state_path = _seed_gateway_state_via_real_producer(
        tmp_path, monkeypatch, stale_first=True
    )
    persisted = json.loads(state_path.read_text())["boot_id"]
    # The producer must have overwritten the stale "999999:1.0" with THIS boot.
    assert persisted == status.get_current_boot_id()
    assert persisted != "999999:1.0"
    # and it carries a real create_time component (not pid-only)
    assert not persisted.endswith(":")


def test_roundtrip_real_script_writes_breadcrumb_gate_marks(tmp_path, monkeypatch):
    """E2E no-mock seam (the load-bearing alias/wrapper-robustness proof): run the
    REAL safe-restart.py as a subprocess — the gateway never sees its command
    line — so the ONLY signal is the breadcrumb file it drops. gateway_state.json
    is produced by the REAL write_runtime_status (not hand-seeded), so the
    script reads the boot_id the real producer persists. Then the real gateway
    gate consumes it and records a mark."""
    import subprocess
    import sys

    script = (
        "/Users/alexgierczyk/.hermes/skills-shared/general/"
        "safe-gateway-restart/scripts/safe-restart.py"
    )
    if not _os.path.exists(script):
        pytest.skip("safe-restart.py skill script not present")

    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "rt").session_key
    # REAL producer writes gateway_state.json (incl. boot_id) at tmp_path —
    # stale_first=True simulates the file surviving a reboot (the NORMAL steady
    # state), so this test exercises the exact stale-boot_id path BLOCKER-1 fixed.
    _seed_gateway_state_via_real_producer(tmp_path, monkeypatch, stale_first=True)

    env = dict(_os.environ)
    env["HERMES_HOME"] = str(tmp_path)
    r = subprocess.run(
        [sys.executable, script, "--no-spawn", "--write-breadcrumb",
         "--session-key", sk, "--session-id", "sidrt", "--chat", "123",
         "--platform", "telegram", "--handoff", "roundtrip"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["breadcrumb_written"] is True

    # The gateway gate now consumes it — boot_id read from the REAL producer's
    # file must match the gate's _current_boot_id() (this is the seam that was
    # broken before BLOCKER-1's fix).
    runner._resumed_this_boot.add(sk)
    runner._apply_post_turn_resume_gate(sk)
    counts = runner._load_restart_failure_counts().get(sk, {})
    assert len(counts.get("replay_marks", [])) == 1


def test_roundtrip_script_boot_id_matches_real_producer(tmp_path, monkeypatch):
    """Pass-2 B-1/I-9 via the REAL producer: the breadcrumb the script writes
    carries exactly the boot_id write_runtime_status persisted — single producer,
    verbatim copy, no second parser. (Replaces the prior hand-seeded version that
    masked BLOCKER-1.)"""
    import subprocess
    import sys

    script = (
        "/Users/alexgierczyk/.hermes/skills-shared/general/"
        "safe-gateway-restart/scripts/safe-restart.py"
    )
    if not _os.path.exists(script):
        pytest.skip("safe-restart.py skill script not present")

    import gateway.status as status
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "rtboot").session_key
    _seed_gateway_state_via_real_producer(tmp_path, monkeypatch, stale_first=True)
    expected_boot = status.get_current_boot_id()

    env = dict(_os.environ)
    env["HERMES_HOME"] = str(tmp_path)
    subprocess.run(
        [sys.executable, script, "--no-spawn", "--write-breadcrumb",
         "--session-key", sk, "--session-id", "s", "--chat", "1",
         "--platform", "telegram", "--handoff", "x"],
        capture_output=True, text=True, env=env, check=True,
    )
    crumb = json.loads(
        (tmp_path / ".restart_initiated" / _restart_initiated_filename(sk)).read_text()
    )
    assert crumb["boot_id"] == expected_boot  # verbatim copy of the producer's id
    assert crumb["boot_id"] != "999999:1.0"   # not the stale prior-boot value


def test_roundtrip_script_dead_pid_writes_no_breadcrumb(tmp_path, monkeypatch):
    """D-4a liveness: if gateway_state.json's pid is dead (stale file), the script
    writes NO breadcrumb (fail-open to C1/F1)."""
    import subprocess
    import sys

    script = (
        "/Users/alexgierczyk/.hermes/skills-shared/general/"
        "safe-gateway-restart/scripts/safe-restart.py"
    )
    if not _os.path.exists(script):
        pytest.skip("safe-restart.py skill script not present")

    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "rtdead").session_key
    # a pid that is almost certainly dead
    state = {"pid": 2, "boot_id": "2:1.0", "gateway_state": "running"}
    (tmp_path / "gateway_state.json").write_text(json.dumps(state), encoding="utf-8")

    env = dict(_os.environ)
    env["HERMES_HOME"] = str(tmp_path)
    r = subprocess.run(
        [sys.executable, script, "--no-spawn", "--write-breadcrumb",
         "--session-key", sk, "--session-id", "s", "--chat", "1",
         "--platform", "telegram", "--handoff", "x"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0
    assert json.loads(r.stdout)["breadcrumb_written"] is False
    assert not (tmp_path / ".restart_initiated" / _restart_initiated_filename(sk)).exists()


def test_no_spawn_without_write_flag_plants_no_breadcrumb(tmp_path, monkeypatch):
    """MINOR-2: a diagnostic --no-spawn (without --write-breadcrumb) must NOT
    drop a real breadcrumb — else a diagnostic run under a live gateway plants a
    crumb the session's next turn consumes as a restart-initiator mark."""
    import subprocess
    import sys

    script = (
        "/Users/alexgierczyk/.hermes/skills-shared/general/"
        "safe-gateway-restart/scripts/safe-restart.py"
    )
    if not _os.path.exists(script):
        pytest.skip("safe-restart.py skill script not present")

    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "nodiag").session_key
    _seed_gateway_state_via_real_producer(tmp_path, monkeypatch)

    env = dict(_os.environ)
    env["HERMES_HOME"] = str(tmp_path)
    r = subprocess.run(
        [sys.executable, script, "--no-spawn",  # NO --write-breadcrumb
         "--session-key", sk, "--session-id", "s", "--chat", "1",
         "--platform", "telegram", "--handoff", "x"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0
    assert json.loads(r.stdout)["breadcrumb_written"] is False
    assert not (tmp_path / ".restart_initiated" / _restart_initiated_filename(sk)).exists()


def test_cross_process_breadcrumb_contract_ci_safe(tmp_path, monkeypatch):
    """CI-GUARANTEED cross-process contract (Required Change #1): the round-trip
    subprocess tests skip when the live skill script isn't on-path (CI), which
    would make the anti-masking proof illusory off-host. This test reproduces the
    SCRIPT's breadcrumb-write contract INLINE (the exact JSON shape + filename
    hash + boot_id read from gateway_state.json) and proves the real gateway gate
    consumes it — so the cross-process file contract is verified on EVERY host,
    with no dependency on the skill script's filesystem location.

    If the real script's write shape ever drifts from this inline replica, the
    subprocess round-trip tests (when present) catch it; this guarantees the gate
    side of the contract is always exercised."""
    import gateway.status as status

    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "ci").session_key
    # real producer persists boot_id (stale_first → exercises the refresh path)
    _seed_gateway_state_via_real_producer(tmp_path, monkeypatch, stale_first=True)

    # Replicate the SCRIPT's write: read boot_id from gateway_state.json (the
    # field the real producer just wrote), write the per-session file with the
    # same {session_key, ts, boot_id} shape and sha8 filename the gate expects.
    gw_state = json.loads((tmp_path / "gateway_state.json").read_text())
    boot_id = gw_state["boot_id"]
    assert boot_id == status.get_current_boot_id()  # producer/consumer agree
    d = runner._restart_initiated_dir()
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    (d / _restart_initiated_filename(sk)).write_text(
        json.dumps({"session_key": sk, "ts": _time.time(), "boot_id": boot_id}),
        encoding="utf-8",
    )

    runner._resumed_this_boot.add(sk)
    runner._apply_post_turn_resume_gate(sk)
    counts = runner._load_restart_failure_counts().get(sk, {})
    assert len(counts.get("replay_marks", [])) == 1


def test_degraded_current_boot_id_rejects_breadcrumb(tmp_path, monkeypatch):
    """MAJOR-1: if the gateway's _current_boot_id() is degraded (pid-only, no
    create_time — psutil failure), the consume side must REJECT every breadcrumb
    (can't prove same-boot → fall back to C1/F1) rather than honor a pid-only
    match (pid reuse across reboots)."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "degraded").session_key
    # force a degraded current boot id
    monkeypatch.setattr(runner, "_current_boot_id", lambda: f"{_os.getpid()}:")
    # write a crumb whose stored boot_id is ALSO pid-only for the same pid
    _write_breadcrumb(runner, sk, boot_id=f"{_os.getpid()}:")
    assert runner._consume_restart_initiated_breadcrumb(sk) is False


def test_sweep_degraded_boot_reaps_all(tmp_path, monkeypatch):
    """Sweep symmetry with consume (Pass-review RC-2): under a degraded current
    boot_id, the sweep treats every crumb as stale (it can't trust a same-pid
    match), keeping the dir from accumulating."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    sk = _entry(runner, "sweepdeg").session_key
    monkeypatch.setattr(runner, "_current_boot_id", lambda: f"{_os.getpid()}:")
    # a crumb stamped with the SAME pid-only id (would survive a naive == check)
    _write_breadcrumb(runner, sk, boot_id=f"{_os.getpid()}:")
    removed = runner._sweep_restart_initiated_breadcrumbs()
    assert removed == 1


# ───────── F2 BACKLOG CLEANUP (spec 2026-06-22_f2-breadcrumb-backlog-cleanup) ─────────
#
# Item 3 (config consistency), Item 2 (contract conformance), Item 1 covered in
# the skill's test_watcher.py. Phase 4 = no-regression source-guard.

from gateway.run import (
    _AGENT_CONFIG_ENV_BRIDGE,
    _bridge_agent_config_to_env,
)


# ---- Phase 1: Item 3 — restart_* config family consistency ----

def test_restart_initiated_ttl_in_bridge_map():
    """The new knob is wired into the single-sourced bridge map (so the startup
    block bridges it without a bespoke if-branch)."""
    assert _AGENT_CONFIG_ENV_BRIDGE["restart_initiated_ttl_secs"] == "HERMES_RESTART_INITIATED_TTL_SECS"
    # the already-shipped siblings stay mapped (no accidental drop in the refactor)
    assert _AGENT_CONFIG_ENV_BRIDGE["restart_loop_threshold"] == "HERMES_RESTART_LOOP_THRESHOLD"
    assert _AGENT_CONFIG_ENV_BRIDGE["restart_loop_window_secs"] == "HERMES_RESTART_LOOP_WINDOW_SECS"


def test_restart_initiated_ttl_bridged_from_config(monkeypatch):
    """The new agent.restart_initiated_ttl_secs config key is bridged to its env
    var by _bridge_agent_config_to_env (the single-sourced startup bridge)."""
    monkeypatch.delenv("HERMES_RESTART_INITIATED_TTL_SECS", raising=False)
    _bridge_agent_config_to_env({"restart_initiated_ttl_secs": 240})
    assert _os.environ["HERMES_RESTART_INITIATED_TTL_SECS"] == "240"


def test_startup_restore_drain_timeout_in_bridge_map():
    """The startup-restore drain-timeout knob is wired into the single-sourced
    bridge map (so the startup block bridges it without a bespoke if-branch)."""
    from gateway.run import _AGENT_CONFIG_ENV_BRIDGE as _M
    assert _M["gateway_startup_restore_drain_timeout"] == "HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT"


def test_startup_restore_drain_timeout_bridged_and_read(monkeypatch):
    """config → env bridge feeds the live helper; a non-positive value opts out
    of the bound (returns the raw config value, which the caller treats as
    'wait forever')."""
    from gateway.run import (
        _bridge_agent_config_to_env as _bridge,
        _startup_restore_drain_timeout_secs as _drain,
        _STARTUP_RESTORE_DRAIN_TIMEOUT_SECS_DEFAULT as _DEFAULT,
    )
    monkeypatch.delenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", raising=False)
    # default when unset
    assert _drain() == float(_DEFAULT)
    # config value bridged and read back
    _bridge({"gateway_startup_restore_drain_timeout": 45})
    assert _os.environ["HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT"] == "45"
    assert _drain() == 45.0
    # opt-out sentinel survives the round-trip (0 => caller waits unbounded)
    _bridge({"gateway_startup_restore_drain_timeout": 0})
    assert _drain() == 0.0
    # malformed env falls back to the module default
    monkeypatch.setenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", "not-a-number")
    assert _drain() == float(_DEFAULT)


def test_startup_restore_drain_default_matches_config(monkeypatch):
    """The DEFAULT_CONFIG agent value equals the live helper's no-env fallback,
    so surfacing it in config changed no effective value."""
    from hermes_cli.config import DEFAULT_CONFIG
    from gateway.run import _startup_restore_drain_timeout_secs as _drain
    monkeypatch.delenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", raising=False)
    assert float(DEFAULT_CONFIG["agent"]["gateway_startup_restore_drain_timeout"]) == _drain()


def test_restart_family_defaults_match_live_helper_fallbacks(monkeypatch):
    """I-2 (pinned to the LIVE helper, not a literal): the DEFAULT_CONFIG agent
    values equal what each helper returns with NO env set — so surfacing them in
    config changed no effective value, and a future helper-fallback edit can't
    drift silently past a test comparing two stale literals."""
    from hermes_cli.config import DEFAULT_CONFIG

    for var in ("HERMES_RESTART_LOOP_THRESHOLD", "HERMES_RESTART_LOOP_WINDOW_SECS",
                "HERMES_RESTART_INITIATED_TTL_SECS"):
        monkeypatch.delenv(var, raising=False)
    agent = DEFAULT_CONFIG["agent"]
    assert agent["restart_loop_threshold"] == _restart_loop_threshold()
    assert agent["restart_loop_window_secs"] == _restart_loop_window_secs()
    assert agent["restart_initiated_ttl_secs"] == _restart_initiated_ttl_secs()


def test_config_wins_over_preset_env_for_ttl(monkeypatch):
    """I-3 (precedence, corrected from ground-truth): config UNCONDITIONALLY wins
    over a pre-set env var (PR #18413). Proven for the new TTL knob AND a shipped
    sibling, so the new one matches established semantics."""
    monkeypatch.setenv("HERMES_RESTART_INITIATED_TTL_SECS", "900")
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "9")
    _bridge_agent_config_to_env(
        {"restart_initiated_ttl_secs": 120, "restart_loop_threshold": 3}
    )
    assert _os.environ["HERMES_RESTART_INITIATED_TTL_SECS"] == "120"  # config wins
    assert _os.environ["HERMES_RESTART_LOOP_THRESHOLD"] == "3"  # sibling-consistent


def test_bridge_absent_key_leaves_env_untouched(monkeypatch):
    """A key absent from config does NOT clobber a pre-set env var (config-when-
    present, else env) — the documented precedence."""
    monkeypatch.setenv("HERMES_RESTART_INITIATED_TTL_SECS", "777")
    _bridge_agent_config_to_env({"max_turns": 50})  # no ttl key
    assert _os.environ["HERMES_RESTART_INITIATED_TTL_SECS"] == "777"


def test_ttl_config_clamps_both_bounds_and_malformed(monkeypatch):
    """Pass-1 R-3: exercise BOTH clamp bounds + malformed, not just the floor."""
    # floor
    monkeypatch.setenv("HERMES_RESTART_INITIATED_TTL_SECS", "5")
    assert _restart_initiated_ttl_secs() == 60.0
    # ceiling
    monkeypatch.setenv("HERMES_RESTART_INITIATED_TTL_SECS", "999999")
    assert _restart_initiated_ttl_secs() == 86400.0
    # malformed → default
    monkeypatch.setenv("HERMES_RESTART_INITIATED_TTL_SECS", "abc")
    assert _restart_initiated_ttl_secs() == 600.0


def test_bridge_noop_on_non_dict():
    """Defensive: a non-dict agent config (malformed yaml) is a no-op, not a crash."""
    _bridge_agent_config_to_env(None)
    _bridge_agent_config_to_env("not a dict")
    _bridge_agent_config_to_env(["list"])


# ---- Phase 2: Item 2 — frozen-contract conformance (gateway side, always-on) ----

# The single source of truth for the cross-repo F2 breadcrumb contract. The skill
# (safe-restart.py, hermes-home) has a mirror test asserting the same literals.
_FROZEN_BREADCRUMB_DIRNAME = ".restart_initiated"
_FROZEN_BREADCRUMB_JSON_KEYS = {"session_key", "ts", "boot_id"}
# Representative key vector (Pass-1 C-2: one sample != byte-equality).
_CONTRACT_KEY_VECTOR = [
    "",
    "agent:main:discord:group:123",
    "x" * 4096,
    "agent:main:test:weird/slash:1",
    "ünïçödé:キー:1",
]


def _frozen_filename(session_key: str) -> str:
    import hashlib
    return hashlib.sha256((session_key or "").encode("utf-8")).hexdigest()[:8]


def test_gateway_breadcrumb_contract_matches_frozen():
    """Always-on (fork CI, never skips): the gateway's breadcrumb constants match
    the frozen contract literals over a key vector. The skill's mirror test asserts
    the same literals in hermes-home CI, so drift in EITHER repo reddens its own
    suite without needing the other checked out."""
    from gateway.run import _RESTART_INITIATED_DIRNAME, _restart_initiated_filename

    assert _RESTART_INITIATED_DIRNAME == _FROZEN_BREADCRUMB_DIRNAME
    for key in _CONTRACT_KEY_VECTOR:
        assert _restart_initiated_filename(key) == _frozen_filename(key), key
    # the consumer reads exactly these JSON keys (from the gate's data.get calls)
    import inspect
    from gateway.run import GatewayRunner
    consume_src = inspect.getsource(GatewayRunner._consume_restart_initiated_breadcrumb)
    for jk in _FROZEN_BREADCRUMB_JSON_KEYS:
        assert f'"{jk}"' in consume_src or f"'{jk}'" in consume_src, jk


def test_breadcrumb_contract_block_documented_in_source():
    """The # F2 BREADCRUMB CONTRACT block is the single documented source of truth;
    its presence is grep-guarded so it can't be silently deleted."""
    import inspect
    import gateway.run as gr
    assert "F2 BREADCRUMB CONTRACT" in inspect.getsource(gr)


# ---- Phase 4: I-1 — no-regression source guard for the shipped #80 path ----

def test_shipped_breadcrumb_bodies_unchanged():
    """I-1 teeth: the load-bearing lines of the four shipped #80 functions are
    present in source, so this hygiene PR provably didn't edit detection logic."""
    import inspect
    import gateway.run as gr
    import gateway.status as gs

    run_src = inspect.getsource(gr)
    # gate: always-consume (no short-circuit) + OR
    assert "breadcrumb = self._consume_restart_initiated_breadcrumb(session_key)" in run_src
    assert "initiated_restart = flag or breadcrumb" in run_src
    # consume: boot gate + degraded fail-safe + always-unlink
    assert "current_boot.endswith(\":\")" in run_src
    assert "path.unlink()" in run_src
    # sweep: current-boot keep
    assert "_sweep_restart_initiated_breadcrumbs" in run_src
    # producer: boot_id refreshed every write
    status_src = inspect.getsource(gs)
    assert 'payload["boot_id"] = current_record["boot_id"]' in status_src


# ─────────────────────────────────────────────────────────────────────────────
# restart_consumed_interrupted (the self-initiated-restart-that-was-also-
# drain-interrupted seam fix). Spec: 2026-07-01_restart-reboot-continuity-*.
# A CLEAN self-restart stays restart_consumed (excluded from auto-resume, F1/F2
# cascade guard). A self-restart that was STILL RUNNING at drain-timeout becomes
# restart_consumed_interrupted → auto-resumes (surface-and-wait) → but still
# records the F2 replay-mark so a genuine loop is bounded.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clean_self_restart_stays_consumed_not_auto_resumed(tmp_path, monkeypatch):
    """AC-2: a self-initiated restart that was NOT interrupted (marked at the
    pre-drain site, interrupted=False) keeps the bare restart_consumed reason and
    is NOT auto-resumed — the F1/F2 cascade guard is preserved."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)

    runner._session_initiated_restart[entry.session_key] = True
    # pre-drain site → interrupted defaults False
    marked, reason, _alert = runner._mark_resume_pending_for_shutdown(entry.session_key)

    assert marked is True
    assert reason == "restart_consumed"
    assert "restart_consumed" not in runner._AUTO_RESUME_REASONS
    assert runner._schedule_resume_pending_sessions() == 0


@pytest.mark.asyncio
async def test_interrupted_self_restart_becomes_interrupted_reason_and_auto_resumes(
    tmp_path, monkeypatch
):
    """AC-1 (core): a self-initiated restart STILL RUNNING at drain-timeout
    (interrupted=True) gets restart_consumed_interrupted, which IS auto-resumed."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)

    runner._session_initiated_restart[entry.session_key] = True
    marked, reason, _alert = runner._mark_resume_pending_for_shutdown(
        entry.session_key, interrupted=True
    )

    assert marked is True
    assert reason == "restart_consumed_interrupted"
    assert "restart_consumed_interrupted" in runner._AUTO_RESUME_REASONS
    assert (
        runner.session_store._entries[entry.session_key].resume_reason
        == "restart_consumed_interrupted"
    )
    # It PROACTIVELY schedules a boot-time surface-and-wait turn (not silent).
    assert runner._schedule_resume_pending_sessions() == 1
    await asyncio.gather(*runner._background_tasks)


def test_reason_discriminator_matrix(tmp_path, monkeypatch):
    """The reason table: self-initiated × interrupted → reason. Non-initiated
    interrupted stays shutdown_timeout (already auto-resumes, unaffected)."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    sk = entry.session_key

    # non-initiated, not interrupted → shutdown_timeout
    assert runner._resume_reason_for_shutdown_mark(sk) == "shutdown_timeout"
    # non-initiated, interrupted → still shutdown_timeout (reboot/plain-shutdown path)
    assert runner._resume_reason_for_shutdown_mark(sk, interrupted=True) == "shutdown_timeout"

    runner._session_initiated_restart[sk] = True
    # self-initiated, not interrupted → restart_consumed (excluded, cascade guard)
    assert runner._resume_reason_for_shutdown_mark(sk) == "restart_consumed"
    # self-initiated, interrupted → the new reason (auto-resumes)
    assert (
        runner._resume_reason_for_shutdown_mark(sk, interrupted=True)
        == "restart_consumed_interrupted"
    )


def test_restart_requested_interrupted_prefers_restart_timeout_when_not_self_initiated(
    tmp_path, monkeypatch
):
    """A /restart-requested shutdown that did NOT set the per-session self-init
    flag still resolves restart_timeout, not the new reason — the new reason is
    strictly for the self-initiated path."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    runner._restart_requested = True
    # not self-initiated for this session
    assert runner._resume_reason_for_shutdown_mark(entry.session_key, interrupted=True) == "restart_timeout"


@pytest.mark.asyncio
async def test_interrupted_reason_still_records_replay_mark_same_pass(tmp_path, monkeypatch):
    """AC-2c (D-2 × D-6 interleaving): on ONE post-timeout pass for a session that
    was _resumed_this_boot, self-restarted, and drain-interrupted again, the mark
    writes restart_consumed_interrupted AND the replay-mark call still increments.
    The reason-tag and the replay-mark are NOT mutually exclusive."""
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    monkeypatch.setenv("HERMES_RESTART_LOOP_WINDOW_SECS", "300")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    sk = entry.session_key

    # Session was auto-resumed this boot, then self-restarts again.
    runner._resumed_this_boot.add(sk)
    runner._session_initiated_restart[sk] = True
    runner._replay_marked_during_stop = set()

    marked, reason, _alert = runner._mark_resume_pending_for_shutdown(sk, interrupted=True)

    assert reason == "restart_consumed_interrupted"
    # the replay-mark was recorded for this session this pass
    counts = runner._load_restart_failure_counts()
    assert len(counts.get(sk, {}).get("replay_marks", [])) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_mode", ["prompt", "auto"])
async def test_interrupted_reason_replay_loop_is_bounded_cross_cycle(
    tmp_path, monkeypatch, resume_mode
):
    """AC-2b (CROSS-CYCLE, the pass-3-sharpest gate): a session that self-restarts
    AND is drain-interrupted every cycle (reason=restart_consumed_interrupted) must
    STILL trip F2/suspend within the 300s window — driven as real resume→re-interrupt
    cycles seeding _resumed_this_boot, NOT 3 marks in one pass. A build that failed to
    wire the new reason through the _resumed_this_boot→replay-mark path would loop
    forever and fail this test (fake-gate rejection)."""
    monkeypatch.setenv("HERMES_RESUME_INTERRUPTED_TURNS", resume_mode)
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    monkeypatch.setenv("HERMES_RESTART_LOOP_WINDOW_SECS", "300")
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner, "iloop")
    sk = entry.session_key

    # Boot 0: session self-initiated a restart AND was drain-interrupted →
    # the new reason, which IS auto-resumed.
    runner._session_initiated_restart[sk] = True
    runner._mark_resume_pending_for_shutdown(sk, interrupted=True)
    runner.session_store._entries[sk].last_resume_marked_at = datetime.now()
    runner.session_store._save()
    assert runner.session_store._entries[sk].resume_reason == "restart_consumed_interrupted"

    scheduled_each_boot = []
    suspended_at = None
    for cycle in range(6):
        booted, _ = _runner(tmp_path, monkeypatch)
        booted.session_store._ensure_loaded()

        n = booted._schedule_resume_pending_sessions()
        scheduled_each_boot.append(n)
        if n:
            await asyncio.gather(*booted._background_tasks)

        # The resumed turn self-restarts again and is drain-interrupted again.
        booted._session_initiated_restart[sk] = True
        booted._replay_marked_during_stop = set()
        booted._mark_resume_pending_for_shutdown(sk, interrupted=True)

        if booted.session_store._entries[sk].suspended:
            suspended_at = cycle
            break

    assert suspended_at is not None, (
        f"restart_consumed_interrupted replay loop never broke; "
        f"scheduled per boot = {scheduled_each_boot}"
    )
    # After suspension a fresh boot schedules ZERO — cascade bounded.
    final, _ = _runner(tmp_path, monkeypatch)
    final.session_store._ensure_loaded()
    assert final.session_store._entries[sk].suspended is True
    assert final._schedule_resume_pending_sessions() == 0


def test_new_reason_has_recovery_phrase(tmp_path, monkeypatch):
    """The new reason maps to a concrete recovery-note phrase (not the generic
    fallback) so the surfaced prompt reads correctly."""
    from gateway.run import _resume_reason_phrase
    assert _resume_reason_phrase("restart_consumed_interrupted") == "a gateway restart"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE= observability (Task B, spec 2026-07-01). The mark decision + boot-resume
# scheduling are logged with session key + reason only (INV-6: no transcript
# content), via the module logger (INV-3/D-8: non-blocking, no fsync in the hot
# path).
# ─────────────────────────────────────────────────────────────────────────────


def test_phase_shutdown_mark_logs_decision(tmp_path, monkeypatch, caplog):
    """AC-3: the per-session mark logs PHASE=shutdown_mark with the chosen reason
    and the interrupted/replay flags — the decision is observable."""
    import logging
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner, "phase1")
    runner._session_initiated_restart[entry.session_key] = True

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        runner._mark_resume_pending_for_shutdown(entry.session_key, interrupted=True)

    lines = [r.getMessage() for r in caplog.records if "PHASE=shutdown_mark" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    assert "reason=restart_consumed_interrupted" in line
    assert "interrupted=True" in line
    assert entry.session_key in line


@pytest.mark.asyncio
async def test_phase_logs_carry_no_transcript_content(tmp_path, monkeypatch, caplog):
    """AC-6/INV-6: PHASE log lines carry session keys + flags only — never
    message/transcript content. NON-TAUTOLOGICAL: the sentinel is seeded into the
    actual session-carried fields the logger has in scope (the transcript tail
    and a pending resume message/handoff), so if any PHASE producer ever
    interpolated message content the assertion WOULD fail."""
    import logging
    from unittest.mock import MagicMock
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner, "phase2")
    SECRET = "TRANSCRIPT_SECRET_marker_should_never_be_logged"

    # Seed the sentinel into every session-scoped surface a PHASE producer could
    # plausibly reach if someone "enriched" a log line: the transcript (keyed by
    # session_id), a pending message, and the running-agent object.
    runner.session_store.append_to_transcript(
        entry.session_id, {"role": "user", "content": SECRET}, skip_db=True
    )
    runner._pending_messages[entry.session_key] = [SECRET]
    _fake_agent = MagicMock()
    _fake_agent.last_user_message = SECRET
    _fake_agent._session_messages = [{"role": "user", "content": SECRET}]
    runner._running_agents[entry.session_key] = _fake_agent

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        runner._mark_resume_pending_for_shutdown(entry.session_key)

    # And exercise the boot-resume PHASE producer too (fresh runner so the
    # SECRET-bearing transcript is loaded from disk, not the mocked agent).
    booted, _ = _runner(tmp_path, monkeypatch)
    booted.session_store._ensure_loaded()
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        n = booted._schedule_resume_pending_sessions()
        if n:
            await asyncio.gather(*booted._background_tasks)

    phase_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("PHASE=")]
    assert phase_lines, "expected at least one PHASE= line"
    for line in phase_lines:
        assert SECRET not in line, f"transcript content leaked into a PHASE line: {line!r}"


def test_phase_mark_log_is_single_emit_and_failopen(tmp_path, monkeypatch):
    """AC-3 (honest gate — replaces an over-claiming 'non-blocking' test). The
    per-session mark PHASE log matches the existing synchronous 'Shutdown phase:'
    logs already in stop() (same FileHandler, same path), so this does NOT claim
    to defeat a pathological blocking handler. What it DOES enforce, and what
    actually protects the drain budget:
      (1) SINGLE emit per mark — no accidental fan-out that would multiply cost
          across N sessions; and
      (2) FAIL-OPEN — a handler that RAISES must not abort the mark (the resume
          flag must still be written), per INV-3.
    Restores the logger level in finally (test-hygiene, review RC).
    """
    import logging
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner, "phase3")

    emit_count = {"n": 0}

    class _CountingRaisingHandler(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith("PHASE=shutdown_mark"):
                emit_count["n"] += 1
                raise RuntimeError("simulated broken log sink")

    lg = logging.getLogger("gateway.run")
    prev_level = lg.level
    # logging swallows handler exceptions unless raiseExceptions is on; force the
    # raise to surface so we PROVE the mark path tolerates it (fail-open).
    prev_raise = logging.raiseExceptions
    logging.raiseExceptions = False
    handler = _CountingRaisingHandler()
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    try:
        marked, reason, _alert = runner._mark_resume_pending_for_shutdown(entry.session_key)
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level)
        logging.raiseExceptions = prev_raise

    # exactly one PHASE=shutdown_mark emit (no fan-out)
    assert emit_count["n"] == 1
    # and the mark still succeeded despite the raising sink (fail-open)
    assert marked is True
    assert reason == "shutdown_timeout"
    assert runner.session_store._entries[entry.session_key].resume_pending is True


@pytest.mark.asyncio
async def test_reason_upgrades_on_repeat_mark_clean_then_interrupted(tmp_path, monkeypatch):
    """Review RC#3: the fix relies on the POST-drain interrupted=True mark
    OVERWRITING a prior pre-drain restart_consumed with the interrupted variant.
    Assert the upgrade directly (mark clean → mark interrupted → reason upgraded),
    so the feature can't silently regress to 'silent' if mark_resume_pending ever
    gains idempotency."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner, "upgrade")
    sk = entry.session_key
    runner._session_initiated_restart[sk] = True

    # pre-drain speculative mark (not interrupted) → clean restart_consumed
    _m, reason1, _a = runner._mark_resume_pending_for_shutdown(sk)
    assert reason1 == "restart_consumed"
    assert runner.session_store._entries[sk].resume_reason == "restart_consumed"

    # post-timeout mark (interrupted) → must UPGRADE to the interrupted variant
    _m, reason2, _a = runner._mark_resume_pending_for_shutdown(sk, interrupted=True)
    assert reason2 == "restart_consumed_interrupted"
    assert (
        runner.session_store._entries[sk].resume_reason == "restart_consumed_interrupted"
    ), "post-drain interrupted mark must overwrite the pre-drain restart_consumed"


@pytest.mark.asyncio
async def test_self_init_reason_does_not_drift_across_unrelated_later_shutdown(
    tmp_path, monkeypatch
):
    """Review residual #31: a session that once self-restarted (interrupted) must
    NOT be stamped a restart_consumed* reason on a later UNRELATED plain shutdown —
    because a successful resume clears resume_reason (→ None) AND the post-turn gate
    pops _session_initiated_restart. Verify the classification returns to
    shutdown_timeout once both signals are cleared."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner, "drift")
    sk = entry.session_key

    # cycle 1: self-initiated + interrupted → interrupted variant
    runner._session_initiated_restart[sk] = True
    _m, reason1, _a = runner._mark_resume_pending_for_shutdown(sk, interrupted=True)
    assert reason1 == "restart_consumed_interrupted"

    # a successful resume clears the flag + nulls resume_reason (the real path)
    runner._session_initiated_restart.pop(sk, None)
    runner.session_store.clear_resume_pending(sk)
    assert runner.session_store._entries[sk].resume_reason is None

    # cycle 2: a later, UNRELATED plain shutdown (no self-init this time)
    _m, reason2, _a = runner._mark_resume_pending_for_shutdown(sk)
    assert reason2 == "shutdown_timeout", (
        "classification drifted: a self-restart reason leaked into a later "
        "unrelated shutdown"
    )


def test_interrupted_mark_logs_at_warning_survives_raised_threshold(tmp_path, monkeypatch, caplog):
    """Review RC (SRE): the restart-continuity audit breadcrumb must survive a
    gateway.run logger raised to WARNING+ — an INTERRUPTED mark is the diagnostic
    case an operator greps when a restart goes wrong, so it logs at WARNING (its
    clean sibling stays INFO)."""
    import logging
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner, "warnlvl")
    runner._session_initiated_restart[entry.session_key] = True

    # Also pin caplog propagation coupling (review residual): gateway.run must
    # propagate for these audit logs to be visible.
    assert logging.getLogger("gateway.run").propagate is True

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._mark_resume_pending_for_shutdown(entry.session_key, interrupted=True)

    warn_marks = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "PHASE=shutdown_mark" in r.getMessage()
        and "reason=restart_consumed_interrupted" in r.getMessage()
    ]
    assert len(warn_marks) == 1, "interrupted mark must log at WARNING (survives raised threshold)"


def test_clean_mark_stays_info(tmp_path, monkeypatch, caplog):
    """The clean (non-interrupted, non-replay) mark stays at INFO — WARNING is
    reserved for the diagnostic cases so routine shutdowns don't spam WARNING."""
    import logging
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner, "infolvl")

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        runner._mark_resume_pending_for_shutdown(entry.session_key)  # shutdown_timeout, clean

    marks = [r for r in caplog.records if "PHASE=shutdown_mark" in r.getMessage()]
    assert len(marks) == 1
    assert marks[0].levelno == logging.INFO
