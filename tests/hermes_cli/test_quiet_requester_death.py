"""Requester-death policy for a spawner-owned delivery child (#93091 follow-up).

A delivery child is a one-shot whose output has exactly one consumer: the process that spawned it.
When that requester dies, nothing reads the child again — yet pre-fix the child kept running,
holding the TARGET profile's ``state.db`` (+ WAL) and its MCP server children, and kept spawning
deliveries. On 2026-09-20 that orphan outlived its requester by 1h46m, reparented to the gateway,
and every later delivery to that profile was refused with the mid-turn ``target_session_live``
text (see ``tools/bot_failure_reasons``) rather than the turn-lock one.

These tests pin the policy's primitives: the child detects the dead requester (from the kernel's
own reparenting signal), ends its own turn, and does so only when a spawner armed it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_cli import quiet_single_query as qsq


# ── arming: the requester pid travels in the child env, and is consumed ──────

def test_requester_pid_is_consumed_before_the_turn(monkeypatch):
    monkeypatch.setenv(qsq.REQUESTER_PID_ENV, "4242")
    assert qsq.peek_requester_pid() == 4242
    assert os.environ[qsq.REQUESTER_PID_ENV] == "4242", "peeking must not consume"
    assert qsq.take_requester_pid() == 4242
    assert qsq.REQUESTER_PID_ENV not in os.environ, (
        "consumed so nothing this run spawns mistakes its own parent for the requester"
    )


@pytest.mark.parametrize("raw", ["", "   ", "not-a-pid", "0", "-3", None])
def test_unusable_requester_pid_never_arms_the_watchdog(monkeypatch, raw):
    environ = {} if raw is None else {qsq.REQUESTER_PID_ENV: raw}
    assert qsq.take_requester_pid(environ) is None


def test_requester_pid_env_names_this_process_by_default():
    assert qsq.requester_pid_env() == {qsq.REQUESTER_PID_ENV: str(os.getpid())}


# ── liveness: reparenting is the signal ─────────────────────────────────────

def test_reparenting_alone_proves_the_requester_was_reaped():
    """The kernel moves a child to a subreaper when its parent is REAPED — exact and free."""
    alive = lambda _pid: True  # the pid itself exists; only the parentage changed
    assert qsq.requester_alive(100, getppid=lambda: 100, pid_exists=alive) is True
    assert qsq.requester_alive(100, getppid=lambda: 1, pid_exists=alive) is False


def test_a_probe_that_cannot_prove_death_does_not_end_the_turn():
    """Never end a live turn on an inconclusive probe: a stripped install has no gateway module."""
    def _boom(_pid):
        raise RuntimeError("no /proc here")

    assert qsq.requester_alive(100, getppid=lambda: 100, pid_exists=_boom) is True


# ── the watchdog fires once, and only for the armed case ────────────────────

def test_watchdog_is_inert_without_a_spawner():
    """A person's ``hermes chat -Q`` sets no requester pid and must never be ended by this."""
    fired = threading.Event()
    stop = qsq.arm_requester_watchdog(None, fired.set, poll_seconds=0.01)
    try:
        assert not fired.wait(0.2), "an unarmed one-shot is untouched"
    finally:
        stop.set()


def test_watchdog_ends_the_run_once_when_the_requester_dies():
    fired = threading.Event()
    calls: list[int] = []

    def _on_death() -> None:
        calls.append(1)
        fired.set()

    # getppid already reports the reparent, i.e. the requester has been reaped.
    stop = qsq.arm_requester_watchdog(
        4242, _on_death, poll_seconds=0.01, getppid=lambda: 1)
    try:
        assert fired.wait(2.0), "the child ends itself instead of outliving its requester"
        time.sleep(0.05)
        assert len(calls) == 1, "exactly once — the watcher returns after firing"
    finally:
        stop.set()


def test_watchdog_survives_a_failing_callback():
    """A failed teardown must not leave the orphan alive: the watcher still returns."""
    fired = threading.Event()

    def _on_death() -> None:
        fired.set()
        raise RuntimeError("interrupt failed")

    stop = qsq.arm_requester_watchdog(
        4242, _on_death, poll_seconds=0.01, getppid=lambda: 1)
    try:
        assert fired.wait(2.0)
    finally:
        stop.set()


# ── the delivery lanes arm the child; a nested delivery re-arms its own ─────

def test_delivery_child_env_arms_the_policy_and_pops_an_inherited_pid(monkeypatch):
    """A NESTED delivery must not inherit the grandparent's pid and mistake its own requester.

    Same contract as the author/dropped-session vars beside it: the requester pid is popped
    first and re-stamped from the process that is actually spawning THIS child.
    """
    import tools.bot_relay as bot_relay
    from agent.turn_author import TURN_AUTHOR_ENV

    monkeypatch.setenv(TURN_AUTHOR_ENV, json.dumps({"id": "bot:previous", "name": "p", "is_bot": True}))
    monkeypatch.setenv(qsq.REQUESTER_PID_ENV, "999999")  # the grandparent's

    env = bot_relay.delivery_env(bot_relay.delivery_turn_author("ops", "ops"))

    assert env[qsq.REQUESTER_PID_ENV] == str(os.getpid()), (
        "the child's requester is whoever spawned THIS delivery, not the grandparent"
    )
    assert qsq.REQUESTER_PID_ENV in os.environ, "the parent's own environ is untouched"


# ── the child's own run: it stops and exits through the normal finalize ─────

def _run_quiet(monkeypatch, tmp_path, *, requester_env: str | None, turn):
    """Run ``_run_quiet_single_query`` against a stub CLI; return (exit_code, report, cli)."""
    import cli as cli_mod

    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    if requester_env is None:
        monkeypatch.delenv(qsq.REQUESTER_PID_ENV, raising=False)
    else:
        monkeypatch.setenv(qsq.REQUESTER_PID_ENV, requester_env)

    report_path = tmp_path / "turn-report.json"
    the_cli = SimpleNamespace(
        agent=SimpleNamespace(run_conversation=turn),
        conversation_history=[],
        session_id="s-1",
        _quiet_notify_linger_done=False,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli_mod, "_emit_interrupted_session_end", lambda *a, **k: None)
        with pytest.raises(SystemExit) as exc:
            cli_mod._run_quiet_single_query(the_cli, "hello")
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return exc.value.code, report, the_cli


def test_requester_death_marks_the_report_and_exits_its_own_code(monkeypatch, tmp_path):
    """The spawner is told WHY the turn ended, and the child leaves immediately.

    The report's exit code is the requester-death code rather than the turn's: the turn was cut
    short, it did not fail — and the spawner must not book it as a delivered answer nobody read.
    """
    monkeypatch.setenv(qsq.TURN_REPORT_FILE_ENV, str(tmp_path / "turn-report.json"))
    # The requester has already been reaped, so the very first poll fires.
    monkeypatch.setattr(qsq, "requester_alive", lambda *a, **k: False)

    def turn(**_kwargs):
        # Give the watcher thread a moment to notice and interrupt.
        time.sleep(0.3)
        return {"final_response": "an answer nobody will read"}

    code, report, the_cli = _run_quiet(monkeypatch, tmp_path, requester_env="4242", turn=turn)

    assert code == qsq.REQUESTER_DEATH_EXIT_CODE
    assert report["exit_code"] == qsq.REQUESTER_DEATH_EXIT_CODE
    assert report["error"] == "requester exited"
    assert the_cli._quiet_notify_linger_done is True, (
        "no linger: the follow-up turns it protects would have no consumer either"
    )


def test_an_ordinary_one_shot_is_unaffected(monkeypatch, tmp_path):
    """No requester pid (a person's ``hermes chat -Q``): normal turn, normal exit code."""
    monkeypatch.setenv(qsq.TURN_REPORT_FILE_ENV, str(tmp_path / "turn-report.json"))
    monkeypatch.setattr(qsq, "requester_alive", lambda *a, **k: True)

    code, report, _cli = _run_quiet(
        monkeypatch, tmp_path, requester_env=None, turn=lambda **_k: {"final_response": "ok"})

    assert code == 0
    assert report["exit_code"] == 0
    assert report["reply"] == "ok"
