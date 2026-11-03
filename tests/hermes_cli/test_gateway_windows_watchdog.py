"""Windows gateway hang watchdog (CORP-836): an OS-scheduled probe outside the gateway process that
restarts a wedged or vanished gateway only after three consecutive failed probes, with a restart cap
and backoff so a gateway that cannot recover is never restarted in a loop.

unittest.TestCase so the file runs under ``python -m unittest`` as well as the pytest runner. Every
schtasks call goes through a fake ``_exec_schtasks``: nothing here registers a real Scheduled Task.
"""

import json
import os
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree

from hermes_cli import gateway_windows
from hermes_cli import gateway_windows_watchdog as watchdog

_ENV_PATCH = None


def setUpModule():
    # Plain unittest has no conftest isolation: keep every HERMES_HOME read inside a temp dir.
    global _ENV_PATCH
    _ENV_PATCH = mock.patch.dict(os.environ, {"HERMES_HOME": tempfile.mkdtemp(prefix="hermes-wd-")})
    _ENV_PATCH.start()


def tearDownModule():
    _ENV_PATCH.stop()


class _Home(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)

    def log_records(self):
        path = self.home / "logs" / "gateway-watchdog.log"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _retained_gateway_state(runtime):
    """Stand-in for ``gateway.status.retained_gateway_state`` (that module needs the full test env)."""
    rt = runtime if isinstance(runtime, dict) else {}
    if rt.get("desired_state") != "stopped":
        if rt.get("gateway_state") == "startup_failed":
            return "startup_failed"
        if rt.get("gateway_state") == "degraded" and rt.get("exit_reason") in {"loop_liveness_watchdog", "shutdown_watchdog"}:
            return "degraded"
    return "stopped"


class _Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class _Probe:
    """Returns the queued verdicts in order (the last one repeats)."""

    def __init__(self, *verdicts, pid=4242):
        self.verdicts, self.pid = list(verdicts), pid

    def __call__(self, home):
        verdict = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        return verdict, (None if verdict in (watchdog.DOWN, watchdog.STOPPED) else self.pid)


class DecideTest(unittest.TestCase):
    def test_restart_only_on_the_third_consecutive_failure(self):
        state = {}
        actions = []
        for _ in range(3):
            state, action = watchdog.decide(state, watchdog.WEDGED, 100.0)
            actions.append(action)
        self.assertEqual(actions, [watchdog.ACTION_NONE, watchdog.ACTION_NONE, watchdog.ACTION_RESTART])

    def test_a_single_failure_between_healthy_probes_never_restarts(self):
        state = {}
        for verdict in (watchdog.WEDGED, watchdog.ALIVE, watchdog.DOWN, watchdog.ALIVE, watchdog.WEDGED, watchdog.WEDGED):
            state, action = watchdog.decide(state, verdict, 100.0)
            self.assertEqual(action, watchdog.ACTION_NONE)
        self.assertEqual(state["consecutive_failures"], 2)

    def test_unknown_neither_counts_nor_resets_and_stopped_resets(self):
        state = {}
        for verdict in (watchdog.WEDGED, watchdog.WEDGED, watchdog.UNKNOWN):
            state, action = watchdog.decide(state, verdict, 100.0)
        self.assertEqual((state["consecutive_failures"], action), (2, watchdog.ACTION_NONE))
        state, action = watchdog.decide(state, watchdog.WEDGED, 100.0)
        self.assertEqual(action, watchdog.ACTION_RESTART)
        state, _ = watchdog.decide({"consecutive_failures": 2}, watchdog.STOPPED, 100.0)
        self.assertEqual(state["consecutive_failures"], 0)

    def test_a_gateway_the_watchdog_stopped_stays_owed_until_it_answers_again(self):
        state = {"owed_token": "t0", "seen_token": "t0"}
        actions = []
        for _ in range(3):
            state, action = watchdog.decide(state, watchdog.STOPPED, 100.0, "t0")
            actions.append(action)
        self.assertEqual(actions[-1], watchdog.ACTION_RESTART)
        state, _ = watchdog.decide(state, watchdog.ALIVE, 100.0, "t0")
        self.assertIsNone(state["owed_token"])
        for _ in range(5):
            state, action = watchdog.decide(state, watchdog.STOPPED, 100.0, "t0")
            self.assertEqual(action, watchdog.ACTION_NONE)

    def test_the_owed_restart_ends_at_a_failed_start_or_any_live_process(self):
        for verdict in (watchdog.STARTUP_FAILED, watchdog.UNKNOWN, watchdog.WEDGED):
            state, action = watchdog.decide({"owed_token": "t0", "seen_token": "t0"}, verdict, 100.0, "t0")
            self.assertIsNone(state["owed_token"], verdict)
            self.assertEqual(action, watchdog.ACTION_NONE)
        for _ in range(5):
            state, action = watchdog.decide(state, watchdog.STARTUP_FAILED, 100.0, "t0")
            self.assertEqual(action, watchdog.ACTION_NONE)

    def test_an_operator_stop_voids_the_owed_restart_and_the_streak_whatever_the_state_file_says(self):
        # The state still claims a debt and a streak, but a stop has happened since (new token).
        state = {"owed_token": "t0", "seen_token": "t0", "consecutive_failures": 2}
        for _ in range(5):
            state, action = watchdog.decide(state, watchdog.STOPPED, 100.0, "t1")
            self.assertEqual(action, watchdog.ACTION_NONE)
        self.assertEqual(state["consecutive_failures"], 0)
        state, action = watchdog.decide({"seen_token": "t0", "consecutive_failures": 2}, watchdog.WEDGED, 100.0, "t1")
        self.assertEqual((state["consecutive_failures"], action), (1, watchdog.ACTION_NONE))

    def test_a_corrupt_state_file_is_reset_instead_of_stopping_the_watchdog(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state" / "gateway-watchdog.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"restart_times": [None]}), encoding="utf-8")
            self.assertEqual(watchdog.load_state(Path(tmp))["restart_times"], [])


class RunOnceTest(_Home):
    def run_probes(self, probe, restart, clock, times, updating=lambda: False):
        return [watchdog.run_once(self.home, probe=probe, restart=restart, clock=clock, updating=updating)
                for _ in range(times)]

    def test_an_update_in_progress_is_never_interrupted_and_clears_the_streak(self):
        restart, probe = mock.Mock(return_value=(True, "alive")), mock.Mock(return_value=(watchdog.DOWN, None))
        self.run_probes(probe, restart, _Clock(), 2)
        self.run_probes(probe, restart, _Clock(), 5, updating=lambda: True)
        restart.assert_not_called()
        self.assertEqual(probe.call_count, 2)  # the gateway is not even probed mid-update
        self.assertEqual(watchdog.load_state(self.home)["consecutive_failures"], 0)

    def test_an_update_that_starts_mid_restart_is_neither_a_failure_nor_counted(self):
        restart = mock.Mock(return_value=(False, watchdog.REASON_UPDATE_IN_PROGRESS))
        self.run_probes(_Probe(watchdog.WEDGED), restart, _Clock(), 3)
        state = watchdog.load_state(self.home)
        self.assertEqual((state["backoff_until"], state["failed_restarts"], state["restart_times"]), (0.0, 0, []))
        result = [r for r in self.log_records() if r["event"] == "restart_result"][-1]
        self.assertEqual(result["result"], "skipped")

    def test_an_operator_stop_after_a_successful_restart_is_never_revived(self):
        restart = mock.Mock(return_value=(True, "alive"))
        self.run_probes(_Probe(watchdog.WEDGED, watchdog.WEDGED, watchdog.WEDGED, watchdog.STOPPED), restart, _Clock(), 8)
        self.assertEqual(restart.call_count, 1)

    def test_hermes_gateway_stop_cancels_an_owed_restart(self):
        restart = mock.Mock(return_value=(False, "not_recovered"))
        clock = _Clock()
        self.run_probes(_Probe(watchdog.WEDGED), restart, clock, 3)
        watchdog.forget_owed_restart(self.home)
        clock.now = watchdog.load_state(self.home)["backoff_until"] + 1
        self.run_probes(_Probe(watchdog.STOPPED), restart, clock, 5)
        self.assertEqual(restart.call_count, 1)

    def test_a_stop_while_the_watchdog_waits_for_recovery_is_not_overwritten(self):
        clock = _Clock()

        def restart(home, pid, **_):
            clock.now += 30
            watchdog.forget_owed_restart(home)  # operator stops the freshly started gateway
            return False, "not_recovered"

        restart_mock = mock.Mock(side_effect=restart)
        self.run_probes(_Probe(watchdog.WEDGED), restart_mock, clock, 3)
        self.assertFalse(watchdog.owed_restart_pending(self.home))
        self.assertEqual(watchdog.load_state(self.home)["backoff_until"], 0.0)
        clock.now += 2 * watchdog.BACKOFF_MAX_S
        self.run_probes(_Probe(watchdog.STOPPED), restart_mock, clock, 5)
        self.assertEqual(restart_mock.call_count, 1)
        result = next(r for r in self.log_records() if r["event"] == "restart_result")
        self.assertEqual(result["result"], "cancelled_by_stop")

    def test_a_stop_while_the_third_probe_runs_prevents_the_restart(self):
        clock = _Clock()
        verdicts = iter([watchdog.WEDGED, watchdog.WEDGED])

        def probe(home):
            verdict = next(verdicts, None)
            if verdict is None:  # the operator stops the hung gateway while the third probe runs
                watchdog.forget_owed_restart(home)
                verdict = watchdog.WEDGED
            return verdict, 4242

        restart = mock.Mock(return_value=(True, "alive"))
        self.run_probes(probe, restart, clock, 3)
        self.run_probes(_Probe(watchdog.STOPPED), restart, clock, 5)
        restart.assert_not_called()
        self.assertFalse(watchdog.owed_restart_pending(self.home))

    def test_a_stop_inside_the_final_state_replace_after_a_failed_restart_is_not_lost(self):
        """Deterministic replay of the review race: the restart fails (a debt is owed), then a
        `hermes gateway stop` lands between the tick's state read and its final replace, and the
        watchdog writes its stale copy (still owing the restart) over the file."""
        clock = _Clock()
        save, saves = watchdog._save_state, []

        def racing_save(home, state):
            saves.append(state)
            if len(saves) == 2:  # tick 3: #1 persists the attempt, #2 is the final save
                watchdog.forget_owed_restart(home)
            save(home, state)

        restart = mock.Mock(return_value=(False, "not_recovered"))
        self.run_probes(_Probe(watchdog.WEDGED), restart, clock, 2)
        with mock.patch.object(watchdog, "_save_state", racing_save):
            self.run_probes(_Probe(watchdog.WEDGED), restart, clock, 1)
        self.assertEqual(restart.call_count, 1)
        self.assertIsNotNone(watchdog.load_state(self.home)["owed_token"])  # the stale debt was written
        for _ in range(4):  # past every backoff: the stale debt must never be honoured
            clock.now += 2 * watchdog.BACKOFF_MAX_S
            self.run_probes(_Probe(watchdog.STOPPED), restart, clock, 3)
        self.assertEqual(restart.call_count, 1)
        self.assertFalse(watchdog.owed_restart_pending(self.home))

    def test_a_stop_between_ticks_beats_an_owed_restart_after_a_failed_one(self):
        clock = _Clock()
        restart = mock.Mock(return_value=(False, "not_recovered"))
        self.run_probes(_Probe(watchdog.WEDGED), restart, clock, 3)
        self.assertTrue(watchdog.owed_restart_pending(self.home))
        # The stop is recorded while the state file claims the debt; the file is never rewritten.
        watchdog.forget_owed_restart(self.home)
        self.assertTrue(watchdog.load_state(self.home)["owed_token"] is not None)
        for _ in range(4):  # across several backoff periods and owed-restart windows
            clock.now += 2 * watchdog.BACKOFF_MAX_S
            self.run_probes(_Probe(watchdog.STOPPED), restart, clock, 3)
        self.assertEqual(restart.call_count, 1)

    def test_every_stop_gets_a_distinct_token(self):
        watchdog.forget_owed_restart(self.home)
        first = watchdog.read_stop_token(self.home)
        watchdog.forget_owed_restart(self.home)
        self.assertNotEqual(first, watchdog.read_stop_token(self.home))
        self.assertNotEqual(first, "")

    def test_an_unreadable_stop_marker_fails_closed(self):
        watchdog.forget_owed_restart(self.home)
        with mock.patch.object(Path, "read_text", side_effect=PermissionError(5, "Access is denied")):
            # Each read looks like a fresh stop: nothing compares equal, so nothing is ever restarted.
            self.assertNotEqual(watchdog.read_stop_token(self.home), watchdog.read_stop_token(self.home))

    def test_forget_owed_restart_never_raises_on_an_unwritable_marker(self):
        with mock.patch.object(os, "replace", side_effect=PermissionError(5, "Access is denied")):
            watchdog.forget_owed_restart(self.home)

    def test_yielding_to_an_update_before_the_stop_leaves_nothing_owed(self):
        restart = mock.Mock(return_value=(False, watchdog.REASON_UPDATE_IN_PROGRESS))
        self.run_probes(_Probe(watchdog.WEDGED), restart, _Clock(), 3)
        self.assertFalse(watchdog.owed_restart_pending(self.home))
        restart.return_value = (False, watchdog.REASON_UPDATE_AFTER_STOP)
        self.run_probes(_Probe(watchdog.WEDGED), restart, _Clock(), 3)
        self.assertTrue(watchdog.owed_restart_pending(self.home))  # we took it down: bring it back after

    def test_the_restart_carries_the_estimated_outage_window(self):
        clock = _Clock(100.0)
        restart = mock.Mock(return_value=(True, "alive"))
        probe = _Probe(watchdog.ALIVE, watchdog.WEDGED, watchdog.WEDGED, watchdog.WEDGED)
        for _ in range(4):
            self.run_probes(probe, restart, clock, 1)
            clock.now += 60
        self.assertEqual(restart.call_args.kwargs["outage"], {"last_healthy_at": 100.0, "detected_at": 160.0})

    def test_without_a_healthy_check_the_outage_start_stays_unknown(self):
        restart = mock.Mock(return_value=(True, "alive"))
        self.run_probes(_Probe(watchdog.WEDGED), restart, _Clock(), 3)
        self.assertIsNone(restart.call_args.kwargs["outage"]["last_healthy_at"])

    def test_a_failed_restart_keeps_the_stopped_gateway_owed_after_backoff(self):
        clock = _Clock()
        restart = mock.Mock(return_value=(False, "not_recovered"))
        self.run_probes(_Probe(watchdog.WEDGED), restart, clock, 3)
        clock.now = watchdog.load_state(self.home)["backoff_until"] + 1
        # The wedged process is gone now; its leftover state is not proof of a crash by itself.
        self.run_probes(_Probe(watchdog.STOPPED), restart, clock, 3)
        self.assertEqual(restart.call_count, 2)
        self.assertEqual(restart.call_args.args, (self.home, None))

    def test_update_probe_failure_fails_closed(self):
        with mock.patch("hermes_cli.update_lock.update_in_progress", side_effect=OSError("locked")):
            self.assertTrue(watchdog.update_in_progress())

    def test_healthy_gateway_is_left_alone_and_logs_nothing_alarming(self):
        restart = mock.Mock()
        actions = self.run_probes(_Probe(watchdog.ALIVE), restart, _Clock(), 5)
        self.assertEqual(set(actions), {watchdog.ACTION_NONE})
        restart.assert_not_called()
        self.assertFalse([r for r in self.log_records() if r["event"].startswith("restart")])

    def test_three_consecutive_wedged_probes_restart_once_and_log_before_and_after(self):
        restart = mock.Mock(return_value=(True, "alive"))
        actions = self.run_probes(_Probe(watchdog.WEDGED, watchdog.WEDGED, watchdog.WEDGED, watchdog.ALIVE),
                                  restart, _Clock(), 4)
        self.assertEqual(actions[2], watchdog.ACTION_RESTART)
        restart.assert_called_once()
        self.assertEqual(restart.call_args.args, (self.home, 4242))
        events = [r for r in self.log_records() if r["event"].startswith("restart")]
        self.assertEqual([r["event"] for r in events], ["restart_begin", "restart_result"])
        self.assertEqual(events[0]["reason"], watchdog.WEDGED)
        self.assertEqual(events[1]["result"], "recovered")
        self.assertTrue(all("ts" in r for r in events))
        state = watchdog.load_state(self.home)
        self.assertEqual((state["consecutive_failures"], state["failed_restarts"]), (0, 0))

    def test_vanished_gateway_that_should_be_running_restarts_without_a_pid(self):
        restart = mock.Mock(return_value=(True, "alive"))
        self.run_probes(_Probe(watchdog.DOWN), restart, _Clock(), 3)
        restart.assert_called_once()
        self.assertEqual(restart.call_args.args, (self.home, None))

    def test_failed_restart_is_logged_and_backs_off(self):
        clock = _Clock()
        restart = mock.Mock(return_value=(False, "not_recovered"))
        probe = _Probe(watchdog.WEDGED)
        self.run_probes(probe, restart, clock, 3)
        result = [r for r in self.log_records() if r["event"] == "restart_result"][-1]
        self.assertEqual((result["result"], result["reason"]), ("failed", "not_recovered"))
        state = watchdog.load_state(self.home)
        self.assertEqual(state["backoff_until"], clock.now + watchdog.BACKOFF_BASE_S)

        clock.now += 60
        self.assertEqual(self.run_probes(probe, restart, clock, 3)[-1], watchdog.ACTION_SUPPRESSED_BACKOFF)
        self.assertEqual(restart.call_count, 1)

        clock.now = state["backoff_until"] + 1
        self.assertEqual(self.run_probes(probe, restart, clock, 1), [watchdog.ACTION_RESTART])
        second = watchdog.load_state(self.home)
        self.assertEqual(second["backoff_until"] - clock.now, 2 * watchdog.BACKOFF_BASE_S)  # exponential

    def test_restart_cap_stops_a_restart_loop_of_gateways_that_recover_then_hang(self):
        clock = _Clock()
        restart = mock.Mock(return_value=(True, "alive"))
        probe = _Probe(watchdog.WEDGED)
        actions = []
        for _ in range(watchdog.RESTART_LIMIT + 2):
            actions.extend(self.run_probes(probe, restart, clock, 3))
            clock.now += 300
        self.assertEqual(restart.call_count, watchdog.RESTART_LIMIT)
        self.assertIn(watchdog.ACTION_SUPPRESSED_LIMIT, actions)
        self.assertTrue(any(r["event"] == "restart_suppressed" for r in self.log_records()))

    def test_restart_exception_is_contained_and_never_logs_its_message_or_secrets(self):
        secret = "xoxb-SECRET-TOKEN-1234567890"

        def boom(home, pid, **_):
            raise RuntimeError(f"connect failed token={secret}")

        with mock.patch.dict(os.environ, {"SLACK_BOT_TOKEN": secret}):
            self.run_probes(_Probe(watchdog.WEDGED), boom, _Clock(), 3)
        text = (self.home / "logs" / "gateway-watchdog.log").read_text(encoding="utf-8")
        self.assertNotIn(secret, text)
        result = [r for r in self.log_records() if r["event"] == "restart_result"][-1]
        self.assertEqual((result["result"], result["reason"]), ("failed", "error:RuntimeError"))
        for record in self.log_records():
            self.assertLessEqual(set(record), set(watchdog.LOG_FIELDS))


class ClassifyTest(_Home):
    def classify(self, pid, runtime=None, loop="alive"):
        probe = mock.Mock(return_value=loop)
        read_runtime = mock.Mock(return_value=runtime)
        verdict = watchdog.classify(self.home, live_pid=lambda home: pid, read_runtime=read_runtime,
                                    loop_probe=probe, retained_state=_retained_gateway_state)
        if pid is None:
            read_runtime.assert_called_once_with(self.home / "gateway_state.json")
        return verdict, probe

    def test_live_pid_is_judged_by_loop_responsiveness_not_process_existence(self):
        for loop, expected in (("alive", watchdog.ALIVE), ("wedged", watchdog.WEDGED), ("unknown", watchdog.UNKNOWN)):
            (verdict, pid), probe = self.classify(77, loop=loop)
            self.assertEqual((verdict, pid), (expected, 77))
            probe.assert_called_once_with(77, home=self.home)

    def test_missing_gateway_is_down_only_when_a_watchdog_proved_it_wedged(self):
        cases = (
            ({"gateway_state": "degraded", "exit_reason": "loop_liveness_watchdog"}, watchdog.DOWN),
            ({"gateway_state": "degraded", "exit_reason": "shutdown_watchdog"}, watchdog.DOWN),
            # Ctrl+C on a foreground run and a force-stopped hung gateway both leave "running":
            # an operator's stop, not evidence of a crash.
            ({"gateway_state": "running"}, watchdog.STOPPED),
            ({"gateway_state": "starting"}, watchdog.STOPPED),
            ({"gateway_state": "degraded", "exit_reason": "loop_liveness_watchdog", "desired_state": "stopped"},
             watchdog.STOPPED),
            ({"gateway_state": "stopped"}, watchdog.STOPPED),
            ({"gateway_state": "startup_failed"}, watchdog.STARTUP_FAILED),
            (None, watchdog.STOPPED),
        )
        for runtime, expected in cases:
            (verdict, pid), _ = self.classify(None, runtime=runtime)
            self.assertEqual((verdict, pid), (expected, None), runtime)


class RestartGatewayTest(_Home):
    def test_stops_the_wedged_pid_then_spawns_and_confirms_recovery(self):
        calls = []
        ok = watchdog.restart_gateway(self.home, 55, wait=lambda home: True,
                                      stop=lambda pid: calls.append(("stop", pid)) or True,
                                      spawn=lambda **kw: calls.append(("spawn", kw)) or 9, updating=lambda: False)
        self.assertEqual(ok, (True, "alive"))
        self.assertEqual(calls, [("stop", 55), ("spawn", {"home": self.home})])

    def test_never_spawns_a_second_gateway_beside_one_it_could_not_stop(self):
        spawn = mock.Mock()
        result = watchdog.restart_gateway(self.home, 55, wait=lambda home: True, stop=lambda pid: False, spawn=spawn,
                                          updating=lambda: False)
        self.assertEqual(result, (False, "stop_failed"))
        spawn.assert_not_called()

    def test_reports_a_gateway_that_does_not_come_back(self):
        stop = mock.Mock()
        result = watchdog.restart_gateway(self.home, None, wait=lambda home: False, stop=stop, spawn=mock.Mock(),
                                          updating=lambda: False)
        self.assertEqual(result, (False, "not_recovered"))
        stop.assert_not_called()  # nothing to stop when the gateway already vanished

    def test_an_update_starting_mid_restart_stops_the_restart_before_it_touches_anything(self):
        stop, spawn = mock.Mock(return_value=True), mock.Mock()
        result = watchdog.restart_gateway(self.home, 55, wait=lambda home: True, stop=stop, spawn=spawn,
                                          updating=lambda: True)
        self.assertEqual(result, (False, watchdog.REASON_UPDATE_IN_PROGRESS))
        stop.assert_not_called()
        spawn.assert_not_called()

        answers = iter([False, True])  # free before the stop, taken before the spawn
        result = watchdog.restart_gateway(self.home, 55, wait=lambda home: True, stop=stop, spawn=spawn,
                                          updating=lambda: next(answers))
        self.assertEqual(result, (False, watchdog.REASON_UPDATE_AFTER_STOP))
        stop.assert_called_once_with(55)
        spawn.assert_not_called()

    def test_a_stop_while_the_wedged_gateway_is_being_killed_prevents_the_start(self):
        spawn = mock.Mock()

        def stop(pid):
            watchdog.forget_owed_restart(self.home)  # operator's `gateway stop` lands mid-kill
            return True

        result = watchdog.restart_gateway(self.home, 55, wait=lambda home: True, stop=stop, spawn=spawn,
                                          updating=lambda: False)
        self.assertEqual(result, (False, watchdog.REASON_OPERATOR_STOPPED))
        spawn.assert_not_called()

    def test_a_stop_after_the_ticks_token_but_before_the_restart_call_still_counts(self):
        tick_token = watchdog.read_stop_token(self.home)
        watchdog.forget_owed_restart(self.home)  # landed after the tick read its token
        stop, spawn = mock.Mock(return_value=True), mock.Mock()
        result = watchdog.restart_gateway(self.home, 55, wait=lambda home: True, stop=stop, spawn=spawn,
                                          updating=lambda: False, stop_token=tick_token)
        self.assertEqual(result, (False, watchdog.REASON_OPERATOR_STOPPED))
        stop.assert_not_called()
        spawn.assert_not_called()

    def test_run_once_hands_its_tick_token_to_the_restart(self):
        watchdog.forget_owed_restart(self.home)
        token = watchdog.read_stop_token(self.home)
        restart, clock = mock.Mock(return_value=(True, "alive")), _Clock()
        for _ in range(3):
            watchdog.run_once(self.home, probe=_Probe(watchdog.WEDGED), restart=restart, clock=clock,
                              updating=lambda: False)
        self.assertEqual(restart.call_args.kwargs["stop_token"], token)

    def test_the_outage_notice_is_queued_right_before_the_start_and_only_then(self):
        calls = []
        outage = {"last_healthy_at": 100.0, "detected_at": 160.0}
        ok = watchdog.restart_gateway(self.home, 55, wait=lambda home: True, stop=lambda pid: True,
                                      spawn=lambda **kw: calls.append("spawn"), updating=lambda: False,
                                      outage=outage, announce=lambda home, record: calls.append(("notice", record)))
        self.assertEqual(ok, (True, "alive"))
        self.assertEqual(calls, [("notice", outage), "spawn"])

        announce = mock.Mock()
        watchdog.restart_gateway(self.home, 55, wait=lambda home: True, stop=lambda pid: False, spawn=mock.Mock(),
                                 updating=lambda: False, outage=outage, announce=announce)
        announce.assert_not_called()  # nothing was restarted: no claim that it was

    def test_a_stop_while_the_notice_is_queued_prevents_the_start(self):
        spawn = mock.Mock()

        def announce(home, record):
            watchdog.forget_owed_restart(home)  # operator's `gateway stop` lands while the notice is written

        result = watchdog.restart_gateway(self.home, 55, wait=lambda home: True, stop=lambda pid: True, spawn=spawn,
                                          updating=lambda: False, outage={"last_healthy_at": 1.0, "detected_at": 2.0},
                                          announce=announce)
        self.assertEqual(result, (False, watchdog.REASON_OPERATOR_STOPPED))
        spawn.assert_not_called()

    def test_a_stop_started_after_the_final_check_still_stops_the_spawned_gateway(self):
        """The interleaving the review named: the watchdog has passed its final stop check and is
        about to start the gateway when `hermes gateway stop` begins. The stop persists its intent,
        then waits for the spawn section; once it gets in, the new gateway is there for its sweep."""
        processes, events = [], []
        stop_finished = threading.Event()

        def run_operator_stop():
            def sweep():
                events.append(("sweep_sees", list(processes)))
                processes.clear()  # the stop's kill of every visible gateway

            watchdog.operator_stop(self.home, sweep)
            stop_finished.set()

        stopper = threading.Thread(target=run_operator_stop)

        def spawn(**_):
            # Past the final check, inside the spawn section: the operator's stop starts now.
            stopper.start()
            deadline = time.monotonic() + 10
            while watchdog.read_stop_token(self.home) == "":  # the stop has persisted its intent
                self.assertLess(time.monotonic(), deadline, "the stop never recorded its intent")
                time.sleep(0.01)
            self.assertFalse(stop_finished.is_set(), "the stop must wait for the spawn section")
            processes.append(4243)  # the new gateway becomes visible before the section ends
            events.append(("spawned", 4243))

        result = watchdog.restart_gateway(self.home, None, wait=lambda home: True, spawn=spawn,
                                          updating=lambda: False)
        stopper.join(timeout=30)
        self.assertTrue(stop_finished.is_set())
        self.assertEqual(result, (True, "alive"))
        self.assertEqual(events, [("spawned", 4243), ("sweep_sees", [4243])])
        self.assertEqual(processes, [], "operator stop always wins: the spawned gateway was stopped")

    def test_a_stop_holding_the_section_before_the_final_check_prevents_the_spawn(self):
        spawn, entered = mock.Mock(), threading.Event()
        release, result = threading.Event(), {}
        watchdog.forget_owed_restart(self.home)  # the stop persists its intent before it locks
        tick_token = ""  # the token the watchdog's tick started from

        def stop_holding_the_lock():
            with watchdog.spawn_lock(self.home) as held:
                self.assertTrue(held)
                entered.set()
                release.wait(timeout=30)

        stopper = threading.Thread(target=stop_holding_the_lock)
        stopper.start()
        entered.wait(timeout=30)
        worker = threading.Thread(target=lambda: result.update(r=watchdog.restart_gateway(
            self.home, None, wait=lambda home: True, spawn=spawn, updating=lambda: False, stop_token=tick_token)))
        worker.start()
        release.set()
        stopper.join(timeout=30)
        worker.join(timeout=30)
        self.assertEqual(result["r"], (False, watchdog.REASON_OPERATOR_STOPPED))
        spawn.assert_not_called()

    def test_a_spawn_section_that_cannot_be_locked_fails_closed(self):
        spawn = mock.Mock()
        with watchdog.spawn_lock(self.home) as held:  # someone else holds the section
            self.assertTrue(held)
            result = watchdog.restart_gateway(self.home, None, wait=lambda home: True, spawn=spawn,
                                              updating=lambda: False, lock_timeout=0.3)
        self.assertEqual(result, (False, watchdog.REASON_SPAWN_LOCK_UNAVAILABLE))
        spawn.assert_not_called()
        with mock.patch.object(watchdog, "_open_lock_file", side_effect=PermissionError(5, "Access is denied")):
            result = watchdog.restart_gateway(self.home, None, wait=lambda home: True, spawn=spawn,
                                              updating=lambda: False)
        self.assertEqual(result, (False, watchdog.REASON_SPAWN_LOCK_UNAVAILABLE))
        spawn.assert_not_called()

    def test_operator_stop_persists_its_intent_before_waiting_and_never_skips_its_sweep(self):
        sweep = mock.Mock()
        with watchdog.spawn_lock(self.home):  # a wedged holder: the stop must still stop
            watchdog.operator_stop(self.home, sweep, lock_timeout=0.3)
        self.assertNotEqual(watchdog.read_stop_token(self.home), "")
        sweep.assert_called_once()

    def test_a_failing_notice_never_blocks_the_restart_and_logs_no_detail(self):
        secret = "xoxb-SECRET-TOKEN-1234567890"
        spawn = mock.Mock()

        def announce(home, record):
            raise OSError(f"disk full writing token={secret}")

        ok = watchdog.restart_gateway(self.home, 55, wait=lambda home: True, stop=lambda pid: True, spawn=spawn,
                                      updating=lambda: False, outage={"last_healthy_at": 1.0, "detected_at": 2.0},
                                      announce=announce)
        self.assertEqual(ok, (True, "alive"))
        spawn.assert_called_once()
        text = (self.home / "logs" / "gateway-watchdog.log").read_text(encoding="utf-8")
        self.assertNotIn(secret, text)
        notice = [r for r in self.log_records() if r["event"] == "notice_failed"]
        self.assertEqual(notice[0]["reason"], "error:OSError")

    def test_the_default_notice_goes_to_the_gateways_planned_restart_marker(self):
        from gateway import watchdog_outage

        watchdog.announce_outage(self.home, {"last_healthy_at": 100.0, "detected_at": 160.0})
        data = json.loads((self.home / watchdog_outage.PLANNED_RESTART_MARKER_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(data[watchdog_outage.OUTAGE_KEY], {"last_healthy_at": 100.0, "detected_at": 160.0})

    def test_wait_for_recovery_polls_until_alive_or_deadline(self):
        ticks = iter(range(0, 1000, 10))
        verdicts = iter([watchdog.DOWN, watchdog.UNKNOWN, watchdog.ALIVE])
        ok = watchdog.wait_for_recovery(self.home, timeout=100, interval=10, sleep=lambda s: None,
                                        monotonic=lambda: next(ticks), probe=lambda home: (next(verdicts), None))
        self.assertTrue(ok)
        ticks = iter(range(0, 1000, 10))
        ok = watchdog.wait_for_recovery(self.home, timeout=30, interval=10, sleep=lambda s: None,
                                        monotonic=lambda: next(ticks), probe=lambda home: (watchdog.WEDGED, 1))
        self.assertFalse(ok)


class _FakeSchtasks:
    def __init__(self, registered=(), xml=None):
        self.calls, self.registered, self.xml, self.created_xml = [], set(registered), xml, {}

    def __call__(self, args, timeout=None):
        self.calls.append(list(args))
        name = args[args.index("/TN") + 1] if "/TN" in args else None
        if args[0] == "/Delete":
            self.registered.discard(name)
            return (0, "SUCCESS", "")
        if args[0] == "/Create":
            self.created_xml[name] = Path(args[args.index("/XML") + 1]).read_text(encoding="utf-16")
            self.registered.add(name)
            return (0, "SUCCESS", "")
        if args[0] == "/Run":
            return (0, "SUCCESS", "") if name in self.registered else (1, "", "ERROR: not found")
        if args[0] == "/Query":
            if name not in self.registered:
                return (1, "", "ERROR: The system cannot find the file specified.")
            return (0, self.xml or self.created_xml.get(name, ""), "") if "/XML" in args else (0, "ok", "")
        raise AssertionError(args)


class _TaskEnv(_Home):
    def setUp(self):
        super().setUp()
        self.schtasks = _FakeSchtasks()
        for name, value in (
            ("_assert_windows", lambda: None),
            ("get_task_name", lambda: "Hermes_Gateway_alice"),
            ("_hermes_home", lambda: self.home),
            ("_resolve_task_user", lambda: r"DOMAIN\alice"),
            ("_launcher_settings", lambda home=None: (r"C:\venv\Scripts\python.exe", str(self.home), str(self.home), "--profile alice")),
            ("_resolve_detached_python", lambda exe: (exe, Path(r"C:\venv"), [])),
            ("_exec_schtasks", self.schtasks),
        ):
            patcher = mock.patch.object(gateway_windows, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class TaskTest(_TaskEnv):
    def leaves(self, xml):
        return gateway_windows._task_xml_leaf_values(xml)

    def test_task_xml_runs_every_minute_from_registration_and_from_logon_through_a_hidden_launcher(self):
        xml = watchdog.build_task_xml(Path(r"C:\h\gateway-service\Hermes_Watchdog_alice.vbs"), r"DOMAIN\alice",
                                      start_boundary="2026-11-02T10:00:00")
        ElementTree.fromstring(xml)
        leaves = self.leaves(xml)
        # A LogonTrigger only fires at the NEXT logon; the TimeTrigger starts probing at install.
        for trigger in ("TimeTrigger", "LogonTrigger"):
            self.assertEqual(leaves[f"Task/Triggers/{trigger}/Enabled"], "true")
            self.assertEqual(leaves[f"Task/Triggers/{trigger}/Repetition/Interval"], "PT1M")
            self.assertEqual(leaves[f"Task/Triggers/{trigger}/Repetition/StopAtDurationEnd"], "false")
        self.assertEqual(leaves["Task/Triggers/TimeTrigger/StartBoundary"], "2026-11-02T10:00:00")
        self.assertEqual(leaves["Task/Settings/MultipleInstancesPolicy"], "IgnoreNew")
        self.assertEqual(leaves["Task/Settings/Enabled"], "true")
        self.assertEqual(leaves["Task/Settings/StartWhenAvailable"], "true")
        self.assertEqual(leaves["Task/Actions/Exec/Command"], "wscript.exe")
        self.assertIn("Hermes_Watchdog_alice.vbs", leaves["Task/Actions/Exec/Arguments"])
        self.assertNotIn("cmd.exe", xml)

    def test_install_registers_a_task_outside_the_gateway_task_namespace(self):
        ok, detail = watchdog.install_task()
        self.assertTrue(ok, detail)
        self.assertEqual(watchdog.task_name(), "Hermes_Watchdog_alice")
        self.assertIn("Hermes_Watchdog_alice", self.schtasks.created_xml)
        self.assertFalse(watchdog.task_name().startswith(gateway_windows._TASK_NAME_DEFAULT))
        launcher = self.home / "gateway-service" / "Hermes_Watchdog_alice.vbs"
        text = launcher.read_text(encoding="utf-16")
        self.assertIn("hermes_cli.gateway_windows_watchdog --once", text)
        self.assertIn(", 0, True", text)  # waits: IgnoreNew then serialises overlapping probes
        self.assertNotIn("gateway run", text)
        self.assertIn(str(launcher), self.schtasks.created_xml["Hermes_Watchdog_alice"])

    def test_launcher_runs_the_watchdog_through_the_managed_launcher_for_the_store_interpreter(self):
        """The bare store interpreter cannot import the watchdog's dependencies (``ruamel``): a managed
        install's launcher must be ``.hermes/bin/hermes --run-module <watchdog> --once``."""
        import subprocess

        from hermes_cli import _launchers, gateway

        account = self.home / "齋藤 厚作リープアローズ"
        root = account / "hermes-agent"
        store_python = account / "tools" / "python-3.14.7+20260901-win32-x64" / "python.exe"
        store_python.parent.mkdir(parents=True)
        store_python.write_text("", encoding="utf-8")
        launcher = root / ".hermes" / "bin" / "hermes.exe"

        def publish():
            launcher.parent.mkdir(parents=True, exist_ok=True)
            launcher.write_text("", encoding="utf-8")

        for target, name, value in (
            (gateway, "PROJECT_ROOT", root),
            (gateway, "_prepare_service_launcher", publish),
            (_launchers, "resolve_store_python",
             lambda repo_root, publication=False: store_python if Path(repo_root) == root else None),
            (gateway_windows, "_launcher_settings",
             lambda home=None: (str(store_python), str(self.home), str(self.home), "--profile alice")),
        ):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        text = watchdog.write_launcher().read_text(encoding="utf-16")

        (run_line,) = [line for line in text.splitlines() if line.startswith("sh.Run ")]
        literal, wait = run_line[len("sh.Run "):].rsplit(", 0, ", 1)
        self.assertEqual(wait, "True")
        self.assertEqual(literal[1:-1].replace('""', '"'), subprocess.list2cmdline(
            [str(launcher), "--run-module", "hermes_cli.gateway_windows_watchdog", "--once"]))
        self.assertNotIn(str(store_python), text)

    def test_uninstall_deletes_only_a_registered_task_and_removes_the_launcher(self):
        self.assertEqual(watchdog.uninstall_task(), [])
        self.assertFalse([c for c in self.schtasks.calls if c[0] == "/Delete"])
        watchdog.install_task()
        messages = watchdog.uninstall_task()
        self.assertIn(["/Delete", "/F", "/TN", "Hermes_Watchdog_alice"], self.schtasks.calls)
        self.assertFalse((self.home / "gateway-service" / "Hermes_Watchdog_alice.vbs").exists())
        self.assertTrue(any("Removed gateway watchdog" in m for m in messages))

    def test_update_refresh_never_enables_and_repairs_only_a_drifted_registration(self):
        self.assertEqual(watchdog.refresh_if_installed(), [])
        self.assertFalse([c for c in self.schtasks.calls if c[0] == "/Create"])

        watchdog.install_task()
        creates = len([c for c in self.schtasks.calls if c[0] == "/Create"])
        watchdog.refresh_if_installed()
        self.assertEqual(len([c for c in self.schtasks.calls if c[0] == "/Create"]), creates)

        # A registration from another minute is not drift (StartBoundary is the install time).
        self.schtasks.xml = re.sub(r"<StartBoundary>[^<]*</StartBoundary>", "<StartBoundary>2020-01-01T00:00:00</StartBoundary>",
                                   self.schtasks.created_xml["Hermes_Watchdog_alice"])
        watchdog.refresh_if_installed()
        self.assertEqual(len([c for c in self.schtasks.calls if c[0] == "/Create"]), creates)

        # A pre-TimeTrigger registration only ever probed after the next logon: repaired.
        self.schtasks.xml = re.sub(r"<TimeTrigger>.*?</TimeTrigger>", "", self.schtasks.created_xml["Hermes_Watchdog_alice"],
                                   flags=re.DOTALL)
        messages = watchdog.refresh_if_installed()
        self.assertEqual(len([c for c in self.schtasks.calls if c[0] == "/Create"]), creates + 1)
        self.assertTrue(any("watchdog" in m for m in messages))

    def test_uninstall_never_creates_the_service_directory(self):
        watchdog.uninstall_task()
        self.assertFalse((self.home / "gateway-service").exists())

    def test_restart_starts_through_the_gateway_task_when_one_is_registered(self):
        self.schtasks.registered.add("Hermes_Gateway_alice")
        with mock.patch.object(gateway_windows, "_spawn_detached") as spawn, \
                mock.patch.object(gateway_windows, "_gateway_pids", return_value=[77]):
            watchdog.start_gateway(home=self.home)
        self.assertIn(["/Run", "/TN", "Hermes_Gateway_alice"], self.schtasks.calls)
        spawn.assert_not_called()

        self.schtasks.registered.clear()
        with mock.patch.object(gateway_windows, "_spawn_detached") as spawn, \
                mock.patch.object(gateway_windows, "_gateway_pids", return_value=[77]):
            watchdog.start_gateway(home=self.home)
        spawn.assert_called_once_with(home=self.home)

    def test_start_returns_only_once_the_new_gateway_is_visible_to_a_stop_sweep(self):
        """The spawn section ends when a stop's PID sweep (``_gateway_pids``) can see the gateway."""
        ticks = iter(range(1000))
        scans = iter([[], [], [88]])
        with mock.patch.object(gateway_windows, "_spawn_detached"), \
                mock.patch.object(gateway_windows, "_gateway_pids", side_effect=lambda: next(scans)) as pids:
            self.assertTrue(watchdog.start_gateway(home=self.home, sleep=lambda s: None, monotonic=lambda: next(ticks)))
        self.assertEqual(pids.call_count, 3)
        ticks = iter(range(0, 1000, 30))
        with mock.patch.object(gateway_windows, "_spawn_detached"), \
                mock.patch.object(gateway_windows, "_gateway_pids", return_value=[]):
            self.assertFalse(watchdog.start_gateway(home=self.home, sleep=lambda s: None, monotonic=lambda: next(ticks)))


class GatewayServiceIntegrationTest(_TaskEnv):
    def test_gateway_install_registers_the_watchdog_after_the_gateway_task(self):
        order = []
        for name, value in (
            ("_prompt_install_choices", lambda *a, **k: (False, True)),
            ("_write_task_script", lambda: self.home / "gateway-service" / "Hermes_Gateway_alice.cmd"),
            ("_is_running_as_admin", lambda: True),
            ("_install_scheduled_task", lambda name, path, task_xml=None: order.append(name) or (True, "created")),
            ("_remove_startup_entries", lambda: ([], [])),
            ("_startup_staging_path", lambda: self.home / "staging.tmp"),
            ("_print_next_steps", lambda: None),
        ):
            patcher = mock.patch.object(gateway_windows, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        with mock.patch.object(watchdog, "install_task", side_effect=lambda: order.append("watchdog") or (True, "ok")), \
                mock.patch("builtins.print"):
            gateway_windows.install()
        self.assertEqual(order, ["Hermes_Gateway_alice", "watchdog"])

    def test_gateway_uninstall_removes_the_watchdog(self):
        startup = self.home / "Startup"
        for name, value in (
            ("get_task_script_path", lambda: self.home / "gateway-service" / "Hermes_Gateway_alice.cmd"),
            ("get_startup_entry_path", lambda: startup / "Hermes_Gateway_alice.vbs"),
            ("_legacy_startup_entry_path", lambda: startup / "Hermes_Gateway_alice.cmd"),
            ("_startup_dir", lambda: startup),
        ):
            patcher = mock.patch.object(gateway_windows, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        watchdog.install_task()
        with mock.patch("builtins.print"):
            gateway_windows.uninstall()
        self.assertNotIn("Hermes_Watchdog_alice", self.schtasks.registered)

    def test_profile_delete_cleanup_removes_the_watchdog_without_elevating(self):
        from hermes_cli import profiles_service_cleanup

        startup = self.home / "Startup"
        for name, value in (
            ("get_startup_entry_path", lambda: startup / "Hermes_Gateway_alice.vbs"),
            ("_legacy_startup_entry_path", lambda: startup / "Hermes_Gateway_alice.cmd"),
            ("_launch_elevated_gateway_command", mock.Mock(side_effect=AssertionError("must not elevate"))),
        ):
            patcher = mock.patch.object(gateway_windows, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        watchdog.install_task()
        with mock.patch("builtins.print"):
            self.assertTrue(profiles_service_cleanup.remove_windows_task())
        self.assertNotIn("Hermes_Watchdog_alice", self.schtasks.registered)


if __name__ == "__main__":
    unittest.main()
