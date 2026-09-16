"""Tests for the surface-agnostic voice supervisor controller.

The controller owns the consult/steer lifecycle between a realtime voice
session and any surface (CLI, Discord VC, browser). Faked session + runner —
no audio, no network, no HermesCLI.
"""

import json
import time
from unittest.mock import MagicMock

from agent.voice_supervisor import (
    MAX_CONSULT_OUTPUT_CHARS,
    VoiceSupervisorController,
)


class FakeSession:
    def __init__(self):
        self.alive = True
        self.last_response_had_audio = False
        self.outputs = []       # (call_id, output)
        self.acks = 0
        self.verbatim = []

    def send_function_output(self, call_id, output):
        self.outputs.append((call_id, output))

    def speak_acknowledgment(self):
        self.acks += 1

    def speak_verbatim(self, text, interruptible=True):
        self.verbatim.append(text)


class FakeRunner:
    def __init__(self):
        self.submitted = []
        self.interrupts = 0
        self.ops = []
        self.busy = False
        self.queue_empty = True
        self.accept = True

    def submit(self, task):
        self.submitted.append(task)
        self.ops.append(("submit", task))
        return self.accept

    def interrupt(self):
        self.interrupts += 1
        self.ops.append(("interrupt",))

    def is_busy(self):
        return self.busy

    def is_queue_empty(self):
        return self.queue_empty


def _make(narrate=True):
    session, runner = FakeSession(), FakeRunner()
    events = []
    ctrl = VoiceSupervisorController(
        session, runner, narrate=narrate,
        on_event=lambda kind, text: events.append((kind, text)),
    )
    return ctrl, session, runner, events


def _consult(ctrl, call_id="c1", task="check disk usage"):
    ctrl.on_function_call("consult_hermes", call_id, json.dumps({"task": task}))


class TestConsult:
    def test_accepted_consult_submits_task_and_emits_event(self):
        ctrl, session, runner, events = _make()
        session.last_response_had_audio = True  # model spoke its own filler
        _consult(ctrl)
        assert runner.submitted == ["check disk usage"]
        assert events == [("consult", "check disk usage")]
        assert ctrl.consult_active
        assert session.outputs == []
        assert session.acks == 0

    def test_silent_tool_call_gets_instant_acknowledgment(self):
        ctrl, session, runner, _ = _make()
        session.last_response_had_audio = False
        _consult(ctrl)
        assert session.acks == 1

    def test_busy_consult_is_rejected_politely(self):
        ctrl, session, runner, _ = _make()
        runner.busy = True
        _consult(ctrl, "c1", "first")
        _consult(ctrl, "c2", "second")
        assert runner.submitted == ["first"]
        assert session.outputs[-1][0] == "c2"
        assert "still working" in session.outputs[-1][1]
        assert "NOT queued" in session.outputs[-1][1]
        assert "steer_hermes" in session.outputs[-1][1]

    def test_stale_dead_consult_is_failed_out_and_replaced(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "first")
        ctrl._consult["at"] = time.monotonic() - 999  # turn died silently
        _consult(ctrl, "c2", "second")
        assert ("c1", "That task failed without producing a result.") in session.outputs
        assert ctrl._consult["call_id"] == "c2"
        assert runner.submitted == ["first", "second"]

    def test_old_consult_not_stale_while_runner_busy_or_queued(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "first")
        ctrl._consult["at"] = time.monotonic() - 999
        runner.busy = True
        assert ctrl._take_stale_consult() is None
        runner.busy = False
        runner.queue_empty = False
        assert ctrl._take_stale_consult() is None

    def test_unknown_tool_and_empty_task_are_answered(self):
        ctrl, session, runner, _ = _make()
        ctrl.on_function_call("mystery", "c1", "{}")
        assert "Unknown tool" in session.outputs[-1][1]
        ctrl.on_function_call("consult_hermes", "c2", "not-json")
        assert session.outputs[-1][0] == "c2"
        assert runner.submitted == []


class TestSteer:
    def test_steer_retargets_interrupts_and_confirms(self):
        ctrl, session, runner, events = _make()
        _consult(ctrl, "c1", "original")
        runner.busy = True
        ctrl.on_function_call(
            "steer_hermes", "s1", json.dumps({"instruction": "also check logs"})
        )
        assert ctrl._consult["task"] == "also check logs"
        assert ctrl._consult["call_id"] == "c1"  # result still answers c1
        assert runner.submitted == ["original", "also check logs"]
        assert runner.interrupts == 1
        # Interrupt must precede the steered submit (avoids cancelling it).
        assert runner.ops[-2:] == [
            ("interrupt",),
            ("submit", "also check logs"),
        ]
        assert session.outputs[-1][0] == "s1"
        assert "Steering applied" in session.outputs[-1][1]
        assert ("steer", "also check logs") in events

    def test_steer_without_consult_reports_nothing_to_steer(self):
        ctrl, session, runner, _ = _make()
        ctrl.on_function_call(
            "steer_hermes", "s1", json.dumps({"instruction": "faster"})
        )
        assert "No Hermes task is running" in session.outputs[-1][1]
        assert runner.submitted == []

    def test_steer_idle_runner_skips_interrupt(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "original")
        runner.busy = False
        ctrl.on_function_call(
            "steer_hermes", "s1", json.dumps({"instruction": "more"})
        )
        assert runner.interrupts == 0
        assert runner.submitted == ["original", "more"]

    def test_failed_steer_keeps_original_task(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "original")
        runner.accept = False
        ctrl.on_function_call(
            "steer_hermes", "s1", json.dumps({"instruction": "nope"})
        )
        assert ctrl._consult["task"] == "original"
        assert "Steering failed" in session.outputs[-1][1]
        assert ctrl.on_turn_complete("original", "done") is True

    def test_presteer_interrupted_partial_is_not_consumed(self):
        # A steer interrupts the running turn, which still completes with the
        # OLD task text and interrupted=True. That partial must never be
        # reported as the consult result — the resubmitted turn owns it.
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "original")
        ctrl.on_function_call(
            "steer_hermes", "s1", json.dumps({"instruction": "also check logs"})
        )
        assert (
            ctrl.on_turn_complete("original", "partial output", interrupted=True)
            is False
        )
        assert ctrl.consult_active
        assert ctrl.on_turn_complete("also check logs", "real answer") is True
        assert session.outputs[-1] == ("c1", "real answer")

    def test_steered_result_drained_under_original_text_completes(self):
        # The gateway drains the steer follow-up inside the interrupted turn
        # and reports its (non-interrupted) result under the ORIGINAL event
        # text. Both texts belong to the consult, so the answer reaches grok.
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "original")
        ctrl.on_function_call(
            "steer_hermes", "s1", json.dumps({"instruction": "also check logs"})
        )
        assert ctrl.owns_turn("original") and ctrl.owns_turn("also check logs")
        assert ctrl.on_turn_complete("original", "steered answer") is True
        assert session.outputs[-1] == ("c1", "steered answer")
        assert not ctrl.consult_active

    def test_failed_steer_forgets_the_instruction_text(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "original")
        runner.accept = False
        ctrl.on_function_call(
            "steer_hermes", "s1", json.dumps({"instruction": "also check logs"})
        )
        assert ctrl.owns_turn("original")
        assert not ctrl.owns_turn("also check logs")


class TestTurnComplete:
    def test_matching_turn_returns_result_and_clears(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "check disk usage")
        assert ctrl.on_turn_complete("check disk usage", "42% full") is True
        assert not ctrl.consult_active
        assert session.outputs[-1] == ("c1", "42% full")

    def test_unrelated_turn_is_ignored(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "check disk usage")
        assert ctrl.on_turn_complete("typed message", "reply") is False
        assert ctrl.consult_active

    def test_merged_turn_containing_task_is_consumed(self):
        # Gateways may coalesce a queued consult with a steering instruction
        # into one turn; first-line equality still routes the result.
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "check disk usage")
        merged = "check disk usage\n\nalso include inode usage"
        assert ctrl.on_turn_complete(merged, "all good") is True
        assert session.outputs[-1] == ("c1", "all good")

    def test_substring_does_not_own_turn(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "check disk usage")
        assert ctrl.owns_turn("please check disk usage now") is False
        assert ctrl.on_turn_complete("please check disk usage now", "nope") is False
        assert ctrl.consult_active

    def test_rejected_submit_does_not_track_consult(self):
        ctrl, session, runner, _ = _make()
        runner.accept = False
        _consult(ctrl)
        assert not ctrl.consult_active
        assert "Could not start" in session.outputs[-1][1]

    def test_fail_active_consult_sends_output_and_clears(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "task")
        ctrl.fail_active_consult("Voice session reconnected; the previous task was dropped.")
        assert not ctrl.consult_active
        assert session.outputs[-1][0] == "c1"
        assert "dropped" in session.outputs[-1][1]

    def test_owns_turn_matches_without_consuming(self):
        # Surfaces use owns_turn to silence classic TTS paths (including
        # streaming TTS, which starts before completion) — it must never
        # mutate consult state.
        ctrl, session, runner, _ = _make()
        assert ctrl.owns_turn("anything") is False  # no consult yet
        _consult(ctrl, "c1", "check disk usage")
        assert ctrl.owns_turn("check disk usage") is True
        assert ctrl.owns_turn("check disk usage\n\nand inodes") is True
        assert ctrl.owns_turn("unrelated typed message") is False
        assert ctrl.owns_turn(None) is False
        assert ctrl.consult_active  # unchanged by any of the above

    def test_no_consult_is_noop(self):
        ctrl, _, _, _ = _make()
        assert ctrl.on_turn_complete("anything", "reply") is False

    def test_dead_session_swallows_result(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "task")
        session.alive = False
        before = list(session.outputs)
        assert ctrl.on_turn_complete("task", "reply") is False
        assert session.outputs == before

    def test_long_output_is_truncated(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "task")
        ctrl.on_turn_complete("task", "x" * (MAX_CONSULT_OUTPUT_CHARS + 500))
        out = session.outputs[-1][1]
        assert len(out) < MAX_CONSULT_OUTPUT_CHARS + 100
        assert "truncated" in out
        assert "session history" in out
        assert "user's screen" not in out

    def test_empty_response_reports_no_output(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl, "c1", "task")
        ctrl.on_turn_complete("task", "   ")
        assert "no text output" in session.outputs[-1][1]


class TestNarrationAndNotify:
    def test_narrate_throttles_and_requires_consult(self):
        ctrl, session, runner, _ = _make()
        ctrl.narrate_tool("run_terminal_cmd")
        assert session.verbatim == []  # no consult yet
        _consult(ctrl)
        ctrl.narrate_tool("run_terminal_cmd")
        assert session.verbatim == ["Running run terminal cmd."]
        ctrl.narrate_tool("read_file")  # inside throttle window
        assert len(session.verbatim) == 1

    def test_narrate_disabled_by_flag(self):
        ctrl, session, runner, _ = _make(narrate=False)
        _consult(ctrl)
        ctrl.narrate_tool("run_terminal_cmd")
        assert session.verbatim == []

    def test_notify_speaks_only_during_consult(self):
        ctrl, session, runner, _ = _make()
        ctrl.notify("Hermes needs your approval.")
        assert session.verbatim == []
        _consult(ctrl)
        ctrl.notify("Hermes needs your approval.")
        assert session.verbatim[-1] == "Hermes needs your approval."

    def test_reset_drops_consult(self):
        ctrl, session, runner, _ = _make()
        _consult(ctrl)
        ctrl.reset()
        assert not ctrl.consult_active

    def test_callback_exceptions_never_propagate(self):
        session, runner = FakeSession(), FakeRunner()
        ctrl = VoiceSupervisorController(
            session, runner,
            on_event=MagicMock(side_effect=RuntimeError("ui died")),
        )
        _consult(ctrl)  # must not raise
        assert ctrl.consult_active
