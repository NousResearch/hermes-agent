"""Activity-aware turn lifetime — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
)
from tests.agent.claude_sdk_fakes import (
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    _text_delta_event,
    ResultMessage,
    _FakeClient,
    _plant_claude_agent_sdk_stand_in,
    _make_hold_open_session,
    _ede_interrupt_ack,
    _wait_for_client,
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


# ---------- activity-aware turn lifetime (the 600s wall-clock fix) ----------
# Production forensics (24/7 gateway deployment, six "turn timed out after
# 600s" retires 2026-07-22 → 2026-08-09): four of six were ACTIVELY-WORKING
# turns — tool loops mid-execution, human approval taps counted as silence —
# killed by a hard wall clock over the whole turn. The lifetime is now evidence-based:
# outstanding tools and pending approvals suspend the rules; the budget only
# fires on a turn that is ALSO quiet; a post-tool quiet watchdog catches
# wedges early; a tripped turn that the CLI acks cleanly keeps its partial
# transcript and resume id instead of retiring.

class TestTurnLifetime:
    def test_active_turn_survives_past_turn_timeout(self):
        # RED on the pre-fix tree: the old hard wall clock kills this turn at
        # 0.5s with "turn timed out after 0s" even though tool results are
        # landing every 50ms. GREEN: activity extends the turn to completion.
        session, holder = _make_hold_open_session(script=[])
        stop = threading.Event()

        def feeder():
            client = _wait_for_client(holder)
            for i in range(18):  # ~0.9s of beats at 50ms, > 0.5s budget
                if stop.is_set():
                    return
                time.sleep(0.05)
                client.feed(
                    AssistantMessage(
                        content=[ToolUseBlock(id=f"t{i}", name="Read", input={})]
                    ),
                    UserMessage(
                        content=[ToolResultBlock(tool_use_id=f"t{i}", content="ok")]
                    ),
                )
            client.feed(
                AssistantMessage(content=[TextBlock("long job done")]),
                ResultMessage(result="long job done", uuid="uuid-long"),
            )

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        try:
            turn = session.run_turn(
                "big task",
                turn_timeout=0.5,
                post_tool_quiet_timeout=0.0,
                watch_poll_interval=0.02,
            )
        finally:
            stop.set()
            thread.join(timeout=5)
            session.close()
        assert turn.error is None
        assert turn.should_retire is False
        assert turn.final_text == "long job done"

    def test_outstanding_tool_suspends_budget(self):
        # A single long-running tool emits NOTHING on the stream. The issued
        # ToolUseBlock keeps the turn suspended past the budget until its
        # result lands. RED on the pre-fix tree (dies at 0.3s).
        session, holder = _make_hold_open_session(
            script=[
                AssistantMessage(
                    content=[ToolUseBlock(id="slow", name="Bash", input={})]
                ),
            ]
        )

        def feeder():
            client = _wait_for_client(holder)
            time.sleep(0.9)  # 3x the budget, tool still "running"
            client.feed(
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="slow", content="done")]
                ),
                AssistantMessage(content=[TextBlock("tool finished")]),
                ResultMessage(result="tool finished", uuid="uuid-slow"),
            )

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        try:
            turn = session.run_turn(
                "run the slow tool",
                turn_timeout=0.3,
                post_tool_quiet_timeout=0.0,
                watch_poll_interval=0.02,
            )
        finally:
            thread.join(timeout=5)
            session.close()
        assert turn.error is None
        assert turn.final_text == "tool finished"

    def test_post_tool_quiet_trips_on_wedge_clean_ack(self):
        # Wedge signature: a tool result lands, then the stream goes silent
        # (alive, no _EOS). The quiet watchdog trips fast, the CLI acks the
        # interrupt with the EDE shape — and the clean ack preserves the
        # partial transcript AND the resume id (no retire), while the trip
        # text WINS over the masked EDE ack (never "SDK result error ...").
        session, holder = _make_hold_open_session(
            script=[
                SystemMessage(session_id="sdk-wedge-1"),
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Grep", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="hits")]
                ),
            ],
            interrupt_ack=[_ede_interrupt_ack()],
        )
        try:
            turn = session.run_turn(
                "search",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.2,
                watch_poll_interval=0.05,
            )
        finally:
            session.close()
        assert turn.error is not None
        assert "turn timed out" in turn.error
        assert "after a tool result" in turn.error
        assert "SDK result error" not in turn.error  # W22 mask + trip wins
        assert turn.interrupted is True
        # should_retire=False is the load-bearing assertion: the runtime
        # persists the resume id ONLY for non-retiring turns. thread_id just
        # has to be present for it to persist (its exact value tracks the
        # last session_id-bearing message — here the ack's own default).
        assert turn.should_retire is False  # clean ack — resumable
        assert turn.thread_id
        assert holder["client"].interrupted is True
        # Partial transcript survived the trip.
        roles = [m["role"] for m in turn.projected_messages]
        assert "assistant" in roles and "tool" in roles

    def test_post_tool_watchdog_resets_on_activity(self):
        # Assistant output after the tool result DISARMS the quiet watchdog
        # (codex-parity reset semantics). The post-disarm silence here (0.7s)
        # EXCEEDS the quiet limit (0.25s) — with the disarm neutered this
        # turn trips; with it, the turn completes untouched.
        session, holder = _make_hold_open_session(script=[])

        def feeder():
            client = _wait_for_client(holder)
            client.feed(
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Read", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="data")]
                ),
            )
            time.sleep(0.1)  # armed, under the limit
            client.feed(AssistantMessage(content=[TextBlock("thinking done")]))
            time.sleep(0.7)  # SILENCE past the limit — trips iff still armed
            client.feed(ResultMessage(result="thinking done", uuid="uuid-r"))

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        try:
            turn = session.run_turn(
                "go",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.25,
                watch_poll_interval=0.02,
            )
        finally:
            thread.join(timeout=5)
            session.close()
        assert turn.error is None
        assert turn.final_text == "thinking done"
        # The load-bearing pin: a DISARMED watchdog never fires an interrupt
        # at all. (Without this, a broken disarm can hide behind the
        # completion-during-grace rescue, which still delivers the pre-trip
        # text — resilient, but the needless interrupt+reconnect cycle is
        # exactly what the disarm exists to avoid.)
        assert holder["client"].interrupted is False

    def test_stream_deltas_disarm_quiet_watchdog(self):
        # Streaming posture — the only posture where the quiet watchdog
        # defaults ON: partial deltas after a tool result prove the model
        # call is alive and DISARM the watchdog. The post-delta silence
        # (0.7s) exceeds the quiet limit (0.25s) — trips iff the StreamEvent
        # branch's disarm is broken.
        session, holder = _make_hold_open_session(script=[])
        session._streaming = True  # the __init__ snapshot, forced for the test

        def feeder():
            client = _wait_for_client(holder)
            client.feed(
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Bash", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="ok")]
                ),
            )
            for _ in range(3):
                time.sleep(0.05)
                client.feed(_text_delta_event("chunk "))
            time.sleep(0.7)  # silence past the limit — armed would trip
            client.feed(
                AssistantMessage(content=[TextBlock("streamed answer")]),
                ResultMessage(result="streamed answer", uuid="uuid-sd"),
            )

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        try:
            turn = session.run_turn(
                "stream it",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.25,
                watch_poll_interval=0.02,
            )
        finally:
            thread.join(timeout=5)
            session.close()
        assert turn.error is None
        assert turn.final_text == "streamed answer"
        assert holder["client"].interrupted is False  # watchdog never fired

    def test_hard_trip_retires_and_clears_interrupt_event(self):
        # The CLI ignores the interrupt for the whole grace: hard-cancel,
        # retire (today's shape, now the rare fallback) — and the interrupt
        # event must NOT leak into the next turn on this session object.
        # RED on the pre-fix tree: the old timeout branch left the event set.
        session, holder = _make_hold_open_session(script=[])  # total silence
        try:
            turn = session.run_turn(
                "hello?",
                turn_timeout=0.2,
                post_tool_quiet_timeout=0.0,
                watch_poll_interval=0.05,
                abort_grace=0.2,
            )
            assert turn.error is not None
            assert "turn timed out after" in turn.error
            assert turn.should_retire is True
            assert turn.interrupted is True
            assert holder["client"].interrupted is True
            assert session._interrupt_event.is_set() is False
        finally:
            session.close()

    def test_completion_during_grace_delivered_in_full(self):
        # The answer text streamed BEFORE the trip; the success ack's uuid
        # then completes the turn inside the grace. Completion wins: no trip
        # error, no retire, the PRE-TRIP text is delivered. (The ack's own
        # result= text is never projected — projection stops at the
        # interrupt — so final_text here is the earlier AssistantMessage's.)
        session, holder = _make_hold_open_session(
            script=[
                AssistantMessage(content=[TextBlock("the answer")]),
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Bash", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="ok")]
                ),
            ],
            interrupt_ack=[
                ResultMessage(result=None, uuid="uuid-late")
            ],
        )
        try:
            turn = session.run_turn(
                "answer then wedge",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.2,
                watch_poll_interval=0.05,
            )
        finally:
            session.close()
        assert turn.error is None
        assert turn.should_retire is False
        assert turn.final_text == "the answer"

    def test_success_ack_without_prior_text_stays_a_trip(self):
        # Negative control for the completion lane's final_text gate: a
        # SUCCESS ack with a uuid but NO answer text anywhere (pure
        # tool-work turn) must remain a trip — voiding it here would turn
        # the timeout into a silent empty delivery.
        session, holder = _make_hold_open_session(
            script=[
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Bash", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="ok")]
                ),
            ],
            interrupt_ack=[
                ResultMessage(result=None, uuid="uuid-empty")
            ],
        )
        try:
            turn = session.run_turn(
                "tool work then wedge",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.2,
                watch_poll_interval=0.05,
            )
        finally:
            session.close()
        assert turn.error is not None
        assert "turn timed out" in turn.error
        assert turn.interrupted is True
        assert turn.should_retire is False  # clean ack still preserves resume

    def test_coroutine_timeout_error_classified_not_spun(self):
        # py3.11 unifies the TimeoutError family: a TimeoutError RAISED BY
        # the turn coroutine (socket/pipe timeout under the CLI) must be
        # classified as a turn failure — the pre-fix tree misread it as the
        # turn hitting its own 600s wall ("turn timed out after 600s").
        class TimeoutRaisingClient(_FakeClient):
            async def query(self, text):
                raise TimeoutError("socket write timed out")

        holder = {}

        def factory(options=None):
            holder["client"] = TimeoutRaisingClient(options=options)
            return holder["client"]

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        try:
            turn = session.run_turn("hi", watch_poll_interval=0.05)
        finally:
            session.close()
        assert turn.error is not None
        assert "claude-agent-sdk turn failed" in turn.error
        assert "socket write timed out" in turn.error
        assert "turn timed out after" not in turn.error

    def test_pending_approval_suspends_watchdogs(self, monkeypatch):
        # A human being asked is not the turn being silent: while the
        # session's own can_use_tool bridge awaits the approval callback,
        # both rules stand down — the quiet watchdog (armed by the tool
        # result) must NOT trip during a 0.5s approval on a 0.15s quiet
        # limit. After the tap the turn completes.
        _plant_claude_agent_sdk_stand_in(monkeypatch)
        released = threading.Event()

        def slow_approval(preview, prompt, **kwargs):
            time.sleep(0.5)
            released.set()
            return "once"

        session, holder = _make_hold_open_session(
            script=[
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Write", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="ok")]
                ),
            ],
            approval_callback=slow_approval,
            permission_mode="default",
        )

        def feeder():
            # The approval prompt fires on the session loop while the turn
            # is in flight — the exact production shape (SDK invoking
            # can_use_tool mid-turn). Event-based sync: wait until the tool
            # result actually ARMED the watchdog, then fire the approval
            # immediately (a sleep here left a ~36ms margin against the
            # quiet limit — flake fuel on loaded CI).
            _wait_for_client(holder)
            deadline = time.monotonic() + 5
            while True:
                watch = session._turn_watch
                if watch is not None and watch.post_tool_armed:
                    break
                if time.monotonic() > deadline:
                    raise AssertionError("watchdog never armed")
                time.sleep(0.01)
            cb = session._make_can_use_tool()
            fut = asyncio.run_coroutine_threadsafe(
                cb("Write", {"file_path": "/x"}, SimpleNamespace(tool_use_id="t1")),
                session._loop,
            )
            fut.result(timeout=5)
            holder["client"].feed(
                AssistantMessage(content=[TextBlock("written")]),
                ResultMessage(result="written", uuid="uuid-appr"),
            )

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        try:
            turn = session.run_turn(
                "write it",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.15,
                watch_poll_interval=0.02,
            )
        finally:
            thread.join(timeout=5)
            session.close()
        assert released.is_set()  # the approval really took 0.5s
        assert turn.error is None
        assert turn.final_text == "written"

    def test_orphaned_approval_decrements_its_own_watch(self, monkeypatch):
        # The F9 shape: an approval wait that OUTLIVES its turn must
        # decrement the watch it suspended — never a later turn's. The
        # callback captures the watch object at entry; a stale decrement
        # lands on the dead watch.
        _plant_claude_agent_sdk_stand_in(monkeypatch)
        from agent.transports import claude_agent_sdk_session_watchdog as session_mod

        release = threading.Event()

        def blocking_cb(preview, prompt, **kwargs):
            release.wait(5)
            return "once"

        session, holder = _make_hold_open_session(
            script=[], approval_callback=blocking_cb,
            permission_mode="default",
        )
        try:
            session.ensure_started()
            w1 = session_mod._TurnWatch()
            session._turn_watch = w1
            cb = session._make_can_use_tool()
            fut = asyncio.run_coroutine_threadsafe(
                cb("Bash", {"command": "true"}, SimpleNamespace(tool_use_id="t1")),
                session._loop,
            )
            deadline = time.monotonic() + 5
            while w1.approvals_pending != 1:
                if time.monotonic() > deadline:
                    raise AssertionError("approval never registered on w1")
                time.sleep(0.01)
            # Turn 1 ends; turn 2 installs a fresh watch while the approval
            # is still pending.
            w2 = session_mod._TurnWatch()
            session._turn_watch = w2
            release.set()
            fut.result(timeout=5)
            assert w1.approvals_pending == 0  # its OWN watch decremented
            assert w2.approvals_pending == 0  # the new turn's never touched
        finally:
            release.set()
            session.close()


class TestTurnLifetimeConfig:
    def _patch_block(self, monkeypatch, block):
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": block}},
            raising=False,
        )

    def test_turn_timeout_reader_validation(self, monkeypatch):
        from agent.transports.claude_agent_sdk_session_config import _configured_turn_timeout

        self._patch_block(monkeypatch, {"turn_timeout": 1500})
        assert _configured_turn_timeout() == 1500.0
        self._patch_block(monkeypatch, {"turn_timeout": "900"})
        assert _configured_turn_timeout() == 900.0
        # 0 = unlimited does NOT exist for the budget; bools are not seconds;
        # garbage and negatives fall back — all with a warning.
        for bad in (0, -5, True, False, "plenty", [600]):
            self._patch_block(monkeypatch, {"turn_timeout": bad})
            assert _configured_turn_timeout() is None
        self._patch_block(monkeypatch, {})
        assert _configured_turn_timeout() is None

    def test_post_tool_quiet_reader_validation(self, monkeypatch):
        from agent.transports.claude_agent_sdk_session_config import (
            _configured_post_tool_quiet_timeout,
        )

        self._patch_block(monkeypatch, {"post_tool_quiet_timeout": 120})
        assert _configured_post_tool_quiet_timeout() == 120.0
        # 0 = explicitly disabled IS a valid value for the quiet watchdog.
        self._patch_block(monkeypatch, {"post_tool_quiet_timeout": 0})
        assert _configured_post_tool_quiet_timeout() == 0.0
        for bad in (-1, True, "off"):
            self._patch_block(monkeypatch, {"post_tool_quiet_timeout": bad})
            assert _configured_post_tool_quiet_timeout() is None

    def test_configured_turn_timeout_reaches_the_watchdog(self, monkeypatch):
        # End-to-end: config.yaml (not the signature default) is what the
        # budget rule enforces. A 0.2s configured budget kills a silent turn
        # fast even though run_turn was called with no explicit timeout.
        self._patch_block(monkeypatch, {"turn_timeout": 0.2})
        session, holder = _make_hold_open_session(script=[])
        try:
            turn = session.run_turn(
                "hi", watch_poll_interval=0.05, abort_grace=0.2
            )
        finally:
            session.close()
        assert turn.error is not None
        assert "turn timed out after" in turn.error

    def test_configured_quiet_reaches_the_watchdog(self, monkeypatch):
        # End-to-end for the second knob: with streaming OFF the quiet
        # watchdog defaults to disabled — a configured value must still
        # reach and arm it.
        self._patch_block(monkeypatch, {"post_tool_quiet_timeout": 0.2})
        session, holder = _make_hold_open_session(
            script=[
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Grep", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="x")]
                ),
            ],
            interrupt_ack=[_ede_interrupt_ack()],
        )
        try:
            turn = session.run_turn(
                "hi", turn_timeout=30.0, watch_poll_interval=0.05
            )
        finally:
            session.close()
        assert turn.error is not None
        assert "after a tool result" in turn.error

    def test_turnwatch_check_semantics(self, monkeypatch):
        # Deterministic unit coverage of the verdict rules (no threads).
        # Module-LOCAL time shadow — patching stdlib time.monotonic
        # process-wide would freeze asyncio loop clocks in concurrent tests.
        from agent.transports import claude_agent_sdk_session_watchdog as session_mod

        clock = {"now": 1000.0}
        monkeypatch.setattr(
            session_mod, "time", SimpleNamespace(monotonic=lambda: clock["now"])
        )
        watch = session_mod._TurnWatch()
        # Budget: needs elapsed >= budget AND idle >= min(30, budget).
        clock["now"] += 599.0
        assert watch.check(budget=600.0, quiet=0.0) is None
        clock["now"] += 2.0  # elapsed 601, idle 601
        assert watch.check(budget=600.0, quiet=0.0) == "budget"
        watch.tick()  # activity: idle 0 — over budget but alive
        assert watch.check(budget=600.0, quiet=0.0) is None
        # Outstanding tool suspends everything.
        clock["now"] += 700.0
        watch.note_tools_issued(1)
        assert watch.check(budget=600.0, quiet=0.0) is None
        watch.note_tools_resolved(1)
        assert watch.check(budget=600.0, quiet=0.0) == "budget"
        # Pending approval suspends everything.
        watch.approval_begin()
        clock["now"] += 700.0
        assert watch.check(budget=600.0, quiet=0.0) is None
        watch.approval_end()  # ticks: idle resets
        assert watch.check(budget=600.0, quiet=0.0) is None
        # Post-tool quiet: armed + idle >= quiet, before the budget.
        watch.arm_post_tool()
        clock["now"] += 91.0
        assert watch.check(budget=60000.0, quiet=90.0) == "post_tool_quiet"
        # Disabled quiet (0) never fires the post-tool rule.
        assert watch.check(budget=60000.0, quiet=0.0) is None
        watch.disarm_post_tool()
        assert watch.check(budget=60000.0, quiet=90.0) is None
        # Rebaseline absorbs a process stall.
        watch.arm_post_tool()
        clock["now"] += 500.0
        watch.rebaseline()
        assert watch.check(budget=60000.0, quiet=90.0) is None
