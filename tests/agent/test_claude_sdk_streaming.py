"""Streaming and stream ownership — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

import asyncio
import logging
import threading
import time

import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
)
from tests.agent.claude_sdk_fakes import (
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    AssistantMessage,
    UserMessage,
    _text_delta_event,
    ResultMessage,
    _FakeClient,
    _make_session,
    _make_turn,
    _make_agent,
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


class TestStreamOwnership:
    """Regression tests for the 2026-07-25 stale-answer incident
    (dasbrow-hermes-coder#2): the Claude Code CLI runs FULL unsolicited turns
    when background Agent tasks complete, leaving unconsumed ResultMessages in
    the shared FIFO. ``receive_response()`` then serves the OLDEST buffered
    result to the next turn — a permanent, silent off-by-N. Every test here is
    RED on the pre-fix implementation."""

    def _wait_unsolicited(self, session, n, timeout=5.0):
        """Sync point for the fixed code (reader routes idle-time messages
        within ms); a bounded no-op on the pre-fix code, which has no reader."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getattr(session, "_unsolicited_results", 0) >= n:
                return True
            time.sleep(0.01)
        return False

    def test_unsolicited_result_while_idle_is_not_served_as_next_answer(self):
        session, holder = _make_session(
            script=[
                AssistantMessage(content=[TextBlock("fresh answer")]),
                ResultMessage(result="fresh answer", uuid="fresh-1"),
            ]
        )
        try:
            session.ensure_started()
            # A background Agent task finished while nobody asked anything:
            # the CLI ran a full turn on its own initiative.
            holder["client"].feed(
                AssistantMessage(
                    content=[TextBlock("stale answer to an earlier question")]
                ),
                ResultMessage(
                    result="stale answer to an earlier question", uuid="stale-1"
                ),
            )
            self._wait_unsolicited(session, 1)
            turn = session.run_turn("new question", turn_timeout=15.0)
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "fresh answer"
        assert getattr(session, "_unsolicited_results", None) == 1

    def test_preloaded_stale_burst_is_drained_before_foreground_claim(self):
        """A resumed client's already-buffered FIFO stays background-owned.

        The stream reader is deliberately held until the foreground claimant
        exists.  On the buggy claim-before-drain path, that makes the stale
        ResultMessage the new query's answer deterministically.  A correct
        reader-mediated claim drains the pre-claim burst as unsolicited first,
        then permits exactly one query whose fast response owns the inbox.
        """
        holder = {}
        delivered = []

        class PreloadedResumeClient(_FakeClient):
            async def receive_messages(self):
                while True:
                    session = holder.get("session")
                    if session is not None and (
                        session._turn_inbox is not None
                        or getattr(session, "_turn_claim_requested", False)
                    ):
                        break
                    await asyncio.sleep(0)
                yield AssistantMessage(content=[TextBlock("stale background answer")])
                yield ResultMessage(
                    result="stale background answer",
                    uuid="stale-preclaim",
                    session_id="sdk-resume-kept",
                )
                async for message in super().receive_messages():
                    yield message

        def factory(options=None):
            client = PreloadedResumeClient(
                options=options,
                script=[
                    AssistantMessage(content=[TextBlock("fresh foreground answer")]),
                    ResultMessage(
                        result="fresh foreground answer",
                        uuid="fresh-foreground",
                        session_id="sdk-resume-kept",
                    ),
                ],
            )
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(
            cwd="/tmp",
            client_factory=factory,
            resume_session_id="sdk-resume-kept",
            on_unsolicited_result=delivered.append,
        )
        holder["session"] = session
        started = time.monotonic()
        try:
            turn = session.run_turn("new foreground question", turn_timeout=15.0)
            elapsed = time.monotonic() - started
        finally:
            session.close()

        assert elapsed < 5.0, f"claim handshake delayed a fast response ({elapsed:.1f}s)"
        assert turn.error is None
        assert turn.final_text == "fresh foreground answer"
        assert holder["client"].queried == ["new foreground question"]
        assert holder["client"].options["resume"] == "sdk-resume-kept"
        assert session._session_id == "sdk-resume-kept"
        assert session._unsolicited_results == 1
        assert session._unsolicited_delivered == {"stale-preclaim"}
        assert delivered == [["stale background answer"]]

    @pytest.mark.parametrize("stream_error", [None, RuntimeError("reader exploded")])
    def test_queued_claim_wakes_when_backlogged_stream_dies(self, stream_error):
        """EOF/exception after backlog must wake a still-queued claim.

        The reader deliberately sees both the backlog and foreground claim as
        ready.  Backlog wins until the stream exits, leaving the claim queued
        unless the death path explicitly resolves every pending acknowledgement.
        """
        holder = {}
        delivered = []

        class BacklogThenDeadClient(_FakeClient):
            async def receive_messages(self):
                while not holder["session"]._turn_claim_requested:
                    await asyncio.sleep(0)
                yield AssistantMessage(content=[TextBlock("background answer")])
                yield ResultMessage(result="background answer", uuid="background-dead")
                if stream_error is not None:
                    raise stream_error

        def factory(options=None):
            client = BacklogThenDeadClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(
            cwd="/tmp",
            client_factory=factory,
            on_unsolicited_result=delivered.append,
        )
        holder["session"] = session
        future = None
        try:
            session.ensure_started()
            started = time.monotonic()
            future = asyncio.run_coroutine_threadsafe(
                session._consume_turn("foreground question"), session._loop
            )
            result = future.result(timeout=2.0)
            elapsed = time.monotonic() - started
        finally:
            if future is not None:
                future.cancel()
            session.close()

        assert elapsed < 1.0
        assert result["stream_ended"] is True
        assert "SDK message stream ended before this turn" in result["error"]
        if stream_error is not None:
            assert "reader exploded" in result["error"]
        assert holder["client"].queried == []
        assert session._turn_inbox is None
        assert session._unsolicited_results == 1
        assert session._unsolicited_delivered == {"background-dead"}
        assert delivered == [["background answer"]]

    def test_stream_death_during_release_preserves_answer_and_retires(self):
        """A terminal answer remains valid when the persistent stream then dies.

        EOF races the reader-mediated release acknowledgement here.  The turn
        must keep the answer, clear its foreground ownership, and retire the
        dead session immediately rather than poisoning one later request.
        """
        holder = {}

        class ResultThenDeadClient(_FakeClient):
            async def receive_messages(self):
                while not self.queried:
                    await asyncio.sleep(0)
                yield ResultMessage(
                    result="answer before stream death",
                    uuid="result-before-death",
                    session_id="dead-after-result",
                )

        def factory(options=None):
            client = ResultThenDeadClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        try:
            turn = session.run_turn(
                "foreground question",
                turn_timeout=2.0,
                watch_poll_interval=0.01,
            )
        finally:
            session.close()

        assert turn.error is None
        assert turn.final_text == "answer before stream death"
        assert turn.should_retire is True
        assert holder["client"].queried == ["foreground question"]
        assert session._stream_ended is not None
        assert session._turn_inbox is None

    def test_stop_during_stream_death_release_handshake_stays_authoritative(self):
        """A non-terminal stream death must observe stops admitted before release."""
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        holder = {}

        class ReleaseRaceSession(ClaudeAgentSdkSession):
            async def _reader_loop(self):
                operation, inbox, claim_ack = await self._turn_claims.get()
                assert operation == "claim"
                self._turn_inbox = inbox
                claim_ack.set_result(None)

                while not self._client.queried:
                    await asyncio.sleep(0)
                inbox.put_nowait(sdk_session_mod._StreamEnd(error=None))

                operation, release_inbox, release_ack = await self._turn_claims.get()
                assert operation == "release"
                assert release_inbox is inbox
                self.request_interrupt()
                self._turn_inbox = None
                self._stream_ended = sdk_session_mod._StreamEnd(error=None)
                release_ack.set_result(None)

        def factory(options=None):
            client = _FakeClient(options=options)
            holder["client"] = client
            return client

        session = ReleaseRaceSession(cwd="/tmp", client_factory=factory)
        try:
            turn = session.run_turn("foreground question")
        finally:
            session.close()

        assert turn.error is not None
        assert turn.final_text == ""
        assert turn.terminal_result_accepted is False
        assert turn.interrupted is True
        assert turn.should_retire is True
        assert session._interrupt_event.is_set() is False
        assert holder["client"].queried == ["foreground question"]
        assert holder["client"].interrupted is True

    def test_late_interrupt_after_terminal_release_does_not_downgrade_answer(
        self, caplog
    ):
        """A stop arriving after terminal commit belongs to the finished turn."""
        holder = {}

        class LateInterruptSession(ClaudeAgentSdkSession):
            async def _consume_turn(self, prompt):
                turn_data = await super()._consume_turn(prompt)
                assert turn_data["result_uuid"] == "late-stop-result"
                assert turn_data["final_text"] == "completed answer"
                self.request_interrupt()
                return turn_data

        def factory(options=None):
            client = _FakeClient(
                options=options,
                script=[
                    ResultMessage(
                        result="completed answer",
                        uuid="late-stop-result",
                        session_id="healthy-after-late-stop",
                    )
                ],
            )
            holder["client"] = client
            return client

        session = LateInterruptSession(cwd="/tmp", client_factory=factory)
        with caplog.at_level(
            logging.INFO, logger="agent.transports.claude_agent_sdk_session"
        ):
            try:
                turn = session.run_turn("foreground question")
                pending_after_turn = session._post_terminal_interrupt_pending
            finally:
                session.close()

        # The decline is not silent: every other interrupt decline logs, and an
        # operator whose /stop appeared to do nothing needs a record saying it
        # was queued rather than dropped.
        assert any(
            "queued for the next turn" in record.getMessage()
            for record in caplog.records
        )
        assert turn.error is None
        assert turn.final_text == "completed answer"
        assert turn.turn_id == "late-stop-result"
        assert turn.thread_id == "healthy-after-late-stop"
        assert turn.terminal_result_accepted is True
        assert turn.interrupted is False
        assert turn.should_retire is False
        assert session._interrupt_event.is_set() is False
        assert pending_after_turn is False
        assert holder["client"].queried == ["foreground question"]
        assert holder["client"].interrupted is False

    def test_terminal_fence_holds_when_commit_guard_is_bypassed(self):
        """The fence is the mapping's snapshot read, not only the commit guard.

        ``request_interrupt`` declines to set ``_interrupt_event`` once the
        terminal result is committed, so a late stop sent through the public
        path leaves the event clear and the mapping is never asked to fence
        anything. Set the event directly instead — the state any other
        admission path would leave behind — so the only thing that can keep the
        completed answer is ``run_turn`` mapping ``interrupt_observed`` from the
        stream consumer's snapshot rather than re-reading the live event.
        """
        holder = {}

        class BypassCommitGuardSession(ClaudeAgentSdkSession):
            async def _consume_turn(self, prompt):
                turn_data = await super()._consume_turn(prompt)
                assert turn_data["terminal_result_accepted"] is True
                assert turn_data["interrupt_observed"] is False
                # Deliberately NOT request_interrupt(): that path would queue
                # the stop instead of setting the event, and the fence under
                # test would go unexercised.
                self._interrupt_event.set()
                return turn_data

        def factory(options=None):
            client = _FakeClient(
                options=options,
                script=[
                    ResultMessage(
                        result="completed answer",
                        uuid="fenced-result",
                        session_id="healthy-after-bypassed-stop",
                    )
                ],
            )
            holder["client"] = client
            return client

        session = BypassCommitGuardSession(cwd="/tmp", client_factory=factory)
        try:
            turn = session.run_turn("foreground question")
            event_after_turn = session._interrupt_event.is_set()
        finally:
            session.close()

        assert turn.error is None
        assert turn.terminal_result_accepted is True
        assert turn.interrupted is False
        assert turn.final_text == "completed answer"
        assert turn.turn_id == "fenced-result"
        assert turn.thread_id == "healthy-after-bypassed-stop"
        assert turn.should_retire is False
        # The fenced stop is consumed with the turn, not carried to the next.
        assert event_after_turn is False

    @pytest.mark.parametrize("exit_shape", ["stream_ended", "billing", "no_claims"])
    def test_preclaim_early_exit_snapshots_interrupt(self, exit_shape):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        session, _ = _make_session(script=[ResultMessage(result="unused")])
        try:
            session.ensure_started()
            session._interrupt_event.set()
            if exit_shape == "stream_ended":
                session._stream_ended = sdk_session_mod._StreamEnd(None)
            elif exit_shape == "billing":
                session._billing_guard_error = "metered billing refused"
            else:
                session._turn_claims = None
            turn_data = session._run_coro(session._consume_turn("hi"), timeout=5.0)
        finally:
            session.close()

        assert turn_data["interrupt_observed"] is True

    def test_interrupt_admitted_during_terminal_projection_is_reported(
        self, monkeypatch
    ):
        """Admission before commit and commit observation are one atomic boundary."""
        # Patch seam follows the consumer's binding: after the transport split
        # ``ClaudeSdkEventProjector`` is imported and instantiated in
        # ``claude_agent_sdk_session_turn``, not in the facade.
        import agent.transports.claude_agent_sdk_session_turn as sdk_session_mod

        projection_entered = threading.Event()
        continue_projection = threading.Event()
        original_projector = sdk_session_mod.ClaudeSdkEventProjector

        class BlockingProjector(original_projector):
            def project(self, message):
                if type(message).__name__ == "ResultMessage":
                    projection_entered.set()
                    assert continue_projection.wait(timeout=5.0)
                return super().project(message)

        monkeypatch.setattr(
            sdk_session_mod, "ClaudeSdkEventProjector", BlockingProjector
        )
        session, holder = _make_session(
            script=[ResultMessage(result="answer", uuid="terminal-race")]
        )
        outcome = {}

        def run_turn():
            outcome["turn"] = session.run_turn("hi")

        worker = threading.Thread(target=run_turn)
        worker.start()
        try:
            assert projection_entered.wait(timeout=5.0)
            session.request_interrupt()
            continue_projection.set()
            worker.join(timeout=10.0)
            assert worker.is_alive() is False
        finally:
            continue_projection.set()
            session.close()

        assert outcome["turn"].interrupted is True
        assert holder["client"].interrupted is True

    def test_offset_does_not_accumulate_across_unsolicited_turns(self):
        # The live incident: 4 unsolicited turns -> every later reply answered
        # a question 4 back. N unsolicited results must be dropped, not queued.
        session, holder = _make_session(
            script=[
                AssistantMessage(content=[TextBlock("the real answer")]),
                ResultMessage(result="the real answer", uuid="real-1"),
            ]
        )
        try:
            session.ensure_started()
            for i in range(3):
                holder["client"].feed(
                    AssistantMessage(content=[TextBlock(f"unsolicited {i}")]),
                    ResultMessage(result=f"unsolicited {i}", uuid=f"u-{i}"),
                )
            self._wait_unsolicited(session, 3)
            turn = session.run_turn("a question", turn_timeout=15.0)
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "the real answer"
        assert getattr(session, "_unsolicited_results", None) == 3

    def test_interrupted_turn_consumes_its_own_result(self):
        # Second entry point to the same corruption: breaking out of the
        # message loop on interrupt used to orphan that turn's ResultMessage
        # in the stream, where it became the NEXT turn's answer.
        holder = {}

        class InterruptingClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                if len(self.queried) == 1:
                    self._pending.append(
                        AssistantMessage(content=[TextBlock("turn1 partial")])
                    )
                    holder["session"]._interrupt_event.set()
                    self._pending.append(
                        ResultMessage(result="turn1 stale result", uuid="r1")
                    )
                else:
                    self._pending.append(
                        AssistantMessage(content=[TextBlock("turn2 answer")])
                    )
                    self._pending.append(
                        ResultMessage(result="turn2 answer", uuid="r2")
                    )

        def factory(options=None):
            client = InterruptingClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        holder["session"] = session
        try:
            turn1 = session.run_turn("first", turn_timeout=15.0)
            assert turn1.interrupted is True
            turn2 = session.run_turn("second", turn_timeout=15.0)
        finally:
            session.close()
        assert turn2.interrupted is False
        assert turn2.final_text == "turn2 answer"  # NOT "turn1 stale result"

    def test_residue_after_result_is_not_carried_into_next_turn(self):
        # A CLI turn that completes WHILE ours is running parks its result
        # behind ours in the stream. It must be routed away as unsolicited,
        # not served as the next turn's answer. (Mid-flight overlap — the one
        # window the idle-time tests above don't cover. Ported from the
        # independent re-derivation of this fix, commit 09537f965.)
        #
        # Updated pin (2026-08-07): the original version routed ALL residue
        # to the background-delivery lane with no content discrimination —
        # which ships a turn's OWN answer as a fake background completion,
        # the 2026-08-06 incident class. New intent: residue never becomes
        # the next turn's answer (unchanged) AND the delivery lane splits by
        # content — genuinely different residue (this test) still delivers
        # as a background burst; own-answer residue is suppressed (see
        # test_own_answer_residue_never_delivered_as_background_result).
        holder = {}
        got = []

        class OverlappingClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                if len(self.queried) == 1:
                    self._pending.append(ResultMessage(result="FIRST", uuid="f-1"))
                    self._pending.append(
                        ResultMessage(
                            result="RESIDUE from an overlapping CLI turn",
                            uuid="res-1",
                        )
                    )
                else:
                    self._pending.append(ResultMessage(result="SECOND", uuid="s-1"))

        def factory(options=None):
            client = OverlappingClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(
            cwd="/tmp", client_factory=factory, on_unsolicited_result=got.append
        )
        try:
            first = session.run_turn("one", turn_timeout=15.0)
            assert first.final_text == "FIRST"
            assert self._wait_unsolicited(session, 1), (
                "residue was left in the stream to poison the next turn"
            )
            second = session.run_turn("two", turn_timeout=15.0)
        finally:
            session.close()
        assert second.final_text == "SECOND"
        # Delivery split: differing residue is a REAL background completion —
        # it must still reach the delivery lane, not be swallowed.
        assert got == [["RESIDUE from an overlapping CLI turn"]]

    def test_own_answer_residue_never_delivered_as_background_result(
        self, caplog
    ):
        # D2 rework (2026-08-06 incident class): a residue ResultMessage that
        # repeats the just-finished turn's OWN answer must be suppressed —
        # dedup-marked and WARN'd, never handed to the background-delivery
        # callback as a fake completion.
        holder = {}
        got = []

        class OwnEchoClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                if len(self.queried) == 1:
                    self._pending.append(ResultMessage(result="FIRST", uuid="f-1"))
                    self._pending.append(
                        ResultMessage(result="FIRST", uuid="own-dup")
                    )
                else:
                    self._pending.append(ResultMessage(result="SECOND", uuid="s-1"))

        def factory(options=None):
            client = OwnEchoClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(
            cwd="/tmp", client_factory=factory, on_unsolicited_result=got.append
        )
        with caplog.at_level(
            logging.WARNING, logger="agent.transports.claude_agent_sdk_session"
        ):
            try:
                first = session.run_turn("one", turn_timeout=15.0)
                assert first.final_text == "FIRST"
                # run_turn returns only after the residue drain — the
                # suppression already happened; a bounded wait just proves
                # nothing arrives late either.
                self._wait(lambda: got, timeout=0.5)
                second = session.run_turn("two", turn_timeout=15.0)
            finally:
                session.close()
        assert got == [], "own-answer residue was delivered as a fake background result"
        assert second.final_text == "SECOND"
        assert session._unsolicited_results == 1  # routed away, still counted
        assert "own-dup" in session._unsolicited_delivered
        assert any(
            "matches this turn's own answer" in r.getMessage()
            for r in caplog.records
        ), "suppression must WARN, never silently drop"

    def test_genuine_overlap_residue_still_delivers_as_burst(self):
        # The delivery split's other half, explicit: residue with DIFFERENT
        # content is deliver_background_results working — never suppressed.
        holder = {}
        got = []

        class OverlapClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                self._pending.append(ResultMessage(result="FIRST", uuid="f-1"))
                self._pending.append(
                    ResultMessage(result="DIFFERENT bg completion", uuid="bg-9")
                )

        def factory(options=None):
            client = OverlapClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(
            cwd="/tmp", client_factory=factory, on_unsolicited_result=got.append
        )
        try:
            first = session.run_turn("one", turn_timeout=15.0)
            assert first.final_text == "FIRST"
            assert self._wait(lambda: got, timeout=5.0)
        finally:
            session.close()
        assert got == [["DIFFERENT bg completion"]]

    def test_stale_unsolicited_text_never_attaches_to_later_result(
        self, caplog
    ):
        # Leak fix: text buffered idle-time whose terminal ResultMessage never
        # arrived is discarded (with WARN) at the next turn's start — a later
        # unrelated result must never pick it up as its own burst.
        got = []
        session, holder = _make_session(
            script=[
                AssistantMessage(content=[TextBlock("fresh answer")]),
                ResultMessage(result="fresh answer", uuid="fresh-1"),
            ],
            on_unsolicited_result=got.append,
        )
        with caplog.at_level(
            logging.WARNING, logger="agent.transports.claude_agent_sdk_session"
        ):
            try:
                session.ensure_started()
                # A background turn started streaming but its result never
                # came (CLI died / mid-burst) — text sits in the buffer.
                holder["client"].feed(
                    AssistantMessage(content=[TextBlock("orphaned partial text")])
                )
                assert self._wait(lambda: session._unsolicited_text)
                turn = session.run_turn("new question", turn_timeout=15.0)
                assert turn.final_text == "fresh answer"
                # A later, unrelated background completion arrives idle-time.
                holder["client"].feed(
                    ResultMessage(result="unrelated bg answer", uuid="bg-x")
                )
                assert self._wait(lambda: got)
            finally:
                session.close()
        assert got == [["unrelated bg answer"]], (
            "stale pre-turn text misattached to an unrelated later result"
        )
        assert any(
            "stale unsolicited text" in r.getMessage()
            for r in caplog.records
        ), "turn-start discard must WARN, never silently drop"

    @staticmethod
    def _wait(cond, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.01)
        return False

    def test_stream_death_mid_turn_fails_fast_instead_of_hanging(self):
        # A script with no ResultMessage models the CLI dying mid-turn. The
        # turn must surface an error promptly — pre-fix the loop ended and the
        # turn returned as an empty SUCCESS; an unguarded reader design would
        # instead hang until turn_timeout.
        session, _holder = _make_session(
            script=[AssistantMessage(content=[TextBlock("half an answer")])]
        )
        started = time.monotonic()
        try:
            turn = session.run_turn("hi", turn_timeout=15.0)
        finally:
            session.close()
        elapsed = time.monotonic() - started
        assert elapsed < 10.0, f"turn took {elapsed:.1f}s — hung on a dead stream"
        assert turn.error is not None and "stream ended" in turn.error


# ---------- streaming deltas (W4, env-gated default OFF) ----------


class TestStreaming:
    def test_env_var_cannot_enable_streaming(self, monkeypatch):
        # AGENTS.md:102-107 keeps behavioural settings out of HERMES_* env
        # vars. The old HERMES_CLAUDE_SDK_STREAMING override is gone, so
        # setting it must have NO effect — config.yaml is the only interface.
        monkeypatch.setenv("HERMES_CLAUDE_SDK_STREAMING", "1")
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert "include_partial_messages" not in holder["client"].options

    def test_option_absent_by_default(self):
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert "include_partial_messages" not in holder["client"].options

    def test_config_yaml_is_the_operator_interface(self, monkeypatch):
        # AGENTS.md: behavioral settings live in config.yaml, not env.
        # agent.claude_agent_sdk.streaming turns the option on without any env.
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": {"streaming": True}}},
        )
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["include_partial_messages"] is True

    def test_env_var_cannot_disable_config_streaming(self, monkeypatch):
        # The mirror of the test above: an explicit env "0" must NOT be able to
        # veto config.yaml either. Together the pair pins the override as fully
        # inert in both directions, so it cannot creep back in unnoticed.
        import hermes_cli.config as cfg

        monkeypatch.setenv("HERMES_CLAUDE_SDK_STREAMING", "0")
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": {"streaming": True}}},
        )
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["include_partial_messages"] is True

    def test_setting_sources_isolated_by_default(self):
        # Absent config → full isolation: the SDK loads NO filesystem
        # settings, so ambient ~/.claude / project files cannot
        # re-permission tools underneath the configured posture.
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["setting_sources"] == []

    def test_setting_sources_config_opt_in(self, monkeypatch):
        # Deployments whose operating model stores tool grants in the
        # operator's own ~/.claude/settings.json (unattended cron turns that
        # must pre-approve WebSearch/MCP tools) opt back in explicitly.
        # Regression: the hardening initially shipped setting_sources
        # hardcoded [] and silently cut a production box's cron jobs off
        # from their allowlist (2026-07-26).
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"setting_sources": ["user"]}}
            },
        )
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["setting_sources"] == ["user"]

    def test_setting_sources_invalid_entries_dropped(self, monkeypatch):
        # A typo must never silently load an unintended source; valid
        # entries survive, invalid ones are dropped (with a warning).
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {
                    "claude_agent_sdk": {
                        "setting_sources": ["user", "bogus", "project"]
                    }
                }
            },
        )
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["setting_sources"] == ["user", "project"]

    def test_deltas_reach_callback_and_never_the_transcript(self):
        got = []
        script = [
            _text_delta_event("Hel"),
            _text_delta_event("lo"),
            AssistantMessage(content=[TextBlock("Hello")]),
            ResultMessage(result="Hello"),
        ]
        session, _ = _make_session(script=script, on_stream_delta=got.append)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert got == ["Hel", "lo"]
        # Display-only: deltas never become transcript rows.
        assert [m["role"] for m in turn.projected_messages] == ["assistant"]
        assert turn.final_text == "Hello"

    def test_tool_adjacent_assistant_text_reaches_interim_callback_once(self):
        commentary = []
        script = [
            AssistantMessage(content=[
                TextBlock("I will inspect the project first."),
                ToolUseBlock(id="t1", name="Bash", input={"command": "pwd"}),
            ]),
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="/tmp")]),
            ResultMessage(result="Final answer"),
        ]
        session, _ = _make_session(script=script, on_interim_assistant=commentary.append)
        try:
            turn = session.run_turn("inspect")
        finally:
            session.close()
        assert commentary == ["I will inspect the project first."]
        assert turn.final_text == "Final answer"

    def test_tool_iteration_callback_runs_before_turn_returns(self):
        iterations = []
        script = [
            AssistantMessage(content=[ToolUseBlock(id="t1", name="Bash", input={"command": "pwd"})]),
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="/tmp")]),
            ResultMessage(result="Final answer"),
        ]
        session, _ = _make_session(script=script, on_tool_iteration=lambda: iterations.append(True))
        try:
            turn = session.run_turn("inspect")
        finally:
            session.close()
        assert iterations == [True]
        assert turn.tool_iterations == 1

    def test_subagent_deltas_are_not_forwarded(self):
        got = []
        script = [
            _text_delta_event("sub", parent_tool_use_id="tool-1"),
            ResultMessage(result="done"),
        ]
        session, _ = _make_session(script=script, on_stream_delta=got.append)
        try:
            session.run_turn("hi")
        finally:
            session.close()
        assert got == []

    def test_runtime_wires_late_bound_stream_callback(self, monkeypatch):
        # The gateway assigns agent.stream_delta_callback per turn AFTER the
        # session exists — the wiring must read it at call time.
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        captured = {}

        class SpySession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn()

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        relay = captured.get("on_stream_delta")
        assert callable(relay)
        cli_seen = []
        gateway_seen = []
        agent.stream_delta_callback = cli_seen.append  # assigned AFTER creation
        agent._stream_callback = gateway_seen.append
        relay("delta-text")
        assert cli_seen == ["delta-text"]
        assert gateway_seen == ["delta-text"]

        # The CLI/TUI callback is cleared between turns, but the gateway sink
        # remains the desktop's message.delta lane and must keep receiving.
        agent.stream_delta_callback = None
        relay("gateway-only")
        assert cli_seen == ["delta-text"]
        assert gateway_seen == ["delta-text", "gateway-only"]

        agent._stream_callback = None  # both cleared → no crash
        relay("dropped")
        assert gateway_seen == ["delta-text", "gateway-only"]


class TestUnsolicitedDelivery:
    """The delivery half of the stream-ownership fix (dasbrow-hermes-coder#2):
    a finished background Agent task's answer must be CAPTURED and handed to
    the delivery callback — never served as a turn result (TestStreamOwnership
    pins that), and never silently discarded either (observed live 2026-07-29:
    14 dropped answers, 32-minute silences until the operator poked)."""

    @staticmethod
    def _wait(cond, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.01)
        return False

    def test_callback_receives_unsolicited_result_text(self):
        got = []
        session, holder = _make_session(on_unsolicited_result=got.append)
        try:
            session.ensure_started()
            holder["client"].feed(
                AssistantMessage(content=[TextBlock("research done: Tupã wins")]),
                ResultMessage(result="research done: Tupã wins", uuid="bg-1"),
            )
            assert self._wait(lambda: got)
        finally:
            session.close()
        assert got == [["research done: Tupã wins"]]
        # Observability unchanged: the counter still ticks.
        assert session._unsolicited_results == 1

    def test_bg_burst_delivers_all_buffered_assistant_messages(self):
        # ×5 incident 2026-08-06: five out-of-turn AssistantMessages were
        # buffered, the terminal ResultMessage carried its own text, and the
        # intermediate "Research landed…" message was silently discarded —
        # only the terminal report reached the callback. The full burst must
        # arrive as an ordered list, result text deduped against the last
        # buffered entry.
        got = []
        session, holder = _make_session(on_unsolicited_result=got.append)
        try:
            session.ensure_started()
            holder["client"].feed(
                AssistantMessage(
                    content=[TextBlock("Research landed — writing up.")]
                ),
                AssistantMessage(content=[TextBlock("the full report")]),
                ResultMessage(result="the full report", uuid="burst-1"),
            )
            assert self._wait(lambda: got)
        finally:
            session.close()
        assert got == [["Research landed — writing up.", "the full report"]]

    def test_falls_back_to_buffered_assistant_text(self):
        # Some CLI results arrive with result=None; the assistant text blocks
        # of the unsolicited turn are the answer then.
        got = []
        session, holder = _make_session(on_unsolicited_result=got.append)
        try:
            session.ensure_started()
            holder["client"].feed(
                AssistantMessage(content=[TextBlock("the long answer body")]),
                ResultMessage(result=None, uuid="bg-2"),
            )
            assert self._wait(lambda: got)
        finally:
            session.close()
        assert got == [["the long answer body"]]

    def test_result_uuid_deduplicated(self):
        got = []
        session, holder = _make_session(on_unsolicited_result=got.append)
        try:
            session.ensure_started()
            holder["client"].feed(ResultMessage(result="answer", uuid="dup-1"))
            assert self._wait(lambda: got)
            holder["client"].feed(ResultMessage(result="answer", uuid="dup-1"))
            self._wait(lambda: len(got) >= 2, timeout=0.5)
        finally:
            session.close()
        assert got == [["answer"]]

    def test_subagent_text_excluded_from_buffer(self):
        # parent_tool_use_id set = subagent stream noise — same gate the
        # stream-delta forwarder uses. Only top-level text is the answer.
        got = []
        session, holder = _make_session(on_unsolicited_result=got.append)
        try:
            session.ensure_started()
            holder["client"].feed(
                AssistantMessage(
                    content=[TextBlock("sub noise")], parent_tool_use_id="t1"
                ),
                AssistantMessage(content=[TextBlock("top-level answer")]),
                ResultMessage(result=None, uuid="bg-3"),
            )
            assert self._wait(lambda: got)
        finally:
            session.close()
        assert got == [["top-level answer"]]

    def test_no_callback_keeps_drop_semantics(self):
        # Without a wired callback the historical WARN+counter drop stands
        # (TestStreamOwnership's pins rely on it).
        session, holder = _make_session()
        try:
            session.ensure_started()
            holder["client"].feed(ResultMessage(result="x", uuid="nc-1"))
            assert self._wait(
                lambda: getattr(session, "_unsolicited_results", 0) >= 1
            )
        finally:
            session.close()
        assert session._unsolicited_results == 1


class TestDeadStreamRetires:
    """A dead SDK stream is permanent on the session object (_stream_ended
    never resets, ensure_started returns early while _client is set) — so a
    non-retiring stream-death error poisons EVERY later turn into an instant
    zero-model-call failure. RED pre-fix: both shapes returned
    should_retire=False (retire fired only on auth hints)."""

    def test_stream_death_mid_turn_retires(self):
        # Base fake: a script with no ResultMessage models the CLI dying
        # mid-turn (_EOS ends the stream).
        session, holder = _make_session(
            script=[AssistantMessage(content=[TextBlock("partial")])]
        )
        try:
            turn = session.run_turn("hi", watch_poll_interval=0.02)
        finally:
            session.close()
        assert turn.error is not None
        assert "stream ended" in turn.error
        assert turn.should_retire is True

    def test_dead_stream_short_circuit_retires(self):
        # The POISONED-SESSION half: a second turn on the same object
        # short-circuits pre-query — it must retire too, or the session
        # errors instantly forever.
        session, holder = _make_session(
            script=[AssistantMessage(content=[TextBlock("partial")])]
        )
        try:
            first = session.run_turn("hi", watch_poll_interval=0.02)
            second = session.run_turn("again", watch_poll_interval=0.02)
        finally:
            session.close()
        assert first.should_retire is True
        assert second.error is not None
        assert "stream ended before this turn" in second.error
        assert second.should_retire is True

    def test_model_level_subtypes_still_do_not_retire(self):
        # The retire widening is stream-death-only: error_max_turns is a
        # model-level outcome on a HEALTHY stream and stays non-retiring.
        session, holder = _make_session(
            script=[
                AssistantMessage(content=[TextBlock("partial work")]),
                ResultMessage(subtype="error_max_turns", is_error=False),
            ]
        )
        try:
            turn = session.run_turn("hi", watch_poll_interval=0.02)
        finally:
            session.close()
        assert turn.error == "SDK turn ended: error_max_turns"
        assert turn.should_retire is False
