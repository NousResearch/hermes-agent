"""Interrupt routing and barge-in — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

from unittest.mock import MagicMock

import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
)
from tests.agent.claude_sdk_fakes import (
    SystemMessage,
    ResultMessage,
    _EOS,
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


# ---------- interrupt routes to the SDK session (W4) ----------


class TestInterruptRoutesToSdkSession:
    """/stop and new-message preemption call AIAgent.interrupt(); the SDK
    session's request_interrupt (event + client.interrupt()) already works —
    this pins the one missing caller."""

    @staticmethod
    def _make_real_agent():
        from run_agent import AIAgent

        return AIAgent(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    def test_interrupt_reaches_live_sdk_session(self):
        agent = self._make_real_agent()
        agent._claude_sdk_session = MagicMock()
        agent.interrupt()
        agent._claude_sdk_session.request_interrupt.assert_called_once()

    def test_interrupt_without_sdk_session_stays_safe(self):
        agent = self._make_real_agent()
        agent._claude_sdk_session = None
        agent.interrupt()  # must not raise

    def test_release_clients_disconnects_sdk_session(self):
        # Adversarial-review HIGH: the gateway's ROUTINE evictions (LRU cap,
        # idle-TTL sweep, model switch) release via release_clients(), which
        # never touched the SDK session — leaking the loop thread + the
        # Claude CLI subprocess per eviction on a 24/7 gateway.
        agent = self._make_real_agent()
        sdk_session = MagicMock()
        agent._claude_sdk_session = sdk_session
        agent.release_clients()
        sdk_session.close.assert_called_once()
        assert agent._claude_sdk_session is None

    def test_pending_interrupt_flag_short_circuits_cold_turn(self, monkeypatch):
        # Adversarial-review MEDIUM: an interrupt landing before the SDK
        # session exists set only agent._interrupt_requested, which the SDK
        # path never read — the turn ran uninterruptible for up to 600s.
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        instances = []

        class SpySession:
            def __init__(self, **kwargs):
                instances.append(self)

            def run_turn(self, user_input):
                return _make_turn()

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._interrupt_requested = True
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert instances == []  # no session created, no subscription burn
        assert result["completed"] is False and result["partial"] is True
        assert agent._interrupt_requested is False  # consumed, next turn runs

    def test_honored_interrupt_consumes_agent_flag(self, monkeypatch):
        # Live-gate catch: after an interrupt was honored mid-turn, the
        # agent-level flag stayed set and the cold-flag check short-circuited
        # the NEXT turn into an empty answer. Honoring must consume it.
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._session_db = None

        class SpySession:
            def __init__(self, **kwargs):
                pass

            def run_turn(self, user_input):
                agent._interrupt_requested = True  # user hit /stop mid-turn
                return _make_turn(interrupted=True, final_text="", projected_messages=[])

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert result["partial"] is True
        assert agent._interrupt_requested is False  # consumed — next turn runs

    def test_thread_id_captured_from_init_message(self):
        # A FIRST-turn interrupt used to lose the resume id (only the final
        # ResultMessage carried it). The SDK announces session_id in its init
        # SystemMessage — capture it from any message.
        session, _ = _make_session(script=[SystemMessage(session_id="sdk-early-7")])
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.thread_id == "sdk-early-7"

    def test_pre_set_interrupt_event_honored_then_next_turn_runs(self):
        # Adversarial-review MEDIUM: run_turn unconditionally CLEARED the
        # interrupt event after connect — an interrupt arriving during the
        # (up to 60s) connect window was silently erased. It must instead be
        # honored by THIS turn, and must not bleed into the next one.
        session, holder = _make_session(
            script=[ResultMessage(result="ok")]
        )
        try:
            session.ensure_started()
            session.request_interrupt()
            turn1 = session.run_turn("first")
            assert turn1.interrupted is True
            assert holder["client"].queried == []  # never reached the model
            turn2 = session.run_turn("second")
            assert turn2.interrupted is False
            assert holder["client"].queried == ["second"]
        finally:
            session.close()

    def test_interrupt_between_completed_turns_targets_the_next_turn(self):
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            turn1 = session.run_turn("first")
            assert turn1.interrupted is False
            session.request_interrupt()
            turn2 = session.run_turn("second")
            assert turn2.interrupted is True
            assert holder["client"].queried == ["first"]
        finally:
            session.close()


    def test_stop_between_turns_still_reaches_the_cli(self):
        """The terminal fence ends when the result reaches the caller.

        Between turns the CLI can still be working -- an unsolicited background
        Agent task under ``deliver_background_results`` -- so a /stop then must
        reach the child, exactly as it did before the fence existed. Leaving
        the commit flag set past the turn would silently drop it.
        """
        import time

        session, holder = _make_session(
            script=[ResultMessage(result="answer", uuid="between-turns-1")]
        )
        try:
            assert session.run_turn("hi").final_text == "answer"

            session.request_interrupt()

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if getattr(holder["client"], "interrupted", False):
                    break
                time.sleep(0.01)
            assert holder["client"].interrupted is True
        finally:
            session.close()


class TestBargeInInterruptHandoff:
    """W22 (2026-08-09 barge-in incident): a mid-turn user message interrupts
    the running turn; the CLI, aborted before any assistant content, returns
    is_error/error_during_execution ("[ede_diagnostic] result_type=user…").
    Two defects made that page the operator with a false "⚠️ Processing
    stopped … Try again": the runtime result dict omitted the "interrupted"
    key (codex_runtime and the native finalizer both return it — the gateway's
    queued-drain needs it to discard, not deliver, the abandoned turn), and
    the transport surfaced the CLI's interrupt-shaped error as a real error.
    The honest error path must NOT weaken: unrequested EDE, auth-hinted
    errors, and CLI death keep surfacing."""

    def test_sdk_result_dict_carries_interrupted_key(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._session_db = None

        class SpySession:
            def __init__(self, **kwargs):
                pass

            def run_turn(self, user_input):
                agent._interrupt_requested = True  # barge-in mid-turn
                return _make_turn(interrupted=True, final_text="",
                                  projected_messages=[])

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert result["interrupted"] is True
        assert result["partial"] is True

    def test_uninterrupted_turn_reports_interrupted_false(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._session_db = None

        class SpySession:
            def __init__(self, **kwargs):
                pass

            def run_turn(self, user_input):
                return _make_turn()

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert result["interrupted"] is False

    def _ede_result(self):
        return ResultMessage(
            subtype="error_during_execution",
            is_error=True,
            errors=["[ede_diagnostic] result_type=user last_content_type=n/a "
                    "stop_reason=null"],
        )

    def test_requested_interrupt_ede_masks_error_with_log(self, caplog):
        import logging
        holder = {}

        class BargeInClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                holder["session"]._interrupt_event.set()
                self._pending.append(ResultMessage(
                    subtype="error_during_execution",
                    is_error=True,
                    errors=["[ede_diagnostic] result_type=user "
                            "last_content_type=n/a stop_reason=null"],
                ))

        def factory(options=None):
            client = BargeInClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        holder["session"] = session
        try:
            with caplog.at_level(
                logging.INFO, logger="agent.transports.claude_agent_sdk_session"
            ):
                turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.interrupted is True
        assert turn.error is None
        assert turn.should_retire is False
        assert any("masked error_during_execution" in r.getMessage()
                   for r in caplog.records)

    def test_unrequested_ede_stays_an_error(self):
        session, _ = _make_session(script=[self._ede_result()])
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert "error_during_execution" in turn.error

    def test_auth_hint_in_ede_outranks_interrupt_mask(self):
        holder = {}

        class AuthFailBargeIn(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                holder["session"]._interrupt_event.set()
                self._pending.append(ResultMessage(
                    subtype="error_during_execution",
                    is_error=True,
                    errors=["401 unauthorized: oauth token has expired"],
                ))

        def factory(options=None):
            client = AuthFailBargeIn(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        holder["session"] = session
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert turn.should_retire is True

    def test_stream_end_during_interrupt_stays_an_error(self):
        holder = {}

        class DyingClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                holder["session"]._interrupt_event.set()
                # No ResultMessage — the CLI dies; the reader sees the
                # stream END. (This _EOS is what makes the comment true:
                # without it the fake merely went silent and the test was
                # riding the old wall clock — 26s of suite time for the
                # wrong mechanism.)
                self._pending.append(_EOS)

        def factory(options=None):
            client = DyingClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        holder["session"] = session
        try:
            turn = session.run_turn("hi", turn_timeout=10.0)
        finally:
            session.close()
        assert turn.interrupted is True
        assert turn.error is not None
        # Stream death retires (the poisoned-session fix), interrupt or not.
        assert turn.should_retire is True
