"""``interrupt()`` records who asked for the stop, so the turn exit reason can attribute it (#112647).

Every system watchdog reaches the agent through ``request_hard_interrupt(..., tool_reason=...)``;
the published ``_tool_interrupt_reason`` is the single source the exit reason is derived from.
"""

from __future__ import annotations

import logging
import threading

from agent.interrupt_compat import request_hard_interrupt
from agent.interrupt_control import interrupt_issuer, interrupt_skip_wording
from tools.interrupt import set_interrupt


def _bare_agent():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_interrupt_reason = None
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.quiet_mode = True
    return agent


def test_system_producer_is_recorded_and_logged_at_publication(caplog):
    """The cron/gateway watchdog shape: the issuer survives to ``interrupt_issuer`` and ONE log line
    names it at the point ``_interrupt_requested`` is set."""
    agent = _bare_agent()
    try:
        with caplog.at_level(logging.INFO, logger="run_agent"):
            assert request_hard_interrupt(agent, "Cron job timed out (inactivity)", tool_reason="cron inactivity watchdog")
        assert agent._interrupt_requested is True
        assert interrupt_issuer(agent) == "cron_inactivity_watchdog"
        published = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Interrupt requested")]
        assert len(published) == 1 and "cron inactivity watchdog" in published[0]
    finally:
        set_interrupt(False)


def test_human_stops_have_no_system_issuer():
    """A plain ``interrupt()`` and a reason-less hard stop (CLI/TUI /stop) stay attributed to the user."""
    agent = _bare_agent()
    try:
        agent.interrupt()
        assert interrupt_issuer(agent) is None
        agent.clear_interrupt()
        assert request_hard_interrupt(agent)
        assert interrupt_issuer(agent) is None
    finally:
        set_interrupt(False)


def test_gateway_lifecycle_producers_name_a_system_issuer():
    """Gateway stop, session eviction and an abandoned SSE run are system stops: none of them may fall
    through to the reason-less default that books ``interrupted_by_user`` (#112647)."""
    import asyncio
    from types import SimpleNamespace

    from gateway.platforms.api_server import _abandon_agent_task
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    from gateway.run_inbound import GatewayInboundMixin
    from gateway.run_shutdown import GatewayShutdownMixin

    try:
        stopping_agent = _bare_agent()
        runner = SimpleNamespace(
            _running_agents={"k": stopping_agent}, _interrupt_api_server_runs=lambda reason: 0,
            _interrupt_deferred_agent_workers=lambda reason: 0,
        )
        GatewayShutdownMixin._interrupt_running_agents(runner, "Gateway shutting down")
        assert interrupt_issuer(stopping_agent) == "gateway_shutdown"

        evicted_agent = _bare_agent()
        runner = SimpleNamespace(
            _peek_session_state=lambda key: SimpleNamespace(turn=SimpleNamespace(agent=evicted_agent)),
            _invalidate_session_run_generation=lambda key, reason="": 1,
            _drop_turn_slot=lambda key, run_generation=None: None,
        )
        runner._interrupt_running_turn = (
            lambda *a, **kw: GatewayAgentCacheMixin._interrupt_running_turn(runner, *a, **kw)
        )
        GatewayInboundMixin._hm_evict_running_agent(runner, "k", "history_reset")
        assert interrupt_issuer(evicted_agent) == "session_evicted"

        sse_agent = _bare_agent()
        asyncio.run(_abandon_agent_task(
            [sse_agent], SimpleNamespace(done=lambda: True), "SSE client disconnected", await_cancel=False))
        assert interrupt_issuer(sse_agent) == "sse_client_disconnected"
    finally:
        set_interrupt(False)


def test_soft_interrupt_with_tool_reason_is_attributed_to_the_system():
    """A system producer that must stop the turn SOFTLY labels itself via ``tool_reason``. The
    message-carrying soft path used to hardcode ``user sent a new message``, so a batch-guard abort
    was booked as a human stop and rendered as the user-stop placeholder (#130207)."""
    import concurrent.futures
    from types import SimpleNamespace
    from unittest.mock import patch

    import agent.tool_executor as te

    agent = _bare_agent()
    agent._touch_activity = lambda *_: None
    batch = SimpleNamespace(authorization_gate=te._ConcurrentToolAuthorizationGate(), executor=None, close=lambda: None)
    prepared = SimpleNamespace(batch=batch, tids=[], future=concurrent.futures.Future(), execution_started=threading.Event())
    try:
        # Drive the REAL batch-timeout guard: a prepared terminal call whose worker never settles.
        with (
            patch("agent.terminal_approval_batch.take_prepared_call", return_value=prepared),
            patch.object(te, "_resolve_sequential_tool_timeout", return_value=0.05),
            patch.object(te, "_emit_terminal_post_tool_call"),
        ):
            te._run_sequential_tool_execution_middleware(
                agent, function_name="terminal", function_args={}, effective_task_id="t",
                tool_call_id="c1", execute=lambda *_a, **_k: None)
        assert agent._interrupt_requested is True
        assert interrupt_issuer(agent) == "terminal_batch_timeout"
        # No message: gateway/CLI re-queue ``_interrupt_message`` as the user's next turn.
        assert agent._interrupt_message is None

        # A USER stop that abandons the same wedged batch (grace elapsed) keeps its own attribution:
        # the guard must not rebook it as a batch timeout nor drop the queued message / redirect.
        set_interrupt(False)
        agent.clear_interrupt()
        agent.interrupt("fix the login bug")
        agent._pending_redirect = "use the staging db"
        prepared.future = concurrent.futures.Future()
        with (
            patch("agent.terminal_approval_batch.take_prepared_call", return_value=prepared),
            patch.object(te, "_resolve_sequential_tool_timeout", return_value=60.0),
            patch.object(te, "_emit_terminal_post_tool_call"),
            patch.object(te.concurrent.futures, "wait", lambda fs, timeout=None: None),
        ):
            te._run_sequential_tool_execution_middleware(
                agent, function_name="terminal", function_args={}, effective_task_id="t",
                tool_call_id="c2", execute=lambda *_a, **_k: None)
        assert interrupt_issuer(agent) is None
        assert agent._interrupt_message == "fix the login bug"
        assert agent._pending_redirect == "use the staging db"

        # Desktop batch PREPARATION has the same two exits: a user stop that cancels it keeps the
        # user's message/redirect, and a preparation timeout is a system stop with no message.
        import agent.terminal_approval_batch as tab

        def _prepare(exc, during=lambda: None):
            class _Batch:
                def __init__(self, *_a):
                    pass

                def start(self):
                    during()
                    raise exc

                def close(self):
                    pass

            ids = iter(["p1", "p2"])

            def _parse(*_a, **_k):
                call_id = next(ids)
                return SimpleNamespace(name="terminal", parse_error=None, ref=lambda _t: SimpleNamespace(call_id=call_id))

            with (
                patch.object(tab, "_TerminalBatch", _Batch),
                patch("gateway.session_context.get_session_env", return_value="desktop"),
                patch("tools.approval._gateway_notify_cb", lambda _k: object()),
                patch.object(te, "_parse_tool_call", _parse),
                tab.terminal_approval_batch(agent, [1, 2], [], "t"),
            ):
                pass

        def _user_stop():
            agent.interrupt("fix the login bug")
            agent._pending_redirect = "use the staging db"

        set_interrupt(False)
        agent.clear_interrupt()
        _prepare(tab._CancelledPreparation("Terminal approval preparation cancelled; command was not started"), _user_stop)
        assert interrupt_issuer(agent) is None
        assert agent._interrupt_message == "fix the login bug"
        assert agent._pending_redirect == "use the staging db"
        set_interrupt(False)
        agent.clear_interrupt()
        _prepare(TimeoutError("Terminal approval preparation timed out; commands were not started"))
        assert interrupt_issuer(agent) == "terminal_batch_preparation_timeout"
        assert agent._interrupt_message is None

        # Sequential skip notices (stop before a call / after one) render the recorded system reason.
        def _calls():
            return [SimpleNamespace(id=f"s{i}", function=SimpleNamespace(name=f"t{i}", arguments="{}")) for i in (1, 2)]

        def _stop_mid_batch(*_a, **_k):
            agent.interrupt(tool_reason="terminal batch timeout")
            return None, 0.0

        agent._vprint, agent.log_prefix = (lambda *_a, **_k: None), ""
        for stop_first in (True, False):
            set_interrupt(False)
            agent.clear_interrupt()
            if stop_first:
                agent.interrupt(tool_reason="terminal batch timeout")
            rows = []
            with (
                patch.object(te, "_budget_for_agent"),
                patch.object(te, "_flush_session_db_after_tool_progress", return_value=True),
                patch.object(te, "_emit_terminal_post_tool_call"),
                patch.object(te, "_resolve_sequential_dispatch"),
                patch.object(te, "_run_sequential_call", _stop_mid_batch),
                patch.object(te, "_publish_sequential_result", return_value=True),
            ):
                te._execute_tool_calls_sequential(agent, SimpleNamespace(tool_calls=_calls()), rows, "t", finalize=False)
            assert rows and all("Turn aborted — terminal batch timeout" in r["content"] for r in rows), rows
            assert not any("User sent a new message" in r["content"] for r in rows)

        # A slot no worker filled after a system stop says so, in both the row and the hook.
        with patch.object(te, "_emit_terminal_post_tool_call") as emitted:
            result, _, _ = te._unfinished_tool_result(
                agent, te._ToolCallRef("t3", {}, "t", "u1", []), timed_out=False, timeout_s=None)
        assert "Turn aborted — terminal batch timeout" in result
        assert emitted.call_args.kwargs["error_message"] == "Tool execution cancelled. Turn aborted — terminal batch timeout"
        # Rendered verbatim: consumers substitute ``{name}`` with str.replace, never str.format.
        agent._tool_interrupt_reason = "guard {x}"
        assert interrupt_skip_wording(agent) == "Turn aborted — guard {x}"
    finally:
        set_interrupt(False)
