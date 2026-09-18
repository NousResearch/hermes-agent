"""``interrupt()`` records who asked for the stop, so the turn exit reason can attribute it (#112647).

Every system watchdog reaches the agent through ``request_hard_interrupt(..., tool_reason=...)``;
the published ``_tool_interrupt_reason`` is the single source the exit reason is derived from.
"""

from __future__ import annotations

import logging
import threading

from agent.interrupt_compat import request_hard_interrupt
from agent.interrupt_control import interrupt_issuer
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
        assert published == ["Interrupt requested (hard): cron inactivity watchdog"]
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


def test_cli_signal_handler_names_a_system_issuer_for_sigterm_and_sighup():
    """cli.py's own OS-signal interrupt path is a system stop too: SIGTERM/SIGHUP mean an external
    process manager (kill, systemd, the kanban dispatcher reaping a worker) ended this process, not
    a human choosing to stop the turn — it must not fall through to the reason-less default that
    books interrupted_by_user (the same #112647 class already fixed for the cron/gateway
    producers, missed for cli.py's own signal handlers). SIGINT stays on the default: in
    single-query mode it is also how a real interactive Ctrl+C arrives, and that IS a user stop."""
    import os
    import signal

    import cli

    prior_grace = os.environ.get("HERMES_SIGTERM_GRACE")
    os.environ["HERMES_SIGTERM_GRACE"] = "0"
    try:
        term_agent = _bare_agent()
        cli._interrupt_agent_for_signal(term_agent, signal.SIGTERM)
        assert interrupt_issuer(term_agent) == "external_signal"

        if hasattr(signal, "SIGHUP"):
            hup_agent = _bare_agent()
            cli._interrupt_agent_for_signal(hup_agent, signal.SIGHUP)
            assert interrupt_issuer(hup_agent) == "external_signal"

        int_agent = _bare_agent()
        cli._interrupt_agent_for_signal(int_agent, signal.SIGINT)
        assert interrupt_issuer(int_agent) is None
    finally:
        if prior_grace is None:
            os.environ.pop("HERMES_SIGTERM_GRACE", None)
        else:
            os.environ["HERMES_SIGTERM_GRACE"] = prior_grace
        set_interrupt(False)
