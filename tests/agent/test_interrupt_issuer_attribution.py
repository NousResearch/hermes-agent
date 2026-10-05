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
    agent = _bare_agent()
    try:
        agent.interrupt("terminal batch tool did not complete", tool_reason="terminal batch aborted")
        assert agent._interrupt_requested is True
        assert interrupt_issuer(agent) == "terminal_batch_aborted"
    finally:
        set_interrupt(False)


def test_soft_interrupt_message_without_tool_reason_stays_a_user_stop():
    """Regression guard for the heuristic itself: a steer/new-message soft interrupt carries no
    system reason and must keep reading as a human stop."""
    agent = _bare_agent()
    try:
        agent.interrupt("what about the other file?")
        assert interrupt_issuer(agent) is None
        agent.clear_interrupt()
        agent.interrupt()
        assert interrupt_issuer(agent) is None
    finally:
        set_interrupt(False)


def test_skip_wording_never_blames_the_user_for_a_system_stop():
    """The skipped-call notice is rendered from the RECORDED reason. A system abort must describe
    itself; only the genuinely user-initiated reasons may say the user did it (#130207)."""
    agent = _bare_agent()
    try:
        # System abort: the batch guard with its own reason.
        agent.interrupt("terminal batch tool did not complete", tool_reason="terminal batch aborted")
        wording = interrupt_skip_wording(agent)
        assert "user" not in wording.lower(), wording
        assert "terminal batch aborted" in wording

        # Genuine user stop keeps the user-facing wording.
        agent.clear_interrupt()
        agent.interrupt("next message")
        assert interrupt_skip_wording(agent) == "User sent a new message"

        # Reason-less turn where nothing was recorded at all.
        agent.clear_interrupt()
        assert interrupt_skip_wording(agent) == "Turn interrupted"
    finally:
        set_interrupt(False)


def test_concurrent_skip_wording_follows_the_recorded_reason():
    """The concurrent batch path renders its skipped-call notices from the RECORDED reason
    like the sequential path does — a system abort must not read as a user stop (#130207)."""
    import types as _types
    from agent.tool_executor import execute_tool_calls_concurrent

    class _Agent:
        _interrupt_requested = True
        _tool_interrupt_reason = "lease lost"
        quiet_mode = False
        log_prefix = ""
        _incremental_persistence_failed = False
        def _vprint(self, *a, **k): pass
        def _safe_print(self, *a, **k): pass
        def _should_emit_quiet_tool_messages(self): return False
        def _touch_activity(self, *a): pass

    tc = _types.SimpleNamespace(function=_types.SimpleNamespace(name="terminal", arguments="{}"), id="c1")
    messages = []
    execute_tool_calls_concurrent(_Agent(), _types.SimpleNamespace(tool_calls=[tc]), messages, "t")
    notice = messages[-1]["content"]
    assert "user" not in notice.lower(), notice
    assert "lease lost" in notice, notice


def test_skip_wording_escapes_braces_from_free_form_reasons():
    """A caller-supplied ``tool_reason`` may contain braces; the wording feeds notice
    templates, so braces must survive as literals and never become format fields."""
    agent = _bare_agent()
    try:
        agent.interrupt("stop", tool_reason="bad {oops} reason")
        wording = interrupt_skip_wording(agent)
        assert "{{oops}}" in wording  # escaped for template interpolation
        # The shipped notice shape renders cleanly through replace-based substitution.
        content = f"[Tool execution cancelled — {{name}} was skipped. {wording}]"
        out = content.replace("{name}", "terminal")
        assert "{oops}" in out and "KeyError" not in out
    finally:
        set_interrupt(False)
