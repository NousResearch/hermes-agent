"""Regression tests for the single-query clarify guard (#94943).

``hermes chat -q`` wires the interactive prompt_toolkit clarify callback
unconditionally: a -q turn never builds the prompt_toolkit application, so
``CLIApp._clarify_callback`` polls its response queue with nothing able to
answer it — the turn hangs until ``agent.clarify_timeout`` expires (default
3600 s, 0 = unlimited). The gateway, cron jobs, the kanban dispatcher and
inter-agent wakeups all deliver work as ``hermes chat -q``, so an agent that
calls ``clarify`` in those turns stalls silently. The oneshot (-z) path
already answers immediately via ``_oneshot_clarify_callback``; this pins the
same headless behavior for -q, wired at the ``CLIAgentSetupMixin`` agent
construction site that already knows it is in single-query mode.
"""

from __future__ import annotations



def test_no_choices_returns_immediate_headless_answer():
    from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback

    result = _single_query_clarify_callback("Which timezone should I use?")
    assert result.startswith("[single-query mode: no user available")
    assert "most reasonable assumption" in result








