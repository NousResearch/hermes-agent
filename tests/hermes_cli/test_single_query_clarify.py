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

def test_returns_immediate_undelivered_reply():
    from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback

    result = _single_query_clarify_callback([{
        "qid": "q0", "question": "Which timezone should I use?", "choices": None,
        "choices_offered": None, "multi_select": False}])
    assert result["answers"] == {}
    assert result["outcome"] == "undelivered"
    assert "no user available" in result["notice"]
