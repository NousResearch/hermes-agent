"""`hermes chat -Q` must not leak presentation output into stdout (#93220)."""

from __future__ import annotations

def test_suppress_status_output_gates_quiet_tool_messages():
    """The executor's [tool]/[done] fallback must stay silent under -Q.

    ``_should_emit_quiet_tool_messages`` is the gate for the quiet-mode
    KawaiiSpinner fallback in agent/tool_executor.py; with the rendering
    callbacks neutralized it would otherwise print ``[tool]``/``[done]``
    lines straight into -Q's captured stdout (#93220).
    """
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.quiet_mode = True
    agent.tool_progress_callback = None
    agent.platform = "cli"

    agent.suppress_status_output = False
    assert agent._should_emit_quiet_tool_messages() is True

    agent.suppress_status_output = True
    assert agent._should_emit_quiet_tool_messages() is False
