"""Terminal result mirrors use explicit control-plane metadata."""

from unittest.mock import MagicMock

from run_agent import AIAgent


def _bare_agent():
    agent = object.__new__(AIAgent)
    agent.log_prefix = ""
    agent._vprint = MagicMock()
    agent.status_callback = MagicMock()
    return agent


def test_terminal_status_uses_typed_final_mirror_event():
    agent = _bare_agent()

    agent._emit_terminal_status("Provider authentication failed.")

    agent.status_callback.assert_called_once_with(
        "terminal_final_mirror",
        "Provider authentication failed.",
    )


def test_regular_status_keeps_lifecycle_event_type():
    agent = _bare_agent()

    agent._emit_status("Trying fallback provider.")

    agent.status_callback.assert_called_once_with(
        "lifecycle",
        "Trying fallback provider.",
    )
