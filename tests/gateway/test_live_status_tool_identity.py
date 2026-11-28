from types import SimpleNamespace
from unittest.mock import Mock

from gateway.run_turn_runner import TurnRunner


def test_live_status_shows_actual_tool_and_clears_on_completion():
    adapter = Mock()
    ctx = SimpleNamespace(
        _live_status_adapter=adapter,
        _live_status_mode="full",
        _run_still_current=lambda: True,
        source=SimpleNamespace(chat_id="C_TEST"),
    )
    runner = TurnRunner(None, ctx)
    runner._progress_live_status("tool.started", "terminal", {"command": "git status --short"})
    adapter.set_status_text.assert_called_once_with(
        "C_TEST", "is running git status --short… [terminal]"
    )
    runner._progress_live_status("tool.completed", "terminal", None)
    adapter.set_status_text.assert_called_with("C_TEST", None)


def test_disabled_live_status_does_not_emit():
    adapter = Mock()
    ctx = SimpleNamespace(_live_status_adapter=adapter, _live_status_mode="off")
    TurnRunner(None, ctx)._progress_live_status("tool.started", "terminal", {})
    adapter.set_status_text.assert_not_called()
