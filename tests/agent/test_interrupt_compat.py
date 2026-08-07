"""Compatibility contract for explicit hard-stop producers."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from agent.interrupt_compat import request_hard_interrupt


class _ModernAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def hard_interrupt(self, message: str | None = None) -> None:
        self.calls.append(("hard", message))

    def interrupt(self, message: str | None = None) -> None:
        self.calls.append(("soft", message))


class _LegacyAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def interrupt(self, message: str | None = None) -> None:
        self.calls.append(("legacy", message))


def test_explicit_producer_prefers_feature_detected_hard_interrupt() -> None:
    agent = _ModernAgent()

    assert request_hard_interrupt(agent, "stop now") is True

    assert agent.calls == [("hard", "stop now")]


def test_explicit_producer_falls_back_to_old_interrupt_signature() -> None:
    agent = _LegacyAgent()

    assert request_hard_interrupt(agent, "stop now") is True

    assert agent.calls == [("legacy", "stop now")]


def test_explicit_producer_reports_unsupported_agent() -> None:
    assert request_hard_interrupt(object(), "stop now") is False


def test_dynamic_proxy_does_not_fabricate_hard_interrupt_support() -> None:
    agent = MagicMock()

    assert request_hard_interrupt(agent, "stop now") is True

    agent.interrupt.assert_called_once_with("stop now")
    agent.hard_interrupt.assert_not_called()


def test_inherited_hard_interrupt_bypasses_legacy_subclass_override() -> None:
    from run_agent import AIAgent

    class LegacySubclass(AIAgent):
        def __init__(self) -> None:
            self.legacy_calls: list[str | None] = []
            self._hard_interrupt_requested = threading.Event()
            self._pending_redirect_lock = threading.RLock()
            self._pending_redirect = None
            self._execution_thread_id = None
            self._interrupt_thread_signal_pending = False
            self._tool_worker_threads: set[int] = set()
            self._tool_worker_threads_lock = threading.Lock()
            self._active_children: list[object] = []
            self._active_children_lock = threading.Lock()
            self.quiet_mode = True
            self.api_mode = "test"

        def interrupt(self, message: str | None = None) -> None:  # type: ignore[override]
            self.legacy_calls.append(message)

    agent = LegacySubclass()

    assert request_hard_interrupt(agent, "stop now") is True

    assert agent.legacy_calls == []
    assert agent._hard_interrupt_requested.is_set()
    assert agent._interrupt_requested is True
    assert agent._interrupt_message == "stop now"


def test_tui_subagent_interrupt_is_an_explicit_hard_stop() -> None:
    import tools.delegate_tool as delegate_tool

    agent = _ModernAgent()
    subagent_id = "sa-hard-stop-test"
    with delegate_tool._active_subagents_lock:
        delegate_tool._active_subagents[subagent_id] = {"agent": agent}
    try:
        assert delegate_tool.interrupt_subagent(subagent_id) is True
    finally:
        with delegate_tool._active_subagents_lock:
            delegate_tool._active_subagents.pop(subagent_id, None)

    assert agent.calls == [("hard", f"Interrupted via TUI ({subagent_id})")]
