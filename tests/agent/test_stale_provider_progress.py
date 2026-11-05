"""Regression tests for stale provider diagnostics reaching delegation parents."""

from agent.chat_completion_helpers import _report_stale_nonstream_kill
from tools.delegate_tool_progress import DelegateEvent


class _Agent:
    provider = "test-provider"
    model = "test-model"

    def __init__(self):
        self.status = []
        self.events = []

    def _buffer_diagnostic_status(self, value):
        self.status.append(value)

    def tool_progress_callback(self, event_type, **kwargs):
        self.events.append((event_type, kwargs))


def test_stale_nonstream_kill_relays_progress_to_parent():
    agent = _Agent()

    _report_stale_nonstream_kill(agent, {"model": "test-model"}, 300, 300)

    assert agent.status
    assert agent.events == [
        (DelegateEvent.TASK_PROGRESS, {"tool_name": "No response from provider for 300s (non-streaming, model: test-model); retrying after stale call. Aborting call."})
    ]
