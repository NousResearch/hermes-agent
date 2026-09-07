"""Complete native/status event copies are masked before their display caps."""
import copy
import queue
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource


@pytest.mark.parametrize("lane", ["native", "live"])
def test_complete_progress_event_masks_before_cap_without_mutating_args(monkeypatch, lane):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    values = []
    adapter = SimpleNamespace(set_status_text=lambda chat_id, value: values.append((chat_id, value)))
    ctx = SimpleNamespace(
        source=source, progress_queue=queue.Queue(), _run_still_current=lambda: True,
        agent_holder=[None], _live_status_adapter=adapter, _live_status_mode="full",
    )
    runner = TurnRunner(SimpleNamespace(), ctx)
    args = {"command": "echo " + "x" * 15 + " github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}
    before = copy.deepcopy(args)
    if lane == "native":
        runner.native_tool_start_callback("call-1", "terminal", args)
        payload = ctx.progress_queue.get_nowait()
        assert payload["tool_call_id"] == "call-1"
        text = payload["preview"]
    else:
        runner._progress_live_status("tool.started", "terminal", args)
        assert values[0][0] == source.chat_id
        text = values[0][1]
    assert args == before
    assert "github_pat_AB" not in text
