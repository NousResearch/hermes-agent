import time
from types import SimpleNamespace

from run_agent import AIAgent
from tools.todo_tool import TodoStore


class _Budget:
    used = 0
    max_total = 10


def test_get_activity_summary_exposes_heartbeat_visibility_fields():
    store = TodoStore()
    store.write([
        {"id": "patch", "content": "Patch heartbeat bubble", "status": "in_progress"}
    ])
    agent = SimpleNamespace(
        _last_activity_ts=time.time() - 5,
        _last_activity_desc="executing tool: terminal",
        _current_tool="terminal",
        _current_tool_preview="pytest tests/gateway/test_heartbeat_status.py -q",
        _current_tool_started_at=time.time() - 7,
        _last_completed_tool={"name": "read_file", "duration": 0.2, "is_error": False, "completed_at": time.time() - 1},
        _recent_tool_activity=[{"label": "read_file", "duration": 0.2, "state": "done"}],
        _todo_store=store,
        _api_call_count=3,
        max_iterations=120,
        iteration_budget=_Budget(),
    )

    summary = AIAgent.get_activity_summary(agent)

    assert summary["current_tool"] == "terminal"
    assert summary["current_tool_preview"] == "pytest tests/gateway/test_heartbeat_status.py -q"
    assert summary["current_tool_elapsed"] >= 6
    assert summary["last_completed_tool"]["name"] == "read_file"
    assert [item["label"] for item in summary["recent_tool_activity"]] == [
        "read_file",
        "pytest tests/gateway/test_heartbeat_status.py -q",
    ]
    assert summary["recent_tool_activity"][-1]["state"] == "running"
    assert summary["current_todo"]["content"] == "Patch heartbeat bubble"
    assert summary["current_todo"]["status"] == "in_progress"
