from types import SimpleNamespace

from agent.activity import (
    current_tool_elapsed,
    mark_concurrent_tools_started,
    mark_tool_completed,
    mark_tool_started,
    reset_turn_activity,
    todo_activity_snapshot,
    tool_activity_history,
    tool_activity_label,
)
from tools.todo_tool import TodoStore


def test_mark_tool_started_stamps_redacted_preview_and_elapsed(monkeypatch):
    agent = SimpleNamespace()
    monkeypatch.setattr("agent.activity.time.time", lambda: 100.0)

    mark_tool_started(agent, "terminal", {"command": "python -m pytest tests/gateway -q"})

    assert agent._current_tool == "terminal"
    assert "pytest" in agent._current_tool_preview
    assert agent._current_tool_started_at == 100.0
    assert current_tool_elapsed(agent, now=145.0) == 45.0


def test_mark_concurrent_tools_started_summarizes_first_three():
    agent = SimpleNamespace()

    mark_concurrent_tools_started(
        agent,
        [
            ("read_file", {"path": "/tmp/alpha.py"}),
            ("search_files", {"pattern": "TODO"}),
            ("terminal", {"command": "pytest tests/foo.py -q"}),
            ("web_search", {"query": "Hermes Agent"}),
        ],
    )

    assert agent._current_tool == "read_file, search_files, terminal, web_search"
    assert "alpha.py" in agent._current_tool_preview
    assert "TODO" in agent._current_tool_preview
    assert "+1 more" in agent._current_tool_preview


def test_mark_tool_completed_clears_active_tool_and_records_last(monkeypatch):
    agent = SimpleNamespace(_current_tool="terminal", _current_tool_preview="pytest", _current_tool_started_at=10.0)
    monkeypatch.setattr("agent.activity.time.time", lambda: 200.0)

    mark_tool_completed(agent, "terminal", 12.34, is_error=False)

    assert agent._current_tool is None
    assert agent._current_tool_preview is None
    assert agent._current_tool_started_at is None
    assert agent._last_completed_tool == {
        "name": "terminal",
        "duration": 12.34,
        "is_error": False,
        "completed_at": 200.0,
    }
    assert agent._recent_tool_activity == [{
        "name": "terminal",
        "label": "terminal",
        "duration": 12.34,
        "is_error": False,
        "state": "done",
        "completed_at": 200.0,
    }]


def test_tool_activity_history_returns_previous_two_plus_current(monkeypatch):
    agent = SimpleNamespace()
    now = 1000.0
    monkeypatch.setattr("agent.activity.time.time", lambda: now)

    for index, name in enumerate(["read_file", "search_files", "terminal"]):
        mark_tool_completed(agent, name, index + 1, is_error=False)
    mark_tool_started(agent, "terminal", {"command": "pytest tests/gateway/test_heartbeat_status.py -q"})
    agent._current_tool_started_at = now - 40

    history = tool_activity_history(agent, now=now)

    assert [item["label"] for item in history] == [
        "search_files",
        "terminal",
        "Running pytest tests/gateway/test_heartbeat_status.py -q",
    ]
    assert history[-1]["state"] == "running"
    assert history[-1]["duration"] == 40


def test_reset_turn_activity_clears_stale_metadata():
    agent = SimpleNamespace(
        _current_tool="terminal",
        _current_tool_preview="pytest",
        _current_tool_started_at=10.0,
        _last_completed_tool={"name": "terminal"},
        _recent_tool_activity=[{"label": "terminal"}],
    )

    reset_turn_activity(agent)

    assert agent._current_tool is None
    assert agent._current_tool_preview is None
    assert agent._current_tool_started_at is None
    assert agent._last_completed_tool is None
    assert agent._recent_tool_activity == []


def test_tool_activity_label_redacts_terminal_password_flags():
    label = tool_activity_label(
        "terminal",
        {"command": "mysql -uroot -psecret123 production"},
    )

    assert "secret123" not in label
    assert "-p***" in label


def test_tool_activity_label_redacts_common_secret_assignments_and_flags():
    terminal_label = tool_activity_label(
        "terminal",
        {"command": "TOKEN=abc123 curl --password hunter2 https://example.test"},
    )
    code_label = tool_activity_label(
        "execute_code",
        {"code": "MYSQL_PWD=swordfish python smoke.py"},
    )

    assert "abc123" not in terminal_label
    assert "hunter2" not in terminal_label
    assert "TOKEN=***" in terminal_label
    assert "--password ***" in terminal_label
    assert "swordfish" not in code_label
    assert "MYSQL_PWD=***" in code_label


def test_todo_activity_snapshot_prefers_in_progress_then_pending():
    store = TodoStore()
    store.write(
        [
            {"id": "a", "content": "already done", "status": "completed"},
            {"id": "b", "content": "patch heartbeat bubble", "status": "in_progress"},
            {"id": "c", "content": "run tests", "status": "pending"},
        ]
    )

    snap = todo_activity_snapshot(store)

    assert snap["status"] == "in_progress"
    assert snap["content"] == "patch heartbeat bubble"
    assert isinstance(snap["elapsed_seconds"], float)

    store.write(
        [
            {"id": "b", "content": "patch heartbeat bubble", "status": "completed"},
            {"id": "c", "content": "run tests", "status": "pending"},
        ]
    )

    snap = todo_activity_snapshot(store)
    assert snap["status"] == "pending"
    assert snap["content"] == "run tests"


def test_todo_activity_snapshot_redacts_gateway_visible_content():
    store = TodoStore()
    store.write(
        [
            {
                "id": "secret-todo",
                "content": "Run TOKEN=abc123 curl --password hunter2 and MYSQL_PWD=swordfish smoke",
                "status": "in_progress",
            }
        ]
    )

    snap = todo_activity_snapshot(store)

    assert "abc123" not in snap["content"]
    assert "hunter2" not in snap["content"]
    assert "swordfish" not in snap["content"]
    assert "TOKEN=***" in snap["content"]
    assert "--password ***" in snap["content"]
    assert "MYSQL_PWD=***" in snap["content"]
