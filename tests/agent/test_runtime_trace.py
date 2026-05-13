"""Tests for agent/runtime_trace.py — safe runtime observability JSONL."""

import json
from pathlib import Path


def test_emit_runtime_event_writes_jsonl_with_required_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from agent.runtime_trace import emit_runtime_event

    emit_runtime_event(
        "assign_agent.resolved",
        session_id="session-1",
        task_id="task-1",
        agent={"name": "smoke-cli"},
    )

    trace_path = tmp_path / ".hermes" / "logs" / "runtime-trace.jsonl"
    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    event = json.loads(lines[0])
    assert event["event"] == "assign_agent.resolved"
    assert event["session_id"] == "session-1"
    assert event["task_id"] == "task-1"
    assert event["data"]["agent"]["name"] == "smoke-cli"
    assert event["ts"].endswith("Z")


def test_emit_runtime_event_redacts_secret_like_keys_recursively(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from agent.runtime_trace import emit_runtime_event, read_runtime_events

    emit_runtime_event(
        "model.request",
        session_id="session-1",
        api_key="sk-secret",
        nested={
            "Authorization": "Bearer secret",
            "safe": "visible",
            "items": [{"password": "p4ss", "token_value": "tok"}],
        },
    )

    [event] = read_runtime_events(session_id="session-1")
    assert event["data"]["api_key"] == "[REDACTED]"
    assert event["data"]["nested"]["Authorization"] == "[REDACTED]"
    assert event["data"]["nested"]["safe"] == "visible"
    assert event["data"]["nested"]["items"][0]["password"] == "[REDACTED]"
    assert event["data"]["nested"]["items"][0]["token_value"] == "[REDACTED]"


def test_read_runtime_events_filters_by_session_agent_and_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from agent.runtime_trace import emit_runtime_event, read_runtime_events

    emit_runtime_event("assign_agent.completed", session_id="s1", agent={"name": "a"}, seq=1)
    emit_runtime_event("assign_agent.completed", session_id="s1", agent={"name": "b"}, seq=2)
    emit_runtime_event("assign_agent.completed", session_id="s1", agent={"name": "a"}, seq=3)
    emit_runtime_event("assign_agent.completed", session_id="s2", agent={"name": "a"}, seq=4)

    events = read_runtime_events(session_id="s1", agent_name="a", limit=1)

    assert len(events) == 1
    assert events[0]["session_id"] == "s1"
    assert events[0]["data"]["agent"]["name"] == "a"
    assert events[0]["data"]["seq"] == 3


def test_emit_runtime_event_is_best_effort_when_log_path_unwritable(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    logs_path = hermes_home / "logs"
    logs_path.parent.mkdir(parents=True)
    logs_path.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.runtime_trace import emit_runtime_event

    # Must not raise even though $HERMES_HOME/logs is a file.
    emit_runtime_event("assign_agent.requested", session_id="s1")
