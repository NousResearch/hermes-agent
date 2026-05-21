from __future__ import annotations

from agent.managed_agents.event_log import (
    EVENT_REVIEW_COMPLETED,
    EVENT_REVIEW_REQUESTED,
    EVENT_TASK_DELEGATED,
    EVENT_TASK_RESULT_RECEIVED,
    EVENT_TOOL_PERMISSION_DENIED,
    ManagedAgentEventLog,
)


def _make_log(tmp_path):
    return ManagedAgentEventLog.from_db_path(tmp_path / "events.db")


def test_logs_delegation_result_and_permission_denial(tmp_path):
    event_log = _make_log(tmp_path)

    event_log.log_task_delegated(
        task_id="task-1",
        session_id="sess-1",
        from_agent="hermes",
        to_agent="claude",
        risk_level="R2",
    )
    event_log.log_task_result_received(
        task_id="task-1",
        session_id="sess-1",
        from_agent="claude",
        status="done",
    )
    event_log.log_tool_permission_denied(
        task_id="task-1",
        session_id="sess-1",
        agent_id="claude",
        tool_name="terminal",
        reason="R4 requires approval",
    )

    events = event_log.replay_task("task-1")

    assert [event["type"] for event in events] == [
        EVENT_TASK_DELEGATED,
        EVENT_TASK_RESULT_RECEIVED,
        EVENT_TOOL_PERMISSION_DENIED,
    ]
    assert events[0]["payload"]["to"] == "claude"
    assert events[1]["payload"]["status"] == "done"
    assert events[2]["payload"]["tool_name"] == "terminal"


def test_replay_filters_by_task_id_and_keeps_order(tmp_path):
    event_log = _make_log(tmp_path)

    event_log.log_task_delegated(
        task_id="task-1",
        session_id="sess-1",
        from_agent="hermes",
        to_agent="claude",
    )
    event_log.log_task_delegated(
        task_id="task-2",
        session_id="sess-1",
        from_agent="hermes",
        to_agent="deepseek",
    )
    event_log.log_review_requested(
        task_id="task-1",
        session_id="sess-1",
        reviewer="codex",
        subject_agent="claude",
    )

    events = event_log.get_timeline("task-1")

    assert [event["type"] for event in events] == [
        EVENT_TASK_DELEGATED,
        EVENT_REVIEW_REQUESTED,
    ]
    assert {event["task_id"] for event in events} == {"task-1"}


def test_export_audit_report_contains_summary_buckets(tmp_path):
    event_log = _make_log(tmp_path)

    event_log.log_task_delegated(
        task_id="task-1",
        session_id="sess-1",
        from_agent="hermes",
        to_agent="claude",
    )
    event_log.log_review_requested(
        task_id="task-1",
        session_id="sess-1",
        reviewer="codex",
        subject_agent="claude",
    )
    event_log.log_review_completed(
        task_id="task-1",
        session_id="sess-1",
        reviewer="codex",
        decision="pass",
    )
    event_log.log_tool_permission_denied(
        task_id="task-1",
        session_id="sess-1",
        agent_id="claude",
        tool_name="terminal",
        reason="not allowed",
    )

    report = event_log.export_audit_report("task-1")

    assert report["event_count"] == 4
    assert len(report["delegations"]) == 1
    assert [event["type"] for event in report["reviews"]] == [
        EVENT_REVIEW_REQUESTED,
        EVENT_REVIEW_COMPLETED,
    ]
    assert len(report["permission_denials"]) == 1
    assert report["final_status"] == "pass"


def test_redacts_secrets_recursively(tmp_path):
    event_log = _make_log(tmp_path)

    event_log.log_task_delegated(
        task_id="task-1",
        session_id="sess-1",
        from_agent="hermes",
        to_agent="claude",
        metadata={
            "api_key": "sk-test",
            "nested": {
                "token": "tok-test",
                "safe": "visible",
            },
            "items": [{"password": "pw"}],
        },
    )

    payload = event_log.replay_task("task-1")[0]["payload"]

    assert payload["metadata"]["api_key"] == "[REDACTED]"
    assert payload["metadata"]["nested"]["token"] == "[REDACTED]"
    assert payload["metadata"]["nested"]["safe"] == "visible"
    assert payload["metadata"]["items"][0]["password"] == "[REDACTED]"
