import inspect
import json
from pathlib import Path

import pytest

import mcp_servers.comms_policy as policy


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    try:
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    except Exception:
        pass
    return tmp_path


class _FakeTool:
    def __init__(self, fn):
        self.name = fn.__name__
        self.description = inspect.getdoc(fn) or ""
        self.fn = fn


class _FakeToolManager:
    def __init__(self):
        self._tools = {}

    def add_tool(self, fn):
        self._tools[fn.__name__] = _FakeTool(fn)

    def list_tools(self):
        return list(self._tools.values())


class _FakeFastMCP:
    def __init__(self, *args, **kwargs):
        self._tool_manager = _FakeToolManager()

    def tool(self):
        def decorator(fn):
            self._tool_manager.add_tool(fn)
            return fn

        return decorator

    def run(self):
        return None


def test_success_failure_and_ready_question_formatting_is_concise():
    success = policy.format_success_notice(
        "Kanban task complete",
        ["tests pass", "PR opened"],
        task_id="t_abc123",
        next_step="Review PR",
    )
    assert success == "✅ Kanban task complete\nTask: t_abc123\n• tests pass\n• PR opened\nNext: Review PR"

    failure = policy.format_failure_notice(
        "Kanban worker failed",
        "pytest timed out after 60s",
        task_id="t_abc123",
        next_step="Retry targeted tests",
    )
    assert failure == (
        "❌ Kanban worker failed\n"
        "Task: t_abc123\n"
        "Error: pytest timed out after 60s\n"
        "Next: Retry targeted tests"
    )

    question = policy.format_ready_question("Deploy now?", context=["staging passed", "prod is quiet"])
    assert question == "❓ Deploy now?\n• staging passed\n• prod is quiet\nReply with the choice or details."


def test_identical_notifications_are_deduped_with_sqlite_state(tmp_path):
    db_path = tmp_path / "notifications.sqlite"
    sent = []

    def sender(target, message):
        sent.append((target, message))
        return {"ok": True, "message_id": len(sent)}

    first = policy.send_policy_notice(
        target="telegram",
        category="success",
        message="✅ Backup complete",
        dedupe_key="backup:daily",
        now=1000,
        sender=sender,
        db_path=db_path,
    )
    assert first["sent"] is True
    assert first["suppressed"] is False

    second = policy.send_policy_notice(
        target="telegram",
        category="success",
        message="✅ Backup complete",
        dedupe_key="backup:daily",
        now=1010,
        sender=sender,
        db_path=db_path,
    )
    assert second["sent"] is False
    assert second["suppressed"] is True
    assert second["reason"] == "duplicate"
    assert sent == [("telegram", "✅ Backup complete")]

    last = policy.check_last_sent_record(
        target="telegram",
        message="✅ Backup complete",
        dedupe_key="backup:daily",
        db_path=db_path,
    )
    assert last["found"] is True
    assert last["status"] == "sent"
    assert last["reason"] == ""


def test_stale_event_is_suppressed_and_not_delivered(tmp_path):
    sent = []

    result = policy.send_policy_notice(
        target="telegram",
        category="failure",
        message="❌ Worker failed",
        dedupe_key="worker:t_123",
        event_ts=1000,
        stale_after_seconds=60,
        now=1120,
        sender=lambda target, message: sent.append((target, message)) or {"ok": True},
        db_path=tmp_path / "notifications.sqlite",
    )

    assert result["sent"] is False
    assert result["suppressed"] is True
    assert result["reason"] == "stale"
    assert result["age_seconds"] == 120
    assert sent == []


def test_queue_notification_records_queued_and_prevents_duplicate_send(tmp_path):
    db_path = tmp_path / "notifications.sqlite"

    queued = policy.send_policy_notice(
        target="telegram",
        category="briefing",
        message="📋 Morning brief\n• Item A",
        dedupe_key="brief:morning",
        dry_run=True,
        now=1000,
        db_path=db_path,
    )
    assert queued["queued"] is True
    assert queued["sent"] is False

    duplicate = policy.dedupe_notification_policy(
        target="telegram",
        category="briefing",
        message="📋 Morning brief\n• Item A",
        dedupe_key="brief:morning",
        now=1001,
        db_path=db_path,
    )
    assert duplicate["should_send"] is False
    assert duplicate["reason"] == "duplicate"


def test_mcp_server_registers_expected_tools(monkeypatch):
    monkeypatch.setattr(policy, "FastMCP", _FakeFastMCP)
    monkeypatch.setattr(policy, "_MCP_SERVER_AVAILABLE", True)

    server = policy.build_server()
    names = {tool.name for tool in server._tool_manager.list_tools()}

    assert {
        "send_success_notice",
        "send_failure_notice",
        "send_briefing",
        "dedupe_notification",
        "check_last_sent",
        "queue_notification",
        "format_ready_question_notice",
    } <= names

    response = server._tool_manager._tools["send_success_notice"].fn(
        title="Tests passed",
        details=["5 passed"],
        dedupe_key="tests:passed",
        dry_run=True,
    )
    payload = json.loads(response)
    assert payload["queued"] is True
    assert payload["message"] == "✅ Tests passed\n• 5 passed"


def test_default_state_path_is_profile_scoped(tmp_path):
    assert policy.default_db_path() == Path(tmp_path) / "comms_policy" / "notifications.sqlite"
