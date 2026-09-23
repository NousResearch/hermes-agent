from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agent.action_mutations import MutationSpec, record_tool_mutation
from gateway.action_journal import ActionJournal
from tools.registry import ToolRegistry

NOW = datetime(2026, 9, 22, 18, 0, tzinfo=UTC)


def test_explicit_mutation_metadata_is_validated_and_not_added_to_schema() -> None:
    registry = ToolRegistry()
    registry.register(
        name="connected_write",
        toolset="connected",
        schema={
            "name": "connected_write",
            "description": "Write something",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kwargs: '{"success": true}',
        mutation=MutationSpec(
            action_type="connected_tool_change",
            provider="Example",
            operation="update_state",
        ),
    )
    entry = registry.get_entry("connected_write")
    assert entry is not None
    assert entry.mutation is not None
    assert "mutation" not in registry.get_definitions({"connected_write"})[0]["function"]


def test_reviewed_calendar_mutation_is_projected_without_raw_data(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky", now=lambda: NOW)
    event = record_tool_mutation(
        function_name="mcp_google_calendar_create_event",
        function_args={"summary": "Secret meeting", "event_id": "private-id"},
        result='{"success": true, "event_id": "private-id", "token": "secret"}',
        status="ok",
        session_id="telegram-session",
        turn_id="turn-1",
        tool_call_id="call-1",
        journal=journal,
        now=lambda: NOW,
    )
    assert event is not None
    assert event.action_type.value == "calendar"
    assert event.status.value == "succeeded"
    assert "Secret meeting" not in event.model_dump_json()
    assert "private-id" not in event.model_dump_json()
    assert "secret" not in event.model_dump_json()


def test_failed_todoist_mutation_is_recorded_with_safe_bounded_projection(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky", now=lambda: NOW)
    event = record_tool_mutation(
        function_name="mcp_todoist_add_tasks",
        function_args={"tasks": [{"content": "private task", "id": "123"}]},
        result={"success": False, "error": "provider secret response"},
        status="error",
        session_id="telegram-session",
        turn_id="turn-2",
        tool_call_id="call-2",
        journal=journal,
        now=lambda: NOW,
    )
    assert event is not None
    assert event.action_type.value == "todoist"
    assert event.status.value == "failed"
    assert len(event.description) <= 1_000
    assert "provider secret" not in event.model_dump_json()


def test_read_search_and_unannotated_tools_are_excluded(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky")
    for name in (
        "mcp_google_calendar_list_events",
        "mcp_todoist_search_tasks",
        "mcp_unknown_update_state",
    ):
        assert record_tool_mutation(
            function_name=name,
            function_args={},
            result={"success": True},
            status="ok",
            session_id="session",
            turn_id="turn",
            tool_call_id=name,
            journal=journal,
        ) is None
    assert journal.list(after_cursor=None, limit=10).events == []


def test_explicit_home_and_network_metadata_support_normal_and_delegated_workflows(
    tmp_path: Path,
) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky")
    specs = {
        "home_change": MutationSpec(
            action_type="home_automation", provider="Home Assistant", operation="call_service"
        ),
        "firewall_change": MutationSpec(
            action_type="network", provider="Firewall", operation="apply_rule"
        ),
    }
    registry = ToolRegistry()
    for name, spec in specs.items():
        registry.register(
            name=name,
            toolset="connected",
            schema={"name": name, "description": name, "parameters": {"type": "object"}},
            handler=lambda args, **kwargs: '{"success": true}',
            mutation=spec,
        )
        event = record_tool_mutation(
            function_name=name,
            function_args={"credential": "do-not-store"},
            result={"success": True},
            status="ok",
            session_id="delegated-session",
            turn_id="turn-3",
            tool_call_id=name,
            journal=journal,
            registry=registry,
            now=lambda: NOW,
        )
        assert event is not None
        assert event.action_type.value == spec.action_type
    assert len(journal.list(after_cursor=None, limit=10).events) == 2


def test_repeated_post_tool_observation_is_idempotent(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky", now=lambda: NOW)
    kwargs = {
        "function_name": "mcp_todoist_add_tasks",
        "function_args": {"tasks": [{"content": "secret"}]},
        "result": {"success": True},
        "status": "ok",
        "session_id": "telegram-session",
        "turn_id": "turn-4",
        "tool_call_id": "call-4",
        "journal": journal,
        "now": lambda: NOW,
    }
    first = record_tool_mutation(**kwargs)
    second = record_tool_mutation(**kwargs)
    assert first is not None and second is not None
    assert first.source_event_key == second.source_event_key
    assert len(journal.list(after_cursor=None, limit=10).events) == 1


def test_common_post_tool_boundary_records_normal_workflow_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    from agent.action_mutations import close_action_journals
    from model_tools import _emit_post_tool_call_hook

    close_action_journals()
    _emit_post_tool_call_hook(
        function_name="mcp_google_calendar_create_event",
        function_args={"summary": "private appointment"},
        result={"success": True, "event_id": "private"},
        session_id="telegram-session",
        turn_id="turn-normal",
        tool_call_id="call-normal",
        status="ok",
    )
    from hermes_constants import hermes_home_key

    journal = ActionJournal(
        tmp_path / "profile" / "gateway" / "becky-actions.sqlite3",
        profile_key=hermes_home_key(tmp_path / "profile"),
    )
    assert len(journal.list(after_cursor=None, limit=10).events) == 1
    close_action_journals()
