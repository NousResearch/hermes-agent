from __future__ import annotations

import pytest

from gateway.becky_loops import should_auto_close_becky_loop


def _result(*events: dict, **overrides: object) -> dict[str, object]:
    return {
        "completed": True,
        "failed": False,
        "partial": False,
        "interrupted": False,
        "final_response": "Done.",
        "turn_exit_reason": "text_response(finish_reason=stop)",
        "turn_tool_events": list(events),
        **overrides,
    }


def _event(
    name: str,
    *,
    success: bool = True,
    arguments: dict | None = None,
    exit_code: int | None = None,
) -> dict:
    event = {
        "name": name,
        "requested_name": name,
        "arguments": arguments or {},
        "success": success,
    }
    if exit_code is not None:
        event["exit_code"] = exit_code
    return event


def test_closes_for_one_successful_todoist_task_add() -> None:
    assert should_auto_close_becky_loop(
        _result(_event("mcp_todoist_add_tasks"))
    ) is True


def test_closes_for_one_successful_google_calendar_event_create() -> None:
    assert should_auto_close_becky_loop(
        _result(_event("google_calendar_create_event"))
    ) is True


def test_closes_for_existing_google_workspace_calendar_command() -> None:
    assert should_auto_close_becky_loop(
        _result(
            _event(
                "terminal",
                arguments={
                    "command": "python google_api.py calendar create --summary Meeting"
                },
                exit_code=0,
            )
        )
    ) is True


def test_accepts_quoted_gapi_calendar_arguments() -> None:
    assert should_auto_close_becky_loop(
        _result(
            _event(
                "terminal",
                arguments={
                    "command": (
                        '$GAPI calendar create --summary "Team Standup" '
                        '--start 2026-03-01T10:00:00-06:00 '
                        '--end 2026-03-01T10:30:00-06:00'
                    )
                },
                exit_code=0,
            )
        )
    ) is True


def test_rejects_truncated_terminal_evidence() -> None:
    assert should_auto_close_becky_loop(
        _result(
            _event(
                "terminal",
                arguments={
                    "command": "gws calendar events insert --summary Meeting",
                    "command_truncated": True,
                },
                exit_code=0,
            )
        )
    ) is False


def test_rejects_an_action_unwrapped_through_tool_search() -> None:
    assert should_auto_close_becky_loop(
        _result(
            {
                "name": "mcp_todoist_add_tasks",
                "requested_name": "tool_call",
                "via_tool_search": True,
                "success": True,
                "arguments": {},
            }
        )
    ) is False


@pytest.mark.parametrize(
    "result",
    [
        _result(_event("mcp_todoist_add_tasks", success=False)),
        _result(_event("google_calendar_create_event"), completed=False),
        _result(_event("google_calendar_create_event"), failed=True),
        _result(_event("google_calendar_create_event"), interrupted=True),
        _result(
            _event("google_calendar_create_event"),
            turn_exit_reason="error_near_max_iterations(provider failed)",
        ),
        _result(_event("google_calendar_create_event"), turn_exit_reason=None),
        _result(_event("google_calendar_create_event"), error="provider failed"),
        _result(_event("mcp_todoist_add_tasks"), final_response=""),
        _result(_event("mcp_todoist_add_tasks"), _event("web_search")),
        _result(_event("web_search")),
        _result(_event("mcp_todoist_delete_and_add_task")),
        _result(_event("delete_todoist_add_tasks")),
        _result(_event("google_calendar_create_event_and_delete")),
        _result(_event("delete_calendar_create_event")),
        _result(
            {
                "name": "mcp_todoist_add_tasks",
                "success": True,
                "arguments": {},
            }
        ),
        _result(
            {
                "name": "google_calendar_create_event",
                "requested_name": "mcp_todoist_add_tasks",
                "success": True,
                "arguments": {},
            }
        ),
        _result(_event("terminal", arguments={"command": "python google_api.py calendar create", "background": True})),
        _result(
            _event(
                "terminal",
                arguments={"command": "python google_api.py calendar create"},
                exit_code=1,
            )
        ),
    ],
)
def test_keeps_complex_failed_or_ambiguous_work_open(result: dict[str, object]) -> None:
    assert should_auto_close_becky_loop(result) is False


def test_accepts_gws_calendar_insert_when_it_is_a_single_successful_command() -> None:
    assert should_auto_close_becky_loop(
        _result(
            _event(
                "terminal",
                arguments={
                    "command": "gws calendar events insert --summary Meeting"
                },
                exit_code=0,
            )
        )
    ) is True


@pytest.mark.parametrize(
    "command",
    [
        "echo done && gws calendar events insert --summary Meeting",
        "gws calendar events insert --summary Meeting | tee /tmp/out",
        "gws calendar events insert --summary Meeting; echo done",
        "python google_api.py calendar create --summary $(touch /tmp/owned)",
        "gws calendar events insert --summary `touch /tmp/owned`",
        "$GAPI calendar create --summary ${UNTRUSTED}",
    ],
)
def test_rejects_composed_terminal_commands(command: str) -> None:
    assert should_auto_close_becky_loop(
        _result(_event("terminal", arguments={"command": command}, exit_code=0))
    ) is False
