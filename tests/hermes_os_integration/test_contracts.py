from hermes_os_integration.contracts import (
    validate_agent_request,
    validate_agent_response,
)


def test_valid_agent_request():
    request, error = validate_agent_request({
        "task_id": "task-1",
        "project_id": "project-1",
        "agent_kind": "research",
        "prompt": "Research this.",
        "working_directory": "/tmp",
        "tool_policy": {"allowed_tools": ["filesystem"]},
    })

    assert error is None
    assert request.task_id == "task-1"
    assert request.agent_kind == "research"


def test_invalid_agent_kind_rejected():
    request, error = validate_agent_request({
        "task_id": "task-1",
        "project_id": "project-1",
        "agent_kind": "unknown",
        "prompt": "Research this.",
        "working_directory": "/tmp",
    })

    assert request is None
    assert error.code == "validation_error"


def test_valid_agent_response():
    response, error = validate_agent_response({
        "task_id": "task-1",
        "status": "completed",
        "output": "done",
    })

    assert error is None
    assert response.status == "completed"
