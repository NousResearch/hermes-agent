from hermes_cli.workflows_capabilities import (
    IMPLEMENTED_NODE_TYPES,
    IMPLEMENTED_TRIGGER_TYPES,
    UNSUPPORTED_NODE_TYPES,
    UNSUPPORTED_TRIGGER_TYPES,
    workflow_capabilities,
)


def test_capabilities_separate_implemented_from_unsupported():
    assert IMPLEMENTED_TRIGGER_TYPES == {"manual", "schedule"}
    assert UNSUPPORTED_TRIGGER_TYPES == {"webhook", "kanban_event"}
    assert IMPLEMENTED_NODE_TYPES == {
        "pass",
        "switch",
        "agent_task",
        "wait",
        "parallel",
        "join",
        "fail",
    }
    assert UNSUPPORTED_NODE_TYPES == {"send_message", "subworkflow"}


def test_capabilities_payload_is_dashboard_friendly():
    payload = workflow_capabilities()
    assert payload["triggers"]["implemented"] == ["manual", "schedule"]
    assert payload["nodes"]["unsupported"] == ["send_message", "subworkflow"]
    assert payload["assistant"]["allowed_triggers"] == ["manual", "schedule"]
    assert "agent_task" in payload["assistant"]["allowed_nodes"]
