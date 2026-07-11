import typing

import pytest

from hermes_cli.workflows_capabilities import (
    DECLARED_NODE_TYPES,
    DECLARED_TRIGGER_TYPES,
    IMPLEMENTED_NODE_TYPES,
    IMPLEMENTED_TRIGGER_TYPES,
    UNSUPPORTED_NODE_TYPES,
    UNSUPPORTED_TRIGGER_TYPES,
    implemented_primitive_errors,
    require_implemented_primitives,
    workflow_capabilities,
)
from hermes_cli.workflows_spec import NodeType, TriggerType, WorkflowSpec


def test_capabilities_implemented_subset_of_declared():
    assert IMPLEMENTED_TRIGGER_TYPES <= DECLARED_TRIGGER_TYPES
    assert IMPLEMENTED_NODE_TYPES <= DECLARED_NODE_TYPES
    assert IMPLEMENTED_TRIGGER_TYPES & UNSUPPORTED_TRIGGER_TYPES == set()
    assert IMPLEMENTED_NODE_TYPES & UNSUPPORTED_NODE_TYPES == set()
    assert UNSUPPORTED_TRIGGER_TYPES == DECLARED_TRIGGER_TYPES - IMPLEMENTED_TRIGGER_TYPES
    assert UNSUPPORTED_NODE_TYPES == DECLARED_NODE_TYPES - IMPLEMENTED_NODE_TYPES
    assert IMPLEMENTED_TRIGGER_TYPES
    assert IMPLEMENTED_NODE_TYPES


def test_capabilities_match_workflows_spec_literals():
    assert set(typing.get_args(TriggerType)) == DECLARED_TRIGGER_TYPES
    assert set(typing.get_args(NodeType)) == DECLARED_NODE_TYPES


def test_capabilities_payload_is_dashboard_friendly():
    payload = workflow_capabilities()
    assert payload["triggers"]["implemented"] == sorted(IMPLEMENTED_TRIGGER_TYPES)
    assert payload["nodes"]["unsupported"] == sorted(UNSUPPORTED_NODE_TYPES)
    assert payload["assistant"]["allowed_triggers"] == sorted(IMPLEMENTED_TRIGGER_TYPES)
    assert payload["assistant"]["allowed_nodes"] == sorted(IMPLEMENTED_NODE_TYPES)


def _spec_with_node(node_type: str) -> WorkflowSpec:
    return WorkflowSpec.model_validate(
        {
            "id": "unsupported_demo",
            "name": "Unsupported Demo",
            "version": 1,
            "triggers": [{"type": "manual"}],
            "nodes": {"start": {"type": node_type, "output": {}}},
            "edges": [],
        }
    )


def test_implemented_primitive_errors_reports_unsupported_node():
    sample = sorted(UNSUPPORTED_NODE_TYPES)[0]
    spec = _spec_with_node(sample)

    assert implemented_primitive_errors(spec) == [
        f"unsupported node type: {sample} on node start"
    ]


def test_require_implemented_primitives_raises_actionable_error():
    sample = sorted(UNSUPPORTED_TRIGGER_TYPES)[0]
    spec = WorkflowSpec.model_validate(
        {
            "id": "unsupported_trigger_demo",
            "name": "Unsupported Trigger Demo",
            "version": 1,
            "triggers": [{"type": sample}],
            "nodes": {"start": {"type": "pass", "output": {}}},
            "edges": [],
        }
    )

    with pytest.raises(ValueError, match=f"unsupported trigger type: {sample}"):
        require_implemented_primitives(spec)


def _agent_task_spec(profile: str = "reviewer") -> WorkflowSpec:
    return WorkflowSpec.model_validate(
        {
            "id": "profile_check",
            "name": "Profile Check",
            "version": 1,
            "triggers": [{"type": "manual"}],
            "nodes": {
                "task": {
                    "type": "agent_task",
                    "profile": profile,
                    "prompt": "Review this.",
                }
            },
        }
    )


def test_require_available_profiles_passes_for_existing_profile():
    from hermes_cli.workflows_capabilities import require_available_profiles

    spec = _agent_task_spec("reviewer")
    require_available_profiles(spec, {"default", "reviewer"})


def test_require_available_profiles_fails_for_missing_profile():
    from hermes_cli.workflows_capabilities import require_available_profiles

    spec = _agent_task_spec("ghost")
    with pytest.raises(ValueError, match="workflow_profile_not_found"):
        require_available_profiles(spec, {"default", "reviewer"})


def test_profile_availability_errors_lists_all_missing_profiles():
    from hermes_cli.workflows_capabilities import profile_availability_errors

    spec = WorkflowSpec.model_validate(
        {
            "id": "multi_profile",
            "name": "Multi Profile",
            "version": 1,
            "triggers": [{"type": "manual"}],
            "nodes": {
                "a": {
                    "type": "agent_task",
                    "profile": "ghost_a",
                    "prompt": "A",
                },
                "b": {
                    "type": "agent_task",
                    "profile": "reviewer",
                    "prompt": "B",
                },
                "c": {
                    "type": "agent_task",
                    "profile": "ghost_b",
                    "prompt": "C",
                },
            },
        }
    )
    errors = profile_availability_errors(spec, {"default", "reviewer"})
    assert len(errors) == 2
    assert all("workflow_profile_not_found" in e for e in errors)
    assert any("ghost_a" in e for e in errors)
    assert any("ghost_b" in e for e in errors)


def test_require_available_profiles_passes_for_pass_only_spec():
    from hermes_cli.workflows_capabilities import require_available_profiles

    spec = WorkflowSpec.model_validate(
        {
            "id": "no_profiles",
            "name": "No Profiles",
            "version": 1,
            "triggers": [{"type": "manual"}],
            "nodes": {"start": {"type": "pass", "output": {}}},
        }
    )
    require_available_profiles(spec, {"default"})


def test_workspace_node_field_rejected_as_unimplemented():
    spec = WorkflowSpec.model_validate(
        {
            "id": "workspace_demo",
            "name": "Workspace Demo",
            "version": 1,
            "triggers": [{"type": "manual"}],
            "nodes": {
                "start": {
                    "type": "pass",
                    "output": {},
                    "workspace": {"cwd": "/tmp/somewhere"},
                }
            },
            "edges": [],
        }
    )

    errors = implemented_primitive_errors(spec)

    assert len(errors) == 1
    assert "workspace" in errors[0]
    assert "workspace_kind/workspace_path" in errors[0]

    with pytest.raises(ValueError, match="unsupported node field: workspace"):
        require_implemented_primitives(spec)
