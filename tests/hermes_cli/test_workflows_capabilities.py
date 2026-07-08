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
