from __future__ import annotations

from hermes_cli.workflow import DagValidationResult, normalize_dag, validate_dag
from hermes_cli.workflow.policy import DEFAULT_POLICY


def _policy(**overrides):
    data = dict(DEFAULT_POLICY)
    data.update(overrides)
    return data


def _valid_large_dag():
    return {
        "schema_version": 1,
        "workflow_id": "wf_example",
        "name": "Example workflow",
        "scale": "large",
        "nodes": [
            {
                "id": "spec-review",
                "title": "Review architecture spec",
                "role": "reviewer",
                "profile": "reviewer",
                "parents": [],
                "scope": {"summary": "Review spec for implementation readiness."},
                "definition_of_done": ["Review artifact exists."],
            },
            {
                "id": "backend-api",
                "title": "Implement backend API",
                "role": "engineer",
                "profile": "engineer",
                "parents": ["spec-review"],
                "scope": {"summary": "Implement normalized backend API."},
                "definition_of_done": ["API tests pass."],
            },
            {
                "id": "frontend-dag-view",
                "title": "Implement DAG view",
                "role": "engineer",
                "profile": "engineer",
                "parents": ["spec-review"],
                "scope": {"summary": "Render workflow DAG nodes."},
                "definition_of_done": ["UI tests pass."],
            },
            {
                "id": "integration",
                "title": "Integrate implementation nodes",
                "role": "integrator",
                "profile": "integrator",
                "parents": ["backend-api", "frontend-dag-view"],
                "scope": {"summary": "Combine backend and frontend outputs."},
                "definition_of_done": ["Integrated tests pass."],
            },
        ],
    }


def test_public_workflow_package_exports_dag_api():
    assert DagValidationResult.__name__ == "DagValidationResult"
    assert callable(validate_dag)
    assert callable(normalize_dag)


def test_normalize_valid_dag_adds_edges_children_and_defaults():
    result = normalize_dag(_valid_large_dag(), policy=_policy())

    assert result.ok
    normalized = result.dag
    assert normalized["workflow_id"] == "wf_example"
    assert normalized["scale"] == "large"
    assert normalized["edges"] == [
        {"source": "spec-review", "target": "backend-api", "kind": "depends_on"},
        {"source": "spec-review", "target": "frontend-dag-view", "kind": "depends_on"},
        {"source": "backend-api", "target": "integration", "kind": "depends_on"},
        {"source": "frontend-dag-view", "target": "integration", "kind": "depends_on"},
    ]
    nodes = {node["id"]: node for node in normalized["nodes"]}
    assert nodes["spec-review"]["children"] == ["backend-api", "frontend-dag-view"]
    assert nodes["backend-api"]["children"] == ["integration"]
    assert nodes["backend-api"]["status"] == "waiting"
    assert nodes["backend-api"]["gate_level"] == 1


def test_validate_dag_rejects_duplicate_node_ids():
    dag = _valid_large_dag()
    dag["nodes"].append(dict(dag["nodes"][0]))

    result = validate_dag(dag, policy=_policy())

    assert not result.ok
    assert [issue.code for issue in result.errors] == ["duplicate_node_id"]
    assert result.errors[0].path == "nodes[4].id"


def test_validate_dag_rejects_missing_parent_reference():
    dag = _valid_large_dag()
    dag["nodes"][1]["parents"] = ["missing-spec"]

    result = validate_dag(dag, policy=_policy())

    assert not result.ok
    assert any(issue.code == "unknown_parent" and issue.path == "nodes[1].parents[0]" for issue in result.errors)


def test_validate_dag_rejects_cycles():
    dag = _valid_large_dag()
    dag["nodes"][0]["parents"] = ["integration"]

    result = validate_dag(dag, policy=_policy())

    assert not result.ok
    assert any(issue.code == "cycle_detected" for issue in result.errors)


def test_validate_dag_rejects_unknown_roles_and_unmapped_profiles():
    dag = _valid_large_dag()
    dag["nodes"][1]["role"] = "wizard"
    policy = _policy(roles={**DEFAULT_POLICY["roles"], "engineer": None})

    result = validate_dag(dag, policy=policy)

    assert not result.ok
    codes = {issue.code for issue in result.errors}
    assert "unknown_role" in codes
    assert "unmapped_profile" in codes


def test_validate_dag_requires_engineer_definition_of_done_and_scope_summary():
    dag = _valid_large_dag()
    dag["nodes"][1]["definition_of_done"] = []
    dag["nodes"][2]["scope"] = {}

    result = validate_dag(dag, policy=_policy())

    assert not result.ok
    codes = {issue.code for issue in result.errors}
    assert "missing_definition_of_done" in codes
    assert "missing_scope_summary" in codes


def test_validate_dag_requires_integrator_for_large_multi_engineer_graphs():
    dag = _valid_large_dag()
    dag["nodes"] = [node for node in dag["nodes"] if node["role"] != "integrator"]

    result = validate_dag(dag, policy=_policy())

    assert not result.ok
    assert any(issue.code == "missing_integrator" for issue in result.errors)


def test_validate_dag_rejects_invalid_gate_level_and_type():
    dag = _valid_large_dag()
    dag["nodes"][0]["gate"] = {"level": 7, "type": ""}

    result = validate_dag(dag, policy=_policy())

    assert not result.ok
    codes = {issue.code for issue in result.errors}
    assert "invalid_gate_level" in codes
    assert "invalid_gate_type" in codes
