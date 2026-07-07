import pytest
from pydantic import ValidationError

from hermes_cli.workflows_spec import EdgeSpec, WorkflowSpec, validate_graph


def _minimal_spec():
    return {
        "id": "demo",
        "name": "Demo",
        "version": 1,
        "enabled": True,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "start": {"type": "pass", "output": {"ok": True}},
            "done": {"type": "pass"},
        },
        "edges": [{"from": "start", "to": "done"}],
    }


def test_minimal_spec_validates():
    spec = WorkflowSpec.model_validate(_minimal_spec())
    validate_graph(spec)
    assert spec.id == "demo"


def test_unknown_edge_target_rejected():
    raw = _minimal_spec()
    raw["edges"] = [{"from": "start", "to": "missing"}]
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="unknown edge target"):
        validate_graph(spec)


def test_unknown_edge_source_rejected():
    raw = _minimal_spec()
    raw["edges"] = [{"from": "missing", "to": "done"}]
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="unknown edge source"):
        validate_graph(spec)


def test_self_catch_rejected():
    raw = _minimal_spec()
    raw["nodes"] = {"start": {"type": "fail", "catch": "start"}}
    raw["edges"] = []
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="cannot catch itself"):
        validate_graph(spec)


def test_empty_workflow_nodes_rejected():
    raw = _minimal_spec()
    raw["nodes"] = {}
    raw["edges"] = []
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="at least one node"):
        validate_graph(spec)


def test_switch_requires_default_or_exhaustive_edges():
    raw = _minimal_spec()
    raw["nodes"]["route"] = {"type": "switch", "cases": []}
    raw["edges"] = [{"from": "start", "to": "route"}]
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="switch node route must define"):
        validate_graph(spec)


def test_workflow_requires_at_least_one_entry_node():
    raw = _minimal_spec()
    raw["nodes"] = {
        "first": {"type": "pass"},
        "second": {"type": "pass"},
    }
    raw["edges"] = [
        {"from": "first", "to": "second"},
        {"from": "second", "to": "first"},
    ]
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="at least one entry node"):
        validate_graph(spec)


def test_switch_case_requires_matching_outgoing_edge():
    raw = _minimal_spec()
    raw["nodes"] = {
        "route": {
            "type": "switch",
            "cases": [{"name": "approved", "when": {"op": "eq", "left": 1, "right": 1}}],
        },
        "done": {"type": "pass"},
    }
    raw["edges"] = [{"from": "route.aproved", "to": "done"}]
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="switch case route.approved requires matching outgoing edge"):
        validate_graph(spec)


def test_bad_workflow_id_rejected():
    raw = _minimal_spec()
    raw["id"] = "Bad ID With Spaces"
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate(raw)


@pytest.mark.parametrize("node_id", ["", "route.any", "Bad", "1start"])
def test_bad_node_id_rejected(node_id):
    raw = _minimal_spec()
    raw["nodes"] = {node_id: {"type": "pass"}}
    raw["edges"] = []
    with pytest.raises(ValidationError, match="invalid node id"):
        WorkflowSpec.model_validate(raw)


def test_dotted_edge_source_requires_switch_or_parallel_node():
    raw = _minimal_spec()
    raw["nodes"]["route"] = {"type": "pass"}
    raw["edges"] = [{"from": "route.any", "to": "done"}]
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="dotted edge source.*switch or parallel"):
        validate_graph(spec)


def test_dotted_edge_source_requires_branch_suffix():
    raw = _minimal_spec()
    raw["nodes"]["route"] = {"type": "switch"}
    raw["edges"] = [{"from": "route.", "to": "done"}]
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="branch suffix"):
        validate_graph(spec)


def test_parallel_edge_source_requires_branch_suffix():
    raw = _minimal_spec()
    raw["nodes"] = {
        "fork": {"type": "parallel"},
        "done": {"type": "pass"},
    }
    raw["edges"] = [{"from": "fork", "to": "done"}]
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="parallel.*branch suffix"):
        validate_graph(spec)


def test_switch_default_target_must_exist():
    raw = _minimal_spec()
    raw["nodes"] = {"route": {"type": "switch", "default": "missing"}}
    raw["edges"] = []
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="unknown switch default target"):
        validate_graph(spec)


@pytest.mark.parametrize("default", ["", "   "])
def test_blank_switch_default_rejected_even_with_outgoing_edge(default):
    raw = _minimal_spec()
    raw["nodes"] = {
        "route": {"type": "switch", "default": default},
        "done": {"type": "pass"},
    }
    raw["edges"] = [{"from": "route.any", "to": "done"}]
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="unknown switch default target"):
        validate_graph(spec)


def test_switch_default_target_satisfies_required_exit():
    raw = _minimal_spec()
    raw["nodes"] = {
        "route": {"type": "switch", "default": "done"},
        "done": {"type": "pass"},
    }
    raw["edges"] = []
    spec = WorkflowSpec.model_validate(raw)
    validate_graph(spec)


@pytest.mark.parametrize(
    ("profile", "prompt", "expected_error"),
    [
        ("", "do it", "agent_task node task requires a non-blank profile"),
        ("worker", "", "agent_task node task requires a non-empty prompt"),
        ("   ", "do it", "agent_task node task requires a non-blank profile"),
        ("worker", "   ", "agent_task node task requires a non-empty prompt"),
    ],
)
def test_agent_task_requires_non_blank_profile_and_prompt(profile, prompt, expected_error):
    raw = _minimal_spec()
    raw["nodes"] = {"task": {"type": "agent_task", "profile": profile, "prompt": prompt}}
    raw["edges"] = []
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match=expected_error):
        validate_graph(spec)


@pytest.mark.parametrize("prompt", [{}, []])
def test_agent_task_prompt_may_be_text_or_structured_but_not_empty_container(prompt):
    raw = _minimal_spec()
    raw["nodes"] = {"task": {"type": "agent_task", "profile": "worker", "prompt": prompt}}
    raw["edges"] = []
    spec = WorkflowSpec.model_validate(raw)
    with pytest.raises(ValueError, match="agent_task node task requires a non-empty prompt"):
        validate_graph(spec)


def test_agent_task_max_retries_must_be_positive():
    raw = _minimal_spec()
    raw["nodes"] = {
        "task": {
            "type": "agent_task",
            "profile": "worker",
            "prompt": "do it",
            "max_retries": 0,
        }
    }
    raw["edges"] = []

    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate(raw)


def test_agent_task_accepts_result_contract():
    spec = WorkflowSpec.model_validate(
        {
            "id": "contract_demo",
            "name": "Contract Demo",
            "version": 1,
            "nodes": {
                "ask": {
                    "type": "agent_task",
                    "profile": "worker",
                    "prompt": "Return JSON",
                    "result_contract": {"summary": "string", "status": "ok|failed"},
                },
                "done": {"type": "pass"},
            },
            "edges": [{"from": "ask", "to": "done"}],
        }
    )

    validate_graph(spec)
    assert spec.nodes["ask"].result_contract["status"] == "ok|failed"
    assert spec.nodes["done"].result_contract == {}
    assert spec.model_dump()["nodes"]["done"]["result_contract"] == {}


def test_edge_from_alias_and_field_name_both_populate_from_():
    by_alias = EdgeSpec.model_validate({"from": "start", "to": "done"})
    by_name = EdgeSpec.model_validate({"from_": "start", "to": "done"})

    assert by_alias.from_ == "start"
    assert by_name.from_ == "start"
