from hermes_cli.workflows_engine import run_in_memory_until_waiting
from hermes_cli.workflows_spec import WorkflowSpec


def test_pass_then_switch_routes_to_matching_branch():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "start": {"type": "pass", "output": {"score": 0.9}},
            "route": {"type": "switch", "cases": [
                {"name": "high", "when": {"op": "gte", "left": {"path": "$.node.start.output.score"}, "right": 0.8}}
            ]},
            "high": {"type": "pass", "output": {"bucket": "high"}},
            "low": {"type": "pass", "output": {"bucket": "low"}},
        },
        "edges": [
            {"from": "start", "to": "route"},
            {"from": "route.high", "to": "high"},
            {"from": "route.default", "to": "low"},
        ],
    })
    result = run_in_memory_until_waiting(spec, input_data={})
    assert result.status == "succeeded"
    assert result.context["node"]["high"]["output"] == {"bucket": "high"}


def test_switch_default_target_does_not_start_as_root_when_case_matches():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {
            "route": {"type": "switch", "default": "low", "cases": [
                {"name": "high", "when": {"op": "eq", "left": 1, "right": 1}}
            ]},
            "high": {"type": "pass", "output": {"bucket": "high"}},
            "low": {"type": "pass", "output": {"bucket": "low"}},
        },
        "edges": [{"from": "route.high", "to": "high"}],
    })

    result = run_in_memory_until_waiting(spec, input_data={})

    assert result.status == "succeeded"
    assert "low" not in result.context["node"]
    assert result.context["node"]["high"]["output"] == {"bucket": "high"}
