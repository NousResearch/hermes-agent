from hermes_cli.workflows_engine import next_edges, run_in_memory_until_waiting
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


def test_no_match_switch_routes_to_default_edge():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {
            "start": {"type": "pass", "output": {"score": 0.1}},
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
    assert "high" not in result.context["node"]
    assert result.context["node"]["low"]["output"] == {"bucket": "low"}


def test_no_match_switch_routes_to_direct_default_target():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {
            "route": {"type": "switch", "default": "low", "cases": [
                {"name": "high", "when": {"op": "eq", "left": 1, "right": 2}}
            ]},
            "high": {"type": "pass", "output": {"bucket": "high"}},
            "low": {"type": "pass", "output": {"bucket": "low"}},
        },
        "edges": [{"from": "route.high", "to": "high"}],
    })

    result = run_in_memory_until_waiting(spec, input_data={})

    assert result.status == "succeeded"
    assert "high" not in result.context["node"]
    assert result.context["node"]["low"]["output"] == {"bucket": "low"}


def test_pass_output_renders_whole_string_templates_recursively():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {
            "start": {"type": "pass", "output": {
                "none": None,
                "nested": ["${ input.score }", {"workflow": "${ workflow.id }"}],
                "literal": "score is ${ input.score }",
            }},
        },
        "edges": [],
    })

    result = run_in_memory_until_waiting(spec, input_data={"score": 7})

    assert result.status == "succeeded"
    assert result.context["node"]["start"]["output"] == {
        "none": None,
        "nested": [7, {"workflow": "demo"}],
        "literal": "score is ${ input.score }",
    }


def test_next_edges_matches_node_and_node_port_exactly():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {
            "route": {"type": "switch", "cases": []},
            "plain": {"type": "pass"},
            "branch": {"type": "pass"},
        },
        "edges": [
            {"from": "route", "to": "plain"},
            {"from": "route.high", "to": "branch"},
        ],
    })

    assert [edge.to for edge in next_edges(spec, "route")] == ["plain"]
    assert [edge.to for edge in next_edges(spec, "route", "high")] == ["branch"]
    assert next_edges(spec, "route", "default") == []


def test_wait_node_returns_waiting():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {"pause": {"type": "wait"}},
        "edges": [],
    })

    result = run_in_memory_until_waiting(spec, input_data={})

    assert result.status == "waiting"
    assert result.waiting_nodes == ["pause"]


def test_fail_node_returns_failed():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {"stop": {"type": "fail", "output": {"reason": "${ input.reason }"}}},
        "edges": [],
    })

    result = run_in_memory_until_waiting(spec, input_data={"reason": "bad"})

    assert result.status == "failed"
    assert result.waiting_nodes == []
    assert result.error == {"node": "stop", "type": "fail", "output": {"reason": "bad"}}


def test_reachable_cycle_trips_max_step_guard():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {
            "start": {"type": "pass"},
            "loop": {"type": "pass"},
        },
        "edges": [
            {"from": "start", "to": "loop"},
            {"from": "loop", "to": "loop"},
        ],
    })

    result = run_in_memory_until_waiting(spec, input_data={})

    assert result.status == "failed"
    assert result.error is not None
    assert "max in-memory steps" in result.error["message"]
