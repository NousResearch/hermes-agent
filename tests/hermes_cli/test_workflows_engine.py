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


def test_completed_node_outputs_are_literal_not_template_rendered():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {
            "ask": {"type": "agent_task", "profile": "worker", "prompt": "do it"},
            "done": {"type": "pass", "output": {"answer": "${ node.ask.output.answer }"}},
        },
        "edges": [{"from": "ask", "to": "done"}],
    })

    result = run_in_memory_until_waiting(
        spec,
        input_data={"secret": "do-not-leak"},
        completed_node_outputs={"ask": {"answer": "${ input.secret }"}},
    )

    assert result.status == "succeeded"
    assert result.context["node"]["ask"]["output"] == {"answer": "${ input.secret }"}
    assert result.context["node"]["done"]["output"] == {"answer": "${ input.secret }"}


def test_parallel_fan_out_joins_branch_outputs():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {
            "fork": {"type": "parallel"},
            "research": {"type": "pass", "output": {"summary": "r"}},
            "implement": {"type": "pass", "output": {"summary": "i"}},
            "merge": {"type": "join"},
            "done": {"type": "pass", "output": {
                "research": "${ node.merge.output.branches.research.summary }",
                "implement": "${ node.merge.output.branches.implement.summary }",
            }},
        },
        "edges": [
            {"from": "fork.research", "to": "research"},
            {"from": "fork.implement", "to": "implement"},
            {"from": "research", "to": "merge"},
            {"from": "implement", "to": "merge"},
            {"from": "merge", "to": "done"},
        ],
    })

    result = run_in_memory_until_waiting(spec, input_data={})

    assert result.status == "succeeded"
    assert result.context["branches"]["fork"] == {
        "research": {"summary": "r"},
        "implement": {"summary": "i"},
    }
    assert result.context["node"]["merge"]["output"]["branches"] == {
        "research": {"summary": "r"},
        "implement": {"summary": "i"},
    }
    assert result.context["node"]["done"]["output"] == {"research": "r", "implement": "i"}


def test_join_waits_until_all_branch_upstreams_succeed():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {
            "fork": {"type": "parallel"},
            "ready": {"type": "pass", "output": {"summary": "r"}},
            "blocked": {"type": "agent_task", "profile": "worker", "prompt": "finish"},
            "merge": {"type": "join"},
            "done": {"type": "pass", "output": {
                "ready": "${ node.merge.output.branches.ready.summary }",
                "blocked": "${ node.merge.output.branches.blocked.summary }",
            }},
        },
        "edges": [
            {"from": "fork.ready", "to": "ready"},
            {"from": "fork.blocked", "to": "blocked"},
            {"from": "ready", "to": "merge"},
            {"from": "blocked", "to": "merge"},
            {"from": "merge", "to": "done"},
        ],
    })

    waiting = run_in_memory_until_waiting(spec, input_data={})

    assert waiting.status == "waiting"
    assert waiting.waiting_nodes == ["blocked"]
    assert "merge" not in waiting.context["node"]

    done = run_in_memory_until_waiting(
        spec,
        input_data={},
        completed_node_outputs={"blocked": {"summary": "b"}},
    )

    assert done.status == "succeeded"
    assert done.context["node"]["merge"]["output"]["branches"] == {
        "ready": {"summary": "r"},
        "blocked": {"summary": "b"},
    }
    assert done.context["node"]["done"]["output"] == {"ready": "r", "blocked": "b"}


def test_parallel_branch_failure_without_catch_fails_execution():
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "nodes": {
            "fork": {"type": "parallel"},
            "ok": {"type": "pass", "output": {"ok": True}},
            "bad": {"type": "fail", "output": {"reason": "boom"}},
            "merge": {"type": "join"},
        },
        "edges": [
            {"from": "fork.ok", "to": "ok"},
            {"from": "fork.bad", "to": "bad"},
            {"from": "ok", "to": "merge"},
            {"from": "bad", "to": "merge"},
        ],
    })

    result = run_in_memory_until_waiting(spec, input_data={})

    assert result.status == "failed"
    assert result.error is not None
    assert result.error["node"] == "bad"


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
        "id": "demo", "name": "Demo", "version": 1, "max_node_runs": 2,
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
    assert "max node runs" in result.error["message"]
