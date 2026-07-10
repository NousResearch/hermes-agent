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
    with pytest.raises(ValueError, match="workflow graph contains cycle"):
        validate_graph(spec)


def test_validate_graph_rejects_self_cycle():
    raw = _minimal_spec()
    raw["nodes"] = {
        "start": {"type": "pass", "output": {}},
        "loop": {"type": "pass", "output": {}},
    }
    raw["edges"] = [
        {"from": "start", "to": "loop"},
        {"from": "loop", "to": "loop"},
    ]
    spec = WorkflowSpec.model_validate(raw)

    with pytest.raises(ValueError, match="workflow graph contains cycle: loop -> loop"):
        validate_graph(spec)


def test_validate_graph_rejects_multi_node_cycle():
    raw = _minimal_spec()
    raw["nodes"] = {
        "start": {"type": "pass", "output": {}},
        "a": {"type": "pass", "output": {}},
        "b": {"type": "pass", "output": {}},
    }
    raw["edges"] = [
        {"from": "start", "to": "a"},
        {"from": "a", "to": "b"},
        {"from": "b", "to": "a"},
    ]
    spec = WorkflowSpec.model_validate(raw)

    with pytest.raises(ValueError, match="workflow graph contains cycle: a -> b -> a"):
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


def test_agent_task_rejects_unknown_result_contract_token_at_deploy_time():
    raw = _minimal_spec()
    raw["nodes"] = {
        "ask": {
            "type": "agent_task",
            "profile": "worker",
            "prompt": "Return JSON",
            "result_contract": {"summary": "strng"},
        }
    }
    raw["edges"] = []
    spec = WorkflowSpec.model_validate(raw)

    with pytest.raises(ValueError, match="invalid result_contract token.*strng"):
        validate_graph(spec)


def test_agent_task_rejects_empty_result_contract_enum_at_deploy_time():
    raw = _minimal_spec()
    raw["nodes"] = {
        "ask": {
            "type": "agent_task",
            "profile": "worker",
            "prompt": "Return JSON",
            "result_contract": {"status": " | "},
        }
    }
    raw["edges"] = []
    spec = WorkflowSpec.model_validate(raw)

    with pytest.raises(ValueError, match="empty result_contract enum"):
        validate_graph(spec)


def test_trigger_ready_when_rejects_malformed_condition_at_deploy_time():
    raw = _minimal_spec()
    raw["triggers"] = [{
        "type": "manual",
        "id": "manual",
        "intake": {"ready_when": {"op": "and", "args": []}},
    }]
    spec = WorkflowSpec.model_validate(raw)

    with pytest.raises(ValueError, match="trigger 'manual' ready_when"):
        validate_graph(spec)


def test_trigger_dedupe_key_rejects_malformed_path_at_deploy_time():
    raw = _minimal_spec()
    raw["triggers"] = [{
        "type": "manual",
        "id": "manual",
        "intake": {"dedupe_key": "input.id"},
    }]
    spec = WorkflowSpec.model_validate(raw)

    with pytest.raises(ValueError, match="trigger 'manual' dedupe_key"):
        validate_graph(spec)


def test_switch_when_rejects_malformed_condition_at_deploy_time():
    raw = _minimal_spec()
    raw["nodes"] = {
        "route": {
            "type": "switch",
            "cases": [{"name": "bad", "when": {"op": "eq", "left": {"path": "input.missing"}, "right": 1}}],
        },
        "done": {"type": "pass"},
    }
    raw["edges"] = [{"from": "route.bad", "to": "done"}]
    spec = WorkflowSpec.model_validate(raw)

    with pytest.raises(ValueError, match="switch case route.bad when"):
        validate_graph(spec)


def test_agent_task_accepts_provider_and_model_fields() -> None:
    spec = WorkflowSpec.model_validate(
        {
            "id": "routing_demo",
            "name": "Routing Demo",
            "version": 1,
            "triggers": [{"type": "manual", "id": "manual"}],
            "nodes": {
                "review": {
                    "type": "agent_task",
                    "profile": "reviewer",
                    "provider": "openai-codex",
                    "model": "gpt-5.5",
                    "prompt": "Review and return JSON.",
                    "result_contract": {"verdict": "approved|changes_requested"},
                }
            },
            "edges": [],
        }
    )

    node = spec.nodes["review"]
    assert node.profile == "reviewer"
    assert node.provider_override == "openai-codex"
    assert node.model_override == "gpt-5.5"
    dumped = spec.model_dump(mode="json", by_alias=True)
    assert dumped["nodes"]["review"]["provider"] == "openai-codex"
    assert dumped["nodes"]["review"]["model"] == "gpt-5.5"


def test_agent_task_rejects_non_string_provider_and_model_fields():
    for field, bad_value in [("provider", False), ("model", {"bad": "type"})]:
        raw = {
            "id": "routing_demo",
            "name": "Routing Demo",
            "version": 1,
            "triggers": [{"type": "manual", "id": "manual"}],
            "nodes": {
                "review": {
                    "type": "agent_task",
                    "profile": "reviewer",
                    "prompt": "Review.",
                    field: bad_value,
                }
            },
            "edges": [],
        }
        with pytest.raises(ValidationError, match=f"{field} must be a string"):
            WorkflowSpec.model_validate(raw)


def test_agent_task_accepts_legacy_provider_and_model_override_aliases() -> None:
    spec = WorkflowSpec.model_validate(
        {
            "id": "legacy_routing_demo",
            "name": "Legacy Routing Demo",
            "version": 1,
            "nodes": {
                "review": {
                    "type": "agent_task",
                    "profile": "reviewer",
                    "provider_override": "xiaomi-token-plan",
                    "model_override": "mimo-vl-7b",
                    "prompt": "Review and return JSON.",
                    "result_contract": {"ok": "boolean"},
                }
            },
            "edges": [],
        }
    )

    node = spec.nodes["review"]
    assert node.provider_override == "xiaomi-token-plan"
    assert node.model_override == "mimo-vl-7b"
    dumped = spec.model_dump(mode="json", by_alias=True)
    assert dumped["nodes"]["review"]["provider"] == "xiaomi-token-plan"
    assert dumped["nodes"]["review"]["model"] == "mimo-vl-7b"
    assert "provider_override" not in dumped["nodes"]["review"]
    assert "model_override" not in dumped["nodes"]["review"]


def test_agent_task_rejects_conflicting_routing_aliases():
    with pytest.raises(ValidationError, match="provider and provider_override must match"):
        WorkflowSpec.model_validate(
            {
                "id": "bad_routing_demo",
                "name": "Bad Routing Demo",
                "version": 1,
                "nodes": {
                    "review": {
                        "type": "agent_task",
                        "profile": "reviewer",
                        "provider": "openai-codex",
                        "provider_override": "minimax",
                        "prompt": "Review.",
                    }
                },
            }
        )

    with pytest.raises(ValidationError, match="model and model_override must match"):
        WorkflowSpec.model_validate(
            {
                "id": "bad_model_routing_demo",
                "name": "Bad Model Routing Demo",
                "version": 1,
                "nodes": {
                    "review": {
                        "type": "agent_task",
                        "profile": "reviewer",
                        "model": "gpt-5.5",
                        "model_override": "minimax-m3",
                        "prompt": "Review.",
                    }
                },
            }
        )


def test_edge_from_alias_and_field_name_both_populate_from_():
    by_alias = EdgeSpec.model_validate({"from": "start", "to": "done"})
    by_name = EdgeSpec.model_validate({"from_": "start", "to": "done"})

    assert by_alias.from_ == "start"
    assert by_name.from_ == "start"


def test_schedule_trigger_rejects_unparsable_cron():
    spec = _minimal_spec()
    spec["triggers"] = [{"type": "schedule", "id": "sched", "cron": "not a cron"}]

    with pytest.raises(ValueError, match="invalid cron expression on trigger 'sched'"):
        validate_graph(WorkflowSpec.model_validate(spec))


def test_schedule_trigger_accepts_valid_cron_and_expr_alias():
    spec = _minimal_spec()
    spec["triggers"] = [{"type": "schedule", "id": "sched", "cron": "0 9 * * 1-5"}]
    validate_graph(WorkflowSpec.model_validate(spec))

    spec["triggers"] = [{"type": "schedule", "id": "sched", "expr": "*/5 * * * *"}]
    validate_graph(WorkflowSpec.model_validate(spec))


def test_unknown_workflow_field_rejected_with_suggestion():
    from hermes_cli.workflows_spec import reject_unknown_spec_fields

    raw = _minimal_spec()
    raw["verion"] = 2

    with pytest.raises(ValueError, match=r"unknown field 'verion' on workflow; did you mean 'version'\?"):
        reject_unknown_spec_fields(raw)


def test_unknown_node_field_typo_rejected_with_location():
    from hermes_cli.workflows_spec import unknown_spec_field_errors

    raw = _minimal_spec()
    raw["nodes"]["start"]["result_contarct"] = {"ok": "boolean"}

    errors = unknown_spec_field_errors(raw)

    assert len(errors) == 1
    assert "unknown field 'result_contarct' on node 'start'" in errors[0]
    assert "result_contract" in errors[0]


def test_unknown_trigger_edge_retry_and_workspace_fields_rejected():
    from hermes_cli.workflows_spec import unknown_spec_field_errors

    raw = _minimal_spec()
    raw["triggers"][0]["cronn"] = "0 9 * * *"
    raw["edges"][0]["too"] = "done"
    raw["nodes"]["start"]["retry"] = {"max_attempt": 2}
    raw["nodes"]["start"]["workspace"] = {"cwdd": "/tmp"}

    errors = unknown_spec_field_errors(raw)

    joined = "; ".join(errors)
    assert "unknown field 'cronn' on trigger [0]" in joined
    assert "unknown field 'too' on edge [0]" in joined
    assert "unknown field 'max_attempt' on node 'start' retry" in joined
    assert "unknown field 'cwdd' on node 'start' workspace" in joined


def test_known_fields_aliases_and_descriptions_pass_strict_check():
    from hermes_cli.workflows_spec import unknown_spec_field_errors

    raw = {
        "id": "strict_ok",
        "name": "Strict OK",
        "version": 1,
        "description": "workflow-level note",
        "max_node_runs": 10,
        "enabled": True,
        "triggers": [
            {"type": "manual", "id": "manual", "description": "trigger note"},
            {"type": "schedule", "id": "sched", "expr": "0 9 * * *", "input": {}},
        ],
        "nodes": {
            "work": {
                "type": "agent_task",
                "profile": "worker",
                "prompt": "Return JSON only.",
                "result_contract": {"ok": "boolean"},
                "provider_override": "openai",
                "model_override": "gpt-5.5",
                "description": "node note",
                "retry": {"max_attempts": 2, "delay_seconds": 1},
            },
            "done": {"type": "pass"},
        },
        "edges": [{"from": "work", "to": "done"}],
    }

    assert unknown_spec_field_errors(raw) == []


def test_trigger_input_schema_and_intake_metadata_validate_and_dump():
    spec = WorkflowSpec.model_validate({
        "id": "intake_demo",
        "name": "Intake Demo",
        "version": 1,
        "triggers": [{
            "type": "manual",
            "id": "kickoff",
            "input": {"static": True},
            "input_schema": {
                "brief": {"kind": "long_text", "label": "Brief", "required": True, "min_length": 10},
                "score": {"kind": "number", "required": True, "min": 0, "max": 1},
                "docs": {"kind": "document", "accepts": ["text/markdown"]},
            },
            "intake": {
                "mode": "continuous",
                "item_source": "kickoff_cell",
                "ready_when": {"op": "exists", "path": "$.input.brief"},
                "dedupe_key": "$.input.source_id",
                "split_strategy": "documents",
            },
        }],
        "nodes": {"start": {"type": "pass"}},
    })

    trigger = spec.triggers[0]
    assert trigger.input == {"static": True}
    assert trigger.input_schema["brief"].kind == "long_text"
    assert trigger.intake.mode == "continuous"
    dumped = spec.model_dump(mode="json")
    assert dumped["triggers"][0]["input_schema"]["score"]["min"] == 0
    assert dumped["triggers"][0]["input_schema"]["score"]["max"] == 1
    assert dumped["triggers"][0]["input_schema"]["docs"]["accepts"] == ["text/markdown"]


def test_unknown_trigger_input_schema_and_intake_fields_rejected():
    from hermes_cli.workflows_spec import unknown_spec_field_errors

    raw = _minimal_spec()
    raw["triggers"][0]["input_schema"] = {"brief": {"kind": "text", "lable": "Brief"}}
    raw["triggers"][0]["intake"] = {"mode": "continuous", "dedup_key": "$.input.id"}

    joined = "; ".join(unknown_spec_field_errors(raw))

    assert "unknown field 'lable' on trigger [0] input_schema 'brief'" in joined
    assert "unknown field 'dedup_key' on trigger [0] intake" in joined


def test_unsupported_batch_or_document_intake_is_rejected_by_graph_validation():
    raw = _minimal_spec()
    raw["triggers"][0]["intake"] = {"mode": "batch"}
    spec = WorkflowSpec.model_validate(raw)

    with pytest.raises(ValueError, match="batch intake is not supported"):
        validate_graph(spec)

    raw = _minimal_spec()
    raw["triggers"][0]["intake"] = {"mode": "continuous", "split_strategy": "documents"}
    spec = WorkflowSpec.model_validate(raw)

    with pytest.raises(ValueError, match="split_strategy"):
        validate_graph(spec)

    raw = _minimal_spec()
    raw["triggers"][0]["intake"] = {"mode": "continuous", "item_source": "documents"}
    spec = WorkflowSpec.model_validate(raw)

    with pytest.raises(ValueError, match="item_source"):
        validate_graph(spec)


def test_load_spec_from_object_strict_ingestion():
    from hermes_cli.workflows_spec import load_spec_from_object

    with pytest.raises(ValueError, match="workflow spec must be an object"):
        load_spec_from_object("not a mapping")

    raw = _minimal_spec()
    raw["nodes"]["start"]["outputt"] = {}
    with pytest.raises(ValueError, match="unknown field 'outputt'"):
        load_spec_from_object(raw)

    spec = load_spec_from_object(_minimal_spec())
    assert spec.id == "demo"
