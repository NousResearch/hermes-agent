import json
from pathlib import Path

import pytest

from plugins.openai_agents import tools as oat


def test_preflight_blocks_high_risk_task_without_explicit_scope():
    with pytest.raises(ValueError, match="requires explicit scope"):
        oat._preflight_request("execute", "delete temporary cache files", [])


def test_preflight_allows_high_risk_task_with_explicit_read_only_scope():
    assert oat._preflight_request(
        "review",
        "review whether deleting cache files would be safe",
        ["read-only analysis; no mutation"],
    ) is None


def test_verified_without_proof_is_downgraded_to_partial():
    output = oat.GovernedAgentOutput(status="verified", summary="done", proof=[])

    result = oat._enforce_postconditions(output, lane="execute")

    assert result.status == "partial"
    assert any("no proof" in risk.lower() for risk in result.risks)


def test_review_lane_mutation_claim_is_downgraded():
    output = oat.GovernedAgentOutput(
        status="verified",
        summary="changed it",
        proof=["self-report"],
        actions_taken=["edited the file"],
    )

    result = oat._enforce_postconditions(output, lane="review")

    assert result.status == "partial"
    assert any("no-mutation" in risk.lower() for risk in result.risks)


def test_blocked_model_aliases_are_rejected():
    with pytest.raises(ValueError, match="blocked by local trust policy"):
        oat._resolve_model("qwen-plus")


def test_receipt_writer_persists_json_without_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    payload = {
        "lane": "verify",
        "model": "gpt-5.5",
        "api_key": "should-not-be-written",
        "governance_contract": oat._GOVERNANCE_CONTRACT,
        "result": {"status": "verified", "proof": ["evidence"]},
    }

    receipt_path = oat._write_receipt(payload)

    path = Path(receipt_path)
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["lane"] == "verify"
    assert "api_key" not in saved
    assert saved["result"]["status"] == "verified"
    assert str(path).startswith(str(tmp_path))


def test_sdk_guardrails_are_attached_to_agent_kwargs():
    kwargs = oat._build_agent_guardrail_kwargs("verify")

    assert kwargs["input_guardrails"]
    assert kwargs["output_guardrails"]


def test_sdk_input_guardrail_is_blocking_per_docs():
    kwargs = oat._build_agent_guardrail_kwargs("execute")

    guardrail = kwargs["input_guardrails"][0]
    assert guardrail.run_in_parallel is False
    assert guardrail.name == "hermes_execute_input_scope"


def test_sdk_output_guardrail_is_proof_enforcing_per_docs():
    kwargs = oat._build_agent_guardrail_kwargs("review")

    guardrail = kwargs["output_guardrails"][0]
    assert guardrail.name == "hermes_review_proof_output"


def test_run_config_intentionally_disables_sensitive_tracing():
    receipt = {
        "trace_sensitive_data": False,
        "sdk_guardrails_attached": True,
        "preflight_enforced": True,
        "postconditions_enforced": True,
    }

    assert receipt["trace_sensitive_data"] is False
    assert receipt["sdk_guardrails_attached"] is True
    assert receipt["preflight_enforced"] is True
    assert receipt["postconditions_enforced"] is True


def test_architecture_workflow_composes_execute_review_verify(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []

    def fake_run(lane, args):
        calls.append((lane, args["task"]))
        return json.dumps({
            "success": True,
            "lane": lane,
            "receipt_path": str(tmp_path / f"{lane}.json"),
            "result": {
                "status": "verified",
                "summary": f"{lane} ok",
                "actions_taken": ["analysis only"],
                "proof": [f"{lane} proof"],
                "risks": [],
                "next_required_action": None,
                "requires_human_approval": False,
            },
        })

    monkeypatch.setattr(oat, "_run_governed_lane", fake_run)

    raw = oat._handle_openai_agents_architecture({
        "task": "Design a safe architecture workflow",
        "context": "SDK optimization only.",
        "acceptance_criteria": ["must produce receipts"],
        "constraints": ["analysis only; no mutation"],
    })

    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["workflow"] == "architecture"
    assert payload["status"] == "verified"
    assert [lane for lane, _ in calls] == ["execute", "review", "verify"]
    assert Path(payload["receipt_path"]).exists()


def test_architecture_schema_registers_architecture_tool_name():
    assert oat.OPENAI_AGENTS_ARCHITECTURE_SCHEMA["name"] == "openai_agents_architecture"
    assert "architecture workflow" in oat.OPENAI_AGENTS_ARCHITECTURE_SCHEMA["description"].lower()


class _UsageObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _ResponseObject:
    def __init__(self, usage):
        self.usage = usage


class _ResultObject:
    def __init__(self, raw_responses):
        self.raw_responses = raw_responses


def test_extract_usage_from_sdk_raw_responses():
    result = _ResultObject([
        _ResponseObject(_UsageObject(input_tokens=10, output_tokens=5, total_tokens=15)),
        _ResponseObject({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
    ])

    usage = oat._extract_usage(result)

    assert usage["available"] is True
    assert usage["input_tokens"] == 13
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 20
    assert usage["cost_estimate_usd"] is None


def test_receipt_sha256_changes_with_content(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = oat._write_receipt({"lane": "verify", "result": {"status": "partial", "proof": ["a"]}})
    second = oat._write_receipt({"lane": "verify", "result": {"status": "partial", "proof": ["b"]}})

    assert oat._sha256_file(first) != oat._sha256_file(second)


def test_quality_contract_contains_perfectionist_standards():
    joined = "\n".join(oat._GOVERNANCE_CONTRACT).lower()

    assert "no claim without proof" in joined
    assert "no success without verification" in joined
    assert "bounded" in joined


def test_optional_pricing_config_estimates_cost(monkeypatch):
    monkeypatch.setattr(oat, "load_config", lambda: {
        "openai_agents": {
            "pricing": {
                "models": {
                    "gpt-5.5": {
                        "input_per_1m": 5.0,
                        "output_per_1m": 30.0,
                        "cached_input_per_1m": 0.5,
                    }
                }
            }
        }
    })
    usage = {"available": True, "input_tokens": 1000, "output_tokens": 100, "cached_tokens": 200}

    estimated = oat._apply_cost_estimate(usage, model="gpt-5.5")

    assert estimated["cost_estimate_status"] == "estimated_from_openai_agents_pricing_config"
    assert estimated["cost_estimate_usd"] == 0.0071


def test_governance_eval_corpus_cases_are_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    corpus_path = Path("evals/openai_agents/governance_cases.json")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

    assert corpus["cases"]
    for case in corpus["cases"]:
        ctype = case["type"]
        if ctype == "preflight_block":
            with pytest.raises(ValueError):
                oat._preflight_request(case["lane"], case["task"], list(case.get("constraints") or []))
        elif ctype == "preflight_allow":
            assert oat._preflight_request(case["lane"], case["task"], list(case.get("constraints") or [])) is None
        elif ctype in {"postcondition_downgrade", "postcondition_mutation_downgrade"}:
            result = oat._enforce_postconditions(oat.GovernedAgentOutput(**case["output"]), lane=case["lane"])
            assert result.status == case["expected_status"]
        elif ctype == "model_block":
            with pytest.raises(ValueError):
                oat._resolve_model(case["model"])
        elif ctype == "receipt_sanitize":
            sanitized = oat._sanitize_for_receipt(case["payload"])
            for key in case["forbidden_keys"]:
                assert key not in sanitized
        elif ctype == "architecture_schema":
            assert oat.OPENAI_AGENTS_ARCHITECTURE_SCHEMA["name"] == case["expected_name"]
        else:
            raise AssertionError(f"unknown eval case type: {ctype}")


def test_receipt_schema_file_accepts_minimal_receipt_shape():
    schema = json.loads(Path("schemas/openai-agents-receipt.schema.json").read_text(encoding="utf-8"))

    assert "success" in schema["required"]
    assert "receipt" in schema["required"]
    assert "result" in schema["required"]
    assert "receipt_sha256" in schema["properties"]
