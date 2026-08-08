import json

import pytest

from agent.token_governance import (
    BudgetPolicy,
    PromptBudget,
    build_task_contract,
    enrich_usage_metrics,
    preflight_workspace,
    validate_handoff,
    validate_task_contract,
    write_handoff,
)


def test_usage_metrics_count_cache_as_prompt_and_processed():
    metrics = enrich_usage_metrics({
        "input_tokens": 10,
        "cache_read_tokens": 30,
        "cache_write_tokens": 5,
        "output_tokens": 7,
        "api_calls": 2,
    })
    assert metrics["new_input_tokens"] == 10
    assert metrics["cache_input_tokens"] == 35
    assert metrics["prompt_tokens"] == 45
    assert metrics["processed_tokens"] == 52
    assert metrics["avg_prompt_tokens_per_call"] == 22.5


def test_task_contract_preserves_original_request_and_detects_tampering(tmp_path):
    original = "完整原始要求\n不得摘要。"
    contract = build_task_contract(
        task_id="T-1",
        requirement_id="R-1",
        revision=1,
        original_request=original,
        business_goal="解决业务问题",
        acceptance_criteria=[{"id": "AC-01", "requirement": "保持原文", "verification": "哈希"}],
    )
    assert contract["original_request"] == original
    assert validate_task_contract(contract)["valid"] is True
    contract["original_request"] = "被改写"
    result = validate_task_contract(contract)
    assert result["valid"] is False
    assert result["reason"] == "contract_hash_mismatch"


def test_task_contract_rejects_invalid_authority_and_criteria():
    contract = build_task_contract(
        task_id="T-1", requirement_id="R-1", revision=1,
        original_request="原文", business_goal="目标",
        acceptance_criteria=[{"id": "AC-01", "requirement": "要求", "verification": "测试"}],
    )
    contract["authority"]["may_push"] = "false"
    contract["contract_hash"] = __import__("hashlib").sha256(
        json.dumps({k: v for k, v in contract.items() if k != "contract_hash"}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert validate_task_contract(contract)["reason"] == "authority_invalid"

    contract = build_task_contract(
        task_id="T-1", requirement_id="R-1", revision=1,
        original_request="原文", business_goal="目标",
        acceptance_criteria=[{"id": "AC-01", "requirement": "", "verification": "测试"}],
    )
    assert validate_task_contract(contract)["reason"] == "acceptance_criteria_invalid"


def test_prompt_budget_thresholds_use_prompt_not_new_input():
    budget = PromptBudget(BudgetPolicy(api_call_limit=10, prompt_token_limit=1000, absolute_prompt_limit=2000))
    assert budget.record(input_tokens=10, cache_read_tokens=690, cache_write_tokens=0, output_tokens=1)["action"] == "summarize"
    assert budget.record(input_tokens=0, cache_read_tokens=150, cache_write_tokens=0, output_tokens=1)["action"] == "freeze_scope"
    result = budget.record(input_tokens=0, cache_read_tokens=150, cache_write_tokens=0, output_tokens=1)
    assert result["action"] == "handoff"
    assert result["status"] == "budget_exhausted"


def test_prompt_budget_absolute_limit_pauses():
    budget = PromptBudget(BudgetPolicy(api_call_limit=80, prompt_token_limit=5000, absolute_prompt_limit=5500))
    result = budget.record(input_tokens=0, cache_read_tokens=5600, cache_write_tokens=0, output_tokens=0)
    assert result["action"] == "pause"
    assert result["status"] == "budget_exhausted"


def test_preflight_empty_workspace_blocks_without_model(tmp_path):
    result = preflight_workspace(tmp_path)
    assert result["status"] == "blocked"
    assert result["workspace_file_count"] == 0
    assert "workspace_empty" in result["missing"]


def test_preflight_superseded_revision_blocks(tmp_path):
    (tmp_path / "app.py").write_text("print('ok')", encoding="utf-8")
    result = preflight_workspace(tmp_path, revision=1, current_revision=2)
    assert result["status"] == "superseded"
    assert result["block_reason"] == "revision_superseded"


def test_preflight_active_run_and_duplicate_are_zero_token_outcomes(tmp_path):
    (tmp_path / "app.py").write_text("print('ok')", encoding="utf-8")
    assert preflight_workspace(tmp_path, active_run=True)["status"] == "blocked"
    assert preflight_workspace(tmp_path, successful_artifact=True)["status"] == "duplicate"


def test_handoff_requires_contract_alignment_and_traceability(tmp_path):
    contract = build_task_contract(
        task_id="T-1", requirement_id="R-1", revision=1,
        original_request="原文", business_goal="目标",
        acceptance_criteria=[{"id": "AC-01", "requirement": "要求", "verification": "测试"}],
    )
    trace = [{"criterion_id": "AC-01", "implementation": ["a.py"], "evidence": ["test.log"], "status": "passed"}]
    path = write_handoff(
        tmp_path / "handoff.json", contract=contract, stage_completed="Discovery",
        next_stage="Planning", acceptance_traceability=trace,
        alignment={"status": "aligned", "criteria_covered": ["AC-01"], "criteria_missing": [], "out_of_scope_changes": [], "unapproved_goals": []},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert validate_handoff(payload, contract)["valid"] is True
    payload["alignment"]["status"] = "deviated"
    assert validate_handoff(payload, contract)["valid"] is False
