import json
from pathlib import Path

from agent.apex_runtimeos_audit import persist_checkpoint
from agent.apex_runtimeos_audit_summary import summarize_audit
from agent.apex_runtimeos_hook import run_apex_runtimeos_completion_hook
from agent.apex_runtimeos_organs import (
    evaluate_recommendation_control,
    run_pre_api_organs,
    run_pre_completion_organs,
)
from gateway.platforms.api_server import _safe_apex_runtimeos_metadata


class DummyAgent:
    session_id = "test-session"
    model = "test-model"
    provider = "test-provider"
    base_url = "http://test"


def _base_result():
    return {
        "final_response": "done",
        "messages": [],
        "api_calls": 1,
        "completed": True,
    }


def test_apex_runtimeos_hook_attaches_metadata_and_report(tmp_path, monkeypatch):
    report_path = tmp_path / "apex_report.json"
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "warn")
    monkeypatch.setenv("APEX_RUNTIMEOS_REPORT_OUTPUT", str(report_path))
    monkeypatch.setenv("APEX_RUNTIMEOS_AUDIT_PATH", str(audit_path))

    result = run_apex_runtimeos_completion_hook(
        result=_base_result(),
        final_response="done",
        completed=True,
        interrupted=False,
        agent=DummyAgent(),
        messages=[],
        turn_exit_reason="unit_test",
    )

    meta = result["apex_runtimeos"]
    assert result["completed"] is True
    assert meta["enabled"] is True
    assert meta["mode"] == "warn"
    assert meta["status"] == "PASS"
    assert meta["dry_run"] is False
    assert meta["runtime_hook_enabled"] is True
    assert meta["runtime_context_recorded"] is True
    assert meta["organs"]["results"]["gene_selector"]["status"] == "PASS"
    assert meta["organs"]["results"]["gene_selector"]["output"]["mutates_memory_or_skills"] is False
    assert meta["organ_audit"]["written"] is True
    assert audit_path.exists()
    assert meta["evm_defect_count"] == 12
    assert meta["blocking"] is False
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["dry_run"] is False
    assert payload["report_type"] == "APEXRuntimeOSCompletionReport"
    assert payload["completion_boundary"]["runtime_hook_enabled"] is True
    assert payload["runtime_context"]["session_id"] == "test-session"
    assert payload["completion_boundary"]["full_agi_claimed"] is False


def test_apex_runtimeos_hook_can_be_disabled(monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_ENABLED", "0")
    result = run_apex_runtimeos_completion_hook(
        result=_base_result(),
        final_response="done",
        completed=True,
        interrupted=False,
        agent=DummyAgent(),
        messages=[],
        turn_exit_reason="unit_test",
    )
    assert result["completed"] is True
    assert result["apex_runtimeos"] == {"enabled": False, "status": "SKIPPED"}


def test_apex_runtimeos_hook_enforce_blocks_on_missing_runtimeos(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    monkeypatch.setenv("APEX_RUNTIMEOS_DIR", str(tmp_path / "missing"))

    # Re-import module-level path by monkeypatching the module constant directly.
    import agent.apex_runtimeos_hook as hook
    monkeypatch.setattr(hook, "APEX_RUNTIMEOS_DIR", tmp_path / "missing")

    result = hook.run_apex_runtimeos_completion_hook(
        result=_base_result(),
        final_response="done",
        completed=True,
        interrupted=False,
        agent=DummyAgent(),
        messages=[],
        turn_exit_reason="unit_test",
    )
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["apex_runtimeos"]["blocking"] is True
    assert result["apex_runtimeos"]["status"] == "ERROR"


def test_safe_apex_runtimeos_metadata_exposes_only_contract_fields():
    result = {
        "apex_runtimeos": {
            "enabled": True,
            "mode": "warn",
            "status": "PASS",
            "runtime_hook_enabled": True,
            "dry_run": False,
            "blocking": False,
            "evm_defect_count": 12,
            "evm_gate_score": 1.0,
            "organ_audit": {"audit_enabled": True, "written": True, "audit_path": "/tmp/secret/audit.jsonl"},
            "autowrite": {
                "enabled": True,
                "written": True,
                "candidate_path": "/tmp/secret/candidates.jsonl",
                "candidate_type": "memory_or_skill_review",
                "promotion_required": True,
                "applied_to_core_memory_or_skill": False,
            },
            "recommendation_control": {
                "status": "WATCH",
                "applied": True,
                "mutates_runtime": False,
                "items": [
                    {
                        "organ": "router",
                        "code": "x",
                        "severity": "warn",
                        "applied": True,
                        "mutates_runtime": False,
                        "reason": "internal detail",
                        "control": {"action": "block", "blocking": True},
                    }
                ],
            },
            "report_path": "/tmp/secret/report.json",
            "error": "stack trace with secret",
            "apex_runtimeos_dir": "/private/path",
        }
    }
    safe = _safe_apex_runtimeos_metadata(result)
    assert safe is not None
    assert safe["decision"] == "allow"
    assert safe["status"] == "PASS"
    assert safe["runtime_hook_enabled"] is True
    assert safe["organ_audit"] == {"audit_enabled": True, "written": True}
    assert safe["autowrite"] == {
        "enabled": True,
        "written": True,
        "candidate_type": "memory_or_skill_review",
        "promotion_required": True,
        "applied_to_core_memory_or_skill": False,
        "reason": None,
    }
    assert safe["recommendation_control"]["applied"] is True
    assert safe["recommendation_control"]["items"][0]["code"] == "x"
    assert "reason" not in safe["recommendation_control"]["items"][0]
    assert "report_path" not in safe
    assert "error" not in safe
    assert "apex_runtimeos_dir" not in safe
    assert "audit_path" not in safe["organ_audit"]


def test_safe_apex_runtimeos_metadata_marks_blocking_failure():
    safe = _safe_apex_runtimeos_metadata(
        {"apex_runtimeos": {"enabled": True, "mode": "enforce", "status": "FAIL", "blocking": True}}
    )
    assert safe is not None
    assert safe["decision"] == "block"
    assert safe["error_code"] == "apex_runtimeos_gate_failed"


def test_router_and_planner_pre_api_organs_are_observational(monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "warn")
    context = {
        "model": "m",
        "provider": "p",
        "api_mode": "chat_completions",
        "tool_count": 3,
        "message_count": 4,
        "api_call_count": 1,
        "approx_input_tokens": 100,
        "request_char_count": 500,
        "max_tokens": 2000,
    }
    result = run_pre_api_organs(context)
    assert result["context_unchanged"] is True
    assert result["blocking"] is False
    assert result["results"]["router"]["output"]["mutates_route"] is False
    assert result["results"]["planner"]["output"]["mutates_plan"] is False
    assert result["results"]["router"]["output"]["recommendation"]["applied"] is False
    assert result["recommendations"]["mutates_runtime"] is False


def test_gene_selector_completion_organ_is_observational(monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "warn")
    context = {"completed": True, "interrupted": False, "turn_exit_reason": "final_response", "message_count": 5}
    result = run_pre_completion_organs(context)
    gene = result["results"]["gene_selector"]
    assert result["context_unchanged"] is True
    assert result["blocking"] is False
    assert gene["output"]["mutates_memory_or_skills"] is False
    assert gene["output"]["recommendation"]["applied"] is False
    assert result["recommendations"]["mutates_runtime"] is False


def test_runtimeos_recommendations_warn_without_mutating_context(monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_ENABLED", "1")
    context = {
        "model": "m",
        "provider": "",
        "api_mode": "chat_completions",
        "tool_count": 100,
        "message_count": 200,
        "api_call_count": 41,
        "approx_input_tokens": 200000,
        "request_char_count": 800000,
        "max_tokens": 2000,
    }
    before = dict(context)
    result = run_pre_api_organs(context)
    assert context == before
    assert result["context_unchanged"] is True
    assert result["blocking"] is False
    assert result["recommendations"]["status"] == "WATCH"
    assert result["recommendations"]["applied"] is False
    assert result["recommendations"]["mutates_runtime"] is False
    codes = {item["code"] for item in result["recommendations"]["items"]}
    assert "router_provider_missing" in codes
    assert "planner_context_heavy" in codes


def test_recommendation_auto_control_is_opt_in_and_blocks_only_in_enforce(monkeypatch):
    rec = {"severity": "warn", "code": "planner_context_heavy"}
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTO_CONTROL_ENABLED", "0")
    assert evaluate_recommendation_control(rec, mode="enforce")["blocking"] is False

    monkeypatch.setenv("APEX_RUNTIMEOS_AUTO_CONTROL_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTO_CONTROL_MIN_SEVERITY", "warn")
    assert evaluate_recommendation_control(rec, mode="warn")["blocking"] is False
    decision = evaluate_recommendation_control(rec, mode="enforce")
    assert decision["blocking"] is True
    assert decision["action"] == "block"
    assert decision["mutates_runtime"] is False


def test_pre_api_auto_control_blocks_without_mutating_context(monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTO_CONTROL_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTO_CONTROL_MIN_SEVERITY", "warn")
    context = {
        "model": "m",
        "provider": "",
        "api_mode": "chat_completions",
        "tool_count": 1,
        "message_count": 2,
        "api_call_count": 1,
        "approx_input_tokens": 100,
        "request_char_count": 500,
        "max_tokens": 2000,
    }
    before = dict(context)
    result = run_pre_api_organs(context)
    assert context == before
    assert result["context_unchanged"] is True
    assert result["blocking"] is True
    assert result["results"]["router"]["status"] == "BLOCKED"
    assert result["results"]["router"]["control"]["action"] == "block"
    assert result["recommendations"]["applied"] is True
    assert result["recommendations"]["mutates_runtime"] is False


def test_audit_persists_auto_control_decision(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("APEX_RUNTIMEOS_AUDIT_PATH", str(audit_path))
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTO_CONTROL_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTO_CONTROL_MIN_SEVERITY", "warn")
    checkpoint = run_pre_api_organs({"model": "m", "provider": "", "api_mode": "chat", "tool_count": 1, "message_count": 2})
    persist_checkpoint("pre_api_request", checkpoint, session_id="s1")
    record = json.loads(audit_path.read_text(encoding="utf-8"))
    control = record["checkpoint"]["results"]["router"]["control"]
    assert control["auto_control_enabled"] is True
    assert control["action"] == "block"
    assert control["mutates_runtime"] is False
    summary = summarize_audit(audit_path)
    assert summary["blocking_records"] == 1
    assert summary["recommendations"]["auto_control"]["blocked"] >= 1
    assert summary["recommendations"]["auto_control"]["enabled_count"] >= 1
    monkeypatch.delenv("APEX_RUNTIMEOS_GATE_MODE", raising=False)
    monkeypatch.delenv("APEX_RUNTIMEOS_AUTO_CONTROL_ENABLED", raising=False)
    monkeypatch.delenv("APEX_RUNTIMEOS_AUTO_CONTROL_MIN_SEVERITY", raising=False)


def test_gene_selector_recommendation_holds_incomplete_completion(monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_ENABLED", "1")
    result = run_pre_completion_organs({"completed": False, "interrupted": False, "final_response_present": False})
    rec = result["results"]["gene_selector"]["output"]["recommendation"]
    assert rec["code"] == "gene_completion_incomplete"
    assert rec["applied"] is False
    assert rec["mutates_runtime"] is False


def test_audit_persists_sanitized_jsonl_without_paths_or_secrets(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("APEX_RUNTIMEOS_AUDIT_PATH", str(audit_path))
    checkpoint = run_pre_api_organs(
        {
            "model": "m",
            "provider": "p",
            "api_mode": "chat_completions",
            "tool_count": 1,
            "message_count": 2,
            "api_call_count": 1,
            "secret_key": "SHOULD_NOT_APPEAR",
            "local_path": "/tmp/private",
        }
    )
    result = persist_checkpoint("pre_api_request", checkpoint, session_id="s1")
    assert result["written"] is True
    raw = audit_path.read_text(encoding="utf-8")
    assert "SHOULD_NOT_APPEAR" not in raw
    assert "/tmp/private" not in raw
    record = json.loads(raw)
    assert record["schema"] == "ApexRuntimeOSCheckpointAudit/v1"
    assert record["stage"] == "pre_api_request"
    assert record["checkpoint"]["results"]["router"]["output"]["model"] == "m"


def test_audit_summary_aggregates_sessions_stages_organs_and_bad_lines(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("APEX_RUNTIMEOS_AUDIT_PATH", str(audit_path))
    pre_api = run_pre_api_organs({"model": "m1", "provider": "p", "api_mode": "chat", "tool_count": 1, "message_count": 2})
    pre_completion = run_pre_completion_organs({"completed": True, "interrupted": False, "turn_exit_reason": "done", "message_count": 3})
    persist_checkpoint("pre_api_request", pre_api, session_id="s1")
    persist_checkpoint("pre_completion", pre_completion, session_id="s1")
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write("not-json\n")
    summary = summarize_audit(audit_path)
    assert summary["records"] == 2
    assert summary["bad_lines"] == 1
    assert summary["stages"]["pre_api_request"]["count"] == 1
    assert summary["stages"]["pre_completion"]["count"] == 1
    assert summary["organs"]["router"]["status"]["PASS"] == 1
    assert summary["organs"]["planner"]["status"]["PASS"] == 1
    assert summary["organs"]["gene_selector"]["status"]["PASS"] == 1
    assert summary["models"]["m1"]["status"]["PASS"] >= 1
