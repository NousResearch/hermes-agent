"""Tests for completion-auditor's deterministic evidence evaluator."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_plugin_package():
    plugin_dir = _repo_root() / "plugins" / "completion-auditor"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    for key in list(sys.modules):
        if key.startswith("hermes_plugins.completion_auditor_verdict_under_test"):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.completion_auditor_verdict_under_test",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.completion_auditor_verdict_under_test"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.completion_auditor_verdict_under_test"] = mod
    spec.loader.exec_module(mod)
    mod._reset_for_tests()
    return mod


def _record_for_response(assistant_response: str, *, tool_name="terminal", args=None, result=None, status="success"):
    mod = _load_plugin_package()
    evidence_mod = sys.modules["hermes_plugins.completion_auditor_verdict_under_test.evidence"]
    report_mod = sys.modules["hermes_plugins.completion_auditor_verdict_under_test.report"]
    config_mod = sys.modules["hermes_plugins.completion_auditor_verdict_under_test.config"]
    evidence_mod.record_tool_call(
        session_id="s1",
        turn_id="t1",
        tool_name=tool_name,
        status=status,
        args=args or {},
        result=result,
    )
    evidence = evidence_mod.pop_turn("s1", "t1")
    return report_mod.build_record(
        settings=config_mod.AuditorConfig(),
        session_id="s1",
        turn_id="t1",
        assistant_response=assistant_response,
        evidence=evidence,
    )


def test_tested_claim_supported_by_successful_structured_command_evidence():
    record = _record_for_response(
        "Tests passed: 21 passed.",
        args={"command": "python -m pytest tests/plugins -q"},
        result={"output": "21 passed", "exit_code": 0},
    )
    assert record["verdict"] == "supported"
    assert record["evidence_tier"] == "tier_1"
    assert record["evidence_refs"][0]["exit_code"] == 0
    assert record["semantic_correctness_guaranteed"] is False


def test_tested_claim_not_supported_by_non_verification_command_success():
    record = _record_for_response(
        "Tests passed: 21 passed.",
        args={"command": "ls plugins"},
        result={"output": "plugins", "exit_code": 0},
    )
    assert record["verdict"] == "weak"
    assert record["evidence_tier"] == "tier_3"
    assert record["evidence_refs"][0]["command_kind"] == "other"


def test_tested_claim_not_supported_by_path_or_echo_words_that_look_like_verification():
    for command in ("ls tests", "echo build", "cat tests/plugins/test_completion_auditor_verdict.py"):
        record = _record_for_response(
            "Tests passed: 21 passed.",
            args={"command": command},
            result={"output": "ok", "exit_code": 0},
        )
        assert record["verdict"] == "weak"
        assert record["evidence_tier"] == "tier_3"
        assert record["evidence_refs"][0]["command_kind"] == "other"


def test_claim_text_is_redacted_before_logging():
    record = _record_for_response("Updated password='super-secret-token'.")
    assert "super-secret-token" not in record["claim_text"]
    assert "[REDACTED]" in record["claim_text"]


def test_tested_claim_fails_on_structured_nonzero_exit_code():
    record = _record_for_response(
        "Tests passed: 21 passed.",
        result={"output": "1 failed", "exit_code": 1},
    )
    assert record["verdict"] == "fail"
    assert record["evidence_tier"] == "tier_1"
    assert record["contradictions"]


def test_modified_claim_supported_by_matching_path_mutation_evidence():
    record = _record_for_response(
        "Updated plugins/completion-auditor/report.py.",
        tool_name="patch",
        args={"path": "plugins/completion-auditor/report.py"},
        result={"success": True},
    )
    assert record["verdict"] == "supported"
    assert record["evidence_tier"] == "tier_1"
    assert record["claim_scope"] == "plugins/completion-auditor/report.py"
    assert record["evidence_refs"][0]["arg_refs"] == ["plugins/completion-auditor/report.py"]


def test_modified_claim_weak_when_mutation_evidence_scope_does_not_match():
    record = _record_for_response(
        "Updated plugins/completion-auditor/report.py.",
        tool_name="patch",
        args={"path": "plugins/completion-auditor/config.py"},
        result={"success": True},
    )
    assert record["verdict"] == "weak"
    assert record["evidence_tier"] == "tier_3"
    assert record["degraded"] is True
    assert record["degrade_reason"] == "mutation evidence does not match claim_scope"


def test_no_completion_claim_remains_not_applicable_even_with_evidence():
    record = _record_for_response(
        "Hermes plugins are opt-in.",
        result={"output": "ok", "exit_code": 0},
    )
    assert record["verdict"] == "not_applicable"
    assert record["evidence_tier"] == "tier_4"
    assert record["claim_text"] is None
