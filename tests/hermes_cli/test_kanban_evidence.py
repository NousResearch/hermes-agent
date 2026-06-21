from __future__ import annotations

from hermes_cli import kanban_evidence as ke


BODY = """**Goal**
- Ship a verifiable fix.

**Approach**
- Inspect the current code path.
- Make the smallest needed change.

**Acceptance criteria**
- The target behavior is covered by a passing check.
- No unrelated behavior changes are introduced.

**Evidence required**
- Record changed files.
- Record commands or tests that verified the result.

**Out of scope**
- None known.
"""


def complete_metadata() -> dict:
    return {
        "changed_files": ["hermes_cli/example.py"],
        "commands_run": ["pytest tests/hermes_cli/test_example.py"],
        "tests": [{"name": "pytest", "status": "passed"}],
        "acceptance": [{"criterion": "covered by passing check", "status": "met"}],
        "artifacts": [],
        "decisions": [],
        "open_questions": [],
        "critic_review": {"verdict": "pass"},
        "temp_files": [],
        "cleanup": {},
        "repair_loop": {},
        "hypothesis_tests": [],
    }


def test_parse_task_contract_sections():
    contract = ke.parse_task_contract(BODY)
    assert contract.missing_sections == []
    assert contract.item_count("acceptance criteria") == 2
    assert contract.item_count("evidence required") == 2


def test_evidence_metadata_passes_when_real_verification_exists():
    report = ke.evaluate_evidence(
        task_id="t_1",
        task_body=BODY,
        task_status="done",
        summary="Implemented and verified.",
        metadata=complete_metadata(),
        strict=True,
        max_attempts=2,
    )
    assert report.verdict == "pass"
    assert report.ok
    assert report.actual_verification_count == 2


def test_hypothesis_tests_do_not_replace_real_verification():
    metadata = complete_metadata()
    metadata["commands_run"] = []
    metadata["tests"] = []
    metadata["hypothesis_tests"] = [
        {
            "hypothesis": "The parser accepts the new section.",
            "expected": "The check would pass.",
            "actual": "Reasoned only.",
        }
    ]
    report = ke.evaluate_evidence(
        task_id="t_1",
        task_body=BODY,
        summary="Reasoned about the result.",
        metadata=metadata,
        strict=True,
    )
    assert not report.ok
    assert "no real tests or commands_run" in "\n".join(report.missing)


def test_virtual_experiments_alias_maps_to_hypothesis_tests():
    metadata = complete_metadata()
    metadata.pop("hypothesis_tests")
    metadata["virtual_experiments"] = [{"scenario": "legacy name"}]
    report = ke.evaluate_evidence(
        task_id="t_1",
        task_body=BODY,
        summary="Implemented and verified.",
        metadata=metadata,
        strict=False,
    )
    assert report.hypothesis_test_count == 1
    assert any("virtual_experiments" in item for item in report.warnings)


def test_temp_files_must_be_ledger_objects():
    metadata = complete_metadata()
    metadata["temp_files"] = ["C:/tmp/scratch.json"]
    report = ke.evaluate_evidence(
        task_id="t_1",
        task_body=BODY,
        summary="Implemented and verified.",
        metadata=metadata,
        strict=True,
    )
    assert not report.ok
    assert any("bare path" in item for item in report.missing)


def test_repair_loop_marks_exhausted_attempts():
    metadata = complete_metadata()
    metadata["commands_run"] = []
    metadata["tests"] = []
    metadata["repair_loop"] = {
        "attempt": 2,
        "max_attempts": 2,
        "failure_reason": "verification failed",
        "next_strategy": "escalate",
    }
    report = ke.evaluate_evidence(
        task_id="t_1",
        task_body=BODY,
        summary="Still failing.",
        metadata=metadata,
        strict=True,
        prior_attempts=2,
        max_attempts=2,
    )
    assert not report.repair_loop["can_retry"]
    assert report.repair_loop["attempts_exhausted"]
    assert "repair loop attempts are exhausted" in report.missing


def test_critic_review_can_create_open_gap():
    metadata = complete_metadata()
    metadata["critic_review"] = {
        "verdict": "needs_work",
        "open_issues": ["acceptance criterion is not proven"],
    }
    report = ke.evaluate_evidence(
        task_id="t_1",
        task_body=BODY,
        summary="Implemented and verified.",
        metadata=metadata,
        strict=True,
    )
    assert not report.ok
    assert report.critic_issue_count == 2
    assert any("critic_review unresolved" in item for item in report.missing)
