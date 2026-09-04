from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from hermes_cli.feature_delivery import (
    ACCEPTANCE_FINAL_MARKER,
    FEATURE_DELIVERY_WORKFLOW,
    LEGAL_TRANSITIONS,
    MAX_FIX_LOOPS,
    AcceptanceCriterionResult,
    AcceptanceReport,
    DeveloperReport,
    FeatureDeliveryState,
    TaskContract,
    TesterReport as FDTesterReport,
    can_transition_to_blocked,
    canonicalize_contract,
    compute_contract_hash,
    count_fix_loops,
    is_legal_transition,
    validate_stage_report,
)


SHA = "a" * 40


def contract_data() -> dict:
    return {
        "task_id": "task-1",
        "title": "Add deterministic delivery guards",
        "objective": "Prevent unaccepted feature delivery",
        "repository": "hermes-agent",
        "base_commit": SHA,
        "branch": "feature/feature-delivery-v1",
        "acceptance_criteria": [
            {"id": "AC-1", "requirement": "Contract validates"},
            {"id": "AC-2", "requirement": "Delivery is acceptance-only"},
        ],
        "constraints": ["No runner"],
        "required_tests": ["feature_delivery"],
        "required_evidence": ["tests", "diff-check"],
        "out_of_scope": ["merge", "deploy"],
        "delivery_gate": "acceptance_agent",
    }


def developer_data() -> dict:
    return {
        "task_id": "task-1",
        "agent": "developer",
        "status": "READY_FOR_TEST",
        "commit": SHA,
        "changed_files": ["hermes_cli/feature_delivery.py"],
        "implementation_summary": "Added guards",
        "self_checks": ["tests passed"],
        "known_risks": [],
    }


def _tester_data() -> dict:
    return {
        "task_id": "task-1",
        "agent": "tester",
        "tested_commit": SHA,
        "status": "TEST_PASS",
        "test_results": ["feature tests passed"],
        "blocking_issues": [],
        "non_blocking_issues": [],
        "evidence": ["tests"],
    }


def acceptance_data() -> dict:
    return {
        "task_id": "task-1",
        "agent": "acceptance",
        "accepted_commit": SHA,
        "status": "ACCEPT",
        "criteria": [
            {"id": "AC-1", "met": True, "evidence": "tests"},
            {"id": "AC-2", "met": True, "evidence": "review"},
        ],
        "blocking_issues": [],
        "evidence": ["tests", "diff-check"],
        "final_marker": ACCEPTANCE_FINAL_MARKER,
    }


def test_feature_delivery_constants_are_exact():
    assert FEATURE_DELIVERY_WORKFLOW == "feature_delivery_v1"
    assert MAX_FIX_LOOPS == 5


def test_valid_task_contract():
    assert TaskContract.model_validate(contract_data()).task_id == "task-1"


@pytest.mark.parametrize("base_commit", ["abc", "A" * 40, "g" * 40, "a" * 39])
def test_invalid_base_commit(base_commit):
    data = contract_data()
    data["base_commit"] = base_commit
    with pytest.raises(ValidationError):
        TaskContract.model_validate(data)


def test_duplicate_acceptance_criterion_id_rejected():
    data = contract_data()
    data["acceptance_criteria"][1]["id"] = "AC-1"
    with pytest.raises(ValidationError, match="unique"):
        TaskContract.model_validate(data)


@pytest.mark.parametrize("field", ["id", "requirement"])
def test_empty_acceptance_criterion_rejected(field):
    data = contract_data()
    data["acceptance_criteria"][0][field] = "  "
    with pytest.raises(ValidationError):
        TaskContract.model_validate(data)


@pytest.mark.parametrize("field", ["task_id", "title", "objective", "repository", "branch"])
def test_empty_required_contract_text_rejected(field):
    data = contract_data()
    data[field] = ""
    with pytest.raises(ValidationError):
        TaskContract.model_validate(data)


def test_contract_requires_acceptance_criterion():
    data = contract_data()
    data["acceptance_criteria"] = []
    with pytest.raises(ValidationError):
        TaskContract.model_validate(data)


@pytest.mark.parametrize("field", ["required_tests", "required_evidence"])
def test_contract_requires_test_and_evidence_entries(field):
    data = contract_data()
    data[field] = []
    with pytest.raises(ValidationError):
        TaskContract.model_validate(data)


def test_invalid_delivery_gate_rejected():
    data = contract_data()
    data["delivery_gate"] = "developer"
    with pytest.raises(ValidationError):
        TaskContract.model_validate(data)


def test_contract_is_frozen():
    contract = TaskContract.model_validate(contract_data())
    with pytest.raises(ValidationError):
        contract.title = "changed"


def test_canonical_contract_hash_is_stable():
    first = TaskContract.model_validate(contract_data())
    reordered = TaskContract.model_validate(dict(reversed(list(contract_data().items()))))
    assert canonicalize_contract(first) == canonicalize_contract(reordered)
    assert compute_contract_hash(first) == compute_contract_hash(reordered)


def test_canonical_contract_is_compact_utf8_json():
    data = contract_data()
    data["title"] = "功能交付"
    contract = TaskContract.model_validate(data)
    encoded = canonicalize_contract(contract)
    assert b'": ' not in encoded
    assert b'", ' not in encoded
    assert json.loads(encoded.decode("utf-8"))["title"] == "功能交付"


def test_contract_change_changes_hash():
    first = TaskContract.model_validate(contract_data())
    data = contract_data()
    data["objective"] = "Different objective"
    assert compute_contract_hash(first) != compute_contract_hash(TaskContract.model_validate(data))


def test_contract_hash_is_lowercase_sha256():
    value = compute_contract_hash(TaskContract.model_validate(contract_data()))
    assert len(value) == 64
    assert value == value.lower()
    int(value, 16)


def test_ready_developer_report_is_valid():
    assert DeveloperReport.model_validate(developer_data()).commit == SHA


def test_blocked_developer_report_allows_null_commit():
    data = developer_data()
    data.update(status="BLOCKED", commit=None)
    assert DeveloperReport.model_validate(data).commit is None


@pytest.mark.parametrize("status", ["ACCEPT", "TEST_PASS", "DELIVERED"])
def test_developer_cannot_submit_other_stage_status(status):
    data = developer_data()
    data["status"] = status
    with pytest.raises(ValidationError):
        DeveloperReport.model_validate(data)


def test_ready_developer_report_requires_commit():
    data = developer_data()
    data["commit"] = None
    with pytest.raises(ValidationError, match="requires"):
        DeveloperReport.model_validate(data)


def test_developer_report_rejects_malformed_commit():
    data = developer_data()
    data["commit"] = "short"
    with pytest.raises(ValidationError):
        DeveloperReport.model_validate(data)


@pytest.mark.parametrize("status", ["TEST_PASS", "TEST_FAIL"])
def test_completed_tester_report_is_valid(status):
    data = _tester_data()
    data["status"] = status
    assert FDTesterReport.model_validate(data).status.value == status


def test_blocked_tester_report_allows_null_commit():
    data = _tester_data()
    data.update(status="BLOCKED", tested_commit=None)
    assert FDTesterReport.model_validate(data).tested_commit is None


@pytest.mark.parametrize("status", ["ACCEPT", "DELIVERED", "READY_FOR_TEST"])
def test_tester_cannot_submit_other_stage_status(status):
    data = _tester_data()
    data["status"] = status
    with pytest.raises(ValidationError):
        FDTesterReport.model_validate(data)


@pytest.mark.parametrize("status", ["TEST_PASS", "TEST_FAIL"])
def test_completed_tester_report_requires_commit(status):
    data = _tester_data()
    data.update(status=status, tested_commit=None)
    with pytest.raises(ValidationError, match="require"):
        FDTesterReport.model_validate(data)


def test_acceptance_accept_is_valid():
    assert AcceptanceReport.model_validate(acceptance_data()).status.value == "ACCEPT"


def test_acceptance_reject_is_valid():
    data = acceptance_data()
    data.update(status="REJECT", final_marker=None)
    assert AcceptanceReport.model_validate(data).status.value == "REJECT"


def test_acceptance_blocked_is_valid_with_null_commit():
    data = acceptance_data()
    data.update(status="BLOCKED", accepted_commit=None, final_marker=None)
    assert AcceptanceReport.model_validate(data).accepted_commit is None


@pytest.mark.parametrize("marker", [None, "accept", "approved", "FINAL ACCEPT"])
def test_acceptance_requires_exact_final_marker(marker):
    data = acceptance_data()
    data["final_marker"] = marker
    with pytest.raises(ValidationError, match="exact"):
        AcceptanceReport.model_validate(data)


@pytest.mark.parametrize("status", ["REJECT", "BLOCKED"])
def test_non_acceptance_cannot_use_accept_marker(status):
    data = acceptance_data()
    data["status"] = status
    if status == "BLOCKED":
        data["accepted_commit"] = None
    with pytest.raises(ValidationError, match="cannot"):
        AcceptanceReport.model_validate(data)


def test_acceptance_report_rejects_duplicate_criteria():
    data = acceptance_data()
    data["criteria"].append(data["criteria"][0])
    with pytest.raises(ValidationError, match="unique"):
        AcceptanceReport.model_validate(data)


@pytest.mark.parametrize(
    ("role", "factory"),
    [
        ("developer", lambda: DeveloperReport.model_validate(developer_data())),
        ("tester", lambda: FDTesterReport.model_validate(_tester_data())),
        ("acceptance", lambda: AcceptanceReport.model_validate(acceptance_data())),
    ],
)
def test_stage_role_accepts_only_matching_report(role, factory):
    assert validate_stage_report(role, factory())


@pytest.mark.parametrize(
    ("role", "factory"),
    [
        ("developer", lambda: FDTesterReport.model_validate(_tester_data())),
        ("developer", lambda: AcceptanceReport.model_validate(acceptance_data())),
        ("tester", lambda: DeveloperReport.model_validate(developer_data())),
        ("tester", lambda: AcceptanceReport.model_validate(acceptance_data())),
        ("acceptance", lambda: DeveloperReport.model_validate(developer_data())),
        ("acceptance", lambda: FDTesterReport.model_validate(_tester_data())),
    ],
)
def test_stage_role_rejects_other_report_types(role, factory):
    assert not validate_stage_report(role, factory())


def test_stage_role_does_not_scan_natural_language_for_acceptance():
    data = developer_data()
    data["implementation_summary"] = "FINAL: ACCEPT"
    assert not validate_stage_report("acceptance", DeveloperReport.model_validate(data))


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (current, target)
        for current, targets in LEGAL_TRANSITIONS.items()
        for target in targets
    ],
)
def test_declared_legal_transitions_pass(current, target):
    assert is_legal_transition(current, target)


@pytest.mark.parametrize(
    "current",
    [
        FeatureDeliveryState.DEVELOPING,
        FeatureDeliveryState.READY_FOR_TEST,
        FeatureDeliveryState.TESTING,
        FeatureDeliveryState.TEST_FAILED,
        FeatureDeliveryState.TEST_PASSED,
        FeatureDeliveryState.REJECTED,
    ],
)
def test_delivery_shortcuts_are_rejected(current):
    assert not is_legal_transition(current, FeatureDeliveryState.DELIVERED)


@pytest.mark.parametrize(
    "terminal",
    [FeatureDeliveryState.BLOCKED, FeatureDeliveryState.DELIVERED],
)
def test_terminal_states_have_no_outgoing_transition(terminal):
    assert LEGAL_TRANSITIONS[terminal] == frozenset()


@pytest.mark.parametrize(
    "state",
    [
        FeatureDeliveryState.CONTRACT_READY,
        FeatureDeliveryState.DEVELOPING,
        FeatureDeliveryState.READY_FOR_TEST,
        FeatureDeliveryState.TESTING,
        FeatureDeliveryState.TEST_FAILED,
        FeatureDeliveryState.TEST_PASSED,
        FeatureDeliveryState.ACCEPTANCE,
        FeatureDeliveryState.REJECTED,
    ],
)
def test_selected_nonterminal_states_can_block(state):
    assert can_transition_to_blocked(state)


@pytest.mark.parametrize(
    "state",
    [
        FeatureDeliveryState.NEW,
        FeatureDeliveryState.BLOCKED,
        FeatureDeliveryState.DELIVERED,
    ],
)
def test_new_and_terminal_states_cannot_block(state):
    assert not can_transition_to_blocked(state)


def test_test_failure_return_counts_as_fix_loop():
    assert count_fix_loops([(FeatureDeliveryState.TEST_FAILED, FeatureDeliveryState.DEVELOPING)]) == 1


def test_rejection_return_counts_as_fix_loop():
    assert count_fix_loops([(FeatureDeliveryState.REJECTED, FeatureDeliveryState.DEVELOPING)]) == 1


def test_other_transitions_do_not_count_as_fix_loops():
    assert count_fix_loops(
        [
            (FeatureDeliveryState.NEW, FeatureDeliveryState.CONTRACT_READY),
            (FeatureDeliveryState.DEVELOPING, FeatureDeliveryState.READY_FOR_TEST),
        ]
    ) == 0


def test_acceptance_criterion_result_rejects_empty_evidence():
    with pytest.raises(ValidationError):
        AcceptanceCriterionResult(id="AC-1", met=True, evidence="")
