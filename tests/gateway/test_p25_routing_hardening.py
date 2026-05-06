"""P2.5 routing hardening tests: docs, route contracts, telemetry, evaluator."""

from __future__ import annotations

from pathlib import Path

import pytest

import gateway.worker_evaluator as worker_evaluator
import tools.skill_usage as skill_usage
from gateway.background_wakeups import clear_background_wake_manifest_cache
from gateway.route_decision import resolve_route_decision
from gateway.worker_evaluator import ROUTE_CONTRACTS, evaluate_background_worker_outcome
from tools.skill_usage import load_usage, log_route_usage_event, summarize_route_usage

REPO_ROOT = Path(__file__).resolve().parents[2]
for _module in (worker_evaluator, skill_usage):
    _module_path = Path(_module.__file__).resolve()
    assert _module_path.is_relative_to(REPO_ROOT), (
        f"{_module.__name__} resolved outside clean worktree: {_module_path}"
    )


@pytest.fixture(autouse=True)
def isolated_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    clear_background_wake_manifest_cache()
    yield tmp_path
    clear_background_wake_manifest_cache()


def test_route_usage_events_are_persisted_and_summarized(isolated_hermes_home):
    log_route_usage_event(
        route_name="repo",
        event="route_decision_resolved",
        details={"decision_type": "auto_dispatch", "score": 4.5, "confidence": 0.9},
    )
    log_route_usage_event(
        route_name="repo",
        event="route_worker_outcome",
        details={"passed": True, "score": 0.95, "issues": []},
    )
    log_route_usage_event(
        route_name="research",
        event="route_worker_outcome",
        details={"passed": False, "score": 0.4, "issues": ["missing_sources"]},
    )

    usage = load_usage()
    assert "_route_usage" in usage
    assert usage["_route_usage"]["repo"]["event_count"] == 2
    assert usage["_route_usage"]["repo"]["events"][-1]["event"] == "route_worker_outcome"

    summary = summarize_route_usage(window_days=30)
    assert summary["repo"]["history_sample_count_30d"] == 1
    assert summary["repo"]["worker_outcome_count_30d"] == 1
    assert summary["repo"]["worker_outcome_signal_30d"] > 0
    assert summary["research"]["worker_outcome_signal_30d"] < 0


def test_route_decision_logs_telemetry_when_source_is_explicit():
    decision = resolve_route_decision(
        "请体系化审查和制定route机制提升计划，阅读开源社区先进案例、Hermes本身机制",
        platform="feishu",
        active_toolsets=("terminal", "file", "skills"),
        telemetry_source="feishu_auto_dispatch",
    )

    assert decision.forced_routes
    summary = summarize_route_usage(window_days=30)
    for route in decision.forced_routes:
        assert summary[route]["history_sample_count_30d"] >= 1
        assert summary[route]["route_signal_score_30d"] != 0


def test_route_contract_registry_covers_live_worker_routes():
    expected_routes = {
        "repo",
        "research",
        "scan",
        "doc_feishu",
        "doc_google",
        "doc_pdf",
        "ppt",
        "automation",
        "multi_agent",
        "difficult_web_extract",
    }

    assert expected_routes <= set(ROUTE_CONTRACTS)
    for route_name in expected_routes:
        contract = ROUTE_CONTRACTS[route_name]
        assert contract.route_name == route_name
        assert contract.description
        assert contract.required_evidence
        assert contract.pass_threshold >= 0.7
        assert contract.to_dict()["route_name"] == route_name


@pytest.mark.parametrize(
    ("route_names", "response", "expected_issue"),
    [
        (("doc_feishu",), "I can draft this later after I get more context.", "missing_document_artifact"),
        (("ppt",), "Looks fine, no slides needed.", "missing_presentation_artifact"),
        (("automation",), "I would automate it conceptually.", "missing_automation_evidence"),
        (("multi_agent",), "One worker result only, no synthesis needed.", "missing_orchestration_summary"),
        (("difficult_web_extract",), "Could not fetch it; maybe use browser later.", "missing_difficult_web_extract_receipt"),
    ],
)
def test_worker_evaluator_enforces_route_specific_artifact_contracts(route_names, response, expected_issue):
    evaluation = evaluate_background_worker_outcome(
        prompt="Run the routed worker and return the artifact.",
        route_names=route_names,
        response=response,
    )

    assert evaluation.passed is False
    assert expected_issue in evaluation.issues
    assert evaluation.score < 0.7
    assert set(route_names) <= set(evaluation.route_contracts)
    assert expected_issue in evaluation.to_dict()["issues"]
    assert "route_contracts" in evaluation.to_dict()


def test_worker_evaluator_accepts_route_specific_evidence_when_present():
    evaluation = evaluate_background_worker_outcome(
        prompt="Research and summarize sources, then inspect repo files.",
        route_names=("research", "repo"),
        response=(
            "Sources: https://example.com/report and https://example.com/docs.\n"
            "Repo evidence: inspected gateway/run.py and tests/gateway/test_background_command.py; "
            "ran pytest tests/gateway/test_background_command.py -q."
        ),
    )

    assert evaluation.passed is True
    assert evaluation.score >= 0.9
    assert evaluation.issues == ()
    assert set(evaluation.route_contracts) == {"research", "repo"}


def test_worker_evaluator_accepts_difficult_web_extract_receipt():
    evaluation = evaluate_background_worker_outcome(
        prompt="Use selector extraction fallback.",
        route_names=("difficult_web_extract",),
        response=(
            "Receipt: backend=scrapling mode=static url=https://example.com "
            "selector=.article fallback_reason=web_extract_empty errors=[]"
        ),
    )

    assert evaluation.passed is True
    assert evaluation.issues == ()
    assert evaluation.score >= 0.9


def test_config_docs_and_example_document_routing_controls():
    configuration_doc = (REPO_ROOT / "website/docs/user-guide/configuration.md").read_text(encoding="utf-8")
    config_example = (REPO_ROOT / "cli-config.yaml.example").read_text(encoding="utf-8")
    telemetry_doc = (REPO_ROOT / "docs/specs/route-telemetry-events.md").read_text(encoding="utf-8")

    for text in (configuration_doc, config_example, telemetry_doc):
        assert "routing:" in text
        assert "feishu_auto_dispatch_enabled" in text
        assert "feishu_route_shadow_hints_enabled" in text
        assert "difficult_web_extract" in text
        assert "route_decision_resolved" in text
        assert "route_selected_for_background" in text
        assert "route_worker_outcome" in text
        assert "worker_evaluation" in text
