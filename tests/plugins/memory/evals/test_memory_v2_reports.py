"""Report rendering and scorecard tests for Memory v2 evals."""

from __future__ import annotations

import json

from plugins.memory.memory_v2.evals.reports import (
    EvalReport,
    EvalScoreRow,
    build_acceptance_scorecard,
    render_markdown_report,
    write_json_report,
)


def test_eval_report_dict_includes_scorecard_fields():
    report = _passing_report()

    payload = report.to_dict()

    assert payload["dataset"] == "local_memory_eval_v1"
    assert set(payload["summary"]) == {"raw_fts", "memory_v2"}
    assert payload["summary"]["memory_v2"]["source_recall_avg"] == 1.0
    assert [row["query_id"] for row in payload["rows"] if row["baseline"] == "memory_v2"] == ["q_pref", "q_irrelevant"]
    assert payload["acceptance"]["passed"] is True
    assert payload["acceptance"]["target_baseline"] == "memory_v2"
    assert {check["name"] for check in payload["acceptance"]["checks"]} >= {
        "source_correctness",
        "irrelevant_suppression",
        "token_budget",
        "memory_v2_vs_raw_fts_source_recall",
    }


def test_scorecard_exposes_per_query_failures():
    report = EvalReport(
        dataset="regression_fixture",
        rows=[
            EvalScoreRow(
                baseline="memory_v2",
                query_id="q_bad_source",
                route="preference_recall",
                source_recall=0.0,
                text_contains=1.0,
                suppression=1.0,
                retrieved_count=1,
                token_estimate=25,
                latency_ms=1.0,
                retrieved_source_refs=["wrong_event"],
            ),
            EvalScoreRow(
                baseline="memory_v2",
                query_id="q_leaky_irrelevant",
                route="no_memory_needed",
                source_recall=1.0,
                text_contains=1.0,
                suppression=0.0,
                retrieved_count=1,
                token_estimate=1,
                latency_ms=1.0,
            ),
        ],
        summary={
            "memory_v2": {
                "query_count": 2,
                "source_recall_avg": 0.5,
                "text_contains_avg": 1.0,
                "suppression_avg": 0.5,
                "token_estimate_total": 26,
                "latency_ms_avg": 1.0,
            }
        },
    )

    scorecard = build_acceptance_scorecard(report)

    assert scorecard["passed"] is False
    source_check = _check_by_name(scorecard, "source_correctness")
    suppression_check = _check_by_name(scorecard, "irrelevant_suppression")
    token_check = _check_by_name(scorecard, "token_budget")
    assert source_check["failed_rows"][0]["query_id"] == "q_bad_source"
    assert suppression_check["failed_rows"][0]["query_id"] == "q_leaky_irrelevant"
    assert token_check["failed_rows"][0]["query_id"] == "q_leaky_irrelevant"


def test_write_json_report_is_stable_and_json_serializable(tmp_path):
    output_path = tmp_path / "report.json"
    report = _passing_report()

    write_json_report(report, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == report.to_dict()
    assert payload["acceptance"]["thresholds"]["source_correctness_min"] == 0.95


def test_render_markdown_report_includes_summary_and_acceptance_status():
    markdown = render_markdown_report(_passing_report())

    assert "# Memory v2 eval: local_memory_eval_v1" in markdown
    assert "memory_v2: source=1.000" in markdown
    assert "## Acceptance: PASS" in markdown


def _passing_report() -> EvalReport:
    return EvalReport(
        dataset="local_memory_eval_v1",
        rows=[
            EvalScoreRow(
                baseline="raw_fts",
                query_id="q_pref",
                route="preference_recall",
                source_recall=1.0,
                text_contains=1.0,
                suppression=1.0,
                retrieved_count=1,
                token_estimate=50,
                latency_ms=1.0,
                retrieved_source_refs=["event_pref"],
            ),
            EvalScoreRow(
                baseline="memory_v2",
                query_id="q_pref",
                route="preference_recall",
                source_recall=1.0,
                text_contains=1.0,
                suppression=1.0,
                retrieved_count=1,
                token_estimate=40,
                latency_ms=1.0,
                retrieved_source_refs=["event_pref"],
            ),
            EvalScoreRow(
                baseline="memory_v2",
                query_id="q_irrelevant",
                route="no_memory_needed",
                source_recall=1.0,
                text_contains=1.0,
                suppression=1.0,
                retrieved_count=0,
                token_estimate=0,
                latency_ms=1.0,
            ),
        ],
        summary={
            "raw_fts": {
                "query_count": 1,
                "source_recall_avg": 1.0,
                "text_contains_avg": 1.0,
                "suppression_avg": 1.0,
                "token_estimate_total": 50,
                "latency_ms_avg": 1.0,
            },
            "memory_v2": {
                "query_count": 2,
                "source_recall_avg": 1.0,
                "text_contains_avg": 1.0,
                "suppression_avg": 1.0,
                "token_estimate_total": 40,
                "latency_ms_avg": 1.0,
            },
        },
    )


def _check_by_name(scorecard: dict, name: str) -> dict:
    return next(check for check in scorecard["checks"] if check["name"] == name)


def test_scorecard_fails_when_average_passes_but_one_row_fails():
    good_rows = [
        EvalScoreRow(
            baseline="memory_v2",
            query_id=f"q_good_{index}",
            route="preference_recall",
            source_recall=1.0,
            text_contains=1.0,
            suppression=1.0,
            retrieved_count=1,
            token_estimate=10,
            latency_ms=1.0,
            retrieved_source_refs=["event"],
        )
        for index in range(19)
    ]
    bad = EvalScoreRow(
        baseline="memory_v2",
        query_id="q_bad",
        route="preference_recall",
        source_recall=0.0,
        text_contains=1.0,
        suppression=1.0,
        retrieved_count=1,
        token_estimate=10,
        latency_ms=1.0,
        retrieved_source_refs=["wrong"],
    )
    report = EvalReport(
        dataset="average_can_hide_failure",
        rows=[bad, *good_rows],
        summary={
            "memory_v2": {
                "query_count": 20,
                "source_recall_avg": 0.95,
                "text_contains_avg": 1.0,
                "suppression_avg": 1.0,
                "token_estimate_total": 200,
                "latency_ms_avg": 1.0,
            }
        },
    )

    scorecard = build_acceptance_scorecard(report)

    assert scorecard["passed"] is False
    assert _check_by_name(scorecard, "source_correctness")["failed_rows"][0]["query_id"] == "q_bad"
