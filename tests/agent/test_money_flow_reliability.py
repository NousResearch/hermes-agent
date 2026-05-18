import json
from datetime import datetime, timezone

import pytest

from agent.money_flow_reliability import (
    ReliabilityStatus,
    evaluate_money_flow_report,
    evaluate_money_flow_report_path,
    fingerprint_money_flow_report,
    load_money_flow_report,
)


NOW = datetime(2026, 5, 12, 0, 0, tzinfo=timezone.utc)


def _valid_report() -> dict:
    return {
        "report_id": "mfr-2026-05-12",
        "generated_at": "2026-05-12T00:00:00Z",
        "source_freshness": {
            "global_liquidity": {
                "status": "fresh",
                "as_of": "2026-05-11T23:00:00Z",
            },
            "sector_flows": {
                "status": "fresh",
                "retrieved_at": "2026-05-11T22:30:00Z",
            },
        },
        "regime": "risk-on liquidity impulse",
        "rotation": "technology and industrials leadership under review",
        "evidence": {
            "regime": [
                {
                    "claim": "Liquidity impulse improved.",
                    "source_id": "liq-dashboard",
                    "as_of": "2026-05-11T23:00:00Z",
                }
            ],
            "rotation": [
                {
                    "claim": "Relative flows favored technology and industrials.",
                    "citations": ["sector-flows-2026-05-11"],
                    "source": "sector flow model",
                }
            ],
        },
        "contradictions": [],
        "watchlist": [
            {
                "symbol": "MSFT",
                "note": "Monitor only: relative flow confirmation is for review; user decides whether any action is appropriate.",
            }
        ],
        "review_ledger_summary": {
            "reviewed_reports": 14,
            "expansion_eligible": True,
        },
    }


def _check(result, name):
    return next(check for check in result.checks if check.name == name)


def test_valid_report_returns_ok_and_json_serializable():
    result = evaluate_money_flow_report(_valid_report(), now=NOW)

    assert result.overall_status == ReliabilityStatus.OK
    assert result.summary == {"OK": 8, "WARN": 0, "FAIL": 0}
    assert len(result.report_fingerprint) == 64

    payload = result.to_dict()
    assert payload["overall_status"] == "OK"
    json.dumps(payload, sort_keys=True)
    assert _check(result, "result_json_serializable").status == ReliabilityStatus.OK


def test_fingerprint_excludes_generated_at_but_preserves_report_content_changes():
    report = _valid_report()
    changed_time = dict(report, generated_at="2026-05-12T01:15:00Z")
    changed_watchlist = dict(report, watchlist=["Monitor only: user decides. New symbol: NVDA."])

    assert fingerprint_money_flow_report(report) == fingerprint_money_flow_report(changed_time)
    assert fingerprint_money_flow_report(report) != fingerprint_money_flow_report(changed_watchlist)


def test_load_json_path_and_evaluate(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_valid_report()), encoding="utf-8")

    loaded = load_money_flow_report(path)
    result = evaluate_money_flow_report_path(path, now=NOW)

    assert loaded["report_id"] == "mfr-2026-05-12"
    assert result.overall_status == ReliabilityStatus.OK


def test_missing_identity_and_contradictions_fail():
    report = _valid_report()
    report.pop("report_id")
    report["generated_at"] = "not-a-timestamp"
    report.pop("contradictions")

    result = evaluate_money_flow_report(report, now=NOW)

    assert result.overall_status == ReliabilityStatus.FAIL
    assert _check(result, "report_identity").status == ReliabilityStatus.FAIL
    assert _check(result, "contradictions_visible").status == ReliabilityStatus.FAIL


def test_stale_source_warns_and_failed_source_fails():
    stale_report = _valid_report()
    stale_report["source_freshness"] = {
        "sector_flows": {
            "status": "fresh",
            "as_of": "2026-05-09T00:00:00Z",
        }
    }
    stale_result = evaluate_money_flow_report(stale_report, now=NOW)

    assert stale_result.overall_status == ReliabilityStatus.WARN
    assert _check(stale_result, "source_freshness").status == ReliabilityStatus.WARN

    failed_report = _valid_report()
    failed_report["source_freshness"] = {
        "global_liquidity": {
            "status": "failed",
            "as_of": "2026-05-11T23:00:00Z",
        }
    }
    failed_result = evaluate_money_flow_report(failed_report, now=NOW)

    assert failed_result.overall_status == ReliabilityStatus.FAIL
    assert _check(failed_result, "source_freshness").status == ReliabilityStatus.FAIL


def test_regime_rotation_evidence_requires_visible_provenance():
    report = _valid_report()
    report["evidence"] = {
        "regime": [{"claim": "Liquidity improved."}],
        "rotation": [{"claim": "Leadership broadened."}],
    }

    result = evaluate_money_flow_report(report, now=NOW)

    check = _check(result, "regime_rotation_evidence")
    assert check.status == ReliabilityStatus.FAIL
    assert "missing provenance" in check.detail


def test_watchlist_forbidden_recommendation_language_fails_and_redacts_secrets():
    report = _valid_report()
    report["watchlist"] = [
        "OPENAI_API_KEY=sk-testsecretvalue1234567890 Buy AAPL at the open with 10% allocation."
    ]

    result = evaluate_money_flow_report(report, now=NOW)

    check = _check(result, "watchlist_safety_boundary")
    assert check.status == ReliabilityStatus.FAIL
    assert "buy/sell/hold" in check.detail
    assert "position sizing" in check.detail
    assert "order timing" in check.detail
    assert "sk-testsecretvalue" not in check.detail
    assert "[REDACTED]" in check.detail


@pytest.mark.parametrize(
    "ledger, expected",
    [
        ({"reviewed_reports": 13, "expansion_eligible": False}, ReliabilityStatus.WARN),
        ({"reviewed_reports": 13, "expansion_eligible": True}, ReliabilityStatus.FAIL),
        ({"reviewed_reports": 14, "expansion_eligible": True}, ReliabilityStatus.OK),
    ],
)
def test_calibration_gate_requires_fourteen_reviewed_reports(ledger, expected):
    report = _valid_report()
    report["review_ledger_summary"] = ledger

    result = evaluate_money_flow_report(report, now=NOW)

    assert _check(result, "calibration_gate").status == expected


def test_calibration_gate_does_not_treat_boolean_flags_as_counts():
    report = _valid_report()
    report["review_ledger_summary"] = {"expansion_eligible": False}

    result = evaluate_money_flow_report(report, now=NOW)

    check = _check(result, "calibration_gate")
    assert check.status == ReliabilityStatus.WARN
    assert "count is not parseable" in check.detail
