import json
from pathlib import Path

from scripts.ci.osv_review_status import generate_review_status


def _write_sarif(path: Path, results: list[dict]) -> None:
    path.write_text(json.dumps({"runs": [{"results": results}]}), encoding="utf-8")


def test_empty_valid_sarif_is_clean(tmp_path):
    sarif = tmp_path / "results.sarif"
    _write_sarif(sarif, [])

    assert generate_review_status("success", sarif) == ([], True)


def test_scan_failure_is_not_reported_as_clean(tmp_path):
    status, evidence_ok = generate_review_status("failure", tmp_path / "missing.sarif")

    assert evidence_ok is False
    assert status[0]["results"][0]["kind"] == "action_required"
    assert "failure" in status[0]["results"][0]["summary"]


def test_missing_or_invalid_sarif_is_not_reported_as_clean(tmp_path):
    missing = tmp_path / "missing.sarif"
    status, evidence_ok = generate_review_status("success", missing)
    assert evidence_ok is False
    assert "missing" in status[0]["results"][0]["summary"]

    invalid = tmp_path / "invalid.sarif"
    invalid.write_text("not json", encoding="utf-8")
    status, evidence_ok = generate_review_status("success", invalid)
    assert evidence_ok is False
    assert "could not be parsed" in status[0]["results"][0]["summary"]


def test_findings_are_reported(tmp_path):
    sarif = tmp_path / "results.sarif"
    _write_sarif(
        sarif,
        [
            {
                "ruleId": "GHSA-test",
                "locations": [
                    {"physicalLocation": {"artifactLocation": {"uri": "uv.lock"}}}
                ],
            }
        ],
    )

    status, evidence_ok = generate_review_status("success", sarif)

    assert evidence_ok is True
    result = status[0]["results"][0]
    assert result["kind"] == "warning"
    assert result["summary"] == "1 known vulnerability found in pinned dependencies."
    assert result["detail"] == "- GHSA-test in uv.lock"
