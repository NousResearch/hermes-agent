import json

from scripts.ci.osv_review_status import generate_review_status, main


def test_empty_valid_sarif_is_clean(tmp_path):
    sarif = tmp_path / "results.sarif"
    sarif.write_text('{"runs":[{"results":[]}]}', encoding="utf-8")

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
    for value in ("not json", '{"runs":[],"value":NaN}'):
        invalid.write_text(value, encoding="utf-8")
        status, evidence_ok = generate_review_status("success", invalid)
        assert evidence_ok is False
        assert "could not be parsed" in status[0]["results"][0]["summary"]


def test_malformed_sarif_containers_are_not_reported_as_clean(tmp_path):
    sarif = tmp_path / "results.sarif"
    for data in (
        [],
        {},
        {"runs": {}},
        {"runs": [None]},
        {"runs": [{}]},
        {"runs": [{"results": {}}]},
        {"runs": [{"results": [None]}]},
        {"runs": [{"results": [{"locations": [{}, None]}]}]},
        *(
            {"runs": [{"results": [{"locations": value}]}]}
            for value in (None, {}, "", 0, False)
        ),
    ):
        sarif.write_text(json.dumps(data), encoding="utf-8")
        status, evidence_ok = generate_review_status("success", sarif)
        assert evidence_ok is False
        assert status[0]["results"][0]["kind"] == "action_required"
        assert "malformed" in status[0]["results"][0]["summary"]


def test_findings_are_reported(tmp_path):
    sarif = tmp_path / "results.sarif"
    finding = {
        "ruleId": "GHSA-test",
        "locations": [{"physicalLocation": {"artifactLocation": {"uri": "uv.lock"}}}],
    }
    sarif.write_text(json.dumps({"runs": [{"results": [finding]}]}), encoding="utf-8")

    status, evidence_ok = generate_review_status("success", sarif)

    assert evidence_ok is True
    result = status[0]["results"][0]
    assert result["kind"] == "warning"
    assert result["summary"].startswith("1 known vulnerability")
    assert result["detail"] == "- GHSA-test in uv.lock"


def test_failure_status_is_written_to_job_and_artifact_outputs(tmp_path, monkeypatch):
    job_output = tmp_path / "github-output"
    artifact_output = tmp_path / "review-status.json"
    job_output.write_text("prior=output\n", encoding="utf-8")
    artifact_output.write_text("stale artifact", encoding="utf-8")
    args = ["--scan-result", "success", "--sarif", str(tmp_path / "missing.sarif")]
    args += ["--output", str(job_output), "--output", str(artifact_output)]
    monkeypatch.setattr("sys.argv", ["osv_review_status.py", *args])

    assert main() == 1
    artifact = artifact_output.read_text(encoding="utf-8")
    assert job_output.read_text(encoding="utf-8") == f"prior=output\n{artifact}"
    assert '"kind":"action_required"' in artifact
