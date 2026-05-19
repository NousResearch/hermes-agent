import json

from institutional_btc_vol.evidence_manifest import build_evidence_manifest, write_evidence_manifest


def test_build_evidence_manifest_hashes_files_and_labels_screen_only(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("SCREEN-ONLY report\n", encoding="utf-8")
    dashboard = tmp_path / "dashboard.html"
    dashboard.write_text("<html>SCREEN-ONLY</html>\n", encoding="utf-8")
    missing = tmp_path / "missing.csv"

    manifest = build_evidence_manifest(
        run_id="btcvol-test",
        as_of_cst="2026-05-15 11:00:00 CDT",
        artifacts={"report": report, "dashboard": dashboard, "missing": missing},
    )

    assert manifest["run_id"] == "btcvol-test"
    assert manifest["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"
    assert manifest["publishability"] == "internal-only until quote-verified and counsel-approved"
    assert manifest["artifact_count"] == 2
    assert manifest["missing_artifacts"] == ["missing"]
    assert manifest["artifacts"][0]["label"] == "dashboard"
    assert manifest["artifacts"][0]["sha256"]
    assert manifest["artifacts"][0]["bytes"] > 0
    assert manifest["artifacts"][1]["label"] == "report"


def test_write_evidence_manifest_outputs_json_and_markdown_index(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("report", encoding="utf-8")
    output_json = tmp_path / "evidence_manifest.json"
    output_md = tmp_path / "evidence_index.md"

    manifest = build_evidence_manifest(
        run_id="btcvol-test",
        as_of_cst="2026-05-15 11:00:00 CDT",
        artifacts={"report": report},
    )

    written = write_evidence_manifest(output_json, output_md, manifest)

    assert written == {"json": output_json, "markdown": output_md}
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "btcvol-test"
    markdown = output_md.read_text(encoding="utf-8")
    assert "# BTC Vol Desk Evidence Manifest" in markdown
    assert "SCREEN-ONLY · NOT EXECUTABLE" in markdown
    assert "report" in markdown
    assert "sha256" in markdown
