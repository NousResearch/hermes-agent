import json
import zipfile

from institutional_btc_vol.evidence_bundle import build_evidence_bundle


def test_build_evidence_bundle_zips_manifest_artifacts_and_writes_summary(tmp_path):
    artifact = tmp_path / "report.md"
    artifact.write_text("SCREEN-ONLY report", encoding="utf-8")
    manifest = tmp_path / "evidence_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": "btcvol-test",
                "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
                "artifacts": [
                    {"label": "report", "path": str(artifact), "sha256": "abc", "bytes": artifact.stat().st_size}
                ],
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "evidence_index.md"
    index.write_text("# Evidence\nSCREEN-ONLY · NOT EXECUTABLE\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"

    summary = build_evidence_bundle(
        bundle,
        run_id="btcvol-test",
        manifest_json=manifest,
        manifest_markdown=index,
    )

    assert summary["run_id"] == "btcvol-test"
    assert summary["bundle_path"] == str(bundle)
    assert summary["bundle_sha256"]
    assert summary["file_count"] == 3
    assert summary["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"
    with zipfile.ZipFile(bundle) as zf:
        names = sorted(zf.namelist())
        assert names == [
            "artifacts/report.md",
            "evidence_index.md",
            "evidence_manifest.json",
        ]
        assert b"SCREEN-ONLY" in zf.read("evidence_index.md")


def test_build_evidence_bundle_rejects_manifest_without_screen_only_status(tmp_path):
    manifest = tmp_path / "evidence_manifest.json"
    manifest.write_text(json.dumps({"run_id": "bad", "artifacts": []}), encoding="utf-8")
    index = tmp_path / "evidence_index.md"
    index.write_text("# Evidence\n", encoding="utf-8")

    try:
        build_evidence_bundle(tmp_path / "bundle.zip", run_id="bad", manifest_json=manifest, manifest_markdown=index)
    except ValueError as exc:
        assert "SCREEN-ONLY" in str(exc)
    else:
        raise AssertionError("Expected missing screen-only status to be rejected")
