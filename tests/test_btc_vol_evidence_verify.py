import json
import zipfile

from institutional_btc_vol.evidence_verify import verify_evidence_bundle


def _write_bundle(tmp_path, manifest, files):
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("evidence_manifest.json", json.dumps(manifest))
        zf.writestr("evidence_index.md", "# Evidence\nSCREEN-ONLY · NOT EXECUTABLE\n")
        for name, data in files.items():
            zf.writestr(name, data)
    return bundle


def test_verify_evidence_bundle_confirms_manifest_hashes_and_screen_only_status(tmp_path):
    content = b"SCREEN-ONLY report"
    import hashlib

    manifest = {
        "run_id": "btcvol-test",
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "artifacts": [
            {"label": "report", "path": "ignored/report.md", "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        ],
    }
    bundle = _write_bundle(tmp_path, manifest, {"artifacts/report.md": content})

    result = verify_evidence_bundle(bundle)

    assert result["ok"] is True
    assert result["run_id"] == "btcvol-test"
    assert result["verified_artifacts"] == 1
    assert result["missing_artifacts"] == []
    assert result["hash_mismatches"] == []
    assert result["orphan_members"] == []
    assert result["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"


def test_verify_evidence_bundle_rejects_unsafe_manifest_labels(tmp_path):
    manifest = {
        "run_id": "btcvol-unsafe",
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "artifacts": [{"label": "../escape", "path": "ignored/report.md", "sha256": "abc", "bytes": 1}],
    }
    bundle = _write_bundle(tmp_path, manifest, {})

    result = verify_evidence_bundle(bundle)

    assert result["ok"] is False
    assert any("unsafe artifact label" in error for error in result["errors"])


def test_verify_evidence_bundle_detects_hash_mismatch_and_missing_file(tmp_path):
    manifest = {
        "run_id": "btcvol-bad",
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "artifacts": [
            {"label": "report", "path": "ignored/report.md", "sha256": "bad", "bytes": 4},
            {"label": "dashboard", "path": "ignored/dashboard.html", "sha256": "abc", "bytes": 10},
        ],
    }
    bundle = _write_bundle(tmp_path, manifest, {"artifacts/report.md": b"real"})

    result = verify_evidence_bundle(bundle)

    assert result["ok"] is False
    assert result["verified_artifacts"] == 0
    assert result["missing_artifacts"] == ["dashboard"]
    assert result["hash_mismatches"] == ["report"]


def test_verify_evidence_bundle_rejects_missing_non_executable_label(tmp_path):
    manifest = {"run_id": "btcvol-bad", "artifacts": []}
    bundle = _write_bundle(tmp_path, manifest, {})

    result = verify_evidence_bundle(bundle)

    assert result["ok"] is False
    assert "missing SCREEN-ONLY" in result["errors"][0]



def test_verify_evidence_bundle_rejects_orphan_zip_members(tmp_path):
    content = b"SCREEN-ONLY report"
    import hashlib

    manifest = {
        "run_id": "btcvol-orphan",
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "artifacts": [
            {"label": "report", "path": "ignored/report.md", "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        ],
    }
    bundle = _write_bundle(tmp_path, manifest, {"artifacts/report.md": content, "artifacts/untracked.md": b"surprise"})

    result = verify_evidence_bundle(bundle)

    assert result["ok"] is False
    assert result["orphan_members"] == ["artifacts/untracked.md"]
    assert "orphan ZIP members present" in result["errors"]


def test_verify_evidence_bundle_requires_index_file(tmp_path):
    bundle = tmp_path / "bundle.zip"
    manifest = {"run_id": "btcvol-no-index", "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE", "artifacts": []}
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("evidence_manifest.json", json.dumps(manifest))

    result = verify_evidence_bundle(bundle)

    assert result["ok"] is False
    assert "missing evidence_index.md" in result["errors"]
