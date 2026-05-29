from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_transfer_verify.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_transfer_verify", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_review_package(root: Path) -> tuple[Path, Path]:
    extracted = root / "extracted" / "signal-room-review"
    extracted.mkdir(parents=True)
    (extracted / "REVIEW_HUB.html").write_text("<html>review</html>")
    (extracted / "fee_machine_v2_draft_with_audio.mp4").write_bytes(b"video")
    bundle = root / "signal_room_fee_machine_v2_review_handoff.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(extracted / "REVIEW_HUB.html", arcname="signal-room-review/REVIEW_HUB.html")
        archive.add(
            extracted / "fee_machine_v2_draft_with_audio.mp4",
            arcname="signal-room-review/fee_machine_v2_draft_with_audio.mp4",
        )
    checksum = root / "signal_room_fee_machine_v2_review_handoff.sha256.json"
    checksum.write_text(
        json.dumps(
            {
                "bundle": {
                    "path": str(bundle),
                    "size_bytes": bundle.stat().st_size,
                    "sha256": sha256_bytes(bundle.read_bytes()),
                },
                "artifact_count": 2,
                "objective_evidence": {
                    "choreography_beats_with_motion": 2,
                    "choreography_beats_without_motion": 0,
                    "choreography_beats_with_valid_review_frames": 2,
                    "choreography_beats_without_valid_review_frames": 0,
                },
                "artifacts": {
                    "REVIEW_HUB.html": {
                        "size_bytes": (extracted / "REVIEW_HUB.html").stat().st_size,
                        "sha256": sha256_bytes((extracted / "REVIEW_HUB.html").read_bytes()),
                    },
                    "fee_machine_v2_draft_with_audio.mp4": {
                        "size_bytes": (extracted / "fee_machine_v2_draft_with_audio.mp4").stat().st_size,
                        "sha256": sha256_bytes((extracted / "fee_machine_v2_draft_with_audio.mp4").read_bytes()),
                    },
                },
            }
        )
    )
    return extracted, checksum


def test_transfer_verify_passes_bundle_archive_and_artifacts(tmp_path: Path) -> None:
    module = load_module()
    extracted, checksum = write_review_package(tmp_path)

    result = module.verify_transfer(checksum, package_dir=extracted)

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["bundle"]["passed"] is True
    assert result["archive"]["passed"] is True
    assert result["artifacts"]["passed"] is True
    assert result["artifact_count"] == 2
    assert result["objective_evidence"]["choreography_beats_with_motion"] == 2


def test_transfer_verify_reports_tampered_extracted_artifact(tmp_path: Path) -> None:
    module = load_module()
    extracted, checksum = write_review_package(tmp_path)
    (extracted / "REVIEW_HUB.html").write_text("<html>changed</html>")

    result = module.verify_transfer(checksum, package_dir=extracted)

    assert result["passed"] is False
    assert "artifact hash mismatch: REVIEW_HUB.html" in result["errors"]


def test_transfer_verify_falls_back_to_bundle_next_to_checksum(tmp_path: Path) -> None:
    module = load_module()
    extracted, checksum = write_review_package(tmp_path)
    data = json.loads(checksum.read_text())
    bundle_name = Path(data["bundle"]["path"]).name
    data["bundle"]["path"] = f"/missing/original/location/{bundle_name}"
    checksum.write_text(json.dumps(data))

    result = module.verify_transfer(checksum, package_dir=extracted)

    assert result["passed"] is True
    assert result["bundle"]["path"] == str(tmp_path / bundle_name)
