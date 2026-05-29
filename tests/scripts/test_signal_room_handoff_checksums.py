from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_handoff_checksums.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_handoff_checksums", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_package(root: Path) -> Path:
    (root / "a.txt").write_text("alpha")
    (root / "nested").mkdir()
    (root / "nested" / "b.txt").write_text("bravo")
    bundle = root / "handoff.tar.gz"
    bundle.write_bytes(b"bundle")
    (root / "handoff_manifest.json").write_text(
        json.dumps(
            {
                "bundle_path": str(bundle),
                "artifacts": ["a.txt", "nested/b.txt"],
                "objective_evidence": {
                    "choreography_beats_with_motion": 2,
                    "choreography_beats_without_motion": 0,
                    "choreography_beats_with_valid_review_frames": 2,
                    "choreography_beats_without_valid_review_frames": 0,
                },
            }
        )
    )
    return bundle


def test_handoff_checksums_hash_manifest_artifacts_and_bundle(tmp_path: Path) -> None:
    module = load_module()
    bundle = write_package(tmp_path)

    result = module.build_checksum_manifest(tmp_path)

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["bundle"]["path"] == str(bundle)
    assert result["bundle"]["sha256"] == hashlib.sha256(b"bundle").hexdigest()
    assert result["artifacts"]["a.txt"]["sha256"] == hashlib.sha256(b"alpha").hexdigest()
    assert result["artifacts"]["nested/b.txt"]["size_bytes"] == 5
    assert result["objective_evidence"]["choreography_beats_with_motion"] == 2
    assert result["objective_evidence"]["choreography_beats_without_valid_review_frames"] == 0


def test_handoff_checksums_fail_missing_artifact(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)
    (tmp_path / "nested" / "b.txt").unlink()

    result = module.build_checksum_manifest(tmp_path)

    assert result["passed"] is False
    assert "missing artifact: nested/b.txt" in result["errors"]


def test_handoff_checksums_write_output(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)
    out = tmp_path / "handoff_checksums.json"

    result = module.write_checksum_manifest(tmp_path, out)

    assert result["passed"] is True
    assert out.exists()
    assert json.loads(out.read_text())["passed"] is True
