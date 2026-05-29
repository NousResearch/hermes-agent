from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_transfer_readme.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_transfer_readme", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_inputs(root: Path) -> Path:
    checksum = root / "handoff.sha256.json"
    checksum.write_text(
        json.dumps(
            {
                "bundle": {
                    "path": str(root / "signal_room_fee_machine_v2_review_handoff.tar.gz"),
                    "size_bytes": 1234,
                    "sha256": "abc123",
                },
                "artifact_count": 51,
            }
        )
    )
    (root / "handoff_manifest.json").write_text(
        json.dumps(
            {
                "primary_review_files": {
                    "review_hub": "REVIEW_HUB.html",
                    "watchable_draft": "fee_machine_v2_draft_with_audio.mp4",
                    "editorial_scorecard": "EDITORIAL_SCORECARD.json",
                    "pose_export_intake": "pose_export_intake/README.md",
                }
            }
        )
    )
    (root / "review_package_index.json").write_text(
        json.dumps({"blockers": ["Blender or Moho is required for local character pose rendering"]})
    )
    return checksum


def test_transfer_readme_renders_bundle_hash_and_review_order(tmp_path: Path) -> None:
    module = load_module()
    checksum = write_inputs(tmp_path)

    markdown = module.render_transfer_readme(tmp_path, checksum)

    assert "# Signal Room Review Handoff Transfer" in markdown
    assert "signal_room_fee_machine_v2_review_handoff.tar.gz" in markdown
    assert "abc123" in markdown
    assert "REVIEW_HUB.html" in markdown
    assert "EDITORIAL_SCORECARD.json" in markdown
    assert "signal_room_transfer_verify.py" in markdown
    assert "sha256sum" in markdown
    assert "Blender or Moho is required" in markdown


def test_transfer_readme_writes_output(tmp_path: Path) -> None:
    module = load_module()
    checksum = write_inputs(tmp_path)
    out = tmp_path / "TRANSFER_README.md"

    result = module.write_transfer_readme(tmp_path, checksum, out)

    assert result["passed"] is True
    assert result["output"] == str(out)
    assert "Signal Room Review Handoff Transfer" in out.read_text()
