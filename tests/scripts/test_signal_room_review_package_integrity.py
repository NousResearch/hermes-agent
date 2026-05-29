from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_review_package_integrity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_review_package_integrity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_package(root: Path) -> None:
    for rel_path in [
        "REVIEW_HUB.html",
        "fee_machine_v2_draft_with_audio.mp4",
        "proof_retention_contact_sheet.svg",
        "EDITORIAL_SCORECARD.json",
        "first_frame_candidates/cold_open_bill.png",
    ]:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset")
    (root / "handoff_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    "REVIEW_HUB.html",
                    "fee_machine_v2_draft_with_audio.mp4",
                    "proof_retention_contact_sheet.svg",
                    "EDITORIAL_SCORECARD.json",
                    "first_frame_candidates/cold_open_bill.png",
                ],
                "primary_review_files": {
                    "review_hub": "REVIEW_HUB.html",
                    "watchable_draft": "fee_machine_v2_draft_with_audio.mp4",
                    "contact_sheet": "proof_retention_contact_sheet.svg",
                    "editorial_scorecard": "EDITORIAL_SCORECARD.json",
                },
            }
        )
    )
    (root / "REVIEW_HUB.html").write_text(
        """
<a href="fee_machine_v2_draft_with_audio.mp4">draft</a>
<a href="proof_retention_contact_sheet.svg">contact</a>
<img src="first_frame_candidates/cold_open_bill.png" alt="cold">
"""
    )


def test_review_package_integrity_passes_existing_relative_links(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)

    result = module.evaluate_package_integrity(tmp_path)

    assert result["passed"] is True
    assert result["errors"] == []
    assert "fee_machine_v2_draft_with_audio.mp4" in result["checked_paths"]


def test_review_package_integrity_fails_missing_and_unsafe_links(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)
    (tmp_path / "proof_retention_contact_sheet.svg").unlink()
    (tmp_path / "REVIEW_HUB.html").write_text(
        """
<a href="../outside.txt">outside</a>
<img src="first_frame_candidates/missing.png" alt="missing">
"""
    )

    result = module.evaluate_package_integrity(tmp_path)

    assert result["passed"] is False
    assert "missing manifest artifact: proof_retention_contact_sheet.svg" in result["errors"]
    assert "unsafe hub path: ../outside.txt" in result["errors"]
    assert "missing hub asset: first_frame_candidates/missing.png" in result["errors"]


def test_review_package_integrity_writes_scorecard(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)
    out = tmp_path / "package_integrity_scorecard.json"

    result = module.write_integrity_scorecard(tmp_path, out)

    assert result["passed"] is True
    assert out.exists()
