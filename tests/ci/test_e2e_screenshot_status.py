"""Tests for scripts/ci/e2e_screenshot_status.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "e2e_screenshot_status.py"
_spec = importlib.util.spec_from_file_location("e2e_screenshot_status", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load e2e_screenshot_status.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_status_selects_only_new_explicit_screenshots_and_all_diffs(tmp_path):
    for name in (
        "explicit-proof.png",
        "test-finished-1.png",
        "visual-actual.png",
        "visual-expected.png",
        "visual-diff.png",
    ):
        (tmp_path / name).write_bytes(b"png")

    base_manifest = tmp_path / "main-manifest.json"
    base_manifest.write_text('{"screenshot_names":["already-on-main.png"]}', encoding="utf-8")
    (tmp_path / "already-on-main.png").write_bytes(b"png")

    selection = _mod.select_evidence(tmp_path, base_manifest)
    status = _mod.build_status(selection, "https://github.test/artifacts/1")

    result = status[0]["results"][0]
    assert result["kind"] == "info"
    assert result["summary"] == "1 new screenshot vs main; 1 visual diff."
    assert _mod.EVIDENCE_START in result["detail"]
    assert "already-on-main.png" not in result["detail"]
    assert result["link"] == "https://github.test/artifacts/1"


def test_cli_output_ends_with_newline_for_github_output_delimiter(tmp_path, monkeypatch):
    output = tmp_path / "review-status.json"
    manifest = tmp_path / "main-manifest.json"
    evidence_dir = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", [
        "e2e_screenshot_status.py",
        "--results-dir", str(tmp_path),
        "--manifest-output", str(manifest),
        "--evidence-dir", str(evidence_dir),
        "--output", str(output),
    ])

    assert _mod.main() == 0
    assert output.read_text(encoding="utf-8") == "[]\n"
    assert manifest.read_text(encoding="utf-8") == '{"screenshot_names": [], "version": 1}\n'
    assert evidence_dir.joinpath("e2e-evidence.json").is_file()