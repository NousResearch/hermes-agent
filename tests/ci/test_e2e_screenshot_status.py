"""Tests for scripts/ci/e2e_screenshot_status.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "e2e_screenshot_status.py"
_spec = importlib.util.spec_from_file_location("e2e_screenshot_status", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load e2e_screenshot_status.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_status_includes_only_explicit_screenshots_and_counts_diffs(tmp_path):
    for name in (
        "explicit-proof.png",
        "test-finished-1.png",
        "visual-actual.png",
        "visual-expected.png",
        "visual-diff.png",
    ):
        (tmp_path / name).write_bytes(b"png")

    status = _mod.build_status(tmp_path, "https://github.test/artifacts/1")

    result = status[0]["results"][0]
    assert result["kind"] == "info"
    assert result["summary"] == "1 screenshot captured; 1 visual diff."
    assert "<details>" in result["detail"]
    assert "explicit-proof.png" in result["detail"]
    assert "test-finished-1.png" not in result["detail"]
    assert "visual-actual.png" not in result["detail"]
    assert "https://github.test/artifacts/1" in result["detail"]