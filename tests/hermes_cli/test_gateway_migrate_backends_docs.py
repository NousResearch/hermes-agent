"""The dashboard guide describes the backend migration this version ships (``--apply`` moves the backend)."""

from __future__ import annotations

from pathlib import Path

_GUIDE = Path(__file__).resolve().parents[2] / "website" / "docs" / "user-guide" / "features" / "web-dashboard.md"


def test_the_dry_run_section_no_longer_says_apply_is_unavailable():
    text = " ".join(_GUIDE.read_text(encoding="utf-8").split())
    assert "not available in this version" not in text
    assert "without `--dry-run` exits 2" not in text
    assert "{#moving-a-dashboard-backend-to-the-gateway}" in text  # the section the dry run points to
