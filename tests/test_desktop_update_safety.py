"""Regression guards for Desktop self-update safety."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_desktop_git_update_checks_have_timeout_and_cleanup():
    main_cjs = (REPO_ROOT / "apps" / "desktop" / "electron" / "main.cjs").read_text(
        encoding="utf-8"
    )

    assert "GIT_OPERATION_TIMEOUT_MS" in main_cjs
    assert "setTimeout" in main_cjs
    assert "child.kill" in main_cjs
    assert "git operation timed out" in main_cjs
