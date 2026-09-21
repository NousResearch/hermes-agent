"""Execution contract for the deterministic long-form review-bundle wrapper."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


WRAPPER = Path(__file__).resolve().parents[2] / "scripts" / "blog-review-bundle.sh"


def test_wrapper_fails_when_declared_builder_is_missing(tmp_path):
    env = os.environ.copy()
    env["REVIEW_BUNDLE_SCRIPT"] = str(tmp_path / "missing.py")
    result = subprocess.run([str(WRAPPER)], text=True, capture_output=True, env=env)

    assert result.returncode != 0
    assert "review bundle builder missing" in result.stderr
