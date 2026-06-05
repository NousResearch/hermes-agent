"""Regression guards for the packaged Hermes Desktop bundle config."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_desktop_dist_is_unpacked_for_python_backend():
    package_json = json.loads(
        (REPO_ROOT / "apps" / "desktop" / "package.json").read_text(encoding="utf-8")
    )
    asar_unpack = package_json.get("build", {}).get("asarUnpack", [])
    assert "dist/**" in asar_unpack
