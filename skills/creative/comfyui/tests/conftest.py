"""Pytest configuration for the comfyui skill test suite.

Adds `scripts/` to sys.path so tests can `from _common import ...`, and
provides a few common fixtures.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
WORKFLOWS = ROOT / "workflows"

sys.path.insert(0, str(SCRIPTS))


@pytest.fixture
def sd15_workflow() -> dict:
    return json.loads((WORKFLOWS / "sd15_txt2img.json").read_text())


@pytest.fixture
def flux_workflow() -> dict:
    return json.loads((WORKFLOWS / "flux_dev_txt2img.json").read_text())


@pytest.fixture
def video_workflow() -> dict:
    return json.loads((WORKFLOWS / "wan_video_t2v.json").read_text())


@pytest.fixture
def workflows_dir() -> Path:
    return WORKFLOWS


@pytest.fixture
def scripts_dir() -> Path:
    return SCRIPTS


@pytest.fixture
def cloud_key() -> str | None:
    """Cloud API key if set, otherwise None.

    Tests that need cloud connectivity should skip when this is None.
    """
    return os.environ.get("COMFY_CLOUD_API_KEY")


def pytest_collection_modifyitems(config, items):
    """Auto-skip cloud tests when no API key is set."""
    if os.environ.get("COMFY_CLOUD_API_KEY"):
        return
    skip_cloud = pytest.mark.skip(reason="Set COMFY_CLOUD_API_KEY to run cloud tests")
    for item in items:
        if "cloud" in item.keywords:
            item.add_marker(skip_cloud)
