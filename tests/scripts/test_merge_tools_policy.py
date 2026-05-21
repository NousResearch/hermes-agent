"""Tests for scripts/merge_tools/upstream_merge_policy.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGE_TOOLS = REPO_ROOT / "scripts" / "merge_tools"
if str(MERGE_TOOLS) not in sys.path:
    sys.path.insert(0, str(MERGE_TOOLS))

from upstream_merge_policy import classify_path_with_context, load_strategy  # noqa: E402


@pytest.fixture
def strategy():
    path = MERGE_TOOLS / "hermes-merge-conflict-strategies.json"
    return load_strategy(path)


def test_brain_preserved(strategy):
    item = classify_path_with_context("brain/SOUL.md", strategy, touched_custom=True)
    assert item.action == "preserve_custom"


def test_uv_lock_upstream(strategy):
    item = classify_path_with_context("uv.lock", strategy, touched_upstream=True)
    assert item.action == "upstream"


def test_gateway_overlap_overlay(strategy):
    item = classify_path_with_context(
        "gateway/run.py",
        strategy,
        touched_upstream=True,
        touched_custom=True,
    )
    assert item.action == "official_with_overlay"


def test_upstream_only_defaults_upstream(strategy):
    item = classify_path_with_context(
        "agent/context_compressor.py",
        strategy,
        touched_upstream=True,
        touched_custom=False,
    )
    assert item.action == "upstream"


def test_removed_gateway_restart_test_follows_upstream(strategy):
    item = classify_path_with_context(
        "tests/hermes_cli/test_update_gateway_restart.py",
        strategy,
        touched_upstream=True,
        touched_custom=True,
    )
    assert item.action == "upstream"


def test_strategy_json_valid():
    path = MERGE_TOOLS / "hermes-merge-conflict-strategies.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "rules" in payload
    assert payload["default_action"] == "manual_api_followup"
