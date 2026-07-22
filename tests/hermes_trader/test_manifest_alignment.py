"""Contract tests between catalog manifest and hermes_trader tool policy."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_trader.tools import LIVE_WRITE_TOOLS, PAPER_MODE_READ_TOOLS


def test_defi_trading_manifest_default_enabled_matches_paper_tools():
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "optional-mcps"
        / "defi-trading"
        / "manifest.yaml"
    )
    if not manifest_path.is_file():
        pytest.skip("defi-trading manifest not present")

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    default_enabled = set(data.get("tools", {}).get("default_enabled") or [])

    assert default_enabled == PAPER_MODE_READ_TOOLS
    assert not (default_enabled & LIVE_WRITE_TOOLS)