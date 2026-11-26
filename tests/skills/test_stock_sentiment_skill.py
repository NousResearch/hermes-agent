from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch


SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "finance"
    / "stock-sentiment"
)
SCRIPT_PATH = SKILL_DIR / "scripts" / "sentiment_client.py"
SKILL_MD = SKILL_DIR / "SKILL.md"


def load_module():
    spec = importlib.util.spec_from_file_location("stock_sentiment_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- Pending-quarter selection (cmd_holders) --------------------------------

def _run_holders(mod, quarters):
    """Invoke cmd_holders with a mocked network; return the params passed to the
    holders GET (so we can assert which reportDate was chosen)."""
    calls = []

    def fake_get(path, **params):
        calls.append((path, params))
        if path == "/api/v1/institutional/quarters":
            return quarters
        return {"data": {"holders": []}}

    with patch.object(mod, "get", side_effect=fake_get):
        mod.cmd_holders(types.SimpleNamespace(ticker="nvda"))

    holders_calls = [c for c in calls if "holders" in c[0]]
    assert holders_calls, "cmd_holders never queried the holders endpoint"
    return holders_calls[0]


def test_cmd_holders_skips_pending_quarter():
    mod = load_module()
    quarters = [
        {"reportDate": "2026-06-30", "pending": True},   # still filing
        {"reportDate": "2026-03-31", "pending": False},  # newest settled
    ]
    path, params = _run_holders(mod, quarters)
    assert "NVDA" in path
    assert params["reportDate"] == "2026-03-31"


def test_cmd_holders_uses_latest_when_not_pending():
    mod = load_module()
    quarters = [
        {"reportDate": "2026-03-31", "pending": False},
        {"reportDate": "2025-12-31", "pending": False},
    ]
    _, params = _run_holders(mod, quarters)
    assert params["reportDate"] == "2026-03-31"


def test_cmd_holders_falls_back_when_all_pending():
    mod = load_module()
    quarters = [
        {"reportDate": "2026-06-30", "pending": True},
        {"reportDate": "2026-03-31", "pending": True},
    ]
    _, params = _run_holders(mod, quarters)
    assert params["reportDate"] == "2026-06-30"  # falls back to [0]


def test_cmd_holders_treats_missing_pending_as_settled():
    mod = load_module()
    quarters = [{"reportDate": "2026-03-31"}]  # no pending key => settled
    _, params = _run_holders(mod, quarters)
    assert params["reportDate"] == "2026-03-31"


# --- Envelope handling (rows / shaped) --------------------------------------

def test_rows_unwraps_data_and_passes_flat_through():
    mod = load_module()
    assert mod.rows([1, 2, 3]) == [1, 2, 3]
    assert mod.rows({"data": [1, 2]}) == [1, 2]
    assert mod.rows({"score": 5}) == {"score": 5}


def test_shaped_preserves_preview_flags():
    mod = load_module()
    envelope = {"isPreview": True, "previewReason": "PRO_REQUIRED", "data": [{"a": 1}]}
    shaped = mod.shaped(envelope)
    assert shaped["isPreview"] is True
    assert shaped["previewReason"] == "PRO_REQUIRED"
    assert shaped["data"] == [{"a": 1}]
    # Non-preview wrapped payloads collapse to their rows.
    assert mod.shaped({"data": [{"a": 1}]}) == [{"a": 1}]


def test_sentiment_scalar_reads_nested_value():
    mod = load_module()
    series = [{"metricValue": {"value": {"value": -0.42}}}]
    assert mod.sentiment_scalar(series) == -0.42
    assert mod.sentiment_scalar([]) is None


# --- Frontmatter / script contracts -----------------------------------------

def test_script_key_url_matches_skill_md():
    """The script's onboarding URL must match SKILL.md's, not the stale one."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "/get-api-key" in source
    assert "/settings/developer" not in source


def test_script_sends_api_key_header():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "X-SentiSense-API-Key" in source


def test_client_is_read_only():
    """No mutating verbs: every request is a GET."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "method=" not in source  # urllib defaults to GET; no POST/PUT/DELETE


def test_skill_frontmatter_names_human_contributor_first():
    lines = SKILL_MD.read_text(encoding="utf-8").splitlines()
    author = next((ln for ln in lines if ln.startswith("author:")), "")
    assert author, "SKILL.md is missing an author line"
    # A human contributor handle must lead the org attribution.
    assert author.index("TheSentiTrader") < author.index("SentiSense")
