"""Phase 3 frozen-golden parity oracle (criterion 4).

The golden was captured (tests/fixtures/_freeze_spend_baseline.py) from the
CURRENT spend_core.session_cost BEFORE any pricing-code edit, under
SPEND_CODEX_TIER=priority. After spend_core becomes a thin shim over the
unified core path, every row must reproduce the golden within ≤ $0.01/row AND
the 7d total delta must be < $0.01 with 0 rows over tolerance.

[B3]: the bar is ≤$0.01/row (not byte-for-byte) because the corrections path
prices codex via the same verbatim float arithmetic but threads it through
Decimal accumulation elsewhere — economically identical, last-digit safe.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERMES_HOME = os.path.expanduser("~/.hermes")
SCRIPTS = os.path.join(HERMES_HOME, "scripts")
FIX_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")

TOLERANCE = 0.01


def _load(name):
    with open(os.path.join(FIX_DIR, name)) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def frozen():
    return _load("spend_frozen_sessions.json")


@pytest.fixture(scope="module")
def golden():
    return _load("spend_core_golden.json")


@pytest.fixture(scope="module")
def spend_core():
    """Import the LIVE spend_core.py (outside the repo). Skips if unavailable
    (e.g. CI without ~/.hermes/scripts)."""
    if not os.path.exists(os.path.join(SCRIPTS, "spend_core.py")):
        pytest.skip("live spend_core.py not present")
    os.environ.setdefault("SPEND_CODEX_TIER", "priority")
    if SCRIPTS not in sys.path:
        sys.path.insert(0, SCRIPTS)
    import spend_core  # noqa

    return spend_core


def test_spend_core_public_names_preserved(spend_core):
    """Every public name other scripts import MUST survive the shim."""
    for name in (
        "normalize_bedrock",
        "codex_cost",
        "session_cost",
        "build_index",
        "root_of",
        "CODEX_PRICING_VERSION",
        "PRICING_OK",
        "CRON_SESSION_RE",
        "REPO",
        "DB",
    ):
        assert hasattr(spend_core, name), name


def test_frozen_golden_parity_per_row(spend_core, frozen, golden):
    """Each frozen row reproduces the golden within ≤ $0.01 (criterion 4)."""
    gold_by_id = {g["id"]: g for g in golden["golden"]}
    over = []
    for row in frozen["rows"]:
        usd, status = spend_core.session_cost(dict(row))
        g = gold_by_id[row["id"]]
        delta = abs((usd or 0.0) - (g["usd"] or 0.0))
        if delta > TOLERANCE:
            over.append((row["id"], g["usd"], usd, delta, g["status"], status))
    assert not over, "rows over $0.01 tolerance:\n" + "\n".join(
        f"  {r[0]}: golden ${r[1]:.6f} [{r[4]}] vs shim ${r[2]:.6f} [{r[5]}] Δ${r[3]:.6f}"
        for r in over
    )


def test_frozen_golden_7d_total_delta(spend_core, frozen, golden):
    """The summed 7d total delta must be < $0.01 (criterion 4)."""
    gold_total = sum((g["usd"] or 0.0) for g in golden["golden"])
    shim_total = sum(
        (spend_core.session_cost(dict(r))[0] or 0.0) for r in frozen["rows"]
    )
    assert abs(shim_total - gold_total) < TOLERANCE, (
        f"7d total delta ${abs(shim_total - gold_total):.6f} "
        f"(golden ${gold_total:.6f} vs shim ${shim_total:.6f})"
    )


def test_frozen_golden_status_preserved(spend_core, frozen, golden):
    """Status (estimated/included/unknown) must match the golden per row."""
    gold_by_id = {g["id"]: g for g in golden["golden"]}
    mism = []
    for row in frozen["rows"]:
        _usd, status = spend_core.session_cost(dict(row))
        g = gold_by_id[row["id"]]
        if status != g["status"]:
            mism.append((row["id"], g["status"], status))
    assert not mism, "status mismatches:\n" + "\n".join(
        f"  {m[0]}: golden [{m[1]}] vs shim [{m[2]}]" for m in mism
    )


# ──────────────────────────────────────────────────────────────────────────
# [B5] Importer smoke + cron-budget-watchdog convergence: the watchdog now
# routes through spend_core.session_cost (unified path) and inherits the C2
# model-family fallback its old hand-rolled _session_cost lacked.
# ──────────────────────────────────────────────────────────────────────────
import importlib.util  # noqa: E402


def _load_script(filename):
    path = os.path.join(SCRIPTS, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not present")
    os.environ.setdefault("SPEND_CODEX_TIER", "priority")
    if SCRIPTS not in sys.path:
        sys.path.insert(0, SCRIPTS)
    spec = importlib.util.spec_from_file_location(
        filename.replace("-", "_").replace(".py", ""), path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cron_cost_report_imports_spend_core():
    mod = _load_script("cron-cost-report.py")
    assert mod._session_cost.__module__ == "spend_core"
    assert mod._PRICING_OK is True


def test_cron_budget_watchdog_uses_spend_core():
    """The watchdog's _session_cost must BE spend_core.session_cost (no fork)."""
    mod = _load_script("cron-budget-watchdog.py")
    assert mod._session_cost.__module__ == "spend_core"
    assert mod._PRICING_OK is True


def test_cron_budget_watchdog_parity_with_spend_core(spend_core, frozen):
    """The watchdog must price every frozen row identically to spend_core
    (it IS spend_core now — guards against a future re-fork)."""
    wd = _load_script("cron-budget-watchdog.py")
    over = []
    for row in frozen["rows"]:
        a = spend_core.session_cost(dict(row))[0] or 0.0
        b = wd._session_cost(dict(row))[0] or 0.0
        if abs(a - b) > TOLERANCE:
            over.append((row["id"], a, b))
    assert not over, "watchdog/spend_core divergence:\n" + "\n".join(
        f"  {o[0]}: spend_core ${o[1]:.6f} vs watchdog ${o[2]:.6f}" for o in over
    )


def test_cron_budget_watchdog_codex_no_longer_zero():
    """C1 regression guard: a codex gpt-5.5 row must price > $0 in the watchdog
    (the company-OAuth $0 bug surface)."""
    wd = _load_script("cron-budget-watchdog.py")
    row = {
        "id": "cron_abc_20260101_000000",
        "parent_session_id": None,
        "model": "gpt-5.5",
        "billing_provider": "openai-codex",
        "billing_base_url": None,
        "input_tokens": 1_000_000,
        "output_tokens": 200_000,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    usd, status = wd._session_cost(row)
    assert usd > 0.0
    assert status == "estimated"
