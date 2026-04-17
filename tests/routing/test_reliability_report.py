"""Tests for scripts/reliability_report.py and dream-scope daily projection."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

WORKTREE = Path(__file__).resolve().parents[2]
SCRIPT = WORKTREE / "scripts" / "reliability_report.py"


# ---------- Wilson CI unit test (imported directly) ----------
def test_wilson_interval_basic():
    sys.path.insert(0, str(WORKTREE))
    from scripts.reliability_report import wilson_interval

    lo, hi = wilson_interval(95, 100)
    assert 0.88 < lo < 0.93
    assert 0.97 < hi <= 1.0

    lo0, hi0 = wilson_interval(0, 0)
    assert lo0 == 0.0 and hi0 == 0.0


# ---------- helpers ----------
def _write_synthetic_jsonl(path: Path, *, total: int = 200) -> Path:
    """Deterministic distribution across last 7 days.

    - 140 gpt-5-mini success, tokens 300/150, 100ms
    -  50 claude-sonnet-4.6 success, tokens 800/400, 600ms
    -   8 claude-opus-4.6 success, tokens 2000/1000, 2500ms
    -   2 claude-opus-4.6 failure TimeoutError, 5000ms
    """
    from agent.routing_telemetry import multiplier_for

    start = datetime.now(timezone.utc) - timedelta(days=7)
    lines = []
    spec = (
        [("gpt-5-mini", "copilot", True, None, 300, 150, 100.0)] * 140
        + [("claude-sonnet-4.6", "copilot", True, None, 800, 400, 600.0)] * 50
        + [("claude-opus-4.7", "copilot", True, None, 2000, 1000, 2500.0)] * 8
        + [("claude-opus-4.7", "copilot", False, "TimeoutError", 0, 0, 5000.0)] * 2
    )
    assert len(spec) == total
    step = timedelta(days=7) / total
    for i, (model, provider, success, err, tin, tout, lat) in enumerate(spec):
        ts = (start + step * i).isoformat()
        mult = multiplier_for(model)
        units = mult * (tin + tout) / 1000.0
        lines.append(json.dumps({
            "timestamp": ts,
            "model": model,
            "provider": provider,
            "domain": "primary",
            "decision_source": "primary",
            "turn_kind": "simple" if model == "gpt-5-mini" else "primary",
            "success": success,
            "error_type": err,
            "latency_ms": lat,
            "tokens_in": tin,
            "tokens_out": tout,
            "premium_multiplier": mult,
            "premium_units": units,
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=WORKTREE,
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------- integration tests ----------
def test_report_from_synthetic_fixture(tmp_path):
    store = _write_synthetic_jsonl(tmp_path / "telemetry.jsonl")
    out = tmp_path / "report.md"

    res = _run_cli("--store", str(store), "--out", str(out), "--synthetic")
    assert res.returncode == 0, res.stderr

    md = out.read_text(encoding="utf-8")
    assert "DATOS SINTÉTICOS" in md
    assert "claude-opus-4.7" in md
    assert "gpt-5-mini" in md
    # Worst should be opus (2 failures out of 10), optimal should be mini
    assert "worst_model" in md
    assert "optimal_model" in md
    assert "`claude-opus-4.7`" in md  # appears as worst
    assert "`gpt-5-mini`" in md  # appears as optimal
    # Dream scope section present
    assert "Dream scope" in md
    assert "AHORRO/día" in md
    # Some savings percentage should be shown and substantial
    import re
    m = re.search(r"AHORRO/día: [\d.]+ unidades \(([\d.]+)%\)", md)
    assert m is not None, f"no savings line found in:\n{md}"
    pct = float(m.group(1))
    assert pct >= 70.0, f"expected ≥70% projected savings, got {pct}"


def test_dream_scope_includes_30d_projection(tmp_path):
    store = _write_synthetic_jsonl(tmp_path / "telemetry.jsonl")
    out = tmp_path / "report.md"
    res = _run_cli("--store", str(store), "--out", str(out))
    assert res.returncode == 0, res.stderr
    md = out.read_text(encoding="utf-8")
    assert "30d conservadora" in md
    assert "30d optimista" in md


def test_report_handles_empty_store_gracefully(tmp_path):
    store = tmp_path / "empty.jsonl"
    store.write_text("", encoding="utf-8")
    res = _run_cli("--store", str(store))
    assert res.returncode == 0
    assert "no events" in (res.stderr + res.stdout).lower()


def test_report_handles_missing_store_gracefully(tmp_path):
    store = tmp_path / "missing.jsonl"
    res = _run_cli("--store", str(store))
    assert res.returncode == 0
    assert "no events" in (res.stderr + res.stdout).lower()


# ---------- dream-scope direct unit test ----------
def test_dream_scope_math(tmp_path):
    sys.path.insert(0, str(WORKTREE))
    from agent.routing_telemetry import load_events
    from scripts.reliability_report import build_dream_scope

    store = _write_synthetic_jsonl(tmp_path / "t.jsonl")
    events = load_events(store=store)
    dream = build_dream_scope(events)

    assert dream["days_span"] > 0
    assert dream["events_per_day"] > 20  # ~200/7
    assert dream["baseline_all_opus_per_day"] > dream["actual_premium_per_day"]
    assert dream["savings_per_day"] > 0
    assert 0 < dream["savings_pct"] <= 1.0
    assert dream["projection_30d_optimistic"] > dream["projection_30d_conservative"]


def test_summarize_by_model_accumulates_premium_units(tmp_path):
    """Regression: total_premium_units per model must sum per-event units."""
    sys.path.insert(0, str(WORKTREE))
    from agent.routing_telemetry import load_events, summarize

    store = _write_synthetic_jsonl(tmp_path / "t.jsonl")
    events = load_events(store=store)
    summary = summarize(events)
    by_model = summary["by_model"]

    # Opus: 8 successful events with tokens 2000+1000=3000, mult=5 → 15 units each → 120
    # Plus 2 failures with 0 tokens → 0 → total still 120
    assert by_model["claude-opus-4.7"]["total_premium_units"] == pytest.approx(120.0, rel=1e-3)
    # Sonnet: 50 events × (800+400) × 1 / 1000 = 60
    assert by_model["claude-sonnet-4.6"]["total_premium_units"] == pytest.approx(60.0, rel=1e-3)
    # Mini: 140 events × 0 multiplier = 0
    assert by_model["gpt-5-mini"]["total_premium_units"] == 0.0
    # Total across models must equal summary-wide total
    per_model_sum = sum(m["total_premium_units"] for m in by_model.values())
    assert per_model_sum == pytest.approx(summary["total_premium_units"], rel=1e-6)
