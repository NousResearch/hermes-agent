"""FASE 3 — cost_tracker CLI tests (RED suite)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample JSONL file with known events (enough for worst test)."""
    store = tmp_path / "telemetry.jsonl"
    events = [
        # gpt-5-mini: 5 success (0% failure)
        {
            "timestamp": "2026-04-16T10:00:00+00:00",
            "model": "gpt-5-mini",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "cheap_route",
            "turn_kind": "simple_turn",
            "success": True,
            "error_type": None,
            "latency_ms": 50.0,
            "tokens_in": 100,
            "tokens_out": 100,
            "premium_multiplier": 0.0,
            "premium_units": 0.0,
        },
        {
            "timestamp": "2026-04-16T10:01:00+00:00",
            "model": "gpt-5-mini",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "cheap_route",
            "turn_kind": "simple_turn",
            "success": True,
            "error_type": None,
            "latency_ms": 60.0,
            "tokens_in": 150,
            "tokens_out": 150,
            "premium_multiplier": 0.0,
            "premium_units": 0.0,
        },
        {
            "timestamp": "2026-04-16T10:01:30+00:00",
            "model": "gpt-5-mini",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "cheap_route",
            "turn_kind": "simple_turn",
            "success": True,
            "error_type": None,
            "latency_ms": 55.0,
            "tokens_in": 120,
            "tokens_out": 120,
            "premium_multiplier": 0.0,
            "premium_units": 0.0,
        },
        {
            "timestamp": "2026-04-16T10:01:45+00:00",
            "model": "gpt-5-mini",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "cheap_route",
            "turn_kind": "simple_turn",
            "success": True,
            "error_type": None,
            "latency_ms": 52.0,
            "tokens_in": 110,
            "tokens_out": 110,
            "premium_multiplier": 0.0,
            "premium_units": 0.0,
        },
        {
            "timestamp": "2026-04-16T10:01:50+00:00",
            "model": "gpt-5-mini",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "cheap_route",
            "turn_kind": "simple_turn",
            "success": True,
            "error_type": None,
            "latency_ms": 58.0,
            "tokens_in": 130,
            "tokens_out": 130,
            "premium_multiplier": 0.0,
            "premium_units": 0.0,
        },
        # claude-opus-4.6: 3 success, 2 failures (40% failure)
        {
            "timestamp": "2026-04-16T10:02:00+00:00",
            "model": "claude-opus-4.6",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "primary_default",
            "turn_kind": "complex_task",
            "success": False,
            "error_type": "timeout",
            "latency_ms": 5000.0,
            "tokens_in": 500,
            "tokens_out": 0,
            "premium_multiplier": 5.0,
            "premium_units": 2.5,
        },
        {
            "timestamp": "2026-04-16T10:02:30+00:00",
            "model": "claude-opus-4.6",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "primary_default",
            "turn_kind": "complex_task",
            "success": False,
            "error_type": "timeout",
            "latency_ms": 5000.0,
            "tokens_in": 500,
            "tokens_out": 0,
            "premium_multiplier": 5.0,
            "premium_units": 2.5,
        },
        {
            "timestamp": "2026-04-16T10:03:00+00:00",
            "model": "claude-opus-4.6",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "primary_default",
            "turn_kind": "complex_task",
            "success": True,
            "error_type": None,
            "latency_ms": 800.0,
            "tokens_in": 600,
            "tokens_out": 400,
            "premium_multiplier": 5.0,
            "premium_units": 5.0,
        },
        {
            "timestamp": "2026-04-16T10:03:30+00:00",
            "model": "claude-opus-4.6",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "primary_default",
            "turn_kind": "complex_task",
            "success": True,
            "error_type": None,
            "latency_ms": 750.0,
            "tokens_in": 550,
            "tokens_out": 350,
            "premium_multiplier": 5.0,
            "premium_units": 4.5,
        },
        {
            "timestamp": "2026-04-16T10:03:45+00:00",
            "model": "claude-opus-4.6",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "primary_default",
            "turn_kind": "complex_task",
            "success": True,
            "error_type": None,
            "latency_ms": 820.0,
            "tokens_in": 620,
            "tokens_out": 380,
            "premium_multiplier": 5.0,
            "premium_units": 5.0,
        },
        # claude-sonnet-4.6: 5 success (0% failure)
        {
            "timestamp": "2026-04-16T10:04:00+00:00",
            "model": "claude-sonnet-4.6",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "primary_default",
            "turn_kind": "complex_task",
            "success": True,
            "error_type": None,
            "latency_ms": 300.0,
            "tokens_in": 400,
            "tokens_out": 300,
            "premium_multiplier": 1.0,
            "premium_units": 0.7,
        },
        {
            "timestamp": "2026-04-16T10:04:15+00:00",
            "model": "claude-sonnet-4.6",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "primary_default",
            "turn_kind": "complex_task",
            "success": True,
            "error_type": None,
            "latency_ms": 310.0,
            "tokens_in": 420,
            "tokens_out": 320,
            "premium_multiplier": 1.0,
            "premium_units": 0.74,
        },
        {
            "timestamp": "2026-04-16T10:04:30+00:00",
            "model": "claude-sonnet-4.6",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "primary_default",
            "turn_kind": "complex_task",
            "success": True,
            "error_type": None,
            "latency_ms": 290.0,
            "tokens_in": 380,
            "tokens_out": 280,
            "premium_multiplier": 1.0,
            "premium_units": 0.66,
        },
        {
            "timestamp": "2026-04-16T10:04:45+00:00",
            "model": "claude-sonnet-4.6",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "primary_default",
            "turn_kind": "complex_task",
            "success": True,
            "error_type": None,
            "latency_ms": 305.0,
            "tokens_in": 410,
            "tokens_out": 310,
            "premium_multiplier": 1.0,
            "premium_units": 0.72,
        },
        {
            "timestamp": "2026-04-16T10:05:00+00:00",
            "model": "claude-sonnet-4.6",
            "provider": "copilot",
            "domain": "code",
            "decision_source": "primary_default",
            "turn_kind": "complex_task",
            "success": True,
            "error_type": None,
            "latency_ms": 295.0,
            "tokens_in": 390,
            "tokens_out": 290,
            "premium_multiplier": 1.0,
            "premium_units": 0.68,
        },
    ]
    with open(store, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return store


def test_summary_prints_table(sample_jsonl, tmp_path):
    """subprocess.run con --store, capture stdout, assert que contiene los nombres de modelos y columna "premium_units"."""
    worktree = Path(__file__).parent.parent.parent
    script = worktree / "scripts" / "cost_tracker.py"
    
    result = subprocess.run(
        [sys.executable, str(script), "summary", "--store", str(sample_jsonl)],
        capture_output=True,
        text=True,
        cwd=str(worktree),
    )
    
    stdout = result.stdout
    # Check that model names appear in output
    assert "gpt-5-mini" in stdout
    assert "claude-opus-4.6" in stdout
    assert "claude-sonnet-4.6" in stdout
    # Check that premium_units column header appears
    assert "premium_units" in stdout.lower() or "premium" in stdout.lower()


def test_compare_reports_savings(sample_jsonl, tmp_path):
    """stdout contiene "savings" y un porcentaje."""
    worktree = Path(__file__).parent.parent.parent
    script = worktree / "scripts" / "cost_tracker.py"
    
    result = subprocess.run(
        [sys.executable, str(script), "compare", "--store", str(sample_jsonl)],
        capture_output=True,
        text=True,
        cwd=str(worktree),
    )
    
    stdout = result.stdout.lower()
    assert "savings" in stdout
    # Should contain a percentage
    assert "%" in stdout or "percent" in stdout


def test_worst_lists_failing_model_first(sample_jsonl, tmp_path):
    """stdout primera línea de datos es el peor modelo."""
    worktree = Path(__file__).parent.parent.parent
    script = worktree / "scripts" / "cost_tracker.py"
    
    result = subprocess.run(
        [sys.executable, str(script), "worst", "--store", str(sample_jsonl)],
        capture_output=True,
        text=True,
        cwd=str(worktree),
    )
    
    stdout = result.stdout
    # claude-opus-4.6 has 2 failures out of 5 requests (40% failure rate)
    # Other models have 0% failure rate
    # So claude-opus-4.6 should be first in the worst list
    lines = stdout.strip().split("\n")
    # Find data lines (containing model names)
    data_lines = [l for l in lines if "claude-opus" in l.lower() or "gpt-5-mini" in l.lower() or "claude-sonnet" in l.lower()]
    assert len(data_lines) > 0
    # The first data line should be claude-opus-4.6 (highest failure rate)
    assert "claude-opus" in data_lines[0].lower()
