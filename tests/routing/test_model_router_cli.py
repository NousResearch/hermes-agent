"""FASE 3 — model_router CLI tests (RED suite)."""
import subprocess
import sys
from pathlib import Path

import pytest


def test_router_simple_returns_mini():
    """subprocess "di hola" → stdout contiene "gpt-5-mini" y "simple_turn"."""
    worktree = Path(__file__).parent.parent.parent
    script = worktree / "scripts" / "model_router.py"
    
    result = subprocess.run(
        [sys.executable, str(script), "di hola"],
        capture_output=True,
        text=True,
        cwd=str(worktree),
    )
    
    stdout = result.stdout.lower()
    assert "gpt-5-mini" in stdout
    assert "simple_turn" in stdout


def test_router_opus_keyword_returns_opus():
    """ "[OPUS] refactor" → contiene "claude-opus" y "opus_keyword"."""
    worktree = Path(__file__).parent.parent.parent
    script = worktree / "scripts" / "model_router.py"
    
    result = subprocess.run(
        [sys.executable, str(script), "[OPUS] refactor this module"],
        capture_output=True,
        text=True,
        cwd=str(worktree),
    )
    
    stdout = result.stdout.lower()
    assert "claude-opus" in stdout
    assert "opus_keyword" in stdout


def test_router_continuation_returns_primary():
    """ "continúa" → contiene "primary_default" o "continuation"."""
    worktree = Path(__file__).parent.parent.parent
    script = worktree / "scripts" / "model_router.py"
    
    result = subprocess.run(
        [sys.executable, str(script), "continúa con lo anterior"],
        capture_output=True,
        text=True,
        cwd=str(worktree),
    )
    
    stdout = result.stdout.lower()
    # Should indicate continuation or primary_default
    assert "primary_default" in stdout or "continuation" in stdout


def test_router_complex_returns_primary():
    """ "debug this traceback and refactor the module extensively" → claude-opus (no mini)."""
    worktree = Path(__file__).parent.parent.parent
    script = worktree / "scripts" / "model_router.py"
    
    result = subprocess.run(
        [sys.executable, str(script), "debug this traceback and refactor the module extensively"],
        capture_output=True,
        text=True,
        cwd=str(worktree),
    )
    
    stdout = result.stdout.lower()
    # Complex task should route to opus, not mini
    assert "claude-opus" in stdout
    assert "gpt-5-mini" not in stdout
