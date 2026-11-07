"""The family clock standard must not disappear quietly.

These tests are the in-repo half of the guard: `~/.hermes/guard/agent_clock_guard.py`
catches an install update that removes the code at runtime, and these catch a change
that removes it before it ships.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import agent_clock  # noqa: E402


def test_resolver_module_is_present():
    assert (REPO / "tools" / "agent_clock.py").exists()


def test_clock_subcommand_is_registered():
    """A file that exists but is unregistered is the same failure wearing a hat."""
    main_py = (REPO / "hermes_cli" / "main.py").read_text(encoding="utf-8")
    assert "build_clock_parser" in main_py
    assert (REPO / "hermes_cli" / "subcommands" / "clock.py").exists()


def test_reading_always_names_its_source(monkeypatch):
    monkeypatch.setattr(agent_clock, "_measure_offset", lambda: (0.0, 1_700_000_000.0))
    monkeypatch.setattr(agent_clock, "_read_cache", lambda: None)
    monkeypatch.setattr(agent_clock, "_write_cache", lambda *a, **k: None)
    reading = agent_clock.probe()
    assert reading.source.startswith("authority:")
    assert reading.degraded is False
    assert reading.stamp().count("clock=") == 1


def test_unreachable_authority_is_degraded_and_says_so(monkeypatch):
    monkeypatch.setattr(agent_clock, "_measure_offset", lambda: None)
    monkeypatch.setattr(agent_clock, "_read_cache", lambda: None)
    reading = agent_clock.probe(force=True)
    assert reading.source == agent_clock.SOURCE_FALLBACK
    assert reading.degraded is True
    # A fallback that looks authoritative is the defect this guards against.
    assert "clock_degraded=true" in reading.stamp()


def test_cached_offset_is_marked_cached(monkeypatch):
    import time

    now = time.time()
    monkeypatch.setattr(
        agent_clock,
        "_read_cache",
        lambda: {"offset_seconds": 1.0, "authority": "https://example.test", "checked_at_epoch": now - 5},
    )
    reading = agent_clock.probe()
    assert reading.source == agent_clock.SOURCE_AUTHORITY_CACHED
    assert reading.degraded is False


def test_hermes_clock_json_contract():
    """The command every seat is told to run must answer with the contract's fields."""
    done = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "clock", "--json"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    for field in ("utc", "source", "degraded", "offset_seconds", "stamp"):
        assert field in report
