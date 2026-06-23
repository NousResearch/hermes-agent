"""Tests for the autopilot ADR decision log (agent/autopilot/adr.py).

The ADR is an append-only markdown record of every autopilot decision (the
moments a human would normally be in the loop). It is OFF by default, fails
soft, and never rewrites prior records.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.autopilot import adr


class _Agent:
    """Minimal stand-in carrying the attributes the ADR reads."""

    def __init__(self, enabled=None, path=None, session_id="sess123"):
        if enabled is not None:
            self._autopilot_adr = enabled
        if path is not None:
            self._autopilot_adr_path = str(path)
        self.session_id = session_id


def _clear_env(monkeypatch):
    monkeypatch.delenv("HERMES_AUTOPILOT_ADR", raising=False)
    monkeypatch.delenv("AUTOPILOT_ADR_PATH", raising=False)
    monkeypatch.delenv("HERMES_WORKSPACE", raising=False)


def test_disabled_by_default(monkeypatch):
    _clear_env(monkeypatch)
    assert adr.adr_enabled(_Agent()) is False
    assert adr.adr_enabled(None) is False


def test_enabled_via_agent_attr(monkeypatch):
    _clear_env(monkeypatch)
    assert adr.adr_enabled(_Agent(enabled=True)) is True


def test_enabled_via_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_AUTOPILOT_ADR", "1")
    assert adr.adr_enabled(None) is True
    monkeypatch.setenv("HERMES_AUTOPILOT_ADR", "off")
    assert adr.adr_enabled(None) is False


def test_record_is_noop_when_disabled(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=False, path=target)
    out = adr.record_decision(agent, kind="completion", goal="do X")
    assert out is None
    assert not target.exists()


def test_record_writes_section_when_enabled(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    out = adr.record_decision(
        agent,
        kind="completion",
        goal="fix all lint errors",
        sent_for_verification="GOAL: fix all lint\nRESULT: ran ruff, 0 errors",
        verdict="allow",
        confidence=0.91,
        chosen="stop — goal verified complete",
        rationale="council verdict=allow",
        source="council",
    )
    assert out == target
    body = target.read_text()
    assert "# Autopilot decision log" in body          # header written once
    assert "## " in body and "— completion" in body
    assert "reviewer: council" in body
    assert "verdict: allow (confidence 0.91)" in body
    assert "fix all lint errors" in body


def test_record_appends_not_overwrites(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    adr.record_decision(agent, kind="completion", goal="first goal", source="aux")
    adr.record_decision(agent, kind="continue", goal="second goal", source="council")
    body = target.read_text()
    # Both records present; header appears exactly once.
    assert body.count("# Autopilot decision log") == 1
    assert "first goal" in body
    assert "second goal" in body
    assert body.count("## ") >= 2


def test_record_logs_options_and_choice(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    adr.record_decision(
        agent,
        kind="clarify",
        goal="Which DB driver?",
        options=["sqlite3", "pysqlite3", "apsw"],
        chosen="sqlite3",
        rationale="stdlib, no extra dep",
        source="aux",
    )
    body = target.read_text()
    assert "options considered:" in body
    assert "sqlite3" in body and "apsw" in body
    assert "chosen path: sqlite3" in body


def test_record_logs_gap_and_required_checks(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    adr.record_decision(
        agent,
        kind="continue",
        goal="ship the feature",
        verdict="deny",
        confidence=0.7,
        gap="no tests were run on the new path",
        required_checks="run pytest on the new module; confirm 0 failures",
        source="council",
    )
    body = target.read_text()
    assert "gap found / why not passing: no tests were run" in body
    assert "required to pass: run pytest" in body


def test_record_fails_soft_on_bad_path(monkeypatch):
    _clear_env(monkeypatch)
    # Point the ADR at a path whose parent cannot be created (a file as a dir).
    agent = _Agent(enabled=True, path="/dev/null/cannot/exist/adr.md")
    # Must not raise; returns None on failure.
    out = adr.record_decision(agent, kind="completion", goal="x")
    assert out is None


def test_path_override_via_env(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "custom" / "log.md"
    monkeypatch.setenv("HERMES_AUTOPILOT_ADR", "1")
    monkeypatch.setenv("AUTOPILOT_ADR_PATH", str(target))
    assert adr.adr_path(None) == target
    adr.record_decision(None, kind="completion", goal="env-path goal", source="aux")
    assert target.exists()
    assert "env-path goal" in target.read_text()


def test_default_path_shape(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_WORKSPACE", str(tmp_path))
    p = adr.adr_path(_Agent(session_id="abc"))
    assert p.parent == tmp_path / ".hermes" / "autopilot" / "adr"
    assert p.name.startswith("AUTOPILOT-abc-")
    assert p.suffix == ".md"
