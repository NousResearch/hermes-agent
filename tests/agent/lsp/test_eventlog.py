"""Tests for the structured logging dedup model.

The contract: a 1000-write session in one project should emit exactly
ONE INFO line ("active for <root>") at the default INFO threshold.
Steady-state events stay at DEBUG; first-time-seen events surface
once at INFO/WARNING.
"""
from __future__ import annotations

import logging

import pytest

from agent.lsp import eventlog


@pytest.fixture(autouse=True)
def _reset():
    eventlog.reset_announce_caches()
    yield
    eventlog.reset_announce_caches()


@pytest.fixture
def caplog_lsp(caplog):
    caplog.set_level(logging.DEBUG, logger="hermes.lint.lsp")
    return caplog


# ---------------------------------------------------------------------------
# Steady-state silence (DEBUG)
# ---------------------------------------------------------------------------


def test_clean_emits_at_debug(caplog_lsp):
    for _ in range(10):
        eventlog.log_clean("pyright", "/proj/x.py")
    info_records = [r for r in caplog_lsp.records if r.levelno >= logging.INFO]
    debug_records = [r for r in caplog_lsp.records if r.levelno == logging.DEBUG]
    assert info_records == []
    assert len(debug_records) == 10


def test_disabled_emits_at_debug(caplog_lsp):
    eventlog.log_disabled("pyright", "/x.py", "feature off")
    eventlog.log_disabled("pyright", "/x.py", "ext not mapped")
    assert all(r.levelno == logging.DEBUG for r in caplog_lsp.records)


# ---------------------------------------------------------------------------
# State transitions: INFO once, DEBUG thereafter
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# Diagnostics events fire INFO every time
# ---------------------------------------------------------------------------


def test_diagnostics_always_info(caplog_lsp):
    for i in range(5):
        eventlog.log_diagnostics("pyright", f"/x{i}.py", 1)
    info = [r for r in caplog_lsp.records if r.levelno == logging.INFO]
    assert len(info) == 5
    assert all("diags" in r.getMessage() for r in info)


# ---------------------------------------------------------------------------
# Action-required: WARNING once, DEBUG thereafter (or per call for novel events)
# ---------------------------------------------------------------------------












def test_spawn_failed_warns(caplog_lsp):
    eventlog.log_spawn_failed("pyright", "/proj", FileNotFoundError("nope"))
    warns = [r for r in caplog_lsp.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "spawn/initialize failed" in warns[0].getMessage()


# ---------------------------------------------------------------------------
# Format: log lines all carry the lsp[<server_id>] prefix for grep
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Steady-state contract: 1000 clean writes → 1 INFO at most
# ---------------------------------------------------------------------------


def test_thousand_clean_writes_emit_one_info(caplog_lsp):
    """A long session writes lots of files cleanly; agent.log should
    show ONE 'active for' INFO and zero other INFO lines."""
    eventlog.log_active("pyright", "/proj")
    for _ in range(1000):
        eventlog.log_clean("pyright", "/proj/x.py")
    info_records = [r for r in caplog_lsp.records if r.levelno == logging.INFO]
    assert len(info_records) == 1
    assert "active for" in info_records[0].getMessage()


# ---------------------------------------------------------------------------
# Path shortening
# ---------------------------------------------------------------------------




def test_short_path_keeps_absolute_when_outside(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path / "a") if (tmp_path / "a").exists() else None
    monkeypatch.chdir(tmp_path)
    other = "/var/log/foo.txt"
    out = eventlog._short_path(other)
    # Outside cwd: keeps absolute (no leading "../")
    assert out == "/var/log/foo.txt" or not out.startswith("..")


