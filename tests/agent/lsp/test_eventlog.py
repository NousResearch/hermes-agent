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








def test_announce_bucket_is_capped(caplog_lsp, monkeypatch):
    """Per-file dedup keys stop at _ANNOUNCE_CAP: the bucket resets (re-firing the first-seen line once)
    rather than holding every file path a long-running process ever touched."""
    monkeypatch.setattr(eventlog, "_ANNOUNCE_CAP", 4)
    for i in range(6):
        eventlog.log_no_project_root("pyright", f"/proj/f{i}.py")
    assert len(eventlog._announced_no_root) <= 4
    eventlog.log_no_project_root("pyright", "/proj/f5.py")  # still deduped after the reset
    info = [r.getMessage() for r in caplog_lsp.records if r.levelno == logging.INFO]
    assert info.count("lsp[pyright] no project root for /proj/f5.py") == 1
