"""Tests for STRICT-READONLY propagation from the autonomous initiator.

Regression matrix coverage:

S1  initiator component — ``objective_spec['strict_readonly']`` flows
    through ``kb.create_task`` and persists on the task row.
S2  ordinary autonomous writable task NOT made strict purely by origin
    — ``created_by == "autonomous_initiator"`` alone does NOT set
    strict mode; an absent / False ``strict_readonly`` key leaves the
    task writable.
    explicit True propagation
    absent/False propagation
    no origin-based inference

These tests use ONLY stdlib + hermes internals; no live subprocess or
network. ``hermes_cli.kanban_db`` is monkey-patched to an isolated
sqlite3 file under ``tmp_path`` so the real ~/.hermes/kanban.db is
never touched.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


PRIMARY = "/home/jr-ubuntu/.hermes/hermes-agent"
if PRIMARY not in sys.path:
    sys.path.insert(0, PRIMARY)


# ---------------------------------------------------------------------------
# Fixtures (parallel to test_autonomous_initiation.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_kanban_db(tmp_path, monkeypatch):
    """Yield a fresh Kanban sqlite3 file under tmp_path, and patch
    hermes_cli.kanban_db to use only this file."""
    db_file = tmp_path / "test_kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_file))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-home"))
    from hermes_cli import kanban_db as kb

    _original_connect = kb.connect
    conn = _original_connect(db_file)
    conn.close()

    @contextmanager
    def _patched_connect_closing():
        c = _original_connect(db_file)
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(kb, "connect_closing", _patched_connect_closing)
    monkeypatch.setattr(
        kb, "connect",
        lambda board=None: _original_connect(db_file),
    )
    # Reset autonomy module state so cross-test contamination is impossible.
    from agent.autonomy import state as _autonomy_state
    _autonomy_state.reset()
    _autonomy_state.enable(policy_version="TEST_V2_STRICT")
    yield db_file
    _autonomy_state.reset()


# ---------------------------------------------------------------------------
# S1 — initiator component (explicit True propagation)
# ---------------------------------------------------------------------------


def test_strict_readonly_true_propagates_from_objective_spec(isolated_kanban_db):
    """S1: when ``objective_spec['strict_readonly']=True``, the resulting
    task row carries ``strict_readonly=1`` and ``kb.get_task(...).strict_readonly``
    is True."""
    from agent.autonomy.initiator import attempt_autonomous_initiation

    spec = {
        "objective_id": "obj_strict_yes",
        "trigger_id": "trig_strict_yes",
        "policy_version": "TEST_V2_STRICT",
        "risk_class": "CLASS_A_AUTONOMOUS_SAFE",
        "title": "Strict autonomous test",
        "body": "Strict body",
        "strict_readonly": True,
    }
    result = attempt_autonomous_initiation(spec)
    assert result.decision == "admit"
    assert result.task_id is not None

    from hermes_cli import kanban_db as kb
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, result.task_id)
    assert task is not None
    assert task.strict_readonly is True
    assert task.created_by == "autonomous_initiator"


# ---------------------------------------------------------------------------
# S2 — absent / False propagation; no origin-based inference
# ---------------------------------------------------------------------------


def test_strict_readonly_absent_yields_writable_task(isolated_kanban_db):
    """S2: ``objective_spec`` without a ``strict_readonly`` key yields a
    writable task (``strict_readonly=False``). Provenance
    ``created_by == "autonomous_initiator"`` alone is NOT enough to
    make the task strict."""
    from agent.autonomy.initiator import attempt_autonomous_initiation

    spec = {
        "objective_id": "obj_strict_absent",
        "trigger_id": "trig_strict_absent",
        "policy_version": "TEST_V2_STRICT",
        "risk_class": "CLASS_A_AUTONOMOUS_SAFE",
        "title": "Ordinary autonomous test",
        "body": "Ordinary body",
    }
    assert "strict_readonly" not in spec  # precondition

    result = attempt_autonomous_initiation(spec)
    assert result.decision == "admit"
    assert result.task_id is not None

    from hermes_cli import kanban_db as kb
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, result.task_id)
    assert task is not None
    # Capability is NOT inferred from created_by.
    assert task.strict_readonly is False
    assert task.created_by == "autonomous_initiator"


def test_strict_readonly_explicit_false_propagates(isolated_kanban_db):
    """S2 (explicit False): ``objective_spec['strict_readonly']=False``
    yields a writable task."""
    from agent.autonomy.initiator import attempt_autonomous_initiation

    spec = {
        "objective_id": "obj_strict_false",
        "trigger_id": "trig_strict_false",
        "policy_version": "TEST_V2_STRICT",
        "risk_class": "CLASS_A_AUTONOMOUS_SAFE",
        "title": "Explicit False",
        "body": "Body",
        "strict_readonly": False,
    }
    result = attempt_autonomous_initiation(spec)
    assert result.decision == "admit"
    assert result.task_id is not None

    from hermes_cli import kanban_db as kb
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, result.task_id)
    assert task is not None
    assert task.strict_readonly is False


# ---------------------------------------------------------------------------
# S1 + S2 (negative) — non-strict autonomous writable tasks remain possible
# ---------------------------------------------------------------------------


def test_writable_autonomous_task_unaffected_by_v2(isolated_kanban_db):
    """S2 + non-regression: an ordinary autonomous task with no
    ``strict_readonly`` opt-in produces a task whose dispatcher env
    WILL NOT contain HERMES_KANBAN_STRICT_READONLY=1 (capability is
    explicit per objective)."""
    from agent.autonomy.initiator import attempt_autonomous_initiation

    spec = {
        "objective_id": "obj_writable_check",
        "trigger_id": "trig_writable_check",
        "policy_version": "TEST_V2_STRICT",
        "risk_class": "CLASS_A_AUTONOMOUS_SAFE",
        "title": "Writable autonomous",
        "body": "Body",
    }
    result = attempt_autonomous_initiation(spec)
    assert result.decision == "admit"
    task_id = result.task_id

    from hermes_cli import kanban_db as kb
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    # Dispatch env for an ordinary autonomous writable task is unchanged:
    # strict_readonly=False => env var MUST NOT be set.
    assert task.strict_readonly is False
    # Indirect check: the dispatcher propagation in _default_spawn only
    # sets the env var when task.strict_readonly is truthy. The
    # underlying boolean field is the single source of truth.
