"""Tests for kanban.assignee_enforcement.

A task assignee that does not correspond to an existing profile is accepted at
write time but has no profiles/<name>/ directory, so it silently stalls in
'ready' at dispatch. WARN mode (the shipped default) canonicalizes + logs so the
stall becomes visible and must NEVER break task creation; REJECT mode is opt-in
and raises at write.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Fresh HERMES_HOME + clean kanban DB (mirrors test_kanban_default_assignee)."""
    test_home = tempfile.mkdtemp(prefix="kanban_assignee_enforcement_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db, test_home


# --- pure decision: (canon, mode, profile_known) -> (should_raise, message) ---

def test_enforcement_off_never_flags(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    assert kb._assignee_enforcement_action("the reviewer", "off", False) == (False, None)


def test_enforcement_warn_flags_unknown_profile_without_raising(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    should_raise, msg = kb._assignee_enforcement_action("the reviewer", "warn", False)
    assert should_raise is False
    assert msg and "the reviewer" in msg


def test_enforcement_warn_silent_for_known_profile(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    assert kb._assignee_enforcement_action("builder", "warn", True) == (False, None)


def test_enforcement_reject_raises_for_unknown_profile(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    should_raise, msg = kb._assignee_enforcement_action("reviewer-typo", "reject", False)
    assert should_raise is True
    assert msg


def test_enforcement_reject_silent_for_known_profile(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    assert kb._assignee_enforcement_action("ops", "reject", True) == (False, None)


def test_enforcement_none_assignee_never_flags(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    assert kb._assignee_enforcement_action(None, "warn", False) == (False, None)


# --- integration: WARN must not break creation; REJECT is opt-in ---

def test_warn_mode_keeps_unknown_assignee_and_warns(isolated_kanban_home, caplog, monkeypatch):
    kb, _ = isolated_kanban_home
    monkeypatch.setattr(kb, "_assignee_enforcement_mode", lambda: "warn")
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        with caplog.at_level(logging.WARNING):
            task_id = kb.create_task(conn, title="t1", assignee="The Reviewer")
        row = conn.execute("SELECT assignee FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert task_id  # WARN must NOT raise / break creation
    assert row["assignee"] == "the reviewer"  # canonicalized, not rejected
    assert any("the reviewer" in r.message.lower() for r in caplog.records)


def test_warn_mode_silent_for_known_profile(isolated_kanban_home, caplog, monkeypatch):
    kb, home = isolated_kanban_home
    os.makedirs(os.path.join(home, "profiles", "builder"), exist_ok=True)
    monkeypatch.setattr(kb, "_assignee_enforcement_mode", lambda: "warn")
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        with caplog.at_level(logging.WARNING):
            task_id = kb.create_task(conn, title="t1", assignee="Builder")
        row = conn.execute("SELECT assignee FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert row["assignee"] == "builder"
    assert not any("not an existing profile" in r.message.lower() for r in caplog.records)


def test_reject_mode_raises_for_unknown_assignee(isolated_kanban_home, monkeypatch):
    kb, _ = isolated_kanban_home
    monkeypatch.setattr(kb, "_assignee_enforcement_mode", lambda: "reject")
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        with pytest.raises(ValueError):
            kb.create_task(conn, title="t1", assignee="The Reviewer")


# --- created_by canonicalization (consistent attribution, no enforcement) ---

def test_canonical_actor_lowercases_and_handles_empty(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    assert kb._canonical_actor("XO") == "xo"
    assert kb._canonical_actor("Orchestrator") == "orchestrator"
    assert kb._canonical_actor("  worker ") == "worker"
    assert kb._canonical_actor(None) is None
    assert kb._canonical_actor("   ") is None


def test_created_by_is_canonicalized_on_create(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="t", assignee="default", created_by="XO")
        row = conn.execute("SELECT created_by FROM tasks WHERE id=?", (tid,)).fetchone()
    assert row["created_by"] == "xo"  # 'XO' canonicalized, not stored mixed-case


def test_created_by_sentinel_and_none_preserved(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        t1 = kb.create_task(conn, title="t1", assignee="default", created_by="worker")
        t2 = kb.create_task(conn, title="t2", assignee="default", created_by=None)
        r1 = conn.execute("SELECT created_by FROM tasks WHERE id=?", (t1,)).fetchone()
        r2 = conn.execute("SELECT created_by FROM tasks WHERE id=?", (t2,)).fetchone()
    assert r1["created_by"] == "worker"
    assert r2["created_by"] is None
