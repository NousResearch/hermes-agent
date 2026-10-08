"""Kernel ready-owner gate (t_f7992f0f): nothing lands in ``ready`` ownerless.

Regression for the 17-card class (4x phantom ``flow`` assignees, 13x NULL
assignees found stacked in ``ready``, never dispatched): every transition
INTO ``ready`` — creation, manual promotion, dependency auto-promotion,
unblock, review reopen, reassignment — must refuse (or divert to ``todo``)
when the assignee is empty or not a profile on disk, and the refusal must
name the profiles that ARE valid so feeders self-correct.

The human lane ``jacob`` is a legitimate owner pulled by other gateways and
must stay allowed. The gate binds only where a managed ``profiles/`` dir
exists; in hermetic test homes without one, creation of ownerless cards is
still allowed (the dispatcher's ``skipped_nonspawnable`` remains the
backstop), so these tests build a real profiles dir to arm the gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def managed_home(tmp_path, monkeypatch):
    """HERMES_HOME with a managed profiles/ dir holding one profile: ops."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "ops").mkdir(parents=True)
    (home / "profiles" / "ops" / "config.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    # Re-arm the real gate (a conftest autouse fixture may neutralize it).
    monkeypatch.setattr(kb, "ready_owner_refusal", kb.ready_owner_refusal_impl)
    return home, kb


def _connect(managed_home):
    from hermes_cli import kanban_db_connect as kbc

    return kbc.connect_closing()


def test_promote_to_ready_without_assignee_is_refused(managed_home):
    home, kb = managed_home
    with _connect(managed_home) as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="parked", assignee="ops", initial_status="blocked")
        conn.execute("UPDATE tasks SET assignee = NULL WHERE id = ?", (tid,))
        conn.commit()
        ok, err = kb.promote_task(conn, tid, actor="test")
    assert ok is False
    assert "ready requires an owner" in err
    assert "ops" in err  # the refusal names a profile that IS on disk


def test_promote_with_phantom_assignee_is_refused_and_names_valid_profiles(managed_home):
    home, kb = managed_home
    with _connect(managed_home) as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="flow card", assignee="ops", initial_status="blocked")
        conn.execute("UPDATE tasks SET assignee = 'flow' WHERE id = ?", (tid,))
        conn.commit()
        ok, err = kb.promote_task(conn, tid, actor="test")
    assert ok is False
    assert "'flow' is not a spawnable profile" in err
    assert "ops" in err


def test_promote_with_real_profile_is_allowed(managed_home):
    home, kb = managed_home
    with _connect(managed_home) as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="ops card", assignee="ops", initial_status="blocked")
        ok, err = kb.promote_task(conn, tid, actor="test")
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert ok is True, err
    assert row["status"] == "ready"


def test_human_lane_jacob_may_sit_in_ready(managed_home):
    home, kb = managed_home
    with _connect(managed_home) as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="human card", assignee="jacob", initial_status="blocked")
        ok, err = kb.promote_task(conn, tid, actor="test")
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert ok is True, err
    assert row["status"] == "ready"


def test_create_ready_with_phantom_assignee_is_refused(managed_home):
    home, kb = managed_home
    with _connect(managed_home) as conn:
        kb.create_board(slug="default", name="Test")
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(conn, title="feeder card", assignee="flow")
    assert "'flow' is not a spawnable profile" in str(excinfo.value)


def test_create_ready_without_assignee_applies_default_assignee_config(managed_home, monkeypatch):
    home, kb = managed_home
    (home / "config.yaml").write_text("kanban:\n  default_assignee: ops\n", encoding="utf-8")
    with _connect(managed_home) as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="feeder card", assignee=None)
        row = conn.execute("SELECT assignee, status FROM tasks WHERE id = ?", (tid,)).fetchone()
        evs = [
            json.loads(r[1]) for r in conn.execute(
                "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind = 'assigned'",
                (tid,),
            )
        ]
    assert (row["assignee"], row["status"]) == ("ops", "ready")
    assert evs and evs[0]["source"] == "kanban.default_assignee"


def test_recompute_ready_refuses_ownerless_card_and_leaves_board_evidence(managed_home):
    home, kb = managed_home
    with _connect(managed_home) as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="orphan", assignee="ops")
        # Park it ownerless in 'todo' as a dependency-gated card would be.
        conn.execute("UPDATE tasks SET assignee = NULL, status = 'todo' WHERE id = ?", (tid,))
        conn.commit()
        promoted = kb.recompute_ready(conn)
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
        refused = list(conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'ready_owner_refused'",
            (tid,),
        ))
    assert promoted == 0
    assert row["status"] == "todo"  # never parked ownerless in ready
    assert refused and "ready requires an owner" in json.loads(refused[0][0])["reason"]


def test_unblock_ownerless_card_lands_in_todo_with_refusal_event(managed_home):
    home, kb = managed_home
    with _connect(managed_home) as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="unblock me", assignee="ops", initial_status="blocked")
        conn.execute("UPDATE tasks SET assignee = NULL WHERE id = ?", (tid,))
        conn.commit()
        assert kb.unblock_task(conn, tid) is True
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
        refused = list(conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'ready_owner_refused'",
            (tid,),
        ))
    assert row["status"] == "todo"
    assert refused


def test_assign_phantom_profile_to_ready_card_is_refused(managed_home):
    home, kb = managed_home
    with _connect(managed_home) as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="ready card", assignee="ops")
        with pytest.raises(ValueError) as excinfo:
            kb.assign_task(conn, tid, "flow")
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert "'flow' is not a spawnable profile" in str(excinfo.value)
    assert row["assignee"] == "ops"  # unchanged


def test_gate_is_inert_without_a_managed_profiles_dir(tmp_path, monkeypatch):
    """No profiles/ dir -> profile discovery is meaningless; creation of an
    ownerless card is allowed (legacy behaviour; dispatcher backstop holds)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    monkeypatch.setattr(kb, "ready_owner_refusal", kb.ready_owner_refusal_impl)
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="legacy ok", assignee=None)
        row = conn.execute("SELECT status, assignee FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert (row["status"], row["assignee"]) == ("ready", None)
