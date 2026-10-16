"""Tests for the session-claim gate on dispatcher-managed tasks (#83736).

A session-side ``kanban claim`` has no heartbeat: once the claim TTL expires
the dispatcher releases the stale lease (it cannot terminate the session's
executor) and respawns the card - two concurrent writers on one workspace,
silent file clobbering. The CLI refuses that combination unless the caller
opts in with ``--allow-session``.

Dispatcher-owned workers carry ``HERMES_KANBAN_RUN_ID`` and control-plane
lanes (assignee is not a real profile) are pulled by terminals by design -
both stay claimable. An unassigned card is dispatcher-managed only when this
home routes it via ``kanban.default_assignee`` (#27145); with no configured
default the dispatcher skips it indefinitely, so it stays claimable too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    kb.init_db()
    return home


def _create_task(title: str, assignee: str | None = None) -> str:
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title=title, assignee=assignee)


def _patch_profile(monkeypatch, *, exists: bool):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: exists)


class TestSessionClaimGate:
    def test_refuses_assigned_dispatcher_managed_task(self, kanban_home, monkeypatch):
        _patch_profile(monkeypatch, exists=True)
        tid = _create_task("dispatcher task", assignee="alice")
        out = kc.run_slash(f"claim {tid}")
        assert "dispatcher-managed" in out
        assert "--allow-session" in out
        with kbc.connect_closing() as conn:
            row = kb.get_task(conn, tid)
        assert row.status == "ready"
        assert row.claim_lock is None

    def test_allow_session_flag_claims_anyway(self, kanban_home, monkeypatch):
        _patch_profile(monkeypatch, exists=True)
        tid = _create_task("dispatcher task", assignee="alice")
        out = kc.run_slash(f"claim {tid} --allow-session")
        assert "Claimed" in out
        with kbc.connect_closing() as conn:
            row = kb.get_task(conn, tid)
        assert row.status == "running"

    def test_control_plane_lane_stays_claimable(self, kanban_home, monkeypatch):
        """Assignee that is not a real profile (e.g. orion-cc) is pulled by
        terminals by design and must not be refused."""
        _patch_profile(monkeypatch, exists=False)
        tid = _create_task("lane task", assignee="orion-cc")
        out = kc.run_slash(f"claim {tid}")
        assert "Claimed" in out

    def test_dispatcher_worker_env_bypasses_gate(self, kanban_home, monkeypatch):
        """A spawned worker carries HERMES_KANBAN_RUN_ID (heartbeat +
        termination semantics) and may claim."""
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
        _patch_profile(monkeypatch, exists=True)
        tid = _create_task("worker task", assignee="alice")
        out = kc.run_slash(f"claim {tid}")
        assert "Claimed" in out

    def test_missing_task_still_reports_no_such_task(self, kanban_home):
        out = kc.run_slash("claim t_missing")
        assert "no such task" in out


class TestUnassignedDefaultAssigneeGate:
    def test_refuses_unassigned_task_routed_by_default_assignee(self, kanban_home):
        # 'default' is always a live profile, so this case runs without the
        # profile_exists stub and exercises the real resolution path.
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  default_assignee: default\n", encoding="utf-8",
        )
        tid = _create_task("unassigned task")
        out = kc.run_slash(f"claim {tid}")
        assert "dispatcher-managed" in out
        assert "kanban.default_assignee" in out
        assert "--allow-session" in out
        with kbc.connect_closing() as conn:
            row = kb.get_task(conn, tid)
        assert row.status == "ready"
        assert row.claim_lock is None

    def test_unassigned_task_claimable_without_default_assignee(self, kanban_home, monkeypatch):
        _patch_profile(monkeypatch, exists=True)
        tid = _create_task("unassigned task")
        out = kc.run_slash(f"claim {tid}")
        assert "Claimed" in out

    def test_unassigned_task_claimable_when_default_profile_missing(self, kanban_home, monkeypatch):
        """A default profile that does not exist never gets auto-assigned by
        the dispatcher, so the card stays a terminal-pull lane."""
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  default_assignee: ghost\n", encoding="utf-8",
        )
        _patch_profile(monkeypatch, exists=False)
        tid = _create_task("unassigned task")
        out = kc.run_slash(f"claim {tid}")
        assert "Claimed" in out

    def test_unassigned_task_claimable_when_default_outside_dispatch_allowlist(self, kanban_home, monkeypatch):
        """A default this home's dispatcher may not claim (kanban.dispatch_profiles
        excludes it) never gets auto-assigned either, so the card stays a
        terminal-pull lane - the guard resolves the default through the same
        allowlist-gated predicate the dispatcher uses (#110995)."""
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  dispatch_profiles:\n    - sage\n  default_assignee: alice\n",
            encoding="utf-8",
        )
        _patch_profile(monkeypatch, exists=True)
        tid = _create_task("unassigned task")
        out = kc.run_slash(f"claim {tid}")
        assert "Claimed" in out

    def test_allow_session_overrides_default_assignee_gate(self, kanban_home, monkeypatch):
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  default_assignee: alice\n", encoding="utf-8",
        )
        _patch_profile(monkeypatch, exists=True)
        tid = _create_task("unassigned task")
        out = kc.run_slash(f"claim {tid} --allow-session")
        assert "Claimed" in out
