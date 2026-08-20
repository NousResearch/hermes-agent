"""Tests for session-claim refusal of dispatcher-managed tasks (#83736).

A session-side ``kanban claim`` has no heartbeat: once the claim TTL
expires the dispatcher reclaims the task (it cannot terminate the
session's executor) and spawns a worker into the same workspace - two
concurrent writers, silent file clobbering. The CLI refuses that
combination unless the caller opts in with ``--allow-session``.
Dispatcher-owned workers carry ``HERMES_KANBAN_RUN_ID``, and control-plane
lanes (assignee is not a real Hermes profile) are pulled by terminals by
design - both stay claimable.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    kb.init_db()
    return home


def _create_task(title: str, assignee: str) -> str:
    with kb.connect() as conn:
        return kb.create_task(conn, title=title, assignee=assignee)


class TestSessionClaimGate:
    def test_refuses_dispatcher_managed_task_from_session(self, kanban_home):
        tid = _create_task("dispatcher task", assignee="alice")
        with patch("hermes_cli.profiles.profile_exists", return_value=True):
            out = kc.run_slash(f"claim {tid}")
        assert "dispatcher-managed" in out
        assert "--allow-session" in out
        # Task unchanged - still ready, unclaimed.
        with kb.connect() as conn:
            row = kb.get_task(conn, tid)
        assert row.status == "ready"
        assert row.claim_lock is None

    def test_allow_session_flag_claims_anyway(self, kanban_home):
        tid = _create_task("dispatcher task", assignee="alice")
        with patch("hermes_cli.profiles.profile_exists", return_value=True):
            out = kc.run_slash(f"claim {tid} --allow-session")
        assert "Claimed" in out
        with kb.connect() as conn:
            row = kb.get_task(conn, tid)
        assert row.status == "running"

    def test_control_plane_lane_stays_claimable(self, kanban_home):
        """Assignee that is not a real profile (e.g. orion-cc) is pulled by
        terminals by design and must not be refused."""
        tid = _create_task("lane task", assignee="orion-cc")
        with patch("hermes_cli.profiles.profile_exists", return_value=False):
            out = kc.run_slash(f"claim {tid}")
        assert "Claimed" in out

    def test_dispatcher_worker_env_bypasses_gate(self, kanban_home, monkeypatch):
        """A spawned worker carries HERMES_KANBAN_RUN_ID (heartbeat +
        termination semantics) and may claim."""
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
        tid = _create_task("worker task", assignee="alice")
        with patch("hermes_cli.profiles.profile_exists", return_value=True):
            out = kc.run_slash(f"claim {tid}")
        assert "Claimed" in out

    def test_missing_task_still_reports_no_such_task(self, kanban_home):
        out = kc.run_slash("claim t_missing")
        assert "no such task" in out
