"""Tests for the gateway /plan slash handler (GatewaySlashCommandsMixin).

Binds ``_handle_plan_command`` onto a lightweight stub that supplies a real
``PlanManager`` — avoids standing up a full GatewayRunner.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.slash_commands import GatewaySlashCommandsMixin


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import plan_mode

    plan_mode._DB_CACHE.clear()
    yield home
    plan_mode._DB_CACHE.clear()


class _Stub(GatewaySlashCommandsMixin):
    def __init__(self, session_id: str):
        from hermes_cli.plan_mode import PlanManager

        self._sid = session_id
        self._mgr = PlanManager(session_id)

    def _get_plan_manager_for_event(self, event):
        return self._mgr, SimpleNamespace(session_id=self._sid)


def _event(args: str):
    return SimpleNamespace(get_command_args=lambda: args)


def _run(stub, args):
    return asyncio.run(stub._handle_plan_command(_event(args)))


def test_enter_and_status(hermes_home):
    from hermes_cli.plan_mode import PlanManager

    stub = _Stub("p1")
    out = _run(stub, "")
    assert "Plan mode on" in out
    assert PlanManager("p1").is_active()
    status = _run(stub, "status")
    assert "planning" in status.lower()


def test_approve(hermes_home):
    from hermes_cli.plan_mode import PlanManager

    stub = _Stub("p2")
    _run(stub, "")
    out = _run(stub, "approve")
    assert "approved" in out.lower()
    assert not PlanManager("p2").is_active()


def test_reject_with_feedback_stays_planning(hermes_home):
    from hermes_cli.plan_mode import PlanManager

    stub = _Stub("p3")
    _run(stub, "")
    out = _run(stub, "reject please add tests")
    assert "please add tests" in out
    assert PlanManager("p3").is_active()


def test_exit_discards_pending_never_approves(hermes_home):
    from hermes_cli.plan_mode import PlanManager, STATUS_APPROVED, load_plan

    stub = _Stub("p4")
    _run(stub, "")
    # move to pending_approval, then exit — must discard, not approve
    stub._mgr.request_approval()
    out = _run(stub, "exit")
    assert "discarded" in out.lower()
    state = load_plan("p4")
    assert state.status != STATUS_APPROVED
    assert not PlanManager("p4").is_active()


def test_show_without_plan(hermes_home):
    stub = _Stub("p5")
    _run(stub, "")
    out = _run(stub, "show")
    assert "No plan" in out
