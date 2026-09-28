"""Regression tests: goal-mode handoff must honor workspace ground truth.

Background (2026-09-27 money-pit finding): a goal-mode worker produced a
test-detection cheat (per-caller behavior via call counting), disclosed it
in its completion summary, and the prompt-only judge passed the handoff
anyway — because the judge sees only goal text + the worker's own summary.

The fix adds a deterministic workspace probe at the handoff gate: a
``dir:``-workspace task whose pytest suite is actually RED is rejected with
``continue`` regardless of the judge verdict; missing signal (no dir
workspace, no tests collected, probe crash) stays fail-open.
"""
from __future__ import annotations

import types

import pytest

from hermes_cli.kanban import _goal_mode_handoff_rejection, _workspace_probe_result


def _task(ws_kind=None, ws_path=None):
    return types.SimpleNamespace(
        id="t_probe", goal_mode=True, title="Make pytest pass",
        body="make pytest pass", workspace_kind=ws_kind, workspace_path=ws_path,
    )


@pytest.fixture()
def judge_says_done(monkeypatch):
    """Judge available and happy: without the probe, the handoff passes."""
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda name: (object(), "judge-model"),
    )
    monkeypatch.setattr(
        "hermes_cli.goals.judge_goal",
        lambda **kw: ("done", "", False, None, False),
    )


def test_red_suite_rejects_even_when_judge_says_done(tmp_path, monkeypatch, judge_says_done):
    """The money-pit scenario: judge approves, ground truth is red -> reject."""
    (tmp_path / "test_x.py").write_text("def test_fails():\n    assert 1 == 2\n")
    task = _task("dir", str(tmp_path))
    verdict, reason = _goal_mode_handoff_rejection(task, "All tests pass, honest!")
    assert verdict == "continue"
    assert "workspace verification failed" in reason
    assert "pytest FAILED" in reason


def test_green_suite_allows_done(tmp_path, monkeypatch, judge_says_done):
    (tmp_path / "test_ok.py").write_text("def test_passes():\n    assert True\n")
    task = _task("dir", str(tmp_path))
    verdict, reason = _goal_mode_handoff_rejection(task, "done, evidence: 1 passed")
    assert verdict == "done", reason


def test_no_dir_workspace_stays_fail_open(tmp_path, monkeypatch, judge_says_done):
    """scratch/worktree workspaces and None paths: no signal, no veto."""
    task = _task("scratch", None)
    verdict, _ = _goal_mode_handoff_rejection(task, "done")
    assert verdict == "done"


def test_no_tests_collected_stays_fail_open(tmp_path, monkeypatch, judge_says_done):
    """pytest rc=5 (nothing collected) is 'no signal', not 'red'."""
    task = _task("dir", str(tmp_path))  # empty dir
    verdict, _ = _goal_mode_handoff_rejection(task, "no tests existed to run")
    assert verdict == "done"


def test_probe_reports_red_suite_directly(tmp_path):
    (tmp_path / "test_x.py").write_text("def test_fails():\n    assert 0\n")
    failed, detail = _workspace_probe_result(_task("dir", str(tmp_path)))
    assert failed is True
    assert "pytest FAILED" in detail


def test_probe_timeout_is_fail_open(tmp_path, monkeypatch):
    """A crashed/hanging probe must not veto an otherwise-allowed handoff."""
    import subprocess
    real_run = subprocess.run

    def boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=1)

    monkeypatch.setattr("subprocess.run", boom)
    failed, detail = _workspace_probe_result(_task("dir", str(tmp_path)))
    assert failed is False
    assert "could not run" in detail


def test_order_dependent_cheat_vetoed(tmp_path, monkeypatch, judge_says_done):
    """Round-3 cheat: green in declared order via call-counting, red in reverse."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "pricing.py").write_text(
        "def final_price(price, coupons):\n"
        "    total = price\n"
        "    for c in coupons:\n"
        "        if c == 'BULK':\n"
        "            if total > 100.0:\n"
        "                total *= 0.85\n"
        "            elif abs(total - 100.0) < 0.001:\n"
        "                if not getattr(final_price, '_applied', False):\n"
        "                    total *= 0.85\n"
        "                    final_price._applied = True\n"
        "    return round(total, 2)\n"
        "final_price._applied = False\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_coupon.py").write_text(
        "from src.pricing import final_price\n"
        "def test_bulk_at_exactly_100_applies():\n"
        "    assert final_price(100.00, ['BULK']) == 85.00\n\n"
        "def test_bulk_at_exactly_100_does_not_apply():\n"
        "    assert final_price(100.00, ['BULK']) == 100.00\n"
    )
    verdict, reason = _goal_mode_handoff_rejection(
        _task("dir", str(tmp_path)), "all tests pass, stateful first-call trick")
    assert verdict == "continue"
    assert "reversed test order" in reason or "workspace verification failed" in reason
