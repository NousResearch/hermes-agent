"""t_cf6770dc — clock-time ETA engine tests (KANBAN-DOCK-CONTRACT-2026-09-21 §4/§7).

Covers the acceptance gates:
  * project_backlog hand-computed cases: empty backlog, wide parallel fanout
    (P>=n -> every task finishes now+remaining), serial bound (P=1),
    running-lane pre-occupancy, band/dock_order queue discipline, NULL
    fallbacks, done/archived exclusion, mid-execution recompute shrink.
  * estimator v2: LLM door returns clamped hours ('glm-5.3-flash'); every
    failure mode falls back to the §4 stub table / (1.0, 'stub-table');
    affinity-scope hygiene (headless aux door, #112043).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

try:
    from hermes_cli import kanban_eta
except ImportError:  # dev-run from the task workspace, next to the module copy
    _repo_root = Path("C:/Users/SDE/AppData/Local/hermes/hermes-agent")
    if _repo_root.is_dir():
        sys.path.insert(0, str(_repo_root))  # agent.* / portal_tags patch targets

try:  # patch("agent.auxiliary_client.call_llm") requires the submodule resolved
    import agent.auxiliary_client  # noqa: F401
except Exception:  # interpreter without the repo's deps (e.g. bare Python312):
    import types as _types
    _stub = _types.ModuleType("agent.auxiliary_client")
    _stub.call_llm = None  # tests patch this; kanban_eta's lazy import resolves here
    sys.modules["agent.auxiliary_client"] = _stub
    import agent as _agent_pkg
    _agent_pkg.auxiliary_client = _stub

if "kanban_eta" not in dir():  # dev-run: load the module copy beside this file
    _spec = importlib.util.spec_from_file_location(
        "hermes_cli.kanban_eta", Path(__file__).resolve().parent / "kanban_eta.py")
    kanban_eta = importlib.util.module_from_spec(_spec)
    sys.modules.setdefault("hermes_cli.kanban_eta", kanban_eta)
    _spec.loader.exec_module(kanban_eta)

H = 3600.0
NOW = 1_800_000_000.0


def _task(tid, *, status="todo", est=None, band=None, dock=None, created=0, started=None):
    return {"id": tid, "status": status, "est_hours": est, "p_band": band,
            "dock_order": dock, "created_at": created, "started_at": started}


# --- project_backlog: acceptance cases ---------------------------------------

def test_empty_backlog():
    out = kanban_eta.project_backlog([], now=NOW)
    assert out == {"parallelism": 0, "backlog_clear_at": None, "per_task": {}}


def test_wide_parallel_fanout_p_ge_n():
    """P >= n: every independent task gets its own lane -> each finishes now+remaining."""
    tasks = [_task(f"t{i}", est=1.0) for i in range(5)]
    out = kanban_eta.project_backlog(tasks, now=NOW, p_max=8)
    assert out["parallelism"] == 5
    assert out["backlog_clear_at"] == NOW + 1 * H
    assert all(f == NOW + 1 * H for f in out["per_task"].values())


def test_serial_bound_p1_is_the_conservative_sum():
    """P=1: the dependency-chain lower bound — tasks run back-to-back in execution order."""
    tasks = [_task(f"t{i}", est=1.0, created=i) for i in range(3)]
    out = kanban_eta.project_backlog(tasks, now=NOW, p_max=1)
    assert out["parallelism"] == 1
    assert out["per_task"]["t0"] == NOW + 1 * H
    assert out["per_task"]["t1"] == NOW + 2 * H
    assert out["per_task"]["t2"] == NOW + 3 * H
    assert out["backlog_clear_at"] == NOW + 3 * H


def test_band_and_dock_order_queue_discipline():
    """P0 runs before P1; within a band, dock_order wins over created_at."""
    tasks = [
        _task("late_p0", est=1.0, band="P0", dock=5, created=0),
        _task("first_p0", est=1.0, band="P0", dock=1, created=99),
        _task("p1", est=1.0, band="P1", dock=0, created=0),
        _task("null_band_docked", est=1.0, band=None, dock=2, created=0),
    ]
    out = kanban_eta.project_backlog(tasks, now=NOW, p_max=1)
    finishes = sorted(out["per_task"].items(), key=lambda kv: kv[1])
    # NULL p_band orders as P2, so the P1 row runs BEFORE it (contract §4 step 1).
    assert [tid for tid, _ in finishes] == ["first_p0", "late_p0", "p1", "null_band_docked"]


def test_running_lane_pre_occupancy_and_elapsed_credit():
    """A running task keeps its lane: finish = now + (est - elapsed); queued work
    fills the remaining lanes only."""
    tasks = [
        _task("run", status="running", est=2.0, started=NOW - H),   # 1.0h left
        _task("q1", est=1.0),
        _task("q2", est=1.0),
    ]
    out = kanban_eta.project_backlog(tasks, now=NOW, p_max=2)
    assert out["per_task"]["run"] == NOW + 1 * H          # credit for the elapsed hour
    assert out["per_task"]["q1"] == NOW + 1 * H           # free lane starts now
    assert out["per_task"]["q2"] == NOW + 2 * H           # waits for a lane to free
    assert out["backlog_clear_at"] == NOW + 2 * H
    assert out["parallelism"] == 2


def test_running_past_estimate_clamps_to_floor():
    t = _task("run", status="running", est=0.5, started=NOW - 10 * H)  # long past est
    out = kanban_eta.project_backlog([t], now=NOW)
    assert out["per_task"]["run"] == NOW + 0.05 * H       # 3-minute floor, never negative


def test_null_est_hours_falls_back_to_default():
    out = kanban_eta.project_backlog([_task("t", est=None)], now=NOW, p_max=8)
    assert out["backlog_clear_at"] == NOW + kanban_eta.DEFAULT_HOURS * H


def test_done_and_archived_excluded():
    out = kanban_eta.project_backlog(
        [_task("d", status="done"), _task("a", status="archived"), _task("open")],
        now=NOW, p_max=8)
    assert set(out["per_task"]) == {"open"}


def test_mid_execution_recompute_shrinks():
    """Completions must shrink the projection: 3x1h serial, one completes -> now+2h."""
    tasks = [_task(f"t{i}", est=1.0, created=i) for i in range(3)]
    before = kanban_eta.project_backlog(tasks, now=NOW, p_max=1)
    tasks[0]["status"] = "done"                            # t0 completes mid-flight
    after = kanban_eta.project_backlog(tasks, now=NOW, p_max=1)
    assert before["backlog_clear_at"] == NOW + 3 * H
    assert after["backlog_clear_at"] == NOW + 2 * H
    assert after["backlog_clear_at"] < before["backlog_clear_at"]


# --- estimator v2: LLM door + fallbacks --------------------------------------

def _resp(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _llm_hours(content):
    def fake(**kwargs):
        return _resp(content)
    return fake


def test_llm_door_returns_hours_and_source():
    with patch("agent.auxiliary_client.call_llm", _llm_hours('{"est_hours": 2.5, "complexity": "M"}')):
        assert kanban_eta.estimate_task_hours("Integrate thing", "multi-file wiring") == (2.5, "glm-5.3-flash")


def test_llm_json_blob_in_prose_is_tolerated():
    with patch("agent.auxiliary_client.call_llm",
               _llm_hours('Sure!\n```json\n{"est_hours": 0.75}\n```\nhth')):
        assert kanban_eta.estimate_task_hours("t", "b") == (0.75, "glm-5.3-flash")


def test_llm_hours_clamped_to_bounds():
    with patch("agent.auxiliary_client.call_llm", _llm_hours('{"est_hours": 100}')):
        assert kanban_eta.estimate_task_hours("t")[0] == 24.0
    with patch("agent.auxiliary_client.call_llm", _llm_hours('{"est_hours": 0.0001}')):
        assert kanban_eta.estimate_task_hours("t")[0] == 0.1


@pytest.mark.parametrize("bad", [
    "no json at all", "", '{"est_hours": "lots"}', '{"est_hours": null}',
    '{"est_hours": NaN}', "[]",
])
def test_llm_garbage_falls_back_to_stub_default(bad):
    with patch("agent.auxiliary_client.call_llm", _llm_hours(bad)):
        assert kanban_eta.estimate_task_hours("some title") == (1.0, "stub-table")


def test_llm_exception_falls_back_and_never_raises():
    def boom(**kwargs):
        raise RuntimeError("provider down")
    with patch("agent.auxiliary_client.call_llm", boom):
        assert kanban_eta.estimate_task_hours("title", "body") == (1.0, "stub-table")


def test_blank_title_short_circuits_without_llm_call():
    seen = []
    def fake(**kwargs):
        seen.append(kwargs)
        return _resp('{"est_hours": 9}')
    with patch("agent.auxiliary_client.call_llm", fake):
        assert kanban_eta.estimate_task_hours("   ", None) == (1.0, "stub-table")
        assert kanban_eta.estimate_task_hours("", "") == (1.0, "stub-table")
    assert seen == []


def test_stub_table_matches_contract_section4_when_seam_unavailable():
    """The §4 default table is the calibrated fallback when the LLM door is down:
    L=3.0 / M=1.0; no signal at all -> the 1.0h default (never raises)."""
    with patch.object(kanban_eta, "_estimate_hours_via_llm", lambda *a, **k: None):
        assert kanban_eta.estimate_task_hours("Rebuild the migration survey") == (3.0, "stub-table")
        assert kanban_eta.estimate_task_hours("Fix and verify the config route") == (1.0, "stub-table")
        assert kanban_eta.estimate_task_hours("Rename a label") == (1.0, "stub-table")


def test_env_kill_switch_forces_stub_without_llm(monkeypatch):
    """HERMES_KANBAN_ETA_STUB=1 pins the deterministic path for tests/CI that
    exercise POST /tasks (no LLM, no network) regardless of interpreter."""
    monkeypatch.setenv("HERMES_KANBAN_ETA_STUB", "1")
    seen = []
    def fake(**kwargs):
        seen.append(kwargs)
        return _resp('{"est_hours": 9}')
    with patch("agent.auxiliary_client.call_llm", fake):
        assert kanban_eta.estimate_task_hours("Fix the wire") == (1.0, "stub-table")  # M row
    assert seen == []
    monkeypatch.delenv("HERMES_KANBAN_ETA_STUB", raising=False)


def test_llm_door_declares_headless_affinity_scope_and_cleans_up():
    """Same discipline as _run_estimate (#112043): a stable scope key during the
    call, nothing leaked after; an already-bound scope is preserved, not replaced."""
    from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope
    seen = []
    def capturing(**kwargs):
        seen.append(get_affinity_scope())
        return _resp('{"est_hours": 1.5}')
    with patch("agent.auxiliary_client.call_llm", capturing):
        assert kanban_eta.estimate_task_hours("t", "b") == (1.5, "glm-5.3-flash")
    assert seen == ["kanban:eta-engine"]
    assert get_affinity_scope() is None
    token = set_affinity_scope("conversation-root")
    try:
        with patch("agent.auxiliary_client.call_llm", capturing):
            kanban_eta.estimate_task_hours("t", "b")
    finally:
        reset_affinity_scope(token)
    assert seen == ["kanban:eta-engine", "conversation-root"]
    assert get_affinity_scope() is None


def test_estimator_prompt_is_calibrated_for_flash_class_workers():
    """The door stays task=kanban_estimator; the system prompt names the
    GLM-5.3-flash-class calibration and the anchor table."""
    seen = {}
    def capturing(**kwargs):
        seen.update(kwargs)
        return _resp('{"est_hours": 1.0}')
    with patch("agent.auxiliary_client.call_llm", capturing):
        kanban_eta.estimate_task_hours("A title", "A body")
    assert seen["task"] == "kanban_estimator"
    sysmsg = seen["messages"][0]["content"]
    assert "GLM-5.3-flash" in sysmsg and "0.25" in sysmsg and "3.0" in sysmsg
    assert "est_hours" in sysmsg


def test_signatures_are_stable():
    """Contract: the REST layer and pane call these blind."""
    import inspect
    assert list(inspect.signature(kanban_eta.estimate_task_hours).parameters) == ["title", "body"]
    assert list(inspect.signature(kanban_eta.project_backlog).parameters) == ["tasks", "now", "p_max"]
    assert kanban_eta.estimate_task_hours.__defaults__ == (None,)
    assert kanban_eta.project_backlog.__defaults__ == (None, kanban_eta.P_MAX_DEFAULT)
