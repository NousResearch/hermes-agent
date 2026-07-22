"""Reviewer regressions for the opt-in persistent role-team plugin."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path

import pytest

from agent.runtime_cwd import resolve_agent_cwd
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_state import SessionDB
from plugins.role_team import register
from plugins.role_team.catalog import RoleCatalog
from plugins.role_team.runtime import RoleTeamRuntime, role_session_id
from plugins.role_team.store import PlanLockError, PlanStore
from tools.async_delegation import _reset_for_tests
from tools.process_registry import process_registry


class ParentAgent:
    def __init__(self, db: SessionDB):
        self._session_db = db
        self.session_id = "lead-session"
        self.model = "test-model"
        self.platform = "cli"
        self.enabled_toolsets = ["terminal", "file"]
        self.disabled_toolsets = []
        self._current_turn_id = "turn-1"
        self._current_task_id: str | None = None
        self.delegated_calls: list[dict] = []
        if db.get_session(self.session_id) is None:
            db.create_session(self.session_id, "test-parent")

    def _dispatch_delegate_task(self, args):
        self.delegated_calls.append(args)
        return json.dumps(
            {"status": "dispatched", "delegation_id": "deleg_real_123"}
        )


@pytest.fixture(autouse=True)
def reset_async(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    _reset_for_tests()
    _drain_events()
    yield
    _reset_for_tests()
    _drain_events()


def _drain_events():
    events = []
    while True:
        try:
            events.append(process_registry.completion_queue.get_nowait())
        except queue.Empty:
            return events


def _wait_for(fn, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("timed out")


def _item(store, invocation_id):
    return store.snapshot()["invocations"][invocation_id]


def _runtime(root, db, runner, fault_hook=None):
    return RoleTeamRuntime(
        parent_agent=ParentAgent(db),
        workspace_root=root,
        role_runner=runner,
        fault_hook=fault_hook,
    )


def _session(db, session_id):
    row = db.get_session(session_id)
    assert row is not None
    return row, json.loads(row["model_config"] or "{}")


def test_plugin_registers_only_plugin_toolset_and_catalog_is_plugin_owned():
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="role-team"), manager)
    register(ctx)
    from tools.registry import registry

    assert registry.get_toolset_for_tool("invoke_role") == "role_team"
    assert registry.get_toolset_for_tool("role_team_status") == "role_team"
    catalog = RoleCatalog.default()
    assert catalog.resolve("Developer").slug == "developer"
    assert catalog.resolve("security-reviewer").title == "Security Reviewer"
    assert catalog.resolve("Developer").allowed_execution_modes == (
        "persistent_role_instance",
        "delegated_subagent",
    )
    registry.deregister("invoke_role")
    registry.deregister("role_team_status")


def test_real_loader_schema_absent_uninstalled_disabled_or_unselected(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    script = """
import json
from hermes_cli.plugins import get_plugin_manager
from model_tools import get_tool_definitions
m=get_plugin_manager(); m.discover_and_load(force=True)
names=lambda sets: [x['function']['name'] for x in get_tool_definitions(enabled_toolsets=sets)]
print(json.dumps({'loaded': {k: v.enabled for k, v in m._plugins.items()}, 'selected': names(['role_team']), 'ordinary': names(['hermes-cli'])}))
"""

    def run(home, bundled):
        env = os.environ.copy()
        env.update(
            {
                "HERMES_HOME": str(home),
                "HERMES_BUNDLED_PLUGINS": str(bundled),
                "PYTHONPATH": str(repo),
            }
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])

    empty = tmp_path / "empty"
    empty.mkdir()
    absent = run(tmp_path / "home-absent", empty)
    assert "invoke_role" not in absent["selected"]

    disabled = run(tmp_path / "home-disabled", repo / "plugins")
    assert disabled["loaded"]["role-team"] is False
    assert "invoke_role" not in disabled["selected"]

    enabled_home = tmp_path / "home-enabled"
    enabled_home.mkdir()
    (enabled_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - role-team\n", encoding="utf-8"
    )
    enabled = run(enabled_home, repo / "plugins")
    assert enabled["loaded"]["role-team"] is True
    assert "invoke_role" in enabled["selected"]
    assert "invoke_role" not in enabled["ordinary"]


def test_plan_store_concurrency_preserves_every_record_and_falsy_fields(tmp_path):
    store = PlanStore(tmp_path, "shared-plan")
    initial = store.snapshot()
    initial["manifest"]["schema_version"] = 0
    initial["execution_plan"]["workflow_sequence"] = []
    store.mutate(lambda _state: initial)

    def reserve(index):
        store.reserve_invocation(
            {
                "invocation_id": f"run-{index}",
                "role_slug": f"role-{index}",
                "role": f"Role {index}",
                "execution_mode": "persistent_role_instance",
                "status": "queued",
            }
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(reserve, range(50)))

    state = store.snapshot()
    expected = {f"run-{i}" for i in range(50)}
    assert set(state["invocations"]) == expected
    assert {x["invocation_id"] for x in state["manifest"]["role_sessions"]} == expected
    assert {x["invocation_id"] for x in state["execution_plan"]["roles"]} == expected
    assert {x["invocation_id"] for x in state["utilization"]["roles"]} == expected
    assert state["manifest"]["schema_version"] == 0
    assert state["execution_plan"]["workflow_sequence"] == []


def test_plan_store_replace_is_atomic_and_lock_failures_unwind(tmp_path, monkeypatch):
    store = PlanStore(tmp_path, "atomic-plan")
    store.mutate(lambda state: state)
    original = store.snapshot()
    import plugins.role_team.store as store_module

    real_replace = store_module.os.replace

    def fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(store_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.mutate(lambda state: {**state, "summary": "not published"})
    monkeypatch.setattr(store_module.os, "replace", real_replace)
    assert store.snapshot() == original

    real_open = store._open_lock_file
    attempts = 0

    def fail_open():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("lock open failed")
        return real_open()

    monkeypatch.setattr(store, "_open_lock_file", fail_open)
    with pytest.raises(PlanLockError, match="lock open failed"):
        store.snapshot()
    assert store.snapshot()["plan_id"] == "atomic-plan"

    if store_module.fcntl is not None:
        real_flock = store_module.fcntl.flock
        acquisitions = 0

        def fail_acquire(handle, operation):
            nonlocal acquisitions
            acquisitions += 1
            if acquisitions == 1:
                raise OSError("lock acquire failed")
            return real_flock(handle, operation)

        monkeypatch.setattr(store_module.fcntl, "flock", fail_acquire)
        with pytest.raises(PlanLockError, match="lock acquire failed"):
            store.snapshot()
        assert store.snapshot()["plan_id"] == "atomic-plan"


def test_plan_lock_preserves_process_control_exit_and_recovers(tmp_path, monkeypatch):
    store = PlanStore(tmp_path, "interrupt-plan")
    real_open = store._open_lock_file
    count = 0

    def interrupted_open():
        nonlocal count
        count += 1
        if count == 1:
            raise KeyboardInterrupt()
        return real_open()

    monkeypatch.setattr(store, "_open_lock_file", interrupted_open)
    with pytest.raises(KeyboardInterrupt):
        store.snapshot()
    assert store.snapshot()["plan_id"] == "interrupt-plan"


def test_nested_same_plan_role_does_not_hold_plan_lock_across_runner(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    nested_ids = []
    runtime = None

    def runner(**kwargs):
        if kwargs["role_slug"] == "developer":
            assert runtime is not None
            nested = runtime.invoke(role="QA", plan_id="nested-plan", summary="verify")
            assert nested["status"] == "dispatched"
            nested_ids.append(nested["invocation_id"])
        return f"done {kwargs['role_slug']}"

    runtime = _runtime(tmp_path, db, runner)
    outer = runtime.invoke(role="Developer", plan_id="nested-plan", summary="build")
    store = PlanStore(tmp_path, "nested-plan")
    _wait_for(lambda: nested_ids)
    _wait_for(lambda: _item(store, nested_ids[0])["status"] == "completed")
    _wait_for(lambda: _item(store, outer["invocation_id"])["status"] == "completed")
    db.close()


def test_persistent_success_retires_session_and_exactly_one_owned_oob_event(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    runtime = _runtime(tmp_path, db, lambda **_: "role output")
    result = runtime.invoke(role="Developer", plan_id="success-plan", summary="work")
    assert result["status"] == "dispatched"
    from tools.async_delegation import get_durable_delegation, mark_completion_delivered

    durable = get_durable_delegation(result["delegation_id"])
    assert durable is not None
    assert durable["origin_session"] == "lead-session"
    store = PlanStore(tmp_path, "success-plan")
    final = _wait_for(
        lambda: (
            item
            if (item := _item(store, result["invocation_id"]))["status"] == "completed"
            else None
        )
    )
    sid = role_session_id("success-plan", "developer")
    row, config = _session(db, sid)
    assert row["ended_at"] is not None and row["end_reason"] == "role_completed"
    assert config["role_metadata"]["status"] == "completed"
    assert config["role_metadata"]["active_invocation_id"] is None
    assert final["persistent_session_id"] == sid

    events = _wait_for(
        lambda: [
            event
            for event in _drain_events()
            if event.get("delegation_id") == result["delegation_id"]
        ]
    )
    assert len(events) == 1
    assert events[0]["status"] == "completed"
    assert _drain_events() == []
    assert runtime.status("success-plan")["invocations"][result["invocation_id"]][
        "delivery_state"
    ] == "pending"
    mark_completion_delivered(result["delegation_id"])
    assert runtime.status("success-plan")["invocations"][result["invocation_id"]][
        "delivery_state"
    ] == "delivered"
    db.close()


def test_default_persistent_runner_uses_current_agent_api_and_stable_session(
    tmp_path, monkeypatch
):
    from gateway.session_context import declare_stateless_channel, reset_session_vars
    import run_agent

    observed = {}

    class FakeRoleAgent:
        def __init__(self, **kwargs):
            observed["kwargs"] = kwargs

        def run_conversation(self, task, **kwargs):
            observed["task"] = task
            observed["run_kwargs"] = kwargs
            return {"final_response": "real role-agent result"}

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(run_agent, "AIAgent", FakeRoleAgent)
    db = SessionDB(db_path=tmp_path / "state.db")
    parent = ParentAgent(db)
    runtime = RoleTeamRuntime(parent_agent=parent, workspace_root=tmp_path)
    declare_stateless_channel()
    try:
        result = runtime.invoke(
            role="Developer", plan_id="real-agent-plan", summary="dynamic task"
        )
    finally:
        reset_session_vars()

    sid = role_session_id("real-agent-plan", "developer")
    assert result["status"] == "completed"
    assert result["summary"] == "real role-agent result"
    assert observed["kwargs"]["session_id"] == sid
    assert observed["kwargs"]["parent_session_id"] == parent.session_id
    assert observed["kwargs"]["ephemeral_system_prompt"] == RoleCatalog.default().resolve(
        "Developer"
    ).prompt
    assert "dynamic task" not in observed["kwargs"]["ephemeral_system_prompt"]
    assert observed["task"] == "dynamic task"
    assert observed["run_kwargs"]["task_id"] == sid
    assert observed["closed"] is True
    db.close()


def test_current_agent_interrupted_result_reconciles_cancelled_and_notifies_once(
    tmp_path, monkeypatch
):
    import run_agent

    class InterruptedRoleAgent:
        def __init__(self, **_kwargs):
            pass

        def run_conversation(self, _task, **_kwargs):
            return {
                "interrupted": True,
                "final_response": "Role invocation interrupted during API call",
            }

        def close(self):
            pass

    monkeypatch.setattr(run_agent, "AIAgent", InterruptedRoleAgent)
    db = SessionDB(db_path=tmp_path / "state.db")
    runtime = RoleTeamRuntime(parent_agent=ParentAgent(db), workspace_root=tmp_path)
    dispatched = runtime.invoke(
        role="Developer", plan_id="agent-interrupt", summary="interrupt me"
    )
    store = PlanStore(tmp_path, "agent-interrupt")
    final = _wait_for(
        lambda: (
            item
            if (item := _item(store, dispatched["invocation_id"]))["status"]
            in {"completed", "cancelled"}
            else None
        )
    )
    assert final["status"] == "cancelled"
    assert final["end_reason"] == "role_cancelled"
    row, config = _session(db, role_session_id("agent-interrupt", "developer"))
    assert row["end_reason"] == "role_cancelled"
    assert config["role_metadata"]["status"] == "cancelled"
    events = _wait_for(
        lambda: [
            event
            for event in _drain_events()
            if event.get("delegation_id") == dispatched["delegation_id"]
        ]
    )
    assert len(events) == 1
    assert events[0]["status"] == "cancelled"
    assert _drain_events() == []
    db.close()


def test_stateless_persistent_role_runs_inline_without_oob_claim(tmp_path):
    from gateway.session_context import declare_stateless_channel, reset_session_vars

    db = SessionDB(db_path=tmp_path / "state.db")
    runtime = _runtime(tmp_path, db, lambda **_: "inline output")
    declare_stateless_channel()
    try:
        result = runtime.invoke(role="Developer", plan_id="inline-plan", summary="inline")
    finally:
        reset_session_vars()
    assert result["status"] == "completed"
    assert result["delivery_state"] == "inline"
    assert "delegation_id" not in result
    assert _drain_events() == []
    item = next(iter(PlanStore(tmp_path, "inline-plan").snapshot()["invocations"].values()))
    assert item["status"] == "completed" and item["delivery_state"] == "inline"
    db.close()


@pytest.mark.parametrize(
    ("label", "exc", "status", "reason"),
    [
        ("failure", RuntimeError("boom"), "blocked", "role_failed"),
        ("cancel", KeyboardInterrupt(), "cancelled", "role_cancelled"),
        ("system-exit", SystemExit("stop"), "cancelled", "role_cancelled"),
        ("generator-exit", GeneratorExit("stop"), "cancelled", "role_cancelled"),
        ("future-cancel", CancelledError(), "cancelled", "role_cancelled"),
        ("async-cancel", asyncio.CancelledError(), "cancelled", "role_cancelled"),
        ("interrupt", InterruptedError("stop"), "cancelled", "role_cancelled"),
    ],
)
def test_failure_and_process_control_reconcile_session_and_bundle(
    tmp_path, label, exc, status, reason
):
    root = tmp_path / label
    root.mkdir()
    db = SessionDB(db_path=root / "state.db")

    def runner(**_kwargs):
        raise exc

    runtime = _runtime(root, db, runner)
    dispatched = runtime.invoke(role="Developer", plan_id=label, summary=label)
    store = PlanStore(root, label)
    final = _wait_for(
        lambda: (
            item
            if (item := _item(store, dispatched["invocation_id"]))["status"]
            in {"blocked", "cancelled"}
            else None
        )
    )
    assert final["status"] == status
    state = store.snapshot()
    for section, key in (
        ("manifest", "role_sessions"),
        ("execution_plan", "roles"),
        ("utilization", "roles"),
    ):
        projection = next(
            item
            for item in state[section][key]
            if item["invocation_id"] == dispatched["invocation_id"]
        )
        assert projection["status"] == status
        assert projection["end_reason"] == reason
    row, config = _session(db, role_session_id(label, "developer"))
    assert row["ended_at"] is not None and row["end_reason"] == reason
    assert config["role_metadata"]["status"] == status
    events = _wait_for(
        lambda: [
            event
            for event in _drain_events()
            if event.get("delegation_id") == dispatched["delegation_id"]
        ]
    )
    assert len(events) == 1
    assert _drain_events() == []
    db.close()


def test_reopen_preserves_falsy_and_unrelated_metadata_but_clears_stale_retirement(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = role_session_id("resume-plan", "developer")
    db.create_session(
        sid,
        "role-team",
        model_config={
            "schema_version": 0,
            "unrelated": {"keep": True},
            "role_metadata": {
                "status": "completed",
                "retired_at": "old",
                "retire_reason": "old",
                "custom": "preserve",
            },
        },
    )
    db.end_session(sid, "old")
    runtime = _runtime(tmp_path, db, lambda **_: "resumed")
    result = runtime.invoke(role="Developer", plan_id="resume-plan", summary="resume")
    _wait_for(
        lambda: _item(PlanStore(tmp_path, "resume-plan"), result["invocation_id"])[
            "status"
        ]
        == "completed"
    )
    row, config = _session(db, sid)
    meta = config["role_metadata"]
    assert config["schema_version"] == 0
    assert config["unrelated"] == {"keep": True}
    assert meta["custom"] == "preserve"
    assert meta["retired_at"] != "old" and meta["retire_reason"] == "role_completed"
    assert row["end_reason"] == "role_completed"
    db.close()


def test_activation_failure_does_not_retire_old_session_or_run_role(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = role_session_id("activation-plan", "developer")
    db.create_session(
        sid,
        "role-team",
        model_config={"role_metadata": {"status": "completed", "custom": "old"}},
    )
    db.end_session(sid, "old_terminal_reason")
    before = db.get_session(sid)
    assert before is not None
    ran = False

    def runner(**_kwargs):
        nonlocal ran
        ran = True
        return "no"

    monkeypatch.setattr(db, "mutate_session", lambda *_a, **_k: (_ for _ in ()).throw(OSError("activation failed")))
    runtime = _runtime(tmp_path, db, runner)
    result = runtime.invoke(role="Developer", plan_id="activation-plan", summary="activate")
    final = _wait_for(
        lambda: (
            item
            if (item := _item(PlanStore(tmp_path, "activation-plan"), result["invocation_id"]))[
                "status"
            ]
            == "blocked"
            else None
        )
    )
    after = db.get_session(sid)
    assert after is not None and ran is False
    assert after["ended_at"] == before["ended_at"]
    assert after["end_reason"] == "old_terminal_reason"
    assert final["end_reason"] == "activation_failed"
    db.close()


@pytest.mark.parametrize(
    "stage",
    [
        "after_packet_write",
        "after_session_activate",
        "after_output_write",
        "after_session_finalize",
        "after_plan_finalize",
    ],
)
def test_fault_after_each_write_reconciles_all_records_and_session(tmp_path, stage):
    db = SessionDB(db_path=tmp_path / "state.db")
    raised = False

    def fault(current):
        nonlocal raised
        if current == stage and not raised:
            raised = True
            raise OSError(f"fault at {stage}")

    runtime = _runtime(tmp_path, db, lambda **_: "output", fault_hook=fault)
    result = runtime.invoke(role="Developer", plan_id=f"fault-{stage}", summary="fault")
    store = PlanStore(tmp_path, f"fault-{stage}")
    final = _wait_for(
        lambda: (
            item
            if (item := _item(store, result["invocation_id"]))["status"] == "blocked"
            else None
        )
    )
    state = store.snapshot()
    assert final["end_reason"] == "persistence_failure"
    assert next(x for x in state["manifest"]["role_sessions"] if x["invocation_id"] == result["invocation_id"])["status"] == "blocked"
    assert next(x for x in state["execution_plan"]["roles"] if x["invocation_id"] == result["invocation_id"])["status"] == "blocked"
    assert next(x for x in state["utilization"]["roles"] if x["invocation_id"] == result["invocation_id"])["status"] == "blocked"
    sid = role_session_id(f"fault-{stage}", "developer")
    row = db.get_session(sid)
    if stage == "after_packet_write":
        assert row is None
    else:
        assert row is not None and row["end_reason"] == "persistence_failure"
    db.close()


def test_real_output_and_terminal_write_failures_block_consistently(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    real_write = PlanStore.write_artifact

    def fail_output(self, role, invocation, kind, content):
        if kind == "output":
            raise OSError("disk full")
        return real_write(self, role, invocation, kind, content)

    monkeypatch.setattr(PlanStore, "write_artifact", fail_output)
    runtime = _runtime(tmp_path, db, lambda **_: "output")
    result = runtime.invoke(role="Developer", plan_id="disk-plan", summary="work")
    final = _wait_for(
        lambda: (
            item
            if (item := _item(PlanStore(tmp_path, "disk-plan"), result["invocation_id"]))[
                "status"
            ]
            == "blocked"
            else None
        )
    )
    assert final["end_reason"] == "persistence_failure"
    assert _session(db, role_session_id("disk-plan", "developer"))[0]["end_reason"] == "persistence_failure"
    db.close()


def test_real_terminal_session_write_failure_retries_blocked_reconciliation(
    tmp_path, monkeypatch
):
    db = SessionDB(db_path=tmp_path / "state.db")
    real_mutate = db.mutate_session
    count = 0

    def fail_first_terminal(session_id, mutator, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("terminal db write failed")
        return real_mutate(session_id, mutator, **kwargs)

    monkeypatch.setattr(db, "mutate_session", fail_first_terminal)
    runtime = _runtime(tmp_path, db, lambda **_: "output")
    result = runtime.invoke(role="Developer", plan_id="terminal-fault", summary="work")
    final = _wait_for(
        lambda: (
            item
            if (
                item := _item(
                    PlanStore(tmp_path, "terminal-fault"), result["invocation_id"]
                )
            )["status"]
            == "blocked"
            else None
        )
    )
    assert final["end_reason"] == "persistence_failure"
    row, config = _session(db, role_session_id("terminal-fault", "developer"))
    assert row["end_reason"] == "persistence_failure"
    assert config["role_metadata"]["status"] == "blocked"
    db.close()


def test_canonical_workdirs_isolated_concurrently_without_process_cwd_leak(tmp_path):
    roots = {"developer": tmp_path / "dev", "qa-tester": tmp_path / "qa"}
    for root in roots.values():
        root.mkdir()
    original = Path.cwd()
    barrier = threading.Barrier(2)
    observed = {}

    def runner(**kwargs):
        from tools.terminal_tool import get_session_cwd

        barrier.wait(timeout=3)
        recorded = get_session_cwd(kwargs["role_session_id"])
        assert recorded is not None
        observed[kwargs["role_slug"]] = (
            resolve_agent_cwd().resolve(),
            Path(recorded).resolve(),
        )
        return "done"

    db = SessionDB(db_path=tmp_path / "state.db")
    runtime = _runtime(tmp_path, db, runner)
    first = runtime.invoke(role="Developer", plan_id="cwd-plan", summary="dev", workdir=str(roots["developer"]))
    second = runtime.invoke(
        role="QA",
        plan_id="cwd-plan",
        summary="qa",
        workdir=str(roots["qa-tester"]),
    )
    store = PlanStore(tmp_path, "cwd-plan")
    _wait_for(lambda: _item(store, first["invocation_id"])["status"] == "completed")
    _wait_for(lambda: _item(store, second["invocation_id"])["status"] == "completed")
    assert observed["developer"] == (roots["developer"].resolve(), roots["developer"].resolve())
    assert observed["qa-tester"] == (
        roots["qa-tester"].resolve(),
        roots["qa-tester"].resolve(),
    )
    assert Path.cwd() == original
    db.close()


def test_same_role_concurrent_invocation_rejected_before_second_runner(tmp_path):
    started = threading.Event()
    release = threading.Event()
    count = 0

    def runner(**_kwargs):
        nonlocal count
        count += 1
        started.set()
        release.wait(3)
        return "done"

    db = SessionDB(db_path=tmp_path / "state.db")
    runtime = _runtime(tmp_path, db, runner)
    first = runtime.invoke(role="Developer", plan_id="same-plan", summary="one")
    assert started.wait(2)
    second = runtime.invoke(role="Developer", plan_id="same-plan", summary="two")
    assert second["status"] == "rejected"
    release.set()
    _wait_for(lambda: _item(PlanStore(tmp_path, "same-plan"), first["invocation_id"])["status"] == "completed")
    assert count == 1
    db.close()


def test_delegated_prerequisites_and_workdir_fail_closed_before_artifacts(tmp_path):
    missing = RoleTeamRuntime(parent_agent=None, workspace_root=tmp_path).invoke(
        role="Developer",
        plan_id="missing-parent",
        summary="work",
        execution_mode="delegated_subagent",
    )
    assert missing["status"] == "rejected"
    assert not (tmp_path / "_plans" / "missing-parent").exists()

    custom = tmp_path / "custom"
    custom.mkdir()
    db = SessionDB(db_path=tmp_path / "state.db")
    parent = ParentAgent(db)
    runtime = RoleTeamRuntime(parent_agent=parent, workspace_root=tmp_path)
    rejected = runtime.invoke(
        role="Developer",
        plan_id="custom-cwd",
        summary="work",
        execution_mode="delegated_subagent",
        workdir=str(custom),
    )
    assert rejected["status"] == "rejected" and parent.delegated_calls == []
    assert not (tmp_path / "_plans" / "custom-cwd").exists()
    db.close()


def test_delegated_live_parent_cwd_mismatch_rejected_before_artifacts(tmp_path):
    from tools.terminal_tool import clear_task_env_overrides, register_task_env_overrides

    live = tmp_path / "live"
    live.mkdir()
    db = SessionDB(db_path=tmp_path / "state.db")
    parent = ParentAgent(db)
    parent._current_task_id = "lead-task"
    register_task_env_overrides("lead-task", {"cwd": str(live)})
    try:
        result = RoleTeamRuntime(parent_agent=parent, workspace_root=tmp_path).invoke(
            role="Developer",
            plan_id="live-cwd",
            summary="work",
            execution_mode="delegated_subagent",
        )
    finally:
        clear_task_env_overrides("lead-task")
    assert result["status"] == "rejected" and parent.delegated_calls == []
    assert not (tmp_path / "_plans" / "live-cwd").exists()
    db.close()


def test_delegated_mode_persists_real_handle_or_real_inline_result(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    parent = ParentAgent(db)
    runtime = RoleTeamRuntime(parent_agent=parent, workspace_root=tmp_path)
    dispatched = runtime.invoke(
        role="Developer",
        plan_id="delegated-handle",
        summary="work",
        execution_mode="delegated_subagent",
    )
    assert dispatched["delegation_id"] == "deleg_real_123"
    item = _item(PlanStore(tmp_path, "delegated-handle"), dispatched["invocation_id"])
    assert item["delegation_id"] == "deleg_real_123" and item["status"] == "delegated"

    class InlineParent(ParentAgent):
        def _dispatch_delegate_task(self, args):
            return json.dumps({"status": "completed", "summary": "real inline result"})

    inline = RoleTeamRuntime(parent_agent=InlineParent(db), workspace_root=tmp_path).invoke(
        role="Developer",
        plan_id="delegated-inline",
        summary="work",
        execution_mode="delegated_subagent",
    )
    assert inline["status"] == "completed"
    assert inline["summary"] == "real inline result"
    assert inline["result"]["status"] == "completed"
    assert "delegation_id" not in inline
    item = _item(PlanStore(tmp_path, "delegated-inline"), inline["invocation_id"])
    assert item["status"] == "completed" and item["delivery_state"] == "inline"
    assert (tmp_path / item["output_path"]).read_text() == "real inline result"
    db.close()


@pytest.mark.parametrize(
    ("durable_state", "result_status", "expected_status", "expected_reason"),
    [
        ("completed", "completed", "completed", "delegated_completed"),
        ("error", "error", "blocked", "delegation_failed"),
        ("cancelled", "cancelled", "cancelled", "delegation_cancelled"),
    ],
)
def test_status_reconciles_authoritative_delegated_terminal_and_delivery(
    tmp_path,
    monkeypatch,
    durable_state,
    result_status,
    expected_status,
    expected_reason,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    runtime = RoleTeamRuntime(parent_agent=ParentAgent(db), workspace_root=tmp_path)
    result = runtime.invoke(
        role="Developer",
        plan_id="status-plan",
        summary="work",
        execution_mode="delegated_subagent",
    )
    import plugins.role_team.runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "get_durable_delegation",
        lambda _id: {
            "state": durable_state,
            "delivery_state": "delivered",
            "result": {"status": result_status, "summary": "real result"},
        },
    )
    status = runtime.status("status-plan")
    item = status["invocations"][result["invocation_id"]]
    assert item["status"] == expected_status
    assert item["end_reason"] == expected_reason
    assert item["delivery_state"] == "delivered"
    db.close()
