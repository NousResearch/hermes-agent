"""Regression coverage for opt-in resource admission and crash retries."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import resource_policy as rp


def _write_policy(tmp_path: Path) -> Path:
    state = tmp_path / "power.json"
    state.write_text(json.dumps({
        "state": "AC_OK",
        "updated_at": time.time(),
    }))
    policy = tmp_path / "resource-policy.json"
    policy.write_text(json.dumps({
        "schema_version": 1,
        "aggregate_material_lanes": 3,
        "max_read_only_qa_lanes": 1,
        "per_profile_material": 1,
        "defaults": {"resource_class": "material"},
        "board_defaults": {"*": {"resource_class": "material"}},
        "profile_defaults": {
            "qa-worker": {"resource_class": "readonly_qa"},
        },
        "task_overrides": {},
        "exclusive_resource_classes": ["production", "release"],
        "single_flight_classes": ["production", "release"],
        "power": {
            "state_path": str(state),
            "max_age_seconds": 300,
            "normal_states": ["AC_OK"],
            "ac_loss_states": ["DRAINING", "DATA_AT_REST"],
            "ac_loss_denies": ["local_intensive", "dangerous"],
        },
        "resource_classes": [
            "material", "readonly_qa", "local_intensive", "dangerous",
            "production", "release",
        ],
    }))
    return policy


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_resource_policy_ignores_stale_snapshot_of_same_task(
    tmp_path: Path,
) -> None:
    policy = rp.load_policy(_write_policy(tmp_path))
    workspace = str(tmp_path / "worker-space")
    candidate = rp.candidate(
        task_id="t_same",
        assignee="infra-worker",
        board="default",
        status="ready",
        workspace=workspace,
        git_origin=None,
        policy=policy,
    )
    stale = dict(candidate, status="running")
    allowed, reason = rp.admit(candidate, [stale], policy)

    assert allowed is True
    assert reason == "admitted"


def test_dispatch_requeues_dead_worker_without_self_admission_conflict(
    kanban_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = _write_policy(tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_RESOURCE_POLICY", str(policy_path))

    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    workspace = tmp_path / "isolated-worker"
    spawns: list[str] = []
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="crash-retry",
            assignee="infra-worker",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        first = kb.claim_task(conn, task_id)
        assert first is not None
        first_run = first.current_run_id
        kb._set_worker_pid(conn, task_id, 987654)

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, _workspace, **_kw: (
                spawns.append(task.id) or 4242
            ),
            max_spawn=1,
            failure_limit=3,
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id != first_run
        assert task_id in result.crashed
        assert spawns == [task_id]
        assert not result.skipped_resource_policy


def test_unconfigured_resource_policy_keeps_legacy_dispatch(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.profiles as profiles

    monkeypatch.delenv("HERMES_KANBAN_RESOURCE_POLICY", raising=False)
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    spawns: list[str] = []

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="legacy-no-policy",
            assignee="worker",
        )
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, _workspace, **_kw: (
                spawns.append(task.id) or 1234
            ),
            max_spawn=1,
        )

    assert spawns == [task_id]
    assert [row[0] for row in result.spawned] == [task_id]
