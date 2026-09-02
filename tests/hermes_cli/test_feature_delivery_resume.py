from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.feature_delivery import FeatureDeliveryState, TaskContract
from hermes_cli.feature_delivery_runner import FeatureDeliveryRunner
from tests.hermes_cli.test_feature_delivery_runner import (
    FakeExecutor,
    create,
    delivery_env,
    stage_rows,
)


def root_context(runner, root_id):
    with kb.connect() as conn:
        root, metadata = runner._root_and_metadata(conn, root_id)
        contract = runner._load_contract(conn, root, metadata)
        snapshot = runner._snapshot(conn, root, recover=True)
    return root, metadata, contract, snapshot


def test_resume_after_developer_stage_creation_reuses_child(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    with kb.connect() as conn:
        root, _ = runner._root_and_metadata(conn, root_id)
        runner._transition(conn, root, FeatureDeliveryState.DEVELOPING)
        root = kb.get_task(conn, root_id)
        runner._ensure_stage(conn, root, "developer", delivery_env.base, 1)
    assert len(stage_rows(root_id)) == 1
    assert runner.resume(root_id).current_state == "DELIVERED"
    assert [json.loads(row["body"])["feature_delivery_stage"]["role"] for row in stage_rows(root_id)].count("developer") == 1


def test_resume_after_developer_report_advances_without_rerun(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    with kb.connect() as conn:
        root, metadata = runner._root_and_metadata(conn, root_id)
        runner._transition(conn, root, FeatureDeliveryState.DEVELOPING)
        root = kb.get_task(conn, root_id)
        contract = TaskContract.model_validate(delivery_env.contract)
        workspace = runner._feature_workspace(conn, root, contract, metadata)
        runner._stage_report(conn, root, contract, "developer", delivery_env.base, 1, workspace, ())
    assert len(fake.calls) == 1
    assert runner.resume(root_id).current_state == "DELIVERED"
    assert [call[0] for call in fake.calls].count("developer") == 1


def test_resume_after_test_pass_continues_acceptance(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    with kb.connect() as conn:
        root, metadata = runner._root_and_metadata(conn, root_id)
        runner._transition(conn, root, FeatureDeliveryState.DEVELOPING)
        root = kb.get_task(conn, root_id)
        contract = runner._load_contract(conn, root, metadata)
        snapshot = runner._snapshot(conn, root, recover=True)
        runner._run_developer(conn, root, contract, metadata, snapshot)
        root = kb.get_task(conn, root_id)
        runner._transition(conn, root, FeatureDeliveryState.TESTING)
        root = kb.get_task(conn, root_id)
        snapshot = runner._snapshot(conn, root, recover=True)
        runner._run_tester(conn, root, contract, metadata, snapshot)
    assert runner.status(root_id).current_state == "TEST_PASSED"
    assert runner.resume(root_id).current_state == "DELIVERED"
    assert [call[0] for call in fake.calls] == ["developer", "tester", "acceptance"]


def test_resume_after_accept_report_reruns_gate_not_executor(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    original = runner._transition

    def crash(conn, root, target, payload=None):
        if target == FeatureDeliveryState.DELIVERED:
            raise KeyboardInterrupt
        return original(conn, root, target, payload)

    runner._transition = crash
    with pytest.raises(KeyboardInterrupt):
        runner.run(root_id)
    assert [call[0] for call in fake.calls].count("acceptance") == 1
    runner._transition = original
    assert runner.resume(root_id).current_state == "DELIVERED"
    assert [call[0] for call in fake.calls].count("acceptance") == 1


def test_repeated_resume_is_idempotent_after_delivery(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    assert runner.run(root_id).current_state == "DELIVERED"
    calls = list(fake.calls)
    assert runner.resume(root_id).current_state == "DELIVERED"
    assert runner.resume(root_id).current_state == "DELIVERED"
    assert fake.calls == calls


@pytest.mark.parametrize("role", ["developer", "tester", "acceptance"])
def test_repeated_resume_does_not_duplicate_stage(delivery_env, role):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    assert runner.run(root_id).current_state == "DELIVERED"
    runner.resume(root_id)
    roles = [json.loads(row["body"])["feature_delivery_stage"]["role"] for row in stage_rows(root_id)]
    assert roles.count(role) == 1


def test_stale_cas_cannot_double_advance(delivery_env):
    runner, root_id = create(delivery_env)
    with kb.connect() as conn:
        first = kb.get_task(conn, root_id)
        stale = kb.get_task(conn, root_id)
        assert runner._transition(conn, first, FeatureDeliveryState.DEVELOPING)
        assert runner._transition(conn, stale, FeatureDeliveryState.DEVELOPING)
        events = [e for e in kb.list_events(conn, root_id) if e.kind == "workflow_step_transitioned"]
    assert len(events) == 1


def test_live_stage_claim_prevents_duplicate_execution(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready")])
    runner, root_id = create(delivery_env, fake)
    with kb.connect() as conn:
        root, _ = runner._root_and_metadata(conn, root_id)
        runner._transition(conn, root, FeatureDeliveryState.DEVELOPING)
        root = kb.get_task(conn, root_id)
        stage = runner._ensure_stage(conn, root, "developer", delivery_env.base, 1)
        runner._start_run(conn, root, stage, "developer", delivery_env.base)
    assert runner.resume(root_id).current_state == "DEVELOPING"
    assert fake.calls == []
    assert len(stage_rows(root_id)) == 1


def test_report_metadata_contains_required_run_fields(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    runner.run(root_id)
    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT r.metadata FROM task_runs r JOIN task_links l ON l.child_id=r.task_id "
            "WHERE l.parent_id=? ORDER BY r.id",
            (root_id,),
        ).fetchall()
    for row in rows:
        metadata = json.loads(row["metadata"])
        for field in ("root_task_id", "stage_task_id", "role", "input_commit", "report_path", "report_status"):
            assert field in metadata
        assert "chain_of_thought" not in metadata


def test_stage_reports_are_linked_to_root(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    runner.run(root_id)
    assert len(stage_rows(root_id)) == 3


def test_tester_and_acceptance_use_detached_exact_commit(delivery_env):
    class InspectingExecutor(FakeExecutor):
        def execute(self, **kwargs):
            if kwargs["role"] in {"tester", "acceptance"}:
                assert git(kwargs["workspace"], "rev-parse", "HEAD") == kwargs["target_commit"]
                assert git(kwargs["workspace"], "branch", "--show-current") == ""
            return super().execute(**kwargs)

    from tests.hermes_cli.test_feature_delivery_runner import git

    fake = InspectingExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    assert runner.run(root_id).current_state == "DELIVERED"


def test_run_records_no_profile_invocation(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    runner.run(root_id)
    assert all(row["assignee"] is None for row in stage_rows(root_id))


def test_root_is_only_task_with_workflow_state(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    runner.run(root_id)
    assert all(row["workflow_template_id"] is None for row in stage_rows(root_id))


def test_status_after_test_failure_reports_fix_loop(delivery_env):
    script = [("developer", "ready"), ("tester", "fail"), ("developer", "blocked")]
    runner, root_id = create(delivery_env, FakeExecutor(delivery_env.repo, script))
    result = runner.run(root_id)
    assert result.fix_loops == 1
    assert result.last_stage == "developer"


def test_report_role_in_metadata_is_validated(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    original = runner._transition

    def crash(conn, root, target, payload=None):
        if target == FeatureDeliveryState.READY_FOR_TEST:
            raise KeyboardInterrupt
        return original(conn, root, target, payload)

    runner._transition = crash
    with pytest.raises(KeyboardInterrupt):
        runner.run(root_id)
    with kb.connect() as conn:
        row = conn.execute("SELECT id, metadata FROM task_runs ORDER BY id DESC LIMIT 1").fetchone()
        metadata = json.loads(row["metadata"])
        metadata["role"] = "acceptance"
        conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(metadata), row["id"]))
        conn.commit()
    runner._transition = original
    assert "invalid_report" in runner.resume(root_id).blocked_reason
