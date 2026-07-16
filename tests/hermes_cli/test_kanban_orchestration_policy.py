from hermes_cli import kanban_db as kb
from concurrent.futures import ThreadPoolExecutor
import threading
import json
import pytest


def _policy(**overrides):
    values = {
        "allowed_assignees": ("planner", "worker"),
        "orchestrator_assignees": ("planner",),
        "max_depth": 2,
        "max_tasks": 8,
        "max_runtime_seconds": 60,
        "max_concurrency": 2,
        "max_wall_clock_seconds": 300,
        "goal_max_turns": 5,
    }
    values.update(overrides)
    return kb.OrchestrationPolicy(**values)


def test_policy_v2_added_fields_are_strict_and_canonical():
    policy = _policy()
    encoded = json.loads(policy.to_json())
    assert encoded == {
        "version": 2,
        "allowed_assignees": ["planner", "worker"],
        "orchestrator_assignees": ["planner"],
        "max_depth": 2,
        "max_tasks": 8,
        "max_runtime_seconds": 60,
        "max_concurrency": 2,
        "max_wall_clock_seconds": 300,
        "goal_max_turns": 5,
    }
    assert kb.OrchestrationPolicy.from_json(policy.to_json()) == policy

    for field, bad in (
        ("max_concurrency", True),
        ("max_concurrency", 1.0),
        ("max_concurrency", "1"),
        ("max_wall_clock_seconds", False),
        ("goal_max_turns", 2.0),
    ):
        payload = dict(encoded)
        payload[field] = bad
        with pytest.raises(ValueError, match="malformed orchestration policy"):
            kb.OrchestrationPolicy.from_json(json.dumps(payload))

    for field, bad in (
        ("max_concurrency", 0), ("max_concurrency", 33),
        ("max_wall_clock_seconds", 0), ("max_wall_clock_seconds", 86401),
        ("goal_max_turns", 0), ("goal_max_turns", 21),
    ):
        with pytest.raises(ValueError):
            _policy(**{field: bad})


def test_policy_a1_json_has_deliberate_bounded_upgrade():
    a1 = json.dumps({
        "allowed_assignees": ["planner"],
        "orchestrator_assignees": ["planner"],
        "max_depth": 1,
        "max_tasks": 2,
        "max_runtime_seconds": 3,
    })
    upgraded = kb.OrchestrationPolicy.from_json(a1)
    assert upgraded.max_concurrency == 1
    assert upgraded.max_wall_clock_seconds == 86400
    assert upgraded.goal_max_turns == 20
    assert json.loads(upgraded.to_json())["version"] == 2


def test_policy_a1_root_replay_compares_upgraded_policy_semantics(tmp_path):
    a1 = json.dumps(
        {
            "allowed_assignees": ["planner"],
            "orchestrator_assignees": ["planner"],
            "max_depth": 1,
            "max_tasks": 2,
            "max_runtime_seconds": 3,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    upgraded = kb.OrchestrationPolicy.from_json(a1)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root = kb.create_task(
            conn,
            title="root",
            assignee="planner",
            idempotency_key="a1-root",
            orchestration_policy=upgraded,
        )
        conn.execute("UPDATE tasks SET orchestration_policy = ? WHERE id = ?", (a1, root))
        replay = kb.create_task(
            conn,
            title="root replay",
            assignee="planner",
            idempotency_key="a1-root",
            orchestration_policy=upgraded,
        )
        assert replay == root
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    finally:
        conn.close()


def test_policy_child_inherits_goal_turn_budget(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root = kb.create_task(conn, title="root", assignee="planner", orchestration_policy=_policy())
        authority = _claim_authority(conn, root)
        child = _bounded_create(
            conn, authority, title="child", assignee="worker", goal_mode=True,
            goal_max_turns=20,
        )
        assert kb.get_task(conn, child).goal_max_turns == 5
    finally:
        conn.close()


def test_policy_root_assignee_must_be_an_orchestrator(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        with pytest.raises(ValueError, match="root assignee"):
            kb.create_task(
                conn,
                title="unusable root",
                assignee="worker",
                orchestration_policy=_policy(),
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    finally:
        conn.close()


def test_program_deadline_rejects_create_and_claim_but_allows_exact_replay(tmp_path, monkeypatch):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root = kb.create_task(conn, title="root", assignee="planner", orchestration_policy=_policy(max_wall_clock_seconds=10))
        authority = _claim_authority(conn, root)
        child = _bounded_create(conn, authority, title="child", assignee="worker", idempotency_key="child-key")
        monkeypatch.setattr(kb.time, "time", lambda: now + 11)
        assert _bounded_create(conn, authority, title="replay", assignee="worker", idempotency_key="child-key") == child
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="deadline"):
            _bounded_create(conn, authority, title="late", assignee="worker", idempotency_key="new-key")
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before
        kb.complete_task(conn, root, result="delegated", expected_run_id=authority.run_id)
        assert kb.claim_task(conn, child) is None
        expired_child = kb.get_task(conn, child)
        assert expired_child is not None
        assert expired_child.status == "blocked"
        assert kb.recompute_ready(conn) == 0
        expired_child = kb.get_task(conn, child)
        assert expired_child is not None
        assert expired_child.status == "blocked"
    finally:
        conn.close()


def test_program_deadline_terminates_running_worker_without_requeue(tmp_path, monkeypatch):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root = kb.create_task(
            conn,
            title="root",
            assignee="planner",
            orchestration_policy=_policy(
                max_runtime_seconds=3600,
                max_wall_clock_seconds=10,
            ),
        )
        claimed = kb.claim_task(conn, root, claimer=f"{kb._claimer_id().split(':', 1)[0]}:deadline")
        assert claimed is not None
        conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (999999, root))
        conn.execute(
            "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
            (999999, claimed.current_run_id),
        )
        monkeypatch.setattr(kb.time, "time", lambda: now + 11)
        signals = []
        assert kb.enforce_max_runtime(
            conn,
            signal_fn=lambda pid, sig: signals.append((pid, sig)),
        ) == [root]
        assert signals
        task = kb.get_task(conn, root)
        assert task is not None
        assert task.status == "blocked"
        run = kb.latest_run(conn, root)
        assert run is not None
        assert run.status == "timed_out"
        assert run.outcome == "timed_out"
        assert kb.recompute_ready(conn) == 0
        task = kb.get_task(conn, root)
        assert task is not None
        assert task.status == "blocked"
    finally:
        conn.close()


def test_program_max_concurrency_is_atomic_and_ignores_ended_runs(tmp_path):
    path = tmp_path / "kanban.db"
    conn = kb.connect(path)
    try:
        root = kb.create_task(conn, title="root", assignee="planner", orchestration_policy=_policy(max_concurrency=1))
        authority = _claim_authority(conn, root)
        first = _bounded_create(conn, authority, title="first", assignee="worker")
        second = _bounded_create(conn, authority, title="second", assignee="worker")
        kb.complete_task(conn, root, result="delegated", expected_run_id=authority.run_id)
    finally:
        conn.close()

    barrier = threading.Barrier(2)
    def claim(task_id):
        c = kb.connect(path)
        try:
            barrier.wait()
            return kb.claim_task(c, task_id)
        finally:
            c.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, (first, second)))
    assert sum(result is not None for result in results) == 1

    winner = next(result for result in results if result is not None)
    loser = second if winner.id == first else first
    conn = kb.connect(path)
    try:
        # Expired-but-not-ended runs are stale and must not hold a slot.
        conn.execute(
            "UPDATE task_runs SET claim_expires=? WHERE id=?",
            (int(kb.time.time()) - 1, winner.current_run_id),
        )
        assert kb.claim_task(conn, loser) is not None
        loser_task = kb.get_task(conn, loser)
        assert kb.complete_task(
            conn, loser, result="done", expected_run_id=loser_task.current_run_id
        )
        # Ended history is likewise free, even while the stale winner row
        # remains marked running for a later reclamation pass.
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL WHERE id=?",
            (loser,),
        )
        assert kb.claim_task(conn, loser) is not None
    finally:
        conn.close()


def test_legacy_unmanaged_claim_and_create_remain_unbounded(tmp_path, monkeypatch):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        parent = kb.create_task(conn, title="legacy", assignee="planner")
        monkeypatch.setattr(kb.time, "time", lambda: now + 100_000)
        assert kb.claim_task(conn, parent) is not None
        child = kb.create_task(conn, title="legacy child", assignee="worker", current_orchestrator_task_id=parent)
        assert kb.get_task(conn, child).orchestration_policy is None
    finally:
        conn.close()


def _claim_authority(conn, task_id):
    claimed = kb.claim_task(conn, task_id, ttl_seconds=300)
    assert claimed is not None
    return kb.CreationAuthority(
        task_id=claimed.id,
        run_id=claimed.current_run_id,
        claim_lock=claimed.claim_lock,
        actor_profile=claimed.assignee,
    )


def _bounded_create(conn, authority, **kwargs):
    return kb.create_task(
        conn,
        current_orchestrator_task_id=authority.task_id,
        creation_authority=authority,
        **kwargs,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"allowed_assignees": ["planner"], "orchestrator_assignees": ["planner"], "max_depth": True, "max_tasks": 2, "max_runtime_seconds": 3},
        {"allowed_assignees": ["planner"], "orchestrator_assignees": ["planner"], "max_depth": 1.0, "max_tasks": 2, "max_runtime_seconds": 3},
        {"allowed_assignees": ["planner"], "orchestrator_assignees": ["planner"], "max_depth": "1", "max_tasks": 2, "max_runtime_seconds": 3},
        {"allowed_assignees": "planner", "orchestrator_assignees": ["planner"], "max_depth": 1, "max_tasks": 2, "max_runtime_seconds": 3},
        {"allowed_assignees": [True], "orchestrator_assignees": [True], "max_depth": 1, "max_tasks": 2, "max_runtime_seconds": 3},
        {"allowed_assignees": ["planner"], "orchestrator_assignees": "planner", "max_depth": 1, "max_tasks": 2, "max_runtime_seconds": 3},
        {"allowed_assignees": ["worker"], "orchestrator_assignees": ["planner"], "max_depth": 1, "max_tasks": 2, "max_runtime_seconds": 3},
        {"allowed_assignees": ["planner"], "max_depth": 1, "max_tasks": 2, "max_runtime_seconds": 3},
        {"allowed_assignees": ["planner"], "orchestrator_assignees": ["planner"], "max_depth": 1, "max_tasks": 2, "max_runtime_seconds": 3, "extra": 4},
    ],
)
def test_policy_json_is_strict(payload):
    with pytest.raises(ValueError, match="malformed orchestration policy"):
        kb.OrchestrationPolicy.from_json(json.dumps(payload))


def test_policy_root_idempotency_is_scope_aware(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker"),
            orchestrator_assignees=("planner",),
            max_depth=2,
            max_tasks=4,
            max_runtime_seconds=60,
        )
        legacy = kb.create_task(
            conn,
            title="legacy",
            assignee="worker",
            idempotency_key="shared-key",
        )
        with pytest.raises(ValueError, match="idempotency key belongs to another program"):
            kb.create_task(
                conn,
                title="bounded root",
                assignee="planner",
                idempotency_key="shared-key",
                orchestration_policy=policy,
            )
        legacy_task = kb.get_task(conn, legacy)
        assert legacy_task is not None
        assert legacy_task.orchestration_policy is None

        root = kb.create_task(
            conn,
            title="bounded root",
            assignee="planner",
            tenant="tenant-a",
            idempotency_key="root-key",
            orchestration_policy=policy,
        )
        replay = kb.create_task(
            conn,
            title="bounded root retry",
            assignee="planner",
            tenant="tenant-a",
            idempotency_key="root-key",
            orchestration_policy=policy,
        )
        assert replay == root

        wider = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker", "other"),
            orchestrator_assignees=("planner",),
            max_depth=2,
            max_tasks=4,
            max_runtime_seconds=60,
        )
        for changed in (
            {"orchestration_policy": wider, "tenant": "tenant-a", "assignee": "planner"},
            {"orchestration_policy": policy, "tenant": "tenant-b", "assignee": "planner"},
            {"orchestration_policy": policy, "tenant": "tenant-a", "assignee": "worker"},
        ):
            with pytest.raises(ValueError, match="idempotency key belongs to another program"):
                kb.create_task(
                    conn,
                    title="conflicting root retry",
                    idempotency_key="root-key",
                    **changed,
                )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2
    finally:
        conn.close()


def test_only_policy_orchestrators_can_fan_out(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "architect", "worker"),
            orchestrator_assignees=("planner", "architect"),
            max_depth=3,
            max_tasks=8,
            max_runtime_seconds=60,
        )
        root = kb.create_task(conn, title="root", assignee="planner", orchestration_policy=policy)
        root_auth = _claim_authority(conn, root)
        worker = _bounded_create(conn, root_auth, title="ordinary", assignee="worker")
        architect = _bounded_create(conn, root_auth, title="approved", assignee="architect")
        assert kb.complete_task(conn, root, result="delegated", expected_run_id=root_auth.run_id)

        worker_auth = _claim_authority(conn, worker)
        with pytest.raises(ValueError, match="not allowed to orchestrate"):
            _bounded_create(conn, worker_auth, title="recursive escape", assignee="worker")
        assert kb.complete_task(conn, worker, result="done", expected_run_id=worker_auth.run_id)

        architect_auth = _claim_authority(conn, architect)
        child = _bounded_create(conn, architect_auth, title="approved fanout", assignee="worker")
        assert kb.get_task(conn, child).orchestration_parent_id == architect
    finally:
        conn.close()


def test_dependency_fan_in_keeps_current_lineage_parent(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("architect", "worker"),
            orchestrator_assignees=("architect",),
            max_depth=3,
            max_tasks=8,
            max_runtime_seconds=60,
        )
        root = kb.create_task(conn, title="root", assignee="architect", orchestration_policy=policy)
        root_auth = _claim_authority(conn, root)
        orchestrator = _bounded_create(conn, root_auth, title="next architect", assignee="architect")
        sibling = _bounded_create(conn, root_auth, title="dependency sibling", assignee="worker")
        assert kb.complete_task(conn, root, result="delegated", expected_run_id=root_auth.run_id)
        sibling_auth = _claim_authority(conn, sibling)
        assert kb.complete_task(conn, sibling, result="done", expected_run_id=sibling_auth.run_id)
        orchestrator_auth = _claim_authority(conn, orchestrator)

        fan_in = _bounded_create(
            conn,
            orchestrator_auth,
            title="fan in",
            assignee="worker",
            parents=[sibling],
        )
        task = kb.get_task(conn, fan_in)
        assert kb.parent_ids(conn, fan_in) == [sibling]
        assert task.orchestration_parent_id == orchestrator
        assert task.orchestration_depth == 2
    finally:
        conn.close()


def test_bounded_project_slug_is_canonical_and_unresolved_fails_closed(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import projects_db

    repo = tmp_path / "repo"
    repo.mkdir()
    with projects_db.connect_closing() as project_conn:
        project_id = projects_db.create_project(
            project_conn, name="Real Project", slug="real-project", primary_path=str(repo)
        )

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker"),
            orchestrator_assignees=("planner",),
            max_depth=2,
            max_tasks=4,
            max_runtime_seconds=60,
        )
        root = kb.create_task(
            conn, title="root", assignee="planner", project_id=project_id,
            orchestration_policy=policy,
        )
        authority = _claim_authority(conn, root)
        child = _bounded_create(
            conn, authority, title="slug child", assignee="worker", project_id="real-project"
        )
        assert kb.get_task(conn, child).project_id == project_id
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="project does not match"):
            _bounded_create(
                conn, authority, title="bad project", assignee="worker", project_id="missing"
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before
    finally:
        conn.close()


def test_bounded_child_requires_exact_active_run_authority(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker"),
            orchestrator_assignees=("planner",),
            max_depth=2,
            max_tasks=4,
            max_runtime_seconds=60,
        )
        root_id = kb.create_task(
            conn, title="root", assignee="planner", orchestration_policy=policy
        )
        authority = _claim_authority(conn, root_id)

        for forged in (
            None,
            kb.CreationAuthority(root_id, authority.run_id + 1, authority.claim_lock, "planner"),
            kb.CreationAuthority(root_id, authority.run_id, "forged", "planner"),
            kb.CreationAuthority(root_id, authority.run_id, authority.claim_lock, "worker"),
        ):
            before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            with pytest.raises(ValueError, match="active orchestration authority"):
                kb.create_task(
                    conn,
                    title="must not exist",
                    assignee="worker",
                    current_orchestrator_task_id=root_id,
                    creation_authority=forged,
                )
            assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before

        child_id = kb.create_task(
            conn,
            title="valid child",
            assignee="worker",
            current_orchestrator_task_id=root_id,
            creation_authority=authority,
        )
        assert kb.get_task(conn, child_id) is not None

        conn.execute("UPDATE task_runs SET status = 'done' WHERE id = ?", (authority.run_id,))
        with pytest.raises(ValueError, match="active orchestration authority"):
            kb.create_task(
                conn,
                title="stale child",
                assignee="worker",
                current_orchestrator_task_id=root_id,
                creation_authority=authority,
            )
    finally:
        conn.close()


def test_orchestrator_child_inherits_program_policy_and_scope(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker"),
            orchestrator_assignees=("planner",),
            max_depth=3,
            max_tasks=8,
            max_runtime_seconds=900,
        )
        root_id = kb.create_task(
            conn,
            title="Program root",
            assignee="planner",
            tenant="tenant-a",
            project_id=None,
            orchestration_policy=policy,
        )
        conn.execute("UPDATE tasks SET project_id = 'project-a' WHERE id = ?", (root_id,))
        authority = _claim_authority(conn, root_id)

        child_id = _bounded_create(
            conn,
            authority,
            title="Delegated work",
            assignee="worker",
        )

        root = kb.get_task(conn, root_id)
        child = kb.get_task(conn, child_id)
        assert root.orchestration_root_id == root_id
        assert root.orchestration_depth == 0
        assert root.orchestration_policy == policy
        assert child.orchestration_root_id == root_id
        assert child.orchestration_depth == 1
        assert child.orchestration_policy == policy
        assert child.tenant == root.tenant
        assert child.project_id == root.project_id
        assert kb.parent_ids(conn, child_id) == [root_id]
        assert child.max_runtime_seconds == policy.max_runtime_seconds
    finally:
        conn.close()

def test_program_task_cap_is_atomic_and_idempotent_replay_is_free(tmp_path):
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    try:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker"),
            orchestrator_assignees=("planner",),
            max_depth=2,
            max_tasks=2,
            max_runtime_seconds=60,
        )
        root_id = kb.create_task(
            conn, title="root", assignee="planner", orchestration_policy=policy
        )
        authority = _claim_authority(conn, root_id)
    finally:
        conn.close()

    barrier = threading.Barrier(2)

    def create_child(key):
        thread_conn = kb.connect(db_path)
        try:
            barrier.wait()
            try:
                return kb.create_task(
                    thread_conn,
                    title=key,
                    assignee="worker",
                    current_orchestrator_task_id=root_id,
                    creation_authority=authority,
                    idempotency_key=key,
                )
            except ValueError as exc:
                return str(exc)
        finally:
            thread_conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create_child, ("child-a", "child-b")))

    ids = [result for result in results if result.startswith("t_")]
    errors = [result for result in results if not result.startswith("t_")]
    assert len(ids) == 1
    assert errors == ["maximum orchestration task count exceeded"]

    conn = kb.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2
        winning_key = "child-a" if results[0].startswith("t_") else "child-b"
        replay = kb.create_task(
            conn,
            title="ignored replay title",
            assignee="worker",
            current_orchestrator_task_id=root_id,
            creation_authority=authority,
            idempotency_key=winning_key,
        )
        assert replay == ids[0]
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("assignee", "root_depth", "message"),
    [
        ("intruder", 0, "assignee is not allowed"),
        ("worker", 1, "maximum orchestration depth"),
    ],
)
def test_orchestrator_rejects_assignee_and_depth_before_insert(
    tmp_path, assignee, root_depth, message
):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker"),
            orchestrator_assignees=("planner",),
            max_depth=1,
            max_tasks=4,
            max_runtime_seconds=60,
        )
        root_id = kb.create_task(
            conn, title="root", assignee="planner", orchestration_policy=policy
        )
        authority = _claim_authority(conn, root_id)
        authority_id = root_id
        if root_depth:
            authority_id = _bounded_create(
                conn,
                authority,
                title="level one",
                assignee="planner",
            )
            assert kb.complete_task(
                conn, root_id, result="delegated", expected_run_id=authority.run_id
            )
            authority = _claim_authority(conn, authority_id)
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

        with pytest.raises(ValueError, match=message):
            kb.create_task(
                conn,
                title="must not exist",
                assignee=assignee,
                current_orchestrator_task_id=authority_id,
                creation_authority=authority,
            )

        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before
    finally:
        conn.close()


def test_child_cannot_widen_or_cross_program_scope(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker"),
            orchestrator_assignees=("planner",),
            max_depth=2,
            max_tasks=6,
            max_runtime_seconds=60,
        )
        root_id = kb.create_task(
            conn,
            title="root",
            assignee="planner",
            tenant="tenant-a",
            orchestration_policy=policy,
        )
        foreign_id = kb.create_task(
            conn, title="foreign", assignee="worker", idempotency_key="foreign-key"
        )
        authority = _claim_authority(conn, root_id)

        with pytest.raises(ValueError, match="tenant does not match"):
            kb.create_task(
                conn,
                title="tenant escape",
                assignee="worker",
                tenant="tenant-b",
                current_orchestrator_task_id=root_id,
                creation_authority=authority,
            )
        with pytest.raises(ValueError, match="outside orchestration program"):
            kb.create_task(
                conn,
                title="foreign parent",
                assignee="worker",
                parents=[foreign_id],
                current_orchestrator_task_id=root_id,
                creation_authority=authority,
            )
        with pytest.raises(ValueError, match="idempotency key belongs to another program"):
            kb.create_task(
                conn,
                title="foreign replay",
                assignee="worker",
                idempotency_key="foreign-key",
                current_orchestrator_task_id=root_id,
                creation_authority=authority,
            )
        wider = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker", "intruder"),
            orchestrator_assignees=("planner",),
            max_depth=3,
            max_tasks=10,
            max_runtime_seconds=120,
        )
        with pytest.raises(ValueError, match="cannot supply orchestration policy"):
            kb.create_task(
                conn,
                title="policy escape",
                assignee="worker",
                orchestration_policy=wider,
                current_orchestrator_task_id=root_id,
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2
    finally:
        conn.close()


def test_legacy_tasks_keep_unrestricted_creation_behavior(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        parent_id = kb.create_task(conn, title="legacy parent", assignee="old")
        child_id = kb.create_task(
            conn,
            title="legacy child",
            assignee="any-profile",
            parents=[parent_id],
            tenant="independent",
            max_runtime_seconds=123,
        )
        child = kb.get_task(conn, child_id)
        assert child.orchestration_policy is None
        assert child.orchestration_root_id is None
        assert child.orchestration_depth is None
        assert child.max_runtime_seconds == 123
        assert child.tenant == "independent"
    finally:
        conn.close()


def test_kanban_create_derives_authority_from_current_worker_context(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    try:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker"),
            orchestrator_assignees=("planner",),
            max_depth=2,
            max_tasks=3,
            max_runtime_seconds=60,
        )
        root_id = kb.create_task(
            conn, title="root", assignee="planner", orchestration_policy=policy
        )
        authority = _claim_authority(conn, root_id)
    finally:
        conn.close()

    from tools import kanban_tools

    monkeypatch.setenv("HERMES_KANBAN_TASK", root_id)
    monkeypatch.setenv("HERMES_PROFILE", "planner")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(authority.run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", authority.claim_lock)
    monkeypatch.setattr(
        kanban_tools, "_connect", lambda **_kwargs: (kb, kb.connect(db_path))
    )
    result = json.loads(
        kanban_tools._handle_create({"title": "child", "assignee": "worker"})
    )
    assert "error" not in result

    conn = kb.connect(db_path)
    try:
        child = kb.get_task(conn, result["task_id"])
        assert child.orchestration_root_id == root_id
        assert child.orchestration_depth == 1
        assert child.orchestration_policy == policy
    finally:
        conn.close()


def test_malformed_and_cross_board_authority_fail_closed(tmp_path, monkeypatch):
    source_path = tmp_path / "source.db"
    foreign_path = tmp_path / "foreign.db"
    conn = kb.connect(source_path)
    try:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker"),
            orchestrator_assignees=("planner",),
            max_depth=2,
            max_tasks=3,
            max_runtime_seconds=60,
        )
        root_id = kb.create_task(
            conn, title="root", assignee="planner", orchestration_policy=policy
        )
        conn.execute(
            "UPDATE tasks SET orchestration_policy = '{bad json' WHERE id = ?",
            (root_id,),
        )
        with pytest.raises(ValueError, match="malformed orchestration policy"):
            kb.create_task(
                conn,
                title="malformed escape",
                assignee="worker",
                current_orchestrator_task_id=root_id,
            )
    finally:
        conn.close()

    from tools import kanban_tools

    monkeypatch.setenv("HERMES_KANBAN_TASK", root_id)
    monkeypatch.setattr(
        kanban_tools, "_connect", lambda **_kwargs: (kb, kb.connect(foreign_path))
    )
    result = json.loads(
        kanban_tools._handle_create(
            {"title": "cross-board escape", "assignee": "worker", "board": "foreign"}
        )
    )
    assert "current task has no orchestration authority" in result["error"]
    conn = kb.connect(foreign_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    finally:
        conn.close()
