import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_swarm import (
    SwarmWorkerSpec,
    create_swarm,
    latest_blackboard,
    post_blackboard_update,
    TOPOLOGY_KEY,
    worker_key,
    worker_result_key,
    worker_status_key,
    worker_handoff_key,
    team_key,
    coordinator_key,
    parse_worker_key,
    parse_namespaced_key,
    worker_prefix,
    worker_fields,
    get_worker_field,
)


def test_create_swarm_builds_parallel_workers_verifier_and_synthesizer(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Map the target market and produce a decision memo.",
            workers=[
                SwarmWorkerSpec(profile="researcher-a", title="Market scan", body="Find competitors"),
                SwarmWorkerSpec(profile="researcher-b", title="Customer scan", body="Find customer pains"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            tenant="intel",
            created_by="orchestrator",
        )

        root = kb.get_task(conn, created.root_id)
        workers = [kb.get_task(conn, tid) for tid in created.worker_ids]
        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)

        assert root is not None
        assert all(task is not None for task in workers)
        workers = [task for task in workers if task is not None]
        assert verifier is not None
        assert synthesizer is not None
        assert root.status == "done"
        assert root.assignee == "orchestrator"
        assert [task.status for task in workers] == ["ready", "ready"]
        assert [task.assignee for task in workers] == ["researcher-a", "researcher-b"]
        assert verifier.status == "todo"
        assert synthesizer.status == "todo"
        assert set(kb.parent_ids(conn, created.verifier_id)) == set(created.worker_ids)
        assert kb.parent_ids(conn, created.synthesizer_id) == [created.verifier_id]
        assert all(created.root_id in (task.body or "") for task in workers)
    finally:
        conn.close()


def test_create_swarm_graph_is_atomic_and_rolls_back_partial_build(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    db_path = tmp_path / "kanban.db"
    writer = kb.connect(db_path)
    reader = kb.connect(db_path)
    original_create = kb.create_task
    original_complete = kb.complete_task
    calls = 0

    def observed_create(*args, **kwargs):
        nonlocal calls
        calls += 1
        task_id = original_create(*args, **kwargs)
        if calls == 1:
            # Releasing the nested create_task savepoint must not expose the
            # root before the whole graph's outer transaction commits.
            visible = reader.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            assert visible == 0
        if calls == 3:
            raise RuntimeError("synthetic graph-construction failure")
        return task_id

    monkeypatch.setattr(kb, "create_task", observed_create)
    try:
        with pytest.raises(RuntimeError, match="synthetic graph-construction failure"):
            create_swarm(
                writer,
                goal="Build atomically",
                workers=[
                    SwarmWorkerSpec(profile="worker-a", title="A", body="A"),
                    SwarmWorkerSpec(profile="worker-b", title="B", body="B"),
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
            )
        assert writer.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        assert reader.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0

        monkeypatch.setattr(kb, "create_task", original_create)
        import hermes_cli.kanban_swarm as ks

        original_activate = ks._activate_root_inline
        monkeypatch.setattr(
            ks, "_activate_root_inline", lambda *args, **kwargs: False
        )
        with pytest.raises(RuntimeError, match="could not activate"):
            create_swarm(
                writer,
                goal="Fail activation atomically",
                workers=[
                    SwarmWorkerSpec(profile="worker-a", title="A", body="A"),
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
            )
        assert writer.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        assert reader.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0

        hooks: list[tuple[str, bool]] = []
        monkeypatch.setattr(ks, "_activate_root_inline", original_activate)
        monkeypatch.setattr(
            kb,
            "_fire_kanban_lifecycle_hook",
            lambda event, *_args, **_kwargs: hooks.append(
                (event, writer.in_transaction)
            ),
        )
        create_swarm(
            writer,
            goal="Commit before lifecycle hook",
            workers=[SwarmWorkerSpec(profile="worker-a", title="A", body="A")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        assert hooks == [("kanban_task_completed", False)]
    finally:
        reader.close()
        writer.close()


def test_plain_write_txn_nesting_raises_and_allow_nested_composes(tmp_path):
    """B1 regression: nesting is explicit opt-in, never silent.

    Plain ``write_txn`` inside an open transaction must raise loudly (the
    historical invariant). ``allow_nested=True`` composes via a savepoint,
    and an outer rollback discards the inner work without any post-commit
    side effects having fired (the workspace directory survives).
    """
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        workspace = tmp_path / "scratch-ws"
        workspace.mkdir()
        tid = kb.create_task(conn, title="ws task", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_path = ? WHERE id = ?",
                (str(workspace), tid),
            )

        # 1) Plain nesting raises loudly.
        with pytest.raises(RuntimeError, match="already inside a transaction"):
            with kb.write_txn(conn):
                with kb.write_txn(conn):
                    pass
        assert not conn.in_transaction

        # 2) allow_nested composes; outer rollback discards inner work
        #    and no side effects (workspace cleanup) fired meanwhile.
        with pytest.raises(RuntimeError, match="outer failure"):
            with kb.write_txn(conn):
                with kb.write_txn(conn, allow_nested=True):
                    conn.execute(
                        "UPDATE tasks SET status = 'done' WHERE id = ?", (tid,)
                    )
                    kb._append_event(conn, tid, "completed", {"result_len": 0})
                # Inner savepoint released, but the outer txn now fails.
                raise RuntimeError("outer failure")
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"  # inner 'done' flip was discarded
        assert not any(
            e.kind == "completed" for e in kb.list_events(conn, tid)
        )
        assert workspace.is_dir()  # no _cleanup_workspace side effect fired
    finally:
        conn.close()


def test_swarm_blackboard_merges_structured_updates(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Collect evidence.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Evidence", body="Find proof")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        post_blackboard_update(
            conn,
            created.root_id,
            author="researcher",
            key="sources",
            value=["https://example.com/a"],
        )
        post_blackboard_update(
            conn,
            created.root_id,
            author="reviewer",
            key="risks",
            value={"missing_primary_source": True},
        )

        board = latest_blackboard(conn, created.root_id)
        assert board["sources"] == ["https://example.com/a"]
        assert board["risks"] == {"missing_primary_source": True}
        assert board["_authors"]["sources"] == "researcher"
    finally:
        conn.close()


def test_swarm_verifier_and_synthesis_are_dependency_gated(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Research two branches then verify and synthesize.",
            workers=[
                SwarmWorkerSpec(profile="a", title="Branch A", body="A"),
                SwarmWorkerSpec(profile="b", title="Branch B", body="B"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        kb.complete_task(
            conn,
            created.worker_ids[0],
            summary="A done",
            metadata={"confidence": 0.8},
        )
        kb.recompute_ready(conn)
        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)
        assert verifier is not None
        assert synthesizer is not None
        assert verifier.status == "todo"
        assert synthesizer.status == "todo"

        kb.complete_task(conn, created.worker_ids[1], summary="B done")
        kb.recompute_ready(conn)
        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)
        assert verifier is not None
        assert synthesizer is not None
        assert verifier.status == "ready"
        assert synthesizer.status == "todo"

        kb.complete_task(
            conn,
            created.verifier_id,
            summary="Verified both branches",
            metadata={"gate": "pass"},
        )
        kb.recompute_ready(conn)
        synthesizer = kb.get_task(conn, created.synthesizer_id)
        assert synthesizer is not None
        assert synthesizer.status == "ready"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Structured key helpers
# ---------------------------------------------------------------------------


def test_worker_key_builds_namespaced_string():
    assert worker_key(0, "status") == "worker:0:status"
    assert worker_key(3, "result") == "worker:3:result"
    assert worker_key(7, "handoff") == "worker:7:handoff"


def test_worker_result_and_status_and_handoff_shortcuts():
    assert worker_result_key(2) == "worker:2:result"
    assert worker_status_key(2) == "worker:2:status"
    assert worker_handoff_key(5) == "worker:5:handoff"


def test_parse_worker_key_returns_index_and_field():
    assert parse_worker_key("worker:3:status") == (3, "status")
    assert parse_worker_key("worker:0:result") == (0, "result")
    assert parse_worker_key("worker:12:handoff") == (12, "handoff")


def test_parse_worker_key_rejects_non_worker_keys():
    assert parse_worker_key("topology") is None
    assert parse_worker_key("team:sources") is None
    assert parse_worker_key("coordinator:decision") is None
    assert parse_worker_key("worker:abc:status") is None  # non-int index
    assert parse_worker_key("worker:3") is None           # no field
    assert parse_worker_key("worker:3:") is None          # empty field
    assert parse_worker_key("worker::status") is None     # empty index


def test_parse_namespaced_key_detects_namespace():
    assert parse_namespaced_key("worker:3:status") == ("worker", "status")
    assert parse_namespaced_key("team:sources") == ("team", "sources")
    assert parse_namespaced_key("coordinator:decision") == ("coordinator", "decision")
    assert parse_namespaced_key("topology") is None
    assert parse_namespaced_key("bareword") is None


def test_team_and_coordinator_key_builders():
    assert team_key("sources") == "team:sources"
    assert team_key("risks") == "team:risks"
    assert coordinator_key("decision") == "coordinator:decision"


def test_worker_prefix():
    assert worker_prefix(3) == "worker:3:"
    assert worker_prefix(0) == "worker:0:"


def test_worker_fields_extracts_all_fields_for_worker():
    board = {
        "worker:3:status": "running",
        "worker:3:result": "passed",
        "worker:5:status": "done",
        "topology": {"worker_ids": ["id-3", "id-5"]},
    }
    fields = worker_fields(board, 3)
    assert fields == {"status": "running", "result": "passed"}
    # worker 5 has no overlapping keys
    assert worker_fields(board, 5) == {"status": "done"}
    # worker that never wrote is empty
    assert worker_fields(board, 99) == {}


def test_get_worker_field_reads_single_value():
    board = {"worker:3:status": "running", "worker:3:result": "passed"}
    assert get_worker_field(board, 3, "status") == "running"
    assert get_worker_field(board, 3, "result") == "passed"
    assert get_worker_field(board, 3, "heartbeat") is None
    assert get_worker_field(board, 99, "status") is None


# ---------------------------------------------------------------------------
# Structured keys wired into the swarm blackboard
# ---------------------------------------------------------------------------


def test_swarm_blackboard_uses_structured_worker_keys(tmp_path):
    """Workers can write status/result via structured keys and read them back."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Write structured status.",
            workers=[SwarmWorkerSpec(profile="worker-a", title="Task A", body="Do A")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        # Worker 0 writes its status
        post_blackboard_update(
            conn,
            created.root_id,
            author="worker-a",
            key=worker_status_key(0),
            value="running",
        )
        post_blackboard_update(
            conn,
            created.root_id,
            author="worker-a",
            key=worker_result_key(0),
            value={"ok": True, "output": "done"},
        )

        board = latest_blackboard(conn, created.root_id)

        # Non-worker keys ("topology") still present
        assert TOPOLOGY_KEY in board
        topology = board[TOPOLOGY_KEY]
        assert isinstance(topology, dict)
        assert topology.get("goal") == "Write structured status."

        # Worker keys readable via structured access
        assert get_worker_field(board, 0, "status") == "running"
        assert get_worker_field(board, 0, "result") == {"ok": True, "output": "done"}

        # Parse the keys from the merged board
        parsed = {k: parse_worker_key(k) for k in board if k.startswith("worker:")}
        assert parsed[worker_status_key(0)] == (0, "status")
        assert parsed[worker_result_key(0)] == (0, "result")
    finally:
        conn.close()
