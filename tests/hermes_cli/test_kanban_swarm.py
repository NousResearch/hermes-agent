import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_swarm import (
    SwarmWorkerSpec,
    create_swarm,
    latest_blackboard,
    post_blackboard_update,
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
        monkeypatch.setattr(kb, "complete_task", lambda *args, **kwargs: False)
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
        monkeypatch.setattr(kb, "complete_task", original_complete)
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
