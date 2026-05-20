import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_swarm as ks
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


def test_create_swarm_is_idempotent_after_partial_failure(tmp_path, monkeypatch):
    """A retry with the same idempotency_key after a crash between graph build
    and the topology write must recover the existing cards, not build a second
    parallel graph under the root."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        swarm_kwargs = dict(
            goal="Research two branches then verify and synthesize.",
            workers=[
                SwarmWorkerSpec(profile="a", title="Branch A", body="A"),
                SwarmWorkerSpec(profile="b", title="Branch B", body="B"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            idempotency_key="swarm-key-1",
        )

        # Simulate a partial failure: the whole graph is written, but the
        # process dies before the topology blackboard comment is posted. The
        # already-committed task rows survive (autocommit + per-call write_txn).
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated crash before topology write")

        monkeypatch.setattr(ks, "post_blackboard_update", _boom)
        with pytest.raises(RuntimeError):
            create_swarm(conn, **swarm_kwargs)
        monkeypatch.undo()

        root_id = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "ORDER BY created_at DESC LIMIT 1",
            ("swarm-key-1",),
        ).fetchone()["id"]

        # The crash left a complete graph but no topology recovery marker.
        first_workers = sorted(kb.child_ids(conn, root_id))
        assert len(first_workers) == 2
        assert "topology" not in latest_blackboard(conn, root_id)
        task_count = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
        assert task_count == 5  # root + 2 workers + verifier + synthesizer

        # Retry with the same key: must re-attach, not duplicate.
        created = create_swarm(conn, **swarm_kwargs)

        assert created.root_id == root_id
        assert sorted(created.worker_ids) == first_workers
        assert sorted(kb.child_ids(conn, root_id)) == first_workers
        # No second worker/verifier/synthesizer set was created.
        retry_count = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
        assert retry_count == 5
        # The verifier still gates on exactly the original two workers.
        assert set(kb.parent_ids(conn, created.verifier_id)) == set(first_workers)
        assert kb.parent_ids(conn, created.synthesizer_id) == [created.verifier_id]
        # Topology marker is now durably written.
        assert latest_blackboard(conn, root_id).get("topology")
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
        assert kb.get_task(conn, created.verifier_id).status == "todo"
        assert kb.get_task(conn, created.synthesizer_id).status == "todo"

        kb.complete_task(conn, created.worker_ids[1], summary="B done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.verifier_id).status == "ready"
        assert kb.get_task(conn, created.synthesizer_id).status == "todo"

        kb.complete_task(
            conn,
            created.verifier_id,
            summary="Verified both branches",
            metadata={"gate": "pass"},
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.synthesizer_id).status == "ready"
    finally:
        conn.close()
