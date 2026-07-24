
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


def test_create_swarm_honors_custom_and_disabled_stage_skills(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        default = create_swarm(
            conn,
            goal="Use default stage skills.",
            workers=[SwarmWorkerSpec(profile="worker", title="Worker", body="Do work")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        custom = create_swarm(
            conn,
            goal="Use custom stage skills.",
            workers=[SwarmWorkerSpec(profile="worker", title="Worker", body="Do work")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            verifier_skills=["qa-reviewer"],
            synthesizer_skills=["writer"],
        )
        disabled = create_swarm(
            conn,
            goal="Disable default stage skills.",
            workers=[SwarmWorkerSpec(profile="worker", title="Worker", body="Do work")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            verifier_skills=[],
            synthesizer_skills=[],
        )

        default_verifier = kb.get_task(conn, default.verifier_id)
        default_synthesizer = kb.get_task(conn, default.synthesizer_id)
        custom_verifier = kb.get_task(conn, custom.verifier_id)
        custom_synthesizer = kb.get_task(conn, custom.synthesizer_id)
        disabled_verifier = kb.get_task(conn, disabled.verifier_id)
        disabled_synthesizer = kb.get_task(conn, disabled.synthesizer_id)

        assert default_verifier is not None
        assert default_synthesizer is not None
        assert custom_verifier is not None
        assert custom_synthesizer is not None
        assert disabled_verifier is not None
        assert disabled_synthesizer is not None
        assert default_verifier.skills == ["requesting-code-review"]
        assert default_synthesizer.skills == ["humanizer"]
        assert custom_verifier.skills == ["qa-reviewer"]
        assert custom_synthesizer.skills == ["writer"]
        assert disabled_verifier.skills == []
        assert disabled_synthesizer.skills == []
    finally:
        conn.close()
