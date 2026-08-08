
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


# ─────────────────────────────────────────────────────────────────────────
# #34273 — per-swarm verifier / synthesizer body + skills overrides
# ─────────────────────────────────────────────────────────────────────────


def test_create_swarm_uses_default_verifier_body_when_unset(tmp_path):
    """Backward compat: with no overrides, the verifier body matches the
    historical code-review default."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="goal",
            workers=[SwarmWorkerSpec(profile="w", title="t", body="b")],
            verifier_assignee="v",
            synthesizer_assignee="s",
            created_by="orch",
        )
        verifier = kb.get_task(conn, created.verifier_id)
        assert "Review every worker handoff" in verifier.body
        # And the historical skill is still attached.
        assert "requesting-code-review" in (verifier.skills or [])
    finally:
        conn.close()


def test_create_swarm_accepts_custom_verifier_body(tmp_path):
    """#34273: caller can supply a custom verifier body (e.g. 'run
    merge_scraped.py'). The swarm context suffix is still appended."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        custom = "Run merge_scraped.py to combine outputs. Validate. Gate pass/block."
        created = create_swarm(
            conn,
            goal="goal",
            workers=[SwarmWorkerSpec(profile="w", title="t", body="b")],
            verifier_assignee="v",
            synthesizer_assignee="s",
            verifier_body=custom,
            created_by="orch",
        )
        verifier = kb.get_task(conn, created.verifier_id)
        # Custom body present.
        assert "Run merge_scraped.py" in verifier.body
        # Swarm context suffix is still appended (root_id reference).
        assert created.root_id in verifier.body
        # Original code-review default phrasing is NOT present.
        assert "Review every worker handoff" not in verifier.body
    finally:
        conn.close()


def test_create_swarm_accepts_custom_verifier_skills(tmp_path):
    """#34273: caller can override verifier skills."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="goal",
            workers=[SwarmWorkerSpec(profile="w", title="t", body="b")],
            verifier_assignee="v",
            synthesizer_assignee="s",
            verifier_skills=["kanban-worker", "data-merge"],
            created_by="orch",
        )
        verifier = kb.get_task(conn, created.verifier_id)
        # Custom skills present, historical default NOT.
        skills = set(verifier.skills or [])
        assert "kanban-worker" in skills
        assert "data-merge" in skills
        assert "requesting-code-review" not in skills
    finally:
        conn.close()


def test_create_swarm_accepts_custom_synthesizer_body(tmp_path):
    """#34273: caller can supply a custom synthesizer body."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        custom = "Run process_reviews.py, then generate_report.py, then push."
        created = create_swarm(
            conn,
            goal="goal",
            workers=[SwarmWorkerSpec(profile="w", title="t", body="b")],
            verifier_assignee="v",
            synthesizer_assignee="s",
            synthesizer_body=custom,
            created_by="orch",
        )
        synth = kb.get_task(conn, created.synthesizer_id)
        assert "process_reviews.py" in synth.body
        # Original phrasing absent.
        assert "Synthesize the verified worker outputs" not in synth.body
    finally:
        conn.close()


def test_create_swarm_accepts_custom_synthesizer_skills(tmp_path):
    """#34273: caller can override synthesizer skills."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="goal",
            workers=[SwarmWorkerSpec(profile="w", title="t", body="b")],
            verifier_assignee="v",
            synthesizer_assignee="s",
            synthesizer_skills=["kanban-worker"],
            created_by="orch",
        )
        synth = kb.get_task(conn, created.synthesizer_id)
        skills = set(synth.skills or [])
        assert "kanban-worker" in skills
        assert "humanizer" not in skills
    finally:
        conn.close()


def test_create_swarm_independent_verifier_and_synthesizer_overrides(tmp_path):
    """#34273: overriding only the verifier leaves the synthesizer default,
    and vice versa."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="goal",
            workers=[SwarmWorkerSpec(profile="w", title="t", body="b")],
            verifier_assignee="v",
            synthesizer_assignee="s",
            verifier_body="Custom verifier only",
            # synthesizer untouched — should keep historical default
            created_by="orch",
        )
        verifier = kb.get_task(conn, created.verifier_id)
        synth = kb.get_task(conn, created.synthesizer_id)
        assert "Custom verifier only" in verifier.body
        # Synthesizer still uses the default text.
        assert "Synthesize the verified worker outputs" in synth.body
    finally:
        conn.close()
