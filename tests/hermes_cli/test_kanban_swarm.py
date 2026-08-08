import json
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_swarm as swarm_mod
from hermes_cli.kanban_swarm import (
    SwarmCreated,
    SwarmWorkerSpec,
    create_swarm,
    get_authoritative_topology,
    latest_blackboard,
    migrate_legacy_swarm_topology,
    post_blackboard_update,
)


def test_create_swarm_builds_parallel_workers_verifier_and_synthesizer(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Map the target market and produce a decision memo.",
            workers=[
                SwarmWorkerSpec(
                    profile="researcher-a", title="Market scan", body="Find competitors"
                ),
                SwarmWorkerSpec(
                    profile="researcher-b",
                    title="Customer scan",
                    body="Find customer pains",
                ),
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
            workers=[
                SwarmWorkerSpec(
                    profile="researcher", title="Evidence", body="Find proof"
                )
            ],
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


def test_forged_blackboard_topology_does_not_override_authoritative_topology(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Protect authoritative topology.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            idempotency_key="swarm:topology:forgery",
        )
        post_blackboard_update(
            conn,
            created.root_id,
            author="compromised-worker",
            key="topology",
            value={
                "root_id": created.root_id,
                "worker_ids": ["t_deadbeef"],
                "verifier_id": "t_bad0001",
                "synthesizer_id": "t_bad0002",
            },
        )

        authoritative = get_authoritative_topology(conn, created.root_id)
        recovered = create_swarm(
            conn,
            goal="Protect authoritative topology.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            idempotency_key="swarm:topology:forgery",
        )

        assert authoritative == created
        assert recovered == created
        assert "topology" not in latest_blackboard(conn, created.root_id)
    finally:
        conn.close()


def test_blackboard_topology_key_is_reserved_for_authoritative_db_state(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Keep topology out of worker comments.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        post_blackboard_update(
            conn,
            created.root_id,
            author="compromised-worker",
            key="topology",
            value={"worker_ids": ["t_deadbeef"]},
        )
        post_blackboard_update(
            conn,
            created.root_id,
            author="researcher",
            key="sources",
            value=["https://example.com/source"],
        )

        blackboard = latest_blackboard(conn, created.root_id)

        assert "topology" not in blackboard
        assert blackboard["sources"] == ["https://example.com/source"]
        assert get_authoritative_topology(conn, created.root_id) == created
    finally:
        conn.close()


def test_swarm_synthesizer_stays_blocked_when_verifier_gate_fails(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Fail closed on verifier failure.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        kb.complete_task(conn, created.worker_ids[0], summary="Branch done")
        kb.complete_task(
            conn,
            created.verifier_id,
            summary="Verifier found unresolved defects",
            metadata={"gate": "fail"},
        )

        assert kb.get_task(conn, created.verifier_id).status == "done"
        assert kb.get_task(conn, created.synthesizer_id).status == "todo"
    finally:
        conn.close()


def test_swarm_synthesizer_stays_blocked_when_verifier_is_archived(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Fail closed when verifier is archived without a pass gate.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        kb.complete_task(conn, created.worker_ids[0], summary="Branch done")
        verifier = kb.get_task(conn, created.verifier_id)
        assert verifier is not None
        assert verifier.status == "ready"

        kb.archive_task(conn, created.verifier_id)

        archived_verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)
        assert archived_verifier is not None
        assert synthesizer is not None
        assert archived_verifier.status == "archived"
        assert synthesizer.status == "todo"
    finally:
        conn.close()


def test_unknown_dependency_gate_is_rejected_and_existing_unknown_gate_fails_closed(
    tmp_path,
):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Unknown gates must never weaken authority.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        with pytest.raises(ValueError, match="unsupported dependency gate"):
            kb.set_dependency_gate(
                conn,
                created.verifier_id,
                created.synthesizer_id,
                "unknown_gate",
            )

        conn.execute(
            "UPDATE task_links SET gate_kind = 'unknown_gate' "
            "WHERE parent_id = ? AND child_id = ?",
            (created.verifier_id, created.synthesizer_id),
        )
        conn.commit()
        kb.complete_task(conn, created.worker_ids[0], summary="Branch done")
        kb.complete_task(
            conn,
            created.verifier_id,
            summary="Verifier passed",
            metadata={"gate": "pass"},
        )

        synthesizer = kb.get_task(conn, created.synthesizer_id)
        assert synthesizer is not None
        assert synthesizer.status == "todo"
        conn.execute(
            "UPDATE tasks SET status = 'ready' WHERE id = ?",
            (created.synthesizer_id,),
        )
        conn.commit()
        assert kb.claim_task(conn, created.synthesizer_id) is None
    finally:
        conn.close()


def test_swarm_creation_writes_typed_gate_before_create_task_returns(
    tmp_path, monkeypatch
):
    conn = kb.connect(tmp_path / "kanban.db")
    original_create_task = kb.create_task

    def observing_create_task(connection, **kwargs):
        task_id = original_create_task(connection, **kwargs)
        if kwargs.get("title") == "Synthesize swarm outputs":
            row = connection.execute(
                "SELECT gate_kind FROM task_links WHERE child_id = ?",
                (task_id,),
            ).fetchone()
            assert row is not None
            assert row["gate_kind"] == "metadata_gate_pass"
        return task_id

    monkeypatch.setattr(kb, "create_task", observing_create_task)
    try:
        create_swarm(
            conn,
            goal="Eliminate the untyped edge race.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
    finally:
        conn.close()


def test_swarm_creation_rolls_back_entire_graph_on_partial_failure(
    tmp_path, monkeypatch
):
    conn = kb.connect(tmp_path / "kanban.db")

    def fail_blackboard(*args, **kwargs):
        raise RuntimeError("injected blackboard failure")

    monkeypatch.setattr(swarm_mod, "post_blackboard_update", fail_blackboard)
    try:
        with pytest.raises(RuntimeError, match="injected blackboard failure"):
            create_swarm(
                conn,
                goal="Rollback every partial swarm write.",
                workers=[
                    SwarmWorkerSpec(profile="researcher", title="Branch", body="B")
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
                idempotency_key="swarm:atomic:rollback",
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM swarm_topologies").fetchone()[0] == 0
    finally:
        conn.close()


@pytest.mark.parametrize(
    "fault_stage",
    [
        "create_task_1",
        "create_task_2",
        "create_task_3",
        "create_task_4",
        "complete_task",
        "gate_event",
        "store_swarm_topology",
        "post_blackboard_update",
    ],
)
def test_swarm_creation_rolls_back_at_each_publication_stage(
    tmp_path, monkeypatch, fault_stage
):
    conn = kb.connect(tmp_path / f"{fault_stage}.db")

    def fail(*args, **kwargs):
        raise RuntimeError(f"injected {fault_stage} failure")

    if fault_stage.startswith("create_task_"):
        fail_at = int(fault_stage.rsplit("_", 1)[1])
        original = kb.create_task
        calls = 0

        def fail_numbered_create(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == fail_at:
                fail(*args, **kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(kb, "create_task", fail_numbered_create)
    elif fault_stage == "gate_event":
        original_append_event = kb._append_event

        def fail_gate_event(connection, task_id, kind, payload, **kwargs):
            if kind == "dependency_gate_set":
                fail()
            return original_append_event(connection, task_id, kind, payload, **kwargs)

        monkeypatch.setattr(kb, "_append_event", fail_gate_event)
    elif fault_stage == "post_blackboard_update":
        monkeypatch.setattr(swarm_mod, fault_stage, fail)
    else:
        monkeypatch.setattr(kb, fault_stage, fail)

    try:
        with pytest.raises(RuntimeError, match=f"injected {fault_stage} failure"):
            create_swarm(
                conn,
                goal="Rollback every publication stage.",
                workers=[
                    SwarmWorkerSpec(profile="researcher", title="Branch", body="B")
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
                idempotency_key=f"swarm:atomic:{fault_stage}",
            )
        for table in (
            "tasks",
            "task_links",
            "task_events",
            "task_comments",
            "swarm_topologies",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        conn.close()


def test_existing_idempotent_root_without_topology_fails_closed(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = kb.create_task(
            conn,
            title="Legacy partial swarm",
            idempotency_key="swarm:legacy:partial",
        )
        post_blackboard_update(
            conn,
            root_id,
            author="untrusted-worker",
            key="topology",
            value={
                "root_id": root_id,
                "worker_ids": ["t_missing_worker"],
                "verifier_id": "t_missing_verifier",
                "synthesizer_id": "t_missing_synthesizer",
            },
        )
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

        with pytest.raises(ValueError, match="lacks valid authoritative topology"):
            create_swarm(
                conn,
                goal="Do not duplicate a legacy partial graph.",
                workers=[
                    SwarmWorkerSpec(profile="researcher", title="Branch", body="B")
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
                idempotency_key="swarm:legacy:partial",
            )

        assert kb.get_task(conn, root_id) is not None
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before
        assert kb.get_swarm_topology(conn, root_id) is None
        assert not any(
            event.kind == "swarm_topology_migrated"
            for event in kb.list_events(conn, root_id)
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda topology, created: topology | {"root_id": "t_wrongroot"},
        lambda topology, created: topology | {"worker_ids": [7]},
        lambda topology, created: topology | {"worker_ids": ["t_missing"]},
    ],
)
def test_authoritative_topology_rejects_malformed_or_unbound_state(tmp_path, mutation):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Bind topology to the real graph.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            idempotency_key="swarm:topology:binding",
        )
        topology = kb.get_swarm_topology(conn, created.root_id)
        assert topology is not None
        conn.execute(
            "UPDATE swarm_topologies SET topology_json = ? WHERE root_id = ?",
            (json.dumps(mutation(topology, created)), created.root_id),
        )
        conn.commit()

        assert get_authoritative_topology(conn, created.root_id) is None
        with pytest.raises(ValueError, match="lacks valid authoritative topology"):
            create_swarm(
                conn,
                goal="Bind topology to the real graph.",
                workers=[
                    SwarmWorkerSpec(profile="researcher", title="Branch", body="B")
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
                idempotency_key="swarm:topology:binding",
            )
    finally:
        conn.close()


def test_swarm_gate_and_topology_mutations_are_audited(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Audit authority mutations.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        root_events = {event.kind for event in kb.list_events(conn, created.root_id)}
        synth_events = {
            event.kind for event in kb.list_events(conn, created.synthesizer_id)
        }

        assert "swarm_topology_stored" in root_events
        assert "dependency_gate_set" in synth_events

        topology_event = next(
            event
            for event in kb.list_events(conn, created.root_id)
            if event.kind == "swarm_topology_stored"
        )
        gate_event = next(
            event
            for event in kb.list_events(conn, created.synthesizer_id)
            if event.kind == "dependency_gate_set"
        )
        assert topology_event.payload["actor"] == "swarm-orchestrator"
        assert len(topology_event.payload["topology_sha256"]) == 64
        assert gate_event.payload["actor"] == "swarm-orchestrator"
        assert len(gate_event.payload["mutation_sha256"]) == 64
    finally:
        conn.close()


def test_authoritative_topology_rejects_cross_tenant_graph(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Keep topology inside one tenant.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            tenant="tenant-a",
        )
        conn.execute(
            "UPDATE tasks SET tenant = 'tenant-b' WHERE id = ?",
            (created.worker_ids[0],),
        )
        conn.commit()

        assert get_authoritative_topology(conn, created.root_id) is None
    finally:
        conn.close()


def test_authoritative_topology_rejects_cross_project_graph(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Keep topology inside one project.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            tenant="tenant-a",
        )
        conn.execute(
            "UPDATE tasks SET project_id = 'project-b' WHERE id = ?",
            (created.worker_ids[0],),
        )
        conn.commit()

        assert get_authoritative_topology(conn, created.root_id) is None
    finally:
        conn.close()


def test_swarm_idempotency_is_scoped_by_tenant(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        first = create_swarm(
            conn,
            goal="Tenant A graph.",
            workers=[SwarmWorkerSpec(profile="researcher", title="A", body="A")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            idempotency_key="same-key",
            tenant="tenant-a",
        )
        second = create_swarm(
            conn,
            goal="Tenant B graph.",
            workers=[SwarmWorkerSpec(profile="researcher", title="B", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            idempotency_key="same-key",
            tenant="tenant-b",
        )

        assert second.root_id != first.root_id
        first_task = kb.get_task(conn, first.root_id)
        second_task = kb.get_task(conn, second.root_id)
        assert first_task is not None
        assert second_task is not None
        assert first_task.tenant == "tenant-a"
        assert second_task.tenant == "tenant-b"
    finally:
        conn.close()


def test_swarm_idempotency_is_scoped_by_project(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from hermes_cli import projects_db as pdb

    projects = {
        project_id: pdb.Project(
            id=project_id,
            slug=project_id,
            name=project_id,
            created_at=0,
            primary_path=str(tmp_path),
        )
        for project_id in ("project-a", "project-b")
    }
    monkeypatch.setattr(pdb, "connect_closing", lambda: nullcontext(object()))
    monkeypatch.setattr(
        pdb, "get_project", lambda _conn, project_id: projects[project_id]
    )

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        first = create_swarm(
            conn,
            goal="Project A graph.",
            workers=[SwarmWorkerSpec(profile="researcher", title="A", body="A")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            idempotency_key="same-project-key",
            tenant="tenant-a",
            project_id="project-a",
        )
        second = create_swarm(
            conn,
            goal="Project B graph.",
            workers=[SwarmWorkerSpec(profile="researcher", title="B", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            idempotency_key="same-project-key",
            tenant="tenant-a",
            project_id="project-b",
        )

        assert second.root_id != first.root_id
        first_task = kb.get_task(conn, first.root_id)
        second_task = kb.get_task(conn, second.root_id)
        assert first_task is not None
        assert second_task is not None
        assert first_task.project_id == "project-a"
        assert second_task.project_id == "project-b"
    finally:
        conn.close()


def test_swarm_explicit_unresolved_project_fails_closed(tmp_path, monkeypatch):
    from hermes_cli import projects_db as pdb

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        unscoped = kb.create_task(
            conn,
            title="Existing unscoped root",
            tenant="tenant-a",
            idempotency_key="same-key",
        )
        monkeypatch.setattr(
            pdb,
            "connect_closing",
            lambda: (_ for _ in ()).throw(OSError("project store unavailable")),
        )

        with pytest.raises(ValueError, match="project not found or unavailable"):
            create_swarm(
                conn,
                goal="Must remain project scoped.",
                workers=[
                    SwarmWorkerSpec(profile="researcher", title="Scoped", body="S")
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
                idempotency_key="same-key",
                tenant="tenant-a",
                project_id="requested-project",
            )

        assert [task.id for task in kb.list_tasks(conn)] == [unscoped]
    finally:
        conn.close()


def test_authoritative_topology_requires_stored_scope_fields(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Require explicit topology scope.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            tenant="tenant-a",
        )
        topology = kb.get_swarm_topology(conn, created.root_id)
        assert topology is not None
        topology.pop("tenant")
        topology.pop("project_id")
        conn.execute(
            "UPDATE swarm_topologies SET topology_json = ? WHERE root_id = ?",
            (json.dumps(topology), created.root_id),
        )
        conn.commit()

        assert get_authoritative_topology(conn, created.root_id) is None
    finally:
        conn.close()


def test_swarm_topology_is_insert_once_and_idempotent(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Do not rewrite authority history.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Branch", body="B")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        topology = kb.get_swarm_topology(conn, created.root_id)
        assert topology is not None
        before = [
            event
            for event in kb.list_events(conn, created.root_id)
            if event.kind == "swarm_topology_stored"
        ]

        kb.store_swarm_topology(
            conn,
            created.root_id,
            topology,
            created_by="different-retry-actor",
        )
        after = [
            event
            for event in kb.list_events(conn, created.root_id)
            if event.kind == "swarm_topology_stored"
        ]
        assert len(after) == len(before)

        with pytest.raises(ValueError, match="conflicting swarm topology"):
            kb.store_swarm_topology(
                conn,
                created.root_id,
                topology | {"verifier_id": created.worker_ids[0]},
                created_by="attacker",
            )
        assert kb.get_swarm_topology(conn, created.root_id) == topology
    finally:
        conn.close()


def test_concurrent_same_key_swarm_creation_yields_one_graph(tmp_path):
    db_path = tmp_path / "kanban.db"
    seed = kb.connect(db_path)
    seed.close()
    barrier = threading.Barrier(2)

    def create_once():
        conn = kb.connect(db_path)
        try:
            barrier.wait(timeout=5)
            return create_swarm(
                conn,
                goal="Create exactly one graph under contention.",
                workers=[
                    SwarmWorkerSpec(profile="researcher", title="Branch", body="B")
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
                idempotency_key="swarm:concurrent:single",
            )
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: create_once(), range(2)))

    assert results[0] == results[1]
    conn = kb.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM swarm_topologies").fetchone()[0] == 1
    finally:
        conn.close()


def test_validated_legacy_topology_migrates_atomically_and_backfills_gate(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root = kb.create_task(
            conn,
            title="Legacy root",
            tenant="legacy-tenant",
            idempotency_key="swarm:legacy:valid",
        )
        worker = kb.create_task(
            conn, title="Legacy worker", tenant="legacy-tenant", parents=[root]
        )
        verifier = kb.create_task(
            conn, title="Legacy verifier", tenant="legacy-tenant", parents=[worker]
        )
        synthesizer = kb.create_task(
            conn,
            title="Legacy synthesizer",
            tenant="legacy-tenant",
            parents=[verifier],
        )
        kb.complete_task(conn, root, summary="legacy planner done")
        topology = {
            "root_id": root,
            "worker_ids": [worker],
            "verifier_id": verifier,
            "synthesizer_id": synthesizer,
        }
        post_blackboard_update(
            conn,
            root,
            author="legacy-orchestrator",
            key="topology",
            value=topology,
        )

        migrated = migrate_legacy_swarm_topology(
            conn, root, created_by="migration-test"
        )
        assert migrated == SwarmCreated(**topology)
        assert kb.get_swarm_topology(conn, root) == topology | {
            "tenant": "legacy-tenant",
            "project_id": None,
        }
        gate = conn.execute(
            "SELECT gate_kind FROM task_links WHERE parent_id = ? AND child_id = ?",
            (verifier, synthesizer),
        ).fetchone()["gate_kind"]
        assert gate == "metadata_gate_pass"
        root_events = kb.list_events(conn, root)
        migrated_event = next(
            event for event in root_events if event.kind == "swarm_topology_migrated"
        )
        assert migrated_event.payload["actor"] == "migration-test"
        assert len(migrated_event.payload["topology_sha256"]) == 64
    finally:
        conn.close()
