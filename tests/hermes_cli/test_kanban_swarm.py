import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_swarm import (
    SwarmWorkerSpec,
    create_swarm,
    latest_blackboard,
    parse_worker_arg,
    post_blackboard_update,
)


def _plant_skill(name: str) -> None:
    from hermes_constants import get_hermes_home

    dest = get_hermes_home() / "skills" / name
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Planted.\n---\n# {name}\n",
        encoding="utf-8",
    )


def test_worker_arg_keeps_loaded_skills_and_refuses_a_colon_in_the_title():
    """profile:title[:skill,skill] splits on the first two colons. A title
    that itself contains ':' was stored as a skill and the worker exited
    Unknown skill (#129349). The third field is a skill list only when every
    token loads the way ``--skills`` loads it, including a name with a space.
    """
    for name in ("requesting-code-review", "humanizer", "deep research"):
        _plant_skill(name)

    spec = parse_worker_arg("planner:Compare options:requesting-code-review,humanizer")
    assert spec.profile == "planner"
    assert spec.title == "Compare options"
    assert spec.skills == ["requesting-code-review", "humanizer"]

    spaced_name = parse_worker_arg("planner:Use the spaced skill:deep research")
    assert spaced_name.title == "Use the spaced skill"
    assert spaced_name.skills == ["deep research"]

    spaced = parse_worker_arg("planner:Plan the work")
    assert spaced.title == "Plan the work"
    assert spaced.skills == []

    with pytest.raises(ValueError, match="compare two options"):
        parse_worker_arg("planner:Plan: compare two options")
    with pytest.raises(ValueError, match="implement"):
        parse_worker_arg("planner:Step 1: implement")


def test_swarm_command_refuses_an_unresolved_skill_before_creating_a_card(capsys):
    """The CLI entry returns before connect, so a title colon never becomes a card."""
    import argparse

    from hermes_cli.kanban import _cmd_swarm
    from hermes_constants import get_hermes_home

    rc = _cmd_swarm(argparse.Namespace(
        worker=["planner:Step 1: implement"],
        goal="unused",
        verifier="reviewer",
        synthesizer="writer",
        tenant=None,
        created_by="orchestrator",
        priority=0,
        idempotency_key=None,
        json=False,
    ))
    assert rc == 2
    assert "implement" in capsys.readouterr().err
    assert not (get_hermes_home() / "kanban.db").exists()


def test_create_swarm_builds_parallel_workers_verifier_and_synthesizer(tmp_path):
    conn = kbc.connect(tmp_path / "kanban.db")
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
    writer = kbc.connect(db_path)
    reader = kbc.connect(db_path)
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
    conn = kbc.connect(tmp_path / "kanban.db")
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
    conn = kbc.connect(tmp_path / "kanban.db")
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
    conn = kbc.connect(tmp_path / "kanban.db")
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
