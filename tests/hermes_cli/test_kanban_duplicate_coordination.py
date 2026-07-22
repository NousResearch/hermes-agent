from __future__ import annotations

from hermes_cli import kanban_db as kb


def _conn(tmp_path, monkeypatch):
    db = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    return kb.connect()


def test_same_workspace_and_implementation_key_allow_only_one_claim(tmp_path, monkeypatch):
    conn = _conn(tmp_path, monkeypatch)
    workspace = str(tmp_path / "shared")
    first = kb.create_task(
        conn, title="first", assignee="worker", workspace_kind="dir",
        workspace_path=workspace, implementation_claim_key="goal:pur-727:round:1:impl",
    )
    second = kb.create_task(
        conn, title="second", assignee="worker", workspace_kind="dir",
        workspace_path=workspace, implementation_claim_key="goal:pur-727:round:1:impl",
    )

    assert kb.claim_task(conn, first, claimer="worker:1") is not None
    assert kb.claim_task(conn, second, claimer="worker:2") is None
    assert kb.get_task(conn, second).status == "ready"


def test_review_requires_changed_input_fingerprint(tmp_path, monkeypatch):
    conn = _conn(tmp_path, monkeypatch)
    task_id = kb.create_task(
        conn, title="review", assignee="reviewer",
        review_input_fingerprint="sha256:a",
    )
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (task_id,))
    conn.commit()

    assert kb.claim_review_task(conn, task_id, claimer="reviewer:1") is not None
    run_id = kb.get_task(conn, task_id).current_run_id
    assert kb.reclaim_task(conn, task_id, signal_fn=lambda *_: None)
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (task_id,))
    conn.commit()
    assert kb.claim_review_task(conn, task_id, claimer="reviewer:2") is None

    conn.execute(
        "UPDATE tasks SET review_input_fingerprint='sha256:b' WHERE id=?", (task_id,)
    )
    conn.commit()
    assert kb.claim_review_task(conn, task_id, claimer="reviewer:3") is not None
    assert kb.get_task(conn, task_id).current_run_id != run_id


def test_review_claim_rechecks_conditional_parent_dependencies(tmp_path, monkeypatch):
    conn = _conn(tmp_path, monkeypatch)
    parent = kb.create_task(conn, title="implementation", assignee="worker")
    review = kb.create_task(
        conn,
        title="review",
        assignee="reviewer",
        parents=[parent],
        parent_outcomes={parent: ["review_required"]},
    )
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (review,))
    conn.commit()

    assert kb.claim_review_task(conn, review, claimer="reviewer:1") is None
    assert kb.get_task(conn, review).status == "todo"
    assert kb.list_runs(conn, review) == []
    event = kb.list_events(conn, review)[-1]
    assert event.kind == "claim_rejected"
    assert event.payload["reason"] == "parents_not_done"


def test_non_review_claim_attempt_does_not_emit_parent_rejection(tmp_path, monkeypatch):
    conn = _conn(tmp_path, monkeypatch)
    parent = kb.create_task(conn, title="parent")
    child = kb.create_task(conn, title="child", parents=[parent])

    assert kb.claim_review_task(conn, child, claimer="reviewer:1") is None
    assert [event.kind for event in kb.list_events(conn, child)] == ["created"]


def test_findings_merge_into_one_bounded_repair(tmp_path, monkeypatch):
    conn = _conn(tmp_path, monkeypatch)
    first = kb.create_or_merge_bounded_repair(
        conn, finding_set_key="review:pur-727", title="repair",
        findings=["duplicate implementation"], assignee="worker",
    )
    second = kb.create_or_merge_bounded_repair(
        conn, finding_set_key="review:pur-727", title="ignored",
        findings=["duplicate implementation", "unchanged review reran"],
        assignee="worker",
    )

    assert second == first
    task = kb.get_task(conn, first)
    assert task.body.count("duplicate implementation") == 1
    assert "unchanged review reran" in task.body
    assert conn.execute(
        "SELECT count(*) AS n FROM tasks WHERE idempotency_key=?",
        ("bounded-repair:review:pur-727",),
    ).fetchone()["n"] == 1


def test_repeated_pattern_is_idempotent_and_escalates(tmp_path, monkeypatch):
    conn = _conn(tmp_path, monkeypatch)
    root = kb.create_task(conn, title="root fix", assignee="worker")
    affected = kb.create_task(conn, title="second sighting", assignee="worker")

    first = kb.record_repeated_pattern(
        conn, pattern_key="duplicate-mainline", affected_task_id=root,
        root_fix_task_id=root, canary="expert",
    )
    second = kb.record_repeated_pattern(
        conn, pattern_key="duplicate-mainline", affected_task_id=affected,
    )

    assert first["occurrences"] == 1
    assert second["occurrences"] == 2
    assert second["state"] == "escalated"
    assert second["root_fix_task_id"] == root
    assert conn.execute(
        "SELECT count(*) AS n FROM task_links WHERE parent_id=? AND child_id=?",
        (root, affected),
    ).fetchone()["n"] == 1


def test_reclaimed_worker_late_completion_isolated_by_run_id(tmp_path, monkeypatch):
    conn = _conn(tmp_path, monkeypatch)
    task_id = kb.create_task(conn, title="work", assignee="worker")
    assert kb.claim_task(conn, task_id, claimer="worker:old") is not None
    old_run = kb.get_task(conn, task_id).current_run_id
    assert kb.reclaim_task(conn, task_id, signal_fn=lambda *_: None)
    assert kb.claim_task(conn, task_id, claimer="worker:new") is not None
    new_run = kb.get_task(conn, task_id).current_run_id

    assert old_run != new_run
    assert kb.complete_task(conn, task_id, expected_run_id=old_run) is False
    assert kb.get_task(conn, task_id).status == "running"
    assert kb.complete_task(conn, task_id, expected_run_id=new_run) is True


def test_cli_block_runtime_entry_merges_second_sighting_to_one_root(tmp_path, monkeypatch):
    from hermes_cli import kanban as kc

    conn = _conn(tmp_path, monkeypatch)
    first = kb.create_task(conn, title="first", assignee="worker")
    second = kb.create_task(conn, title="second", assignee="worker")
    conn.close()

    assert "Blocked" in kc.run_slash(
        f"block {first} first --pattern-key provider-quota --root-fix {first}"
    )
    assert "Blocked" in kc.run_slash(
        f"block {second} second --pattern-key provider-quota"
    )
    conn = kb.connect()
    row = conn.execute(
        "SELECT * FROM repeated_patterns WHERE pattern_key='provider-quota'"
    ).fetchone()
    assert row["occurrences"] == 2
    assert row["root_fix_task_id"] == first
    assert conn.execute(
        "SELECT count(*) AS n FROM task_links WHERE parent_id=? AND child_id=?",
        (first, second),
    ).fetchone()["n"] == 1


def test_cli_block_without_explicit_root_fails_closed(tmp_path, monkeypatch):
    from hermes_cli import kanban as kc

    conn = _conn(tmp_path, monkeypatch)
    first = kb.create_task(conn, title="first failure", assignee="worker")
    second = kb.create_task(conn, title="second failure", assignee="worker")
    conn.close()

    assert "Blocked" in kc.run_slash(
        f"block {first} first --pattern-key pur-727-duplicate"
    )
    assert "Blocked" in kc.run_slash(
        f"block {second} second --pattern-key pur-727-duplicate"
    )
    conn = kb.connect()
    row = conn.execute(
        "SELECT * FROM repeated_patterns WHERE pattern_key='pur-727-duplicate'"
    ).fetchone()
    assert row["occurrences"] == 2
    assert row["state"] == "escalated"
    assert row["root_fix_task_id"] is None
    assert conn.execute(
        "SELECT count(*) AS n FROM task_links WHERE child_id IN (?, ?)",
        (first, second),
    ).fetchone()["n"] == 0


def test_cli_high_risk_first_sighting_escalates_without_inventing_root(
    tmp_path, monkeypatch
):
    from hermes_cli import kanban as kc

    conn = _conn(tmp_path, monkeypatch)
    affected = kb.create_task(conn, title="high risk failure", assignee="worker")
    conn.close()

    assert "Blocked" in kc.run_slash(
        f"block {affected} urgent --pattern-key auth-leak --high-risk"
    )
    conn = kb.connect()
    row = conn.execute(
        "SELECT * FROM repeated_patterns WHERE pattern_key='auth-leak'"
    ).fetchone()
    assert row["occurrences"] == 1
    assert row["state"] == "escalated"
    assert row["high_risk"] == 1
    assert row["root_fix_task_id"] is None
    assert conn.execute(
        "SELECT count(*) AS n FROM task_links WHERE child_id=?", (affected,)
    ).fetchone()["n"] == 0


def test_cli_review_findings_merge_and_changed_input_requeues(tmp_path, monkeypatch):
    from hermes_cli import kanban as kc

    conn = _conn(tmp_path, monkeypatch)
    review = kb.create_task(
        conn, title="review", assignee="reviewer",
        review_input_fingerprint="sha256:a",
    )
    conn.close()

    first = kc.run_slash("repair review:pur-727 repair duplicate-mainline --assignee worker")
    second = kc.run_slash("repair review:pur-727 ignored unchanged-review --assignee worker")
    assert first.split()[-1] == second.split()[-1]
    repair_id = first.split()[-1]
    conn = kb.connect()
    assert "duplicate-mainline" in kb.get_task(conn, repair_id).body
    assert "unchanged-review" in kb.get_task(conn, repair_id).body
    conn.execute(
        "UPDATE tasks SET status='review', review_last_fingerprint='sha256:a' WHERE id=?",
        (review,),
    )
    conn.commit()
    assert kb.claim_review_task(conn, review, claimer="same") is None
    conn.close()

    assert "Review queued" in kc.run_slash(f"review-input {review} sha256:b")
    conn = kb.connect()
    assert kb.claim_review_task(conn, review, claimer="reviewer:new") is not None
    assert conn.execute(
        "SELECT count(*) AS n FROM task_runs WHERE task_id=?", (review,)
    ).fetchone()["n"] == 1


def test_expert_pur_727_cli_dispatcher_canary(tmp_path, monkeypatch):
    from hermes_cli import kanban as kc
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    conn = _conn(tmp_path, monkeypatch)
    root = kb.create_task(conn, title="PUR-727 root fix")
    first = kb.create_task(conn, title="PUR-727 failure one")
    second = kb.create_task(conn, title="PUR-727 failure two")
    review = kb.create_task(
        conn, title="PUR-727 review", assignee="reviewer",
        review_input_fingerprint="sha256:pur-727-a",
    )
    legacy_review = kb.create_task(
        conn, title="legacy review without fingerprint", assignee="reviewer",
    )
    ordinary = kb.create_task(
        conn, title="ordinary scratch task", assignee="worker",
    )
    conn.execute(
        "UPDATE tasks SET status='review' WHERE id IN (?, ?)",
        (review, legacy_review),
    )
    conn.commit()
    conn.close()

    assert "Blocked" in kc.run_slash(
        f"block {first} first --pattern-key pur-727-canary --root-fix {root}"
    )
    assert "Blocked" in kc.run_slash(
        f"block {second} second --pattern-key pur-727-canary"
    )
    repair_one = kc.run_slash(
        "repair review:pur-727-canary repair duplicate-mainline"
    ).split()[-1]
    repair_two = kc.run_slash(
        "repair review:pur-727-canary ignored unchanged-review"
    ).split()[-1]
    assert repair_two == repair_one

    spawned = []

    def fake_spawn(task, workspace):
        spawned.append((task.id, workspace))
        return 10000 + len(spawned)

    conn = kb.connect()
    first_dispatch = kb.dispatch_once(conn, spawn_fn=fake_spawn)
    assert {task_id for task_id, _, _ in first_dispatch.spawned} == {
        ordinary, review, legacy_review,
    }
    ordinary_task = kb.get_task(conn, ordinary)
    assert ordinary_task is not None
    assert ordinary_task.workspace_kind == "scratch"
    assert conn.execute(
        "SELECT count(*) AS n FROM repeated_patterns"
    ).fetchone()["n"] == 1

    assert kb.reclaim_task(conn, review, signal_fn=lambda *_: None)
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (review,))
    conn.commit()
    run_count = conn.execute(
        "SELECT count(*) AS n FROM task_runs WHERE task_id=?", (review,)
    ).fetchone()["n"]
    assert review not in {
        task_id for task_id, _, _ in kb.dispatch_once(
            conn, spawn_fn=fake_spawn
        ).spawned
    }
    assert conn.execute(
        "SELECT count(*) AS n FROM task_runs WHERE task_id=?", (review,)
    ).fetchone()["n"] == run_count
    conn.close()

    assert "Review queued" in kc.run_slash(
        f"review-input {review} sha256:pur-727-b"
    )
    conn = kb.connect()
    assert review in {
        task_id for task_id, _, _ in kb.dispatch_once(
            conn, spawn_fn=fake_spawn
        ).spawned
    }
    pattern = conn.execute(
        "SELECT * FROM repeated_patterns WHERE pattern_key='pur-727-canary'"
    ).fetchone()
    assert pattern["root_fix_task_id"] == root
    assert pattern["occurrences"] == 2
    assert conn.execute(
        "SELECT count(*) AS n FROM task_links WHERE parent_id=? "
        "AND child_id IN (?, ?)", (root, first, second),
    ).fetchone()["n"] == 2
    repair_task = kb.get_task(conn, repair_one)
    assert repair_task is not None
    body = repair_task.body or ""
    assert "duplicate-mainline" in body
    assert "unchanged-review" in body
    assert conn.execute(
        "SELECT count(*) AS n FROM tasks WHERE idempotency_key=?",
        ("bounded-repair:review:pur-727-canary",),
    ).fetchone()["n"] == 1
    assert conn.execute(
        "SELECT count(*) AS n FROM task_runs WHERE task_id=?", (legacy_review,)
    ).fetchone()["n"] == 1
