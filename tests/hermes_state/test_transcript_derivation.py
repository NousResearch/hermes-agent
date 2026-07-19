"""Durable, atomic transcript-boundary derivation contracts."""

import pytest

from hermes_state import (
    SessionDB,
    TranscriptDerivationConflictError,
    TranscriptDerivationValidationError,
)


@pytest.fixture
def db(tmp_path):
    state = SessionDB(tmp_path / "state.db")
    try:
        yield state
    finally:
        state.close()


def _source_turns(db: SessionDB, session_id: str = "source"):
    db.create_session(
        session_id,
        "api_server",
        model="openai-codex/gpt-5",
        model_config={"reasoning_effort": "high"},
        system_prompt="stable prompt",
        user_id="owner",
        cwd="/workspace",
    )
    db.set_session_title(session_id, "Original")
    first_user = db.append_message(session_id, "user", "First request")
    first_assistant = db.append_message(session_id, "assistant", "First answer")
    second_user = db.append_message(session_id, "user", "Second request")
    second_assistant = db.append_message(session_id, "assistant", "Second answer")
    return first_user, first_assistant, second_user, second_assistant


def _contents(db: SessionDB, session_id: str):
    return [(row["role"], row["content"]) for row in db.get_messages(session_id)]


def _raw_source_snapshot(db: SessionDB, session_id: str):
    session = tuple(
        db._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    )
    messages = [
        tuple(row)
        for row in db._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    ]
    return session, messages


def test_branch_copies_through_terminal_assistant_without_mutating_source(db):
    _, first_assistant, _, _ = _source_turns(db)
    before = _raw_source_snapshot(db, "source")

    result = db.derive_session_at_boundary(
        "source",
        "branch-child",
        "op-branch",
        "branch",
        first_assistant,
        "after_turn",
    )

    assert result == {
        "operation_id": "op-branch",
        "kind": "branch",
        "boundary": "after_turn",
        "target_message_id": f"msg:v1:{first_assistant}",
        "source_requested_session_id": "source",
        "source_resolved_session_id": "source",
        "child_session_id": "branch-child",
        "child_parent_session_id": "source",
    }
    assert _contents(db, "branch-child") == [
        ("user", "First request"),
        ("assistant", "First answer"),
    ]
    child = db.get_session("branch-child")
    assert child["parent_session_id"] == "source"
    assert child["model"] == "openai-codex/gpt-5"
    assert child["system_prompt"] == "stable prompt"
    assert _raw_source_snapshot(db, "source") == before


def test_edit_and_retry_stop_before_the_canonical_user_turn(db):
    _, _, second_user, second_assistant = _source_turns(db)

    edit = db.derive_session_at_boundary(
        "source",
        "edit-child",
        "op-edit",
        "edit",
        second_user,
        "before_turn",
    )
    retry = db.derive_session_at_boundary(
        "source",
        "retry-child",
        "op-retry",
        "retry",
        second_assistant,
        "before_turn",
    )

    expected_prefix = [
        ("user", "First request"),
        ("assistant", "First answer"),
    ]
    assert _contents(db, "edit-child") == expected_prefix
    assert _contents(db, "retry-child") == expected_prefix
    assert edit["target_message_id"] == f"msg:v1:{second_user}"
    assert retry["retry_user_message_id"] == f"msg:v1:{second_user}"
    assert retry["retry_user_content"] == "Second request"
    assert retry["retry_user_attachments"] == []


def test_tool_turn_is_copied_as_one_causal_group_and_nonterminal_target_is_rejected(db):
    db.create_session("tools", "api_server")
    user = db.append_message("tools", "user", "Look it up")
    tool_call = db.append_message(
        "tools",
        "assistant",
        None,
        tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
    )
    db.append_message("tools", "tool", "result", tool_call_id="call-1", tool_name="search")
    final = db.append_message("tools", "assistant", "Here is the answer")

    with pytest.raises(TranscriptDerivationValidationError, match="terminal"):
        db.derive_session_at_boundary(
            "tools", "bad-child", "op-bad", "branch", tool_call, "after_turn"
        )

    db.derive_session_at_boundary(
        "tools", "tools-child", "op-tools", "branch", final, "after_turn"
    )
    child = db.get_messages("tools-child")
    assert [row["id"] for row in child] != [user, tool_call, final]
    assert [(row["role"], row["content"]) for row in child] == [
        ("user", "Look it up"),
        ("assistant", None),
        ("tool", "result"),
        ("assistant", "Here is the answer"),
    ]


def test_compression_lineage_resolves_tip_and_copies_ancestor_anchor_once(db):
    db.create_session("root", "api_server")
    db.append_message("root", "user", "Root request")
    root_answer = db.append_message("root", "assistant", "Root answer")
    db.end_session("root", "compression")
    db.create_session("tip", "api_server", parent_session_id="root")
    db.append_message("tip", "user", "Tip request")
    db.append_message("tip", "assistant", "Tip answer")

    result = db.derive_session_at_boundary(
        "root", "root-branch", "op-root", "branch", root_answer, "after_turn"
    )

    assert result["source_resolved_session_id"] == "tip"
    assert result["child_parent_session_id"] == "tip"
    assert _contents(db, "root-branch") == [
        ("user", "Root request"),
        ("assistant", "Root answer"),
    ]


def test_operation_replay_is_idempotent_and_conflicting_reuse_fails(db):
    _, assistant, _, _ = _source_turns(db)
    kwargs = dict(
        source_session_id="source",
        child_session_id="child",
        operation_id="same-op",
        kind="branch",
        target_message_id=assistant,
        boundary="after_turn",
    )
    first = db.derive_session_at_boundary(**kwargs)
    second = db.derive_session_at_boundary(**kwargs)
    assert second == first
    assert db._conn.execute(
        "SELECT COUNT(*) FROM transcript_derivations WHERE operation_id = 'same-op'"
    ).fetchone()[0] == 1
    assert _contents(db, "child") == [
        ("user", "First request"),
        ("assistant", "First answer"),
    ]

    with pytest.raises(TranscriptDerivationConflictError, match="another request"):
        db.derive_session_at_boundary(
            **{**kwargs, "target_message_id": kwargs["target_message_id"] + 2}
        )


def test_deleted_child_id_remains_tombstoned_and_cannot_spoof_replay(db):
    _, assistant, _, _ = _source_turns(db)
    kwargs = dict(
        source_session_id="source",
        child_session_id="child",
        operation_id="original-op",
        kind="branch",
        target_message_id=assistant,
        boundary="after_turn",
    )
    db.derive_session_at_boundary(**kwargs)
    assert db.delete_session("child") is True

    with pytest.raises(TranscriptDerivationConflictError, match="deleted"):
        db.derive_session_at_boundary(**kwargs)
    with pytest.raises(TranscriptDerivationConflictError, match="already reserved"):
        db.derive_session_at_boundary(
            **{**kwargs, "operation_id": "replacement-op"}
        )

    db.create_session("child", "api_server")
    with pytest.raises(TranscriptDerivationConflictError, match="belongs to another"):
        db.derive_session_at_boundary(**kwargs)


def test_hidden_foreign_and_context_snapshot_rows_are_not_branchable(db):
    _, assistant, _, _ = _source_turns(db)
    db.create_session("foreign", "api_server")
    foreign = db.append_message("foreign", "assistant", "Foreign")
    snapshot = db.append_message(
        "source", "assistant", "Hidden snapshot", context_snapshot=True
    )
    db._conn.execute("UPDATE messages SET active = 0 WHERE id = ?", (assistant,))

    for index, target in enumerate((foreign, snapshot, assistant)):
        with pytest.raises(TranscriptDerivationValidationError, match="display lineage"):
            db.derive_session_at_boundary(
                "source",
                f"hidden-{index}",
                f"op-hidden-{index}",
                "branch",
                target,
                "after_turn",
            )


def test_failure_rolls_back_child_messages_and_operation(db, monkeypatch):
    _, assistant, _, _ = _source_turns(db)

    def fail_after_child_insert(conn, child_session_id, source_message_ids):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', 'partial', 1)",
            (child_session_id,),
        )
        raise RuntimeError("injected failure")

    monkeypatch.setattr(db, "_copy_derivation_prefix_in_transaction", fail_after_child_insert)
    with pytest.raises(RuntimeError, match="injected failure"):
        db.derive_session_at_boundary(
            "source", "rollback-child", "op-rollback", "branch", assistant, "after_turn"
        )

    assert db.get_session("rollback-child") is None
    assert db._conn.execute(
        "SELECT 1 FROM transcript_derivations WHERE operation_id = 'op-rollback'"
    ).fetchone() is None
    assert db._conn.execute(
        "SELECT 1 FROM messages WHERE session_id = 'rollback-child'"
    ).fetchone() is None


def test_invalid_boundary_and_non_text_retry_fail_closed(db):
    _, assistant, _, _ = _source_turns(db)
    with pytest.raises(TranscriptDerivationValidationError, match="kind or boundary"):
        db.derive_session_at_boundary(
            "source", "wrong-boundary", "op-wrong", "branch", assistant, "before_turn"
        )

    db.create_session("image", "api_server")
    db.append_message(
        "image",
        "user",
        [{"type": "text", "text": "Describe"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
    )
    image_answer = db.append_message("image", "assistant", "Description")
    with pytest.raises(TranscriptDerivationValidationError, match="non-text"):
        db.derive_session_at_boundary(
            "image", "image-retry", "op-image", "retry", image_answer, "before_turn"
        )


def test_existing_schema_worker_connection_opens_while_writer_lock_is_held(db):
    db._conn.execute("BEGIN IMMEDIATE")
    worker = None
    try:
        worker = SessionDB(db.db_path, initialize_schema=False)
        assert worker.get_session("missing") is None
    finally:
        if worker is not None:
            worker.close()
        db._conn.rollback()
