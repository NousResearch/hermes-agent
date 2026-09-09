"""Display revision and compression-lineage identity contracts."""

import json
import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def test_missing_revision_is_implicit_zero(db):
    db.create_session("root", "desktop")

    assert db.get_display_revision("root") == 0


def test_compression_child_shares_root_but_branch_is_independent(db):
    db.create_session("root", "desktop")
    db.end_session("root", "compression")
    db.create_session("continued", "desktop", parent_session_id="root")

    db.create_session(
        "branch",
        "desktop",
        parent_session_id="continued",
        model_config={"_branched_from": "continued"},
    )

    assert db.get_display_lineage_identity("root") == ("root", "continued")
    assert db.get_display_lineage_identity("continued") == ("root", "continued")
    assert db.get_display_lineage_identity("branch") == ("branch", "branch")


def test_inherited_branch_marker_keeps_compression_continuation_in_lineage(db):
    db.create_session("root", "desktop")
    db.end_session("root", "compression")
    db.create_session("continued", "desktop", parent_session_id="root")
    db.end_session("continued", "compression")
    db.create_session(
        "branch",
        "desktop",
        parent_session_id="continued",
        model_config={"_branched_from": "continued"},
    )
    db.end_session("branch", "compression")
    db.create_session(
        "branch-tip",
        "desktop",
        parent_session_id="branch",
        model_config={"_branched_from": "continued"},
    )

    assert db.get_display_lineage_identity("branch") == ("branch", "branch-tip")
    assert db.get_display_lineage_identity("branch-tip") == ("branch", "branch-tip")


def test_inherited_delegate_marker_keeps_compression_continuation_in_lineage(db):
    db.create_session("root", "desktop")
    db.end_session("root", "compression")
    db.create_session("continued", "desktop", parent_session_id="root")
    db.end_session("continued", "compression")
    db.create_session(
        "delegate",
        "delegate",
        parent_session_id="continued",
        model_config={"_delegate_from": "continued"},
    )
    db.end_session("delegate", "compression")
    db.create_session(
        "delegate-tip",
        "delegate",
        parent_session_id="delegate",
        model_config={"_delegate_from": "continued"},
    )

    assert db.get_display_lineage_identity("delegate") == ("delegate", "delegate-tip")
    assert db.get_display_lineage_identity("delegate-tip") == ("delegate", "delegate-tip")


@pytest.mark.parametrize(
    ("kind", "marker"),
    [("branch", "_branched_from"), ("delegate", "_delegate_from")],
)
def test_inherited_fork_marker_is_not_a_child_of_the_compression_tip(
    db, kind, marker
):
    """Compression copies model_config, but not the original fork edge.

    The first child is a real branch/delegate and keeps its existing picker
    semantics.  Its continuation inherits the marker for historical reasons;
    it must still be followed as that fork's compression tip.
    """
    db.create_session("root", "desktop")
    db.create_session(
        kind,
        "desktop",
        parent_session_id="root",
        model_config={marker: "root"},
    )
    db.append_message(kind, "user", f"{kind} before compression")
    db.end_session(kind, "compression")
    db.create_session(
        f"{kind}-tip",
        "desktop",
        parent_session_id=kind,
        model_config={marker: "root"},
    )
    db.append_message(f"{kind}-tip", "assistant", f"{kind} continuation")

    assert db.resolve_resume_session_id(kind) == f"{kind}-tip"
    assert db.get_display_lineage_identity(kind) == (kind, f"{kind}-tip")
    assert db.get_display_lineage_identity(f"{kind}-tip") == (
        kind,
        f"{kind}-tip",
    )

    before = db.get_display_revision(kind)
    db.append_message(f"{kind}-tip", "user", "visible mutation")
    assert db.get_display_revision(kind) == before + 1

    listed = {row["id"] for row in db.list_sessions_rich(limit=20)}
    recent = {
        row["id"]
        for row in db.list_sessions_rich(limit=20, order_by_last_active=True)
    }
    if kind == "branch":
        assert f"{kind}-tip" in listed
        assert f"{kind}-tip" in recent
        assert kind not in listed
        assert db.session_count(exclude_children=True) == 2
    else:
        assert kind not in listed
        assert f"{kind}-tip" not in listed
        assert kind not in recent
        assert f"{kind}-tip" not in recent
        assert db.session_count(exclude_children=True) == 1


def test_batch_revision_lookup_uses_projected_root_ids(db):
    db.create_session("a", "desktop")
    db.create_session("b", "desktop")

    assert db._execute_write(lambda conn: db._bump_display_revision(conn, "a")) == 1

    assert db.get_display_revisions(["a", "b", "a", ""]) == {"a": 1, "b": 0}


def _assert_one_bump(db, session_id, mutate):
    before = db.get_display_revision(session_id)

    result = mutate()

    assert db.get_display_revision(session_id) == before + 1
    return result


def test_append_message_bumps_revision_once(db):
    db.create_session("root", "desktop")

    _assert_one_bump(
        db,
        "root",
        lambda: db.append_message("root", "user", "hello"),
    )


def test_append_messages_batch_bumps_once_for_multiple_rows(db):
    db.create_session("root", "desktop")

    inserted = _assert_one_bump(
        db,
        "root",
        lambda: db.append_messages_batch(
            "root",
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow-up"},
            ],
        ),
    )

    assert inserted == 3


def test_empty_append_messages_batch_does_not_bump(db):
    db.create_session("root", "desktop")
    before = db.get_display_revision("root")

    assert db.append_messages_batch("root", []) == 0

    assert db.get_display_revision("root") == before


def test_replace_messages_bumps_revision_once(db):
    """A documented transcript replacement is one logical visible mutation."""
    db.create_session("root", "desktop")
    db.append_message("root", "user", "old")

    _assert_one_bump(
        db,
        "root",
        lambda: db.replace_messages(
            "root", [{"role": "user", "content": "replacement"}]
        ),
    )


def test_empty_replace_on_empty_session_does_not_bump(db):
    db.create_session("root", "desktop")
    before = db.get_display_revision("root")

    db.replace_messages("root", [])

    assert db.get_display_revision("root") == before


def test_archive_and_compact_bumps_revision_once(db):
    db.create_session("root", "desktop")
    db.append_messages_batch(
        "root",
        [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ],
    )

    inserted = _assert_one_bump(
        db,
        "root",
        lambda: db.archive_and_compact(
            "root", [{"role": "assistant", "content": "summary"}]
        ),
    )

    assert inserted == 1


def test_empty_archive_and_compact_on_empty_session_does_not_bump(db):
    db.create_session("root", "desktop")
    before = db.get_display_revision("root")

    assert db.archive_and_compact("root", []) == 0

    assert db.get_display_revision("root") == before


def test_display_kind_change_bumps_once_and_repeated_assignment_is_noop(db):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "hello")

    changed = _assert_one_bump(
        db,
        "root",
        lambda: db.set_latest_matching_message_display_kind(
            "root",
            role="user",
            content="hello",
            display_kind="command",
            display_metadata={"label": "shell"},
        ),
    )
    assert changed is True

    before = db.get_display_revision("root")
    changed = db.set_latest_matching_message_display_kind(
        "root",
        role="user",
        content="hello",
        display_kind="command",
        display_metadata={"label": "shell"},
    )

    assert changed is False
    assert db.get_display_revision("root") == before


def test_missing_display_kind_target_does_not_bump(db):
    db.create_session("root", "desktop")
    before = db.get_display_revision("root")

    assert not db.set_latest_matching_message_display_kind(
        "root", role="user", content="missing", display_kind="command"
    )

    assert db.get_display_revision("root") == before


def test_display_kind_semantically_equal_metadata_order_is_noop(db):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "hello")
    assert db.set_latest_matching_message_display_kind(
        "root",
        role="user",
        content="hello",
        display_kind="command",
        display_metadata={"a": 1, "b": 2},
    )
    before = db.get_display_revision("root")

    changed = db.set_latest_matching_message_display_kind(
        "root",
        role="user",
        content="hello",
        display_kind="command",
        display_metadata={"b": 2, "a": 1},
    )

    assert changed is False
    assert db.get_display_revision("root") == before


def test_display_kind_legacy_empty_metadata_matches_absence_without_rewrite(db):
    db.create_session("root", "desktop")
    message_id = db.append_message("root", "user", "hello")
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET display_kind = ?, display_metadata = ? WHERE id = ?",
            ("command", "{}", message_id),
        )
    )
    before = db.get_display_revision("root")

    changed = db.set_latest_matching_message_display_kind(
        "root",
        role="user",
        content="hello",
        display_kind="command",
        display_metadata={},
    )

    assert changed is False
    assert db.get_display_revision("root") == before
    row = db._conn.execute(
        "SELECT display_metadata FROM messages WHERE id = ?", (message_id,)
    ).fetchone()
    assert row["display_metadata"] == "{}"
    assert db.get_messages("root")[0]["display_metadata"] == {}


def test_visible_reaction_change_bumps_revision_once(db):
    db.create_session("root", "desktop")
    message_id = db.append_message("root", "assistant", "answer")

    reactions = _assert_one_bump(
        db,
        "root",
        lambda: db.set_message_reaction("root", message_id, "👍"),
    )

    assert [reaction["emoji"] for reaction in reactions] == ["👍"]


def test_clearing_missing_reaction_does_not_bump(db):
    """Repeating an emoji toggles it off; clearing an absent author is the no-op."""
    db.create_session("root", "desktop")
    message_id = db.append_message("root", "assistant", "answer")
    before = db.get_display_revision("root")

    assert db.set_message_reaction("root", message_id, None) == []

    assert db.get_display_revision("root") == before


def test_rewind_to_message_bumps_once_only_when_rows_become_inactive(db):
    db.create_session("root", "desktop")
    db.append_messages_batch(
        "root",
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    )
    target_id = db.get_messages("root")[0]["id"]

    result = _assert_one_bump(
        db, "root", lambda: db.rewind_to_message("root", target_id)
    )
    assert result["rewound_count"] == 2

    before = db.get_display_revision("root")
    result = db.rewind_to_message("root", target_id)

    assert result["rewound_count"] == 0
    assert db.get_display_revision("root") == before


def test_restore_rewound_bumps_once_only_when_rows_become_active(db):
    db.create_session("root", "desktop")
    target_id = db.append_message("root", "user", "question")
    db.rewind_to_message("root", target_id)

    restored = _assert_one_bump(
        db, "root", lambda: db.restore_rewound("root", target_id)
    )
    assert restored == 1

    before = db.get_display_revision("root")
    assert db.restore_rewound("root", target_id) == 0
    assert db.get_display_revision("root") == before


def test_clear_messages_bumps_once_only_when_rows_are_deleted(db):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "hello")

    _assert_one_bump(db, "root", lambda: db.clear_messages("root"))

    before = db.get_display_revision("root")
    db.clear_messages("root")
    assert db.get_display_revision("root") == before


def test_publish_compression_child_bumps_parent_root_once(db):
    db.create_session("root", "desktop")
    before = db.get_display_revision("root")

    db.publish_compression_child(
        parent_session_id="root",
        child_session_id="continued",
        source="desktop",
        messages=[{"role": "assistant", "content": "handoff"}],
        require_compression_lease=False,
    )

    assert db.get_display_lineage_identity("continued") == ("root", "continued")
    assert db.get_display_revision("root") == before + 1
    assert db.get_display_revision("continued") == before + 1


def test_delete_session_bumps_tombstoned_root_once(db):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "hello")
    before = db.get_display_revision("root")

    assert db.delete_session("root") is True

    assert db.get_display_revision("root") == before + 1


def test_delete_missing_session_does_not_bump(db):
    before = db.get_display_revision("missing")

    assert db.delete_session("missing") is False

    assert db.get_display_revision("missing") == before


def test_delete_sessions_deduplicates_one_display_root_per_call(db):
    db.create_session("root", "desktop")
    db.end_session("root", "compression")
    db.create_session("continued", "desktop", parent_session_id="root")
    db.append_message("continued", "user", "hello")
    before = db.get_display_revision("root")

    assert db.delete_sessions(["root", "continued", "continued"]) == 2

    assert db.get_display_revision("root") == before + 1


def test_delete_sessions_with_only_missing_ids_does_not_bump(db):
    before = db.get_display_revision("missing")

    assert db.delete_sessions(["missing"]) == 0

    assert db.get_display_revision("missing") == before


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda db: db.set_session_title("root", "A title"), id="title"),
        pytest.param(lambda db: db.set_session_pinned("root", True), id="pinned"),
        pytest.param(lambda db: db.set_session_archived("root", True), id="archived"),
        pytest.param(lambda db: db.set_session_read("root", read=True), id="read"),
    ],
)
def test_non_transcript_session_metadata_does_not_bump(db, mutate):
    db.create_session("root", "desktop")
    before = db.get_display_revision("root")

    assert mutate(db)

    assert db.get_display_revision("root") == before


def test_model_only_api_content_sidecar_does_not_bump(db):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "visible")
    before = db.get_display_revision("root")

    assert db.set_latest_user_api_content("root", "visible", "model-only sidecar") == 1

    assert db.get_display_revision("root") == before


def test_marking_reactions_seen_does_not_bump(db):
    db.create_session("root", "desktop")
    message_id = db.append_message("root", "assistant", "answer")
    db.set_message_reaction("root", message_id, "👍")
    before = db.get_display_revision("root")

    pending = db.take_unseen_reactions("root")

    assert [reaction["emoji"] for reaction in pending] == ["👍"]
    assert db.get_display_revision("root") == before


def test_append_message_rolls_back_when_revision_bump_fails(db, monkeypatch):
    db.create_session("root", "desktop")

    def fail_bump(conn, session_id):
        raise sqlite3.OperationalError("display revision write failed")

    monkeypatch.setattr(db, "_bump_display_revision", fail_bump)

    with pytest.raises(sqlite3.OperationalError, match="display revision write failed"):
        db.append_message("root", "user", "hello")

    assert db.get_messages("root") == []
    session = db.get_session("root")
    assert session["message_count"] == 0


def test_display_message_page_returns_changed_latest_page(db):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "hello")

    page = db.get_display_message_page(
        "root",
        limit=120,
        latest=True,
        include_compacted=True,
        known_display_revision=0,
    )

    assert page["session_id"] == "root"
    assert page["lineage_root_id"] == "root"
    assert page["resolved_tip_id"] == "root"
    assert page["display_revision"] == 1
    assert page["unchanged"] is False
    assert [message["content"] for message in page["messages"]] == ["hello"]
    assert page["pagination"] == {
        "limit": 120,
        "offset": 0,
        "order": "latest",
        "returned": 1,
    }


def test_display_message_page_unchanged_skips_row_selection_and_decode(
    db, monkeypatch
):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "hello")

    def unexpected(*args, **kwargs):
        raise AssertionError("unchanged page must not select or decode messages")

    monkeypatch.setattr(db, "_get_message_rows_from_conn", unexpected, raising=False)
    monkeypatch.setattr(db, "_decode_message_rows", unexpected, raising=False)
    monkeypatch.setattr(db, "_decode_content", unexpected)

    page = db.get_display_message_page(
        "root",
        limit=120,
        latest=True,
        include_compacted=True,
        known_display_revision=1,
    )

    assert page["unchanged"] is True
    assert page["messages"] == []
    assert page["pagination"]["returned"] == 0


def test_display_message_page_true_revision_is_not_treated_as_integer_one(db):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "hello")

    page = db.get_display_message_page(
        "root", limit=120, known_display_revision=True
    )

    assert page["display_revision"] == 1
    assert page["unchanged"] is False
    assert [message["content"] for message in page["messages"]] == ["hello"]


def test_display_message_page_false_revision_is_not_treated_as_integer_zero(db):
    db.create_session("root", "desktop")

    page = db.get_display_message_page(
        "root", limit=120, known_display_revision=False
    )

    assert page["display_revision"] == 0
    assert page["unchanged"] is False


@pytest.mark.parametrize("known_revision", [-1, 0.0, "0"])
def test_display_message_page_invalid_core_revision_is_treated_as_unknown(
    db, known_revision
):
    db.create_session("root", "desktop")

    page = db.get_display_message_page(
        "root", limit=120, known_display_revision=known_revision
    )

    assert page["display_revision"] == 0
    assert page["unchanged"] is False


def test_display_message_page_missing_requested_session_raises_key_error(db):
    with pytest.raises(KeyError, match="missing"):
        db.get_display_message_page("missing", limit=120)


def test_display_message_page_resolves_compression_tip_and_compacted_history(db):
    db.create_session("root", "desktop")
    db.end_session("root", "compression")
    db.create_session("continued", "desktop", parent_session_id="root")
    db.append_messages_batch(
        "continued",
        [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ],
    )
    db.archive_and_compact(
        "continued",
        [
            {"role": "assistant", "content": "summary"},
            {"role": "user", "content": "live question"},
        ],
    )

    page = db.get_display_message_page(
        "root", limit=120, latest=False, include_compacted=True
    )
    active_page = db.get_display_message_page(
        "root", limit=120, latest=False, include_compacted=False
    )

    assert page["session_id"] == "continued"
    assert page["lineage_root_id"] == "root"
    assert page["resolved_tip_id"] == "continued"
    assert [message["content"] for message in page["messages"]] == [
        "old question",
        "old answer",
        "summary",
        "live question",
    ]
    assert [message["content"] for message in active_page["messages"]] == [
        "summary",
        "live question",
    ]


@pytest.mark.parametrize(
    ("limit", "offset", "latest", "expected"),
    [
        pytest.param(
            3,
            0,
            False,
            ["parent question", "parent answer", "compressed summary"],
            id="oldest-crosses-parent-child-boundary",
        ),
        pytest.param(
            3,
            0,
            True,
            ["parent answer", "compressed summary", "continued answer"],
            id="latest-crosses-parent-child-boundary",
        ),
        pytest.param(
            2,
            1,
            False,
            ["parent answer", "compressed summary"],
            id="offset-crosses-parent-child-boundary",
        ),
    ],
)
def test_display_message_page_keeps_real_compression_parent_history(
    db, limit, offset, latest, expected
):
    db.create_session("root", "desktop")
    db.append_messages_batch(
        "root",
        [
            {"role": "user", "content": "parent question"},
            {"role": "assistant", "content": "parent answer"},
        ],
    )
    db.publish_compression_child(
        parent_session_id="root",
        child_session_id="continued",
        source="desktop",
        messages=[
            {"role": "assistant", "content": "compressed summary"},
            {"role": "assistant", "content": "continued answer"},
        ],
        require_compression_lease=False,
    )

    page = db.get_display_message_page(
        "root",
        limit=limit,
        offset=offset,
        latest=latest,
        include_compacted=True,
    )

    assert page["lineage_root_id"] == "root"
    assert page["resolved_tip_id"] == "continued"
    assert [message["content"] for message in page["messages"]] == expected


def test_display_message_page_matches_filtered_noncompression_resume_descendant(db):
    db.create_session("donor", "desktop")
    db.append_message("donor", "user", "donor history")
    db.create_session("legacy-continuation", "desktop", parent_session_id="donor")
    db.create_session(
        "explicit-branch",
        "desktop",
        parent_session_id="donor",
        model_config={"_branched_from": "donor"},
    )
    db.append_message("explicit-branch", "assistant", "unsafe branch history")
    db.create_session(
        "deep-continuation",
        "desktop",
        parent_session_id="legacy-continuation",
    )
    db.append_message("deep-continuation", "assistant", "recovered answer")

    assert db.resolve_resume_session_id("donor") == "deep-continuation"

    donor_page = db.get_display_message_page(
        "donor", limit=120, latest=False, include_compacted=True
    )
    continuation_page = db.get_display_message_page(
        "legacy-continuation", limit=120, latest=False, include_compacted=True
    )

    assert donor_page["resolved_tip_id"] == "deep-continuation"
    assert [message["content"] for message in donor_page["messages"]] == [
        "donor history",
        "recovered answer",
    ]
    assert continuation_page["lineage_root_id"] == "legacy-continuation"
    assert continuation_page["resolved_tip_id"] == "deep-continuation"
    assert [message["content"] for message in continuation_page["messages"]] == [
        "recovered answer"
    ]


def _assert_conditional_page_changed(db, requested_id, known_revision):
    page = db.get_display_message_page(
        requested_id,
        limit=120,
        latest=False,
        include_compacted=True,
        known_display_revision=known_revision,
    )
    assert page["unchanged"] is False
    return page


def test_adopt_orphaned_gateway_session_invalidates_donor_display_page(db):
    db.create_session(
        "donor", "telegram", session_key="agent:main:telegram:dm:peer"
    )
    db.append_message("donor", "user", "before repair")
    db.create_session("orphan", "telegram")
    db.append_message("orphan", "assistant", "after repair")
    before = db.get_display_message_page("donor", limit=120, latest=False)

    assert db.adopt_orphaned_gateway_session("orphan", "donor") is True

    page = _assert_conditional_page_changed(
        db, "donor", before["display_revision"]
    )
    assert page["resolved_tip_id"] == "orphan"
    assert [message["content"] for message in page["messages"]] == [
        "before repair",
        "after repair",
    ]


def test_end_session_invalidates_old_and_new_display_roots(db):
    db.create_session("parent", "desktop")
    db.append_message("parent", "user", "parent history")
    db.create_session("child", "desktop", parent_session_id="parent")
    db.append_message("child", "assistant", "child history")
    before = db.get_display_message_page("child", limit=120, latest=False)
    assert before["lineage_root_id"] == "child"

    db.end_session("parent", "compression")

    page = _assert_conditional_page_changed(db, "child", before["display_revision"])
    assert page["lineage_root_id"] == "parent"
    assert page["resolved_tip_id"] == "child"
    assert [message["content"] for message in page["messages"]] == [
        "parent history",
        "child history",
    ]


def test_reopen_session_invalidates_old_and_new_display_roots(db):
    db.create_session("parent", "desktop")
    db.append_message("parent", "user", "parent history")
    db.end_session("parent", "compression")
    db.create_session("child", "desktop", parent_session_id="parent")
    db.append_message("child", "assistant", "child history")
    before = db.get_display_message_page("child", limit=120, latest=False)
    assert before["lineage_root_id"] == "parent"
    child_revision = db.get_display_revisions(["child"])["child"]
    for _ in range(before["display_revision"] - child_revision):
        db._execute_write(
            lambda conn: db._bump_display_revision_root(conn, "child")
        )
    assert db.get_display_revisions(["child"])["child"] == before[
        "display_revision"
    ]

    db.reopen_session("parent")

    page = _assert_conditional_page_changed(db, "child", before["display_revision"])
    assert page["lineage_root_id"] == "child"
    assert page["resolved_tip_id"] == "child"
    assert [message["content"] for message in page["messages"]] == [
        "child history"
    ]


def test_import_parent_attachment_invalidates_existing_parent_display_page(db):
    db.create_session("parent", "desktop")
    db.append_message("parent", "user", "parent history")
    db.end_session("parent", "compression")
    before = db.get_display_message_page("parent", limit=120, latest=False)

    result = db.import_sessions(
        [
            {
                "id": "imported-child",
                "source": "desktop",
                "parent_session_id": "parent",
                "messages": [
                    {"role": "assistant", "content": "imported continuation"}
                ],
            }
        ]
    )
    assert result["ok"] is True

    page = _assert_conditional_page_changed(
        db, "parent", before["display_revision"]
    )
    assert page["resolved_tip_id"] == "imported-child"
    assert [message["content"] for message in page["messages"]] == [
        "parent history",
        "imported continuation",
    ]


def test_append_after_adoption_invalidates_donor_ancestor_page(db):
    db.create_session(
        "donor", "telegram", session_key="agent:main:telegram:dm:peer"
    )
    db.append_message("donor", "user", "before repair")
    db.create_session("orphan", "telegram")
    db.append_message("orphan", "assistant", "recovered answer")
    assert db.adopt_orphaned_gateway_session("orphan", "donor") is True
    before = db.get_display_message_page("donor", limit=120, latest=False)
    assert before["resolved_tip_id"] == "orphan"

    db.append_message("orphan", "user", "follow-up after repair")

    page = _assert_conditional_page_changed(
        db, "donor", before["display_revision"]
    )
    assert [message["content"] for message in page["messages"]] == [
        "before repair",
        "recovered answer",
        "follow-up after repair",
    ]


def test_append_to_legal_continuation_invalidates_ancestor_page(db):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "root history")
    db.create_session("continued", "desktop", parent_session_id="root")
    db.append_message("continued", "assistant", "continued answer")
    before = db.get_display_message_page("root", limit=120, latest=False)
    assert before["resolved_tip_id"] == "continued"

    db.append_message("continued", "user", "continued follow-up")

    page = _assert_conditional_page_changed(
        db, "root", before["display_revision"]
    )
    assert [message["content"] for message in page["messages"]] == [
        "root history",
        "continued answer",
        "continued follow-up",
    ]


@pytest.mark.parametrize("lineage_kind", ["adopted", "markerless"])
@pytest.mark.parametrize("mutation", ["clear", "replace-empty"])
def test_destructive_descendant_rewrite_invalidates_ancestor_page(
    db, lineage_kind, mutation
):
    if lineage_kind == "adopted":
        db.create_session(
            "root", "telegram", session_key="agent:main:telegram:dm:peer"
        )
        db.create_session("leaf", "telegram")
    else:
        db.create_session("root", "desktop")
        db.create_session("leaf", "desktop", parent_session_id="root")
    db.append_message("root", "user", "root history")
    db.append_message("leaf", "assistant", "leaf history")
    if lineage_kind == "adopted":
        assert db.adopt_orphaned_gateway_session("leaf", "root") is True

    before = db.get_display_message_page("root", limit=120, latest=False)
    assert before["resolved_tip_id"] == "leaf"
    assert [message["content"] for message in before["messages"]] == [
        "root history",
        "leaf history",
    ]

    if mutation == "clear":
        db.clear_messages("leaf")
    else:
        db.replace_messages("leaf", [])

    page = _assert_conditional_page_changed(
        db, "root", before["display_revision"]
    )
    assert [message["content"] for message in page["messages"]] == [
        "root history"
    ]


@pytest.mark.parametrize("delete_method", ["single", "bulk"])
def test_delete_descendant_invalidates_ancestor_page(db, delete_method):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "root history")
    db.create_session("leaf", "desktop", parent_session_id="root")
    db.append_message("leaf", "assistant", "leaf history")
    before = db.get_display_message_page("root", limit=120, latest=False)
    assert before["resolved_tip_id"] == "leaf"

    if delete_method == "single":
        assert db.delete_session("leaf") is True
    else:
        assert db.delete_sessions(["leaf"]) == 1

    page = _assert_conditional_page_changed(
        db, "root", before["display_revision"]
    )
    assert [message["content"] for message in page["messages"]] == [
        "root history"
    ]


@pytest.mark.parametrize("delete_method", ["single", "bulk", "prune"])
def test_delete_parent_invalidates_surviving_child_new_root(db, delete_method):
    db.create_session("parent", "desktop")
    db.append_message("parent", "user", "parent history")
    db.create_session("child", "desktop", parent_session_id="parent")
    db.append_message("child", "assistant", "child history")
    db.end_session("parent", "compression")
    revisions = db.get_display_revisions(["parent", "child"])
    assert revisions["parent"] == revisions["child"]
    before = db.get_display_message_page("child", limit=120, latest=False)
    assert before["display_revision"] == revisions["parent"]
    assert [message["content"] for message in before["messages"]] == [
        "parent history",
        "child history",
    ]

    if delete_method == "single":
        assert db.delete_session("parent") is True
    elif delete_method == "bulk":
        assert db.delete_sessions(["parent"]) == 1
    else:
        assert db.prune_sessions(
            older_than_days=None, started_before=4_000_000_000.0
        ) == 1

    page = _assert_conditional_page_changed(
        db, "child", before["display_revision"]
    )
    assert page["lineage_root_id"] == "child"
    assert page["resolved_tip_id"] == "child"
    assert [message["content"] for message in page["messages"]] == [
        "child history"
    ]


def test_purge_descendant_marker_invalidates_ancestor_page(db):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "root history")
    db.create_session("leaf", "desktop", parent_session_id="root")
    db.append_message(
        "leaf",
        "assistant",
        "[memory]",
        tool_calls=[{"id": "call-1", "type": "function"}],
    )
    before = db.get_display_message_page("root", limit=120, latest=False)

    result = db.purge_stale_tool_call_markers(backup=False)

    assert result["rows_affected"] == 1
    page = _assert_conditional_page_changed(
        db, "root", before["display_revision"]
    )
    assert [message["content"] for message in page["messages"]] == [
        "root history",
        "",
    ]


@pytest.mark.parametrize(
    ("limit", "offset", "latest", "expected"),
    [
        pytest.param(2, 0, False, ["parent message", "child message"], id="oldest"),
        pytest.param(1, 1, False, ["child message"], id="oldest-offset"),
        pytest.param(1, 0, True, ["child message"], id="latest"),
        pytest.param(1, 1, True, ["parent message"], id="latest-offset"),
    ],
)
def test_display_page_orders_segments_by_lineage_not_global_message_id(
    db, limit, offset, latest, expected
):
    result = db.import_sessions(
        [
            {
                "id": "child",
                "source": "desktop",
                "parent_session_id": "parent",
                "messages": [
                    {"role": "assistant", "content": "child message"}
                ],
            },
            {
                "id": "parent",
                "source": "desktop",
                "ended_at": 2.0,
                "end_reason": "compression",
                "messages": [{"role": "user", "content": "parent message"}],
            },
        ]
    )
    assert result["ok"] is True
    child_row_id = db.get_messages("child")[0]["id"]
    parent_row_id = db.get_messages("parent")[0]["id"]
    assert child_row_id < parent_row_id

    page = db.get_display_message_page(
        "parent",
        limit=limit,
        offset=offset,
        latest=latest,
        include_compacted=True,
    )

    assert page["resolved_tip_id"] == "child"
    assert [message["content"] for message in page["messages"]] == expected


def test_display_dedupe_keeps_reasoning_distinct_messages(db):
    db.create_session("parent", "desktop")
    db.append_message(
        "parent",
        "assistant",
        "same visible answer",
        reasoning="parent reasoning",
        timestamp=1234.5,
    )
    db.end_session("parent", "compression")
    db.create_session("child", "desktop", parent_session_id="parent")
    db.append_message(
        "child",
        "assistant",
        "same visible answer",
        reasoning="child reasoning",
        timestamp=1234.5,
    )

    page = db.get_display_message_page(
        "parent", limit=120, latest=False, include_compacted=True
    )

    assert [message["reasoning"] for message in page["messages"]] == [
        "parent reasoning",
        "child reasoning",
    ]


def test_display_keeps_identical_active_messages_across_compression_segments(db):
    db.create_session("parent", "desktop")
    db.append_message(
        "parent", "assistant", "same answer", timestamp=1234.5
    )
    db.end_session("parent", "compression")
    db.create_session("child", "desktop", parent_session_id="parent")
    db.append_message(
        "child", "assistant", "same answer", timestamp=1234.5
    )

    page = db.get_display_message_page(
        "parent", limit=120, latest=False, include_compacted=True
    )

    assert [message["content"] for message in page["messages"]] == [
        "same answer",
        "same answer",
    ]
    assert [message["session_id"] for message in page["messages"]] == [
        "parent",
        "child",
    ]


def test_display_hides_only_the_recorded_rotation_tail_source(db):
    db.create_session("parent", "desktop")
    db.append_message("parent", "user", "before compression")
    watermark = db.get_active_message_watermark("parent")
    db.append_message("parent", "assistant", "concurrent tail")

    db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="desktop",
        messages=[{"role": "assistant", "content": "summary"}],
        watermark=watermark,
        require_compression_lease=False,
    )

    page = db.get_display_message_page(
        "parent", limit=120, latest=False, include_compacted=True
    )
    tail_rows = [
        message
        for message in page["messages"]
        if message["content"] == "concurrent tail"
    ]

    assert len(tail_rows) == 1
    assert tail_rows[0]["session_id"] == "child"


@pytest.mark.parametrize(
    ("selected_successor", "expected_tail_session"),
    [("clone", "clone"), ("sibling", "parent")],
)
def test_clone_provenance_only_suppresses_sources_on_selected_lineage(
    db, selected_successor, expected_tail_session
):
    db.create_session("parent", "desktop")
    db.append_message("parent", "user", "before compression")
    watermark = db.get_active_message_watermark("parent")
    db.append_message("parent", "assistant", "shared parent tail")
    db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="clone",
        source="desktop",
        messages=[{"role": "assistant", "content": "clone summary"}],
        watermark=watermark,
        require_compression_lease=False,
    )
    db.create_session("sibling", "desktop", parent_session_id="parent")
    db.append_message("sibling", "assistant", "sibling answer")
    if selected_successor == "clone":
        # Compression-ended successors sort ahead of live siblings.
        db.end_session("clone", "compression")

    assert db.resolve_resume_session_id("parent") == selected_successor
    page = db.get_display_message_page(
        "parent", limit=120, latest=False, include_compacted=True
    )
    tail_rows = [
        message
        for message in page["messages"]
        if message["content"] == "shared parent tail"
    ]

    assert page["resolved_tip_id"] == selected_successor
    assert len(tail_rows) == 1
    assert tail_rows[0]["session_id"] == expected_tail_session


def test_rewound_rotation_clone_does_not_resurrect_parent_source(db):
    db.create_session("parent", "desktop")
    db.append_message("parent", "assistant", "before compression")
    watermark = db.get_active_message_watermark("parent")
    db.append_message("parent", "user", "concurrent tail")
    db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="desktop",
        messages=[{"role": "assistant", "content": "summary"}],
        watermark=watermark,
        require_compression_lease=False,
    )
    clone = next(
        message
        for message in db.get_messages("child")
        if message["content"] == "concurrent tail"
    )

    result = db.rewind_to_message("child", clone["id"])

    assert result["rewound_count"] == 1
    page = db.get_display_message_page(
        "child", limit=120, latest=False, include_compacted=True
    )
    assert "concurrent tail" not in [
        message["content"] for message in page["messages"]
    ]

    db.clear_messages("child")
    restored_source_page = db.get_display_message_page(
        "parent", limit=120, latest=False, include_compacted=True
    )
    source_rows = [
        message
        for message in restored_source_page["messages"]
        if message["content"] == "concurrent tail"
    ]
    assert len(source_rows) == 1
    assert source_rows[0]["session_id"] == "parent"


def test_clone_provenance_round_trips_with_exported_lineage(db, tmp_path):
    db.create_session("parent", "desktop")
    db.append_message("parent", "assistant", "before compression")
    watermark = db.get_active_message_watermark("parent")
    db.append_message("parent", "user", "concurrent tail")
    db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="desktop",
        messages=[{"role": "assistant", "content": "summary"}],
        watermark=watermark,
        require_compression_lease=False,
    )
    exported = json.loads(json.dumps(db.export_session_lineage("parent")))

    imported = SessionDB(tmp_path / "imported.db")
    try:
        result = imported.import_sessions(exported["segments"])
        assert result["ok"] is True
        page = imported.get_display_message_page(
            "parent", limit=120, latest=False, include_compacted=True
        )
        tail_rows = [
            message
            for message in page["messages"]
            if message["content"] == "concurrent tail"
        ]
        assert len(tail_rows) == 1
        assert tail_rows[0]["session_id"] == "child"
    finally:
        imported.close()


def test_clone_provenance_round_trips_with_export_all(db, tmp_path):
    db.create_session("parent", "desktop")
    db.append_message("parent", "assistant", "before compression")
    watermark = db.get_active_message_watermark("parent")
    db.append_message("parent", "user", "concurrent tail")
    db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="desktop",
        messages=[{"role": "assistant", "content": "summary"}],
        watermark=watermark,
        require_compression_lease=False,
    )

    exported = json.loads(json.dumps(db.export_all()))
    provenance_carriers = [
        session
        for session in exported
        if session.get("message_clone_lineage")
    ]
    assert len(provenance_carriers) == 1
    assert provenance_carriers[0]["message_clone_lineage_version"] == 1
    assert len(provenance_carriers[0]["message_clone_lineage"]) == 1

    imported = SessionDB(tmp_path / "imported-all.db")
    try:
        result = imported.import_sessions(exported)
        assert result["ok"] is True
        page = imported.get_display_message_page(
            "parent", limit=120, latest=False, include_compacted=True
        )
        tail_rows = [
            message
            for message in page["messages"]
            if message["content"] == "concurrent tail"
        ]
        assert len(tail_rows) == 1
        assert tail_rows[0]["session_id"] == "child"
    finally:
        imported.close()


def test_clone_provenance_partial_import_conflict_fails_atomically(db, tmp_path):
    db.create_session("parent", "desktop")
    db.append_message("parent", "assistant", "exported parent")
    watermark = db.get_active_message_watermark("parent")
    db.append_message("parent", "user", "concurrent tail")
    db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="desktop",
        messages=[{"role": "assistant", "content": "summary"}],
        watermark=watermark,
        require_compression_lease=False,
    )
    exported = json.loads(json.dumps(db.export_all()))

    target = SessionDB(tmp_path / "partial-target.db")
    try:
        target.create_session("parent", "desktop")
        target.append_message("parent", "user", "existing target parent")

        result = target.import_sessions(exported)

        assert result["ok"] is False
        assert result["imported"] == 0
        assert "message clone provenance crosses skipped and imported sessions" in (
            result["errors"][0]["error"]
        )
        assert target.get_session("child") is None
        assert [message["content"] for message in target.get_messages("parent")] == [
            "existing target parent"
        ]
        mapping_count = target._execute_write(
            lambda conn: conn.execute(
                "SELECT COUNT(*) FROM message_clone_lineage"
            ).fetchone()[0]
        )
        assert mapping_count == 0
    finally:
        target.close()


def test_legacy_import_ignores_non_integer_message_id_without_provenance(db):
    result = db.import_sessions(
        [
            {
                "id": "legacy",
                "source": "desktop",
                "messages": [
                    {"id": "legacy-opaque-id", "role": "user", "content": "hello"}
                ],
            }
        ]
    )

    assert result["ok"] is True
    assert [message["content"] for message in db.get_messages("legacy")] == [
        "hello"
    ]


@pytest.mark.parametrize(
    ("limit", "offset", "latest", "include_compacted"),
    [
        pytest.param(2, 1, False, False, id="oldest-active-offset"),
        pytest.param(2, 1, True, False, id="latest-active-offset"),
        pytest.param(3, 1, False, True, id="oldest-compacted-offset"),
        pytest.param(3, 1, True, True, id="latest-compacted-offset"),
    ],
)
def test_display_message_page_paging_matches_get_messages(
    db, limit, offset, latest, include_compacted
):
    db.create_session("root", "desktop")
    db.append_messages_batch(
        "root",
        [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ],
    )
    db.archive_and_compact(
        "root",
        [
            {"role": "assistant", "content": "summary"},
            {"role": "user", "content": "live question"},
            {"role": "assistant", "content": "live answer"},
        ],
    )

    expected = db.get_messages(
        "root",
        limit=limit,
        offset=offset,
        latest=latest,
        include_compacted=include_compacted,
    )
    page = db.get_display_message_page(
        "root",
        limit=limit,
        offset=offset,
        latest=latest,
        include_compacted=include_compacted,
    )

    assert page["messages"] == expected
    assert page["pagination"] == {
        "limit": limit,
        "offset": offset,
        "order": "latest" if latest else "oldest",
        "returned": len(expected),
    }


@pytest.mark.requires_wal
def test_display_message_page_revision_and_rows_share_one_sqlite_snapshot(
    db, monkeypatch
):
    db.create_session("root", "desktop")
    db.append_message("root", "user", "before snapshot")
    writer = SessionDB(db.db_path)
    original = db._display_revision_from_conn
    interleaved = False

    def append_after_revision_read(conn, lineage_root_id):
        nonlocal interleaved
        revision = original(conn, lineage_root_id)
        if not interleaved:
            interleaved = True
            writer.append_message("root", "assistant", "after snapshot")
        return revision

    monkeypatch.setattr(db, "_display_revision_from_conn", append_after_revision_read)
    try:
        page = db.get_display_message_page(
            "root", limit=120, latest=False, include_compacted=True
        )
    finally:
        writer.close()

    assert interleaved
    assert page["display_revision"] == 1
    assert [message["content"] for message in page["messages"]] == [
        "before snapshot"
    ]
