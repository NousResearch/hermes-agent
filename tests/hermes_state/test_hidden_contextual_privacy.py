"""Internal contextual transcript rows stay private on public state surfaces."""

import pytest

from hermes_state import SessionDB


def _preview(rows, session_id: str) -> str:
    return next(row["preview"] for row in rows if row["id"] == session_id)


def test_search_export_and_preview_hide_internal_contextual_rows(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message(
        "s1",
        role="user",
        content="contextual-secret-needle",
        display_kind="hidden",
    )
    db.append_message("s1", role="user", content="visible opening")
    db.append_message("s1", role="assistant", content="visible response")

    assert db.search_messages("contextual-secret-needle") == []
    privileged = db.search_messages(
        "contextual-secret-needle", include_hidden=True
    )
    assert len(privileged) == 1
    assert "contextual-secret-needle" in privileged[0]["snippet"]

    exported = db.export_session("s1")
    assert exported is not None
    assert [row["content"] for row in exported["messages"]] == [
        "visible opening",
        "visible response",
    ]
    assert exported["message_count"] == len(exported["messages"]) == 2

    privileged_export = db.export_session("s1", include_hidden=True)
    assert privileged_export is not None
    assert "contextual-secret-needle" in [
        row["content"] for row in privileged_export["messages"]
    ]
    assert privileged_export["message_count"] == len(
        privileged_export["messages"]
    ) == 3

    rich = db.get_session_rich_row("s1")
    assert rich is not None
    assert rich["preview"] == "visible opening"


@pytest.mark.parametrize("order_by_last_active", [False, True])
def test_list_session_preview_filters_hidden_before_limit(
    tmp_path, order_by_last_active
):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message(
        "s1", role="user", content="HIDDEN-PREVIEW", display_kind="hidden"
    )
    db.append_message("s1", role="user", content="visible preview")

    rows = db.list_sessions_rich(
        order_by_last_active=order_by_last_active,
        min_message_count=1,
    )
    assert _preview(rows, "s1") == "visible preview"
    assert "HIDDEN-PREVIEW" not in repr(rows)


def test_pinned_backfill_preview_filters_hidden_before_limit(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("old", source="cli")
    db.append_message(
        "old", role="user", content="HIDDEN-PINNED", display_kind="hidden"
    )
    db.append_message("old", role="user", content="visible pinned")
    db.set_session_pinned("old", True)

    db.create_session("new", source="cli")
    db.append_message("new", role="user", content="newer visible")

    rows = db.list_sessions_rich(
        limit=1,
        min_message_count=1,
        order_by_last_active=True,
        include_pinned=True,
    )
    assert _preview(rows, "old") == "visible pinned"
    assert "HIDDEN-PINNED" not in repr(rows)


@pytest.mark.parametrize("create_binding_table", [False, True])
def test_telegram_session_picker_preview_filters_hidden_before_limit(
    tmp_path, create_binding_table
):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("tg", source="telegram", user_id="user-1")
    db.append_message(
        "tg", role="user", content="HIDDEN-TELEGRAM", display_kind="hidden"
    )
    db.append_message("tg", role="user", content="visible telegram")
    if create_binding_table:
        with db._lock:
            assert db._conn is not None
            db._conn.execute(
                "CREATE TABLE telegram_dm_topic_bindings (session_id TEXT)"
            )
            db._conn.commit()

    rows = db.list_unlinked_telegram_sessions_for_user(
        chat_id="chat-1", user_id="user-1"
    )
    assert _preview(rows, "tg") == "visible telegram"
    assert "HIDDEN-TELEGRAM" not in repr(rows)


def test_search_context_hydration_skips_hidden_neighbors(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message("s1", role="user", content="visible before")
    db.append_message(
        "s1", role="assistant", content="HIDDEN-BEFORE", display_kind="hidden"
    )
    db.append_message("s1", role="user", content="visible unique match")
    db.append_message(
        "s1", role="assistant", content="HIDDEN-AFTER", display_kind="hidden"
    )
    db.append_message("s1", role="assistant", content="visible after")

    result = db.search_messages(
        "visible unique match", fields=("snippet", "context")
    )
    assert len(result) == 1
    assert [row["content"] for row in result[0]["context"]] == [
        "visible before",
        "visible unique match",
        "visible after",
    ]
    assert "HIDDEN-" not in repr(result)

    privileged = db.search_messages(
        "visible unique match",
        fields=("snippet", "context"),
        include_hidden=True,
    )
    assert [row["content"] for row in privileged[0]["context"]] == [
        "HIDDEN-BEFORE",
        "visible unique match",
        "HIDDEN-AFTER",
    ]


def test_first_assistant_text_ignores_hidden_scaffold(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message(
        "s1", role="assistant", content="HIDDEN-TITLE", display_kind="hidden"
    )
    db.append_message("s1", role="assistant", content="visible title source")

    assert db.get_first_assistant_text("s1") == "visible title source"


def test_hidden_rows_do_not_affect_public_counts_filters_exports_or_recency(
    tmp_path,
):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("older-visible", source="cli")
    db.append_message(
        "older-visible", role="user", content="visible old", timestamp=100
    )
    db.append_message(
        "older-visible",
        role="assistant",
        content="visible tool request",
        tool_calls=[{"id": "visible-call"}],
        timestamp=101,
    )
    db.append_message(
        "older-visible",
        role="assistant",
        content="hidden late activity",
        tool_calls=[{"id": "hidden-1"}, {"id": "hidden-2"}],
        display_kind="hidden",
        timestamp=300,
    )

    db.create_session("newer-visible", source="cli")
    db.append_message(
        "newer-visible", role="user", content="visible new", timestamp=200
    )

    db.create_session("hidden-only", source="cli")
    db.append_message(
        "hidden-only",
        role="user",
        content="hidden only",
        display_kind="hidden",
        timestamp=400,
    )

    rows = db.list_sessions_rich(
        order_by_last_active=True,
        min_message_count=1,
        project_compression_tips=False,
    )
    assert [row["id"] for row in rows] == ["newer-visible", "older-visible"]
    older = next(row for row in rows if row["id"] == "older-visible")
    assert older["message_count"] == 2
    assert older["tool_call_count"] == 1
    assert older["last_active"] == 101

    rich = db.get_session_rich_row("older-visible")
    assert rich is not None
    assert rich["message_count"] == 2
    assert rich["tool_call_count"] == 1
    assert rich["last_active"] == 101

    exported = next(
        row for row in db.export_all() if row["id"] == "older-visible"
    )
    assert exported["message_count"] == len(exported["messages"]) == 2
    assert exported["tool_call_count"] == 1
