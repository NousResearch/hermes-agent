"""Search behavior against a real PostgreSQL database, including bounded-index tails."""

import pytest

from hermes_state_common import FTS_TOOL_CONTENT_PREFIX_CHARS
from hermes_state_postgres_schema import SEARCH_CONTENT_CHARS, SEARCH_TOOL_CALLS_CHARS, SEARCH_TOOL_NAME_CHARS


def _ids(db, query, **kwargs):
    return [row["id"] for row in db.search_messages(query, fields={"id"}, **kwargs)]


def test_boolean_phrase_prefix_and_searchable_tool_metadata(postgres_db):
    db = postgres_db
    db.create_session("search", "cli", model="test-model")
    phrase = db.append_message("search", "user", "PostgreSQL connection pool", timestamp=1)
    separate = db.append_message("search", "assistant", "PostgreSQL safe connection with pool", timestamp=2)
    metadata = db.append_message("search", "assistant", "done", tool_name="inspect_schema",
                                 tool_calls=[{"function": {"name": "inspect_schema", "arguments": "migrationreceipt"}}])

    assert set(_ids(db, "PostgreSQL AND pool")) == {phrase, separate}
    assert _ids(db, '"connection pool"') == [phrase]
    assert _ids(db, "PostgreSQL NOT safe") == [phrase]
    assert set(_ids(db, "connect* OR migrationreceipt")) == {phrase, separate, metadata}
    assert _ids(db, "inspect_schema") == [metadata]
    assert _ids(db, '"?"*') == []
    assert _ids(db, "OR NOT missing") == []
    assert _ids(db, "PostgreSQL", sort="newest", limit=1, offset=1) == [phrase]
    assert _ids(db, "PostgreSQL", sort="oldest") == [phrase, separate]


def test_search_visibility_projection_context_and_filters(postgres_db):
    db = postgres_db
    db.create_session("visible", "cli", model="test-model")
    db.create_session("excluded", "cron")
    before = db.append_message("visible", "user", "before", timestamp=1)
    visible = db.append_message("visible", "assistant", "needle visible", timestamp=2)
    after = db.append_message("visible", "user", "after", timestamp=3)
    rewind = db.append_message("visible", "assistant", "needle rewound", timestamp=4)
    compacted = db.append_message("visible", "assistant", "needle compacted", timestamp=5)
    db.append_message("visible", "assistant", "needle hidden", display_kind="hidden")
    other = db.append_message("excluded", "user", "needle other")
    db._write_sql("UPDATE messages SET active = 0 WHERE id IN (?, ?)", (rewind, compacted))
    db._write_sql("UPDATE messages SET compacted = 1 WHERE id = ?", (compacted,))

    assert set(_ids(db, "needle")) == {visible, compacted, other}
    assert set(_ids(db, "needle", include_inactive=True, source_filter=["cli"])) == {visible, compacted, rewind}
    assert set(_ids(db, "needle", exclude_sources=["cron"], role_filter=["assistant"])) == {visible, compacted}
    assert _ids(db, "needle", source_filter=[]) == []
    assert set(_ids(db, "needle", exclude_sources=[])) == {visible, compacted, other}
    hit = db.search_messages("visible", fields={"id", "snippet", "context"})[0]
    assert set(hit) == {"id", "snippet", "context"}
    assert hit["id"] == visible
    assert "visible" in hit["snippet"]
    assert [row["content"] for row in hit["context"]] == ["before", "needle visible", "after"]
    assert before < visible < after
    with pytest.raises(ValueError, match="unknown search result field"):
        db.search_messages("needle", fields={"not_a_field"})
    with pytest.raises(TypeError, match="collection"):
        db.search_messages("needle", fields="id")


def test_unicode_substrings_and_boolean_fallback(postgres_db):
    db = postgres_db
    db.create_session("unicode", "cli")
    chinese = db.append_message("unicode", "user", "修改youer服务端 数据库连接成功")
    excluded = db.append_message("unicode", "user", "数据库连接失败")
    literal = db.append_message("unicode", "user", "完成率100% 数据库版本_v2")

    assert _ids(db, "youer") == [chinese]
    assert _ids(db, "数据库 NOT 失败", sort="oldest") == [chinese, literal]
    assert set(_ids(db, "成功 OR 失败")) == {chinese, excluded}
    assert _ids(db, "100% 数据库") == [literal]
    assert _ids(db, '"数据库版本_v2"') == [literal]


def test_index_bounds_preserve_long_messages_and_not_exclusions(postgres_db):
    db = postgres_db
    db.create_session("long", "cli")
    short = db.append_message("long", "user", "needle alpha beta")
    long = db.append_message("long", "assistant", "x " * SEARCH_CONTENT_CHARS + "needle tailreceipt")
    excluded = db.append_message("long", "assistant", "needle " + "x " * SEARCH_CONTENT_CHARS + "forbidden")
    metadata = db.append_message("long", "assistant", "done", tool_calls=[{
        "function": {"name": "inspect", "arguments": "x " * SEARCH_TOOL_CALLS_CHARS + "metadatareceipt"}}])
    tool_name = db.append_message("long", "tool", "done", tool_name="x " * SEARCH_TOOL_NAME_CHARS + "namereceipt")
    late_phrase = db.append_message("long", "assistant", "x " * 17_000 + "alpha beta")

    assert set(_ids(db, "needle NOT forbidden")) == {short, long}
    assert set(_ids(db, "needle")) == {short, long, excluded}
    tail = db.search_messages("tailreceipt")[0]
    assert tail["id"] == long
    assert "tailreceipt" in tail["snippet"]
    assert _ids(db, "metadatareceipt") == [metadata]
    assert _ids(db, "namereceipt") == [tool_name]
    assert set(_ids(db, '"alpha beta"')) == {short, late_phrase}


def test_large_tool_body_requires_explicit_role_and_maintenance_keeps_results(postgres_db):
    db = postgres_db
    db.create_session("tool", "cli")
    tool = db.append_message("tool", "tool", "prefixreceipt " + "x " * FTS_TOOL_CONTENT_PREFIX_CHARS + "tailreceipt")

    assert _ids(db, "prefixreceipt") == [tool]
    assert _ids(db, "tailreceipt") == []
    assert _ids(db, "tailreceipt", role_filter=["tool"]) == [tool]
    assert db.fts_rebuild_status() is None
    assert db.fts_cjk_rebuild_status() is None
    assert db.fts_rebuild_step() is False
    assert db.fts_cjk_rebuild_step() is False
    assert db.optimize_fts() == 1
    assert db.rebuild_fts() == 1
    assert _ids(db, "prefixreceipt") == [tool]
    db.replace_messages("tool", [{"role": "user", "content": "replacementreceipt"}])
    assert _ids(db, "prefixreceipt") == []
    assert _ids(db, "replacementreceipt")


def test_bounded_recent_browse_and_timeout_leave_connection_usable(postgres_db):
    db = postgres_db
    db.create_session("recent", "cli")
    db.append_message("recent", "user", "browse receipt")

    assert [row["id"] for row in db.list_recent_sessions_bounded()] == ["recent"]
    with pytest.raises(TimeoutError, match="recent-session browse exceeded"):
        db._read_with_timeout("SELECT pg_sleep(?)", (0.05,), timeout_seconds=0.01)
    assert [row["id"] for row in db.list_recent_sessions_bounded()] == ["recent"]
