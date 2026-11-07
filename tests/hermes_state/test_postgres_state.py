"""Exercise SessionDB's existing persistence contracts on SQLite and PostgreSQL."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from copy import deepcopy
import json
import threading

import pytest

from hermes_state import SessionDB
from hermes_state_errors import SessionTurnLeaseLostError


@pytest.fixture(params=["sqlite", "postgres"])
def state_db(request, tmp_path):
    if request.param == "postgres":
        yield request.getfixturevalue("postgres_db")
        return
    with SessionDB(tmp_path / "state.db") as database:
        yield database


@pytest.mark.parametrize("content", [
    [{"type": "text", "text": "مرحبا 世界 👋"}],
    {"text": "Structured content 世界"},
    "Text containing a NUL: \x00 and its suffix",
    "\x01json:literal user text",
])
def test_message_fields_and_prompt_survive_reopen(state_db, content):
    session_id = "round-trip"
    model_config = {"label": "مرحبا 世界", "temperature": 0.25}
    state_db.create_session(
        session_id, "cli", model="test-model", model_config=model_config,
        system_prompt="Preserve this prompt exactly.",
    )
    timestamp = 1_800_000_000.123456
    message_id = state_db.append_message(
        session_id, "user", content, timestamp=timestamp,
        platform_message_id="platform-1", display_metadata={"gateway_input_owner": "owner"},
    )
    state_db.append_message(session_id, "assistant", "Response", reasoning="Internal reasoning")
    assert state_db.set_session_title(session_id, "Stored conversation")
    state_db.close()

    with SessionDB(state_db.db_path) as reopened:
        session = reopened.get_session(session_id)
        assert json.loads(session["model_config"]) == model_config
        assert session["system_prompt"] == "Preserve this prompt exactly."
        assert session["title"] == "Stored conversation"
        messages = reopened.get_messages(session_id)
        assert messages[0]["id"] == message_id
        assert messages[0]["content"] == content
        assert messages[0]["timestamp"] == pytest.approx(timestamp, abs=0.000001, rel=0)
        assert messages[0]["display_metadata"] == {"gateway_input_owner": "owner"}
        assert messages[1]["reasoning"] == "Internal reasoning"
        assert reopened.has_platform_message_id(session_id, "platform-1")
        assert reopened.get_messages_as_conversation(session_id)[-1]["content"] == "Response"


def test_delete_removes_messages_and_usage_without_affecting_another_session(state_db):
    for session_id in ("removed", "retained"):
        state_db.create_session(session_id, "cli", model="test-model")
        state_db.append_message(session_id, "user", f"Message for {session_id}")
        state_db.update_token_counts(session_id, input_tokens=11, api_call_count=1, model="test-model")

    state_db.delete_session("removed")

    assert state_db.get_session("removed") is None
    assert state_db.get_messages("removed") == []
    assert state_db._read_one(
        "SELECT COUNT(*) FROM session_model_usage WHERE session_id = ?", ("removed",)
    )[0] == 0
    assert state_db.get_messages("retained")[0]["content"] == "Message for retained"


def test_failed_transaction_rolls_back_all_changes(state_db):
    state_db.create_session("rollback", "cli")

    def interrupted_write(connection):
        connection.execute("UPDATE sessions SET title = ? WHERE id = ?", ("Uncommitted", "rollback"))
        connection.execute("INSERT INTO state_meta(key, value) VALUES (?, ?)", ("uncommitted", "value"))
        raise RuntimeError("interrupted transaction")

    with pytest.raises(RuntimeError, match="interrupted transaction"):
        state_db._execute_write(interrupted_write)

    assert state_db.get_session("rollback")["title"] is None
    assert state_db.get_meta("uncommitted") is None
    state_db.append_message("rollback", "user", "The connection can still write")
    assert state_db.message_count("rollback") == 1


def test_nullable_costs_and_repeated_model_usage_accumulate(state_db):
    state_db.create_session("usage", "cli", model="test-model")
    for cost in (None, 0.25, None):
        state_db.update_token_counts(
            "usage", input_tokens=10, output_tokens=2, actual_cost_usd=cost,
            model="test-model", billing_provider="test-provider", api_call_count=1,
        )

    session = state_db.get_session("usage")
    assert session["input_tokens"] == 30
    assert session["output_tokens"] == 6
    assert session["actual_cost_usd"] == pytest.approx(0.25)
    usage = state_db._read_one(
        "SELECT input_tokens, api_call_count, actual_cost_usd FROM session_model_usage WHERE session_id = ?",
        ("usage",),
    )
    assert (usage["input_tokens"], usage["api_call_count"]) == (30, 3)
    assert usage["actual_cost_usd"] == pytest.approx(0.25)


def test_flush_waits_for_a_claimed_token_batch(state_db, monkeypatch):
    state_db.create_session("queued-usage", "cli")
    writing = threading.Event()
    release = threading.Event()
    update_token_counts = state_db.update_token_counts

    def delayed_update(session_id, **kwargs):
        writing.set()
        assert release.wait(timeout=15)
        update_token_counts(session_id, **kwargs)

    monkeypatch.setattr(state_db, "update_token_counts", delayed_update)
    try:
        state_db.queue_token_counts("queued-usage", input_tokens=7)
        assert writing.wait(timeout=10)
        assert state_db.flush_token_counts(timeout=2) is False
    finally:
        release.set()
    assert state_db.flush_token_counts(timeout=10) is True
    assert state_db.flush_token_counts() is True
    assert state_db.get_session("queued-usage")["input_tokens"] == 7


def test_close_drains_inflight_usage_before_closing_connections(state_db, monkeypatch):
    state_db.create_session("close-usage", "cli")
    writing = threading.Event()
    release = threading.Event()
    closing = threading.Event()
    update_token_counts = state_db.update_token_counts

    def delayed_update(session_id, **kwargs):
        writing.set()
        assert release.wait(timeout=15)
        update_token_counts(session_id, **kwargs)

    def close():
        closing.set()
        state_db.close()

    monkeypatch.setattr(state_db, "update_token_counts", delayed_update)
    with ThreadPoolExecutor(max_workers=1) as executor:
        state_db.queue_token_counts("close-usage", input_tokens=9)
        try:
            assert writing.wait(timeout=10)
            closed = executor.submit(close)
            assert closing.wait(timeout=10)
            with pytest.raises(TimeoutError):
                closed.result(timeout=2)
        finally:
            release.set()
        closed.result(timeout=10)

    with SessionDB(state_db.db_path) as reopened:
        assert reopened.get_session("close-usage")["input_tokens"] == 9


def test_readonly_store_rejects_public_and_direct_writes(state_db):
    state_db.create_session("readonly", "cli")
    with SessionDB(state_db.db_path, read_only=True) as reader:
        assert reader.get_session("readonly")["id"] == "readonly"
        with pytest.raises(Exception, match="read.only|readonly"):
            reader.append_message("readonly", "user", "Must not persist")
        with pytest.raises(Exception, match="read.only|readonly"):
            with reader._read_ctx() as connection:
                connection.execute("UPDATE sessions SET title = ? WHERE id = ?", ("Must not persist", "readonly"))
    assert state_db.get_messages("readonly") == []
    assert state_db.get_session("readonly")["title"] is None


def test_independent_writers_preserve_message_ids_and_counters(state_db):
    state_db.create_session("concurrent", "cli")
    barrier = threading.Barrier(4)

    def write_messages(worker):
        with SessionDB(state_db.db_path) as database:
            barrier.wait(timeout=15)
            return [database.append_message("concurrent", "user", f"{worker}:{index}") for index in range(6)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        message_ids = [message_id for batch in executor.map(write_messages, range(4)) for message_id in batch]

    assert len(set(message_ids)) == len(message_ids)
    assert state_db.message_count("concurrent") == len(message_ids)
    assert state_db.get_session("concurrent")["message_count"] == len(message_ids)
    assert {message["content"] for message in state_db.get_messages("concurrent")} == {
        f"{worker}:{index}" for worker in range(4) for index in range(6)
    }


def test_turn_lease_allows_only_one_independent_owner(state_db):
    state_db.create_session("leased", "cli")
    barrier = threading.Barrier(2)

    def claim(holder):
        with SessionDB(state_db.db_path) as database:
            barrier.wait(timeout=15)
            return holder, database.try_acquire_session_turn_lease("leased", holder)

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = dict(executor.map(claim, ("worker-a", "worker-b")))
    assert sum(claims.values()) == 1
    owner = next(holder for holder, acquired in claims.items() if acquired)
    other = next(holder for holder, acquired in claims.items() if not acquired)
    with pytest.raises(SessionTurnLeaseLostError):
        state_db.append_message("leased", "user", "Unowned write", turn_lease_holder=other)
    state_db.append_message("leased", "user", "Owned write", turn_lease_holder=owner)
    state_db.release_session_turn_lease("leased", owner)
    assert state_db.try_acquire_session_turn_lease("leased", other)


def test_branch_rewind_does_not_change_parent_history(state_db):
    state_db.create_session("parent", "cli", model="test-model")
    state_db.append_message("parent", "user", "Original question")
    state_db.append_message("parent", "assistant", "Original answer")
    original = state_db.get_messages_as_conversation("parent")
    state_db.create_session(
        "branch", "cli", parent_session_id="parent", model_config={"_branched_from": "parent"}
    )
    state_db.append_messages_batch("branch", deepcopy(original))
    target = state_db.append_message("branch", "user", "Branch question")
    state_db.append_message("branch", "assistant", "Branch answer")

    result = state_db.rewind_to_message("branch", target)

    assert result["rewound_count"] == 2
    assert state_db.get_messages_as_conversation("parent") == original
    assert state_db.get_messages_as_conversation("branch") == original
    assert state_db.get_session("branch")["parent_session_id"] == "parent"
    assert state_db.is_explicit_fork_child("branch")


def test_compacted_display_deduplicates_tail_and_refreshes_edited_identity(state_db):
    state_db.create_session("compacted", "cli")
    for role, content in (
        ("user", "First question"), ("assistant", "First answer"),
        ("user", "Tail question"), ("assistant", "Tail answer"),
    ):
        state_db.append_message("compacted", role, content)
    tail = state_db.get_messages_as_conversation("compacted")[-2:]
    state_db.archive_and_compact(
        "compacted", [{"role": "user", "content": "Summary of earlier turns"}, *tail]
    )
    displayed = state_db.get_messages("compacted", include_compacted=True)
    assert sum(message["content"] == "Tail question" for message in displayed) == 1
    active_tail = next(message for message in state_db.get_messages("compacted") if message["content"] == "Tail question")
    state_db._execute_write(lambda connection: connection.execute(
        "UPDATE messages SET content = ? WHERE id = ?", ("Edited tail question", active_tail["id"])
    ))

    displayed = state_db.get_messages("compacted", include_compacted=True)
    assert sum(message["content"] == "Tail question" for message in displayed) == 1
    assert sum(message["content"] == "Edited tail question" for message in displayed) == 1
    _, resumed_display = state_db.get_resume_conversations("compacted")
    assert [(message["role"], message["content"]) for message in resumed_display] == [
        (message["role"], message["content"]) for message in displayed
    ]


def test_gateway_owner_probe_ignores_malformed_metadata(state_db):
    state_db.create_session("accepted-input", "telegram")
    state_db.append_message(
        "accepted-input", "user", "Accepted", display_metadata={"gateway_input_owner": "owner-1"}
    )
    malformed = state_db.append_message("accepted-input", "user", "Legacy metadata")
    state_db._execute_write(lambda connection: connection.execute(
        "UPDATE messages SET display_metadata = ? WHERE id = ?", ("{invalid json", malformed)
    ))
    assert state_db.has_gateway_input_owner("accepted-input", "owner-1")
    assert not state_db.has_gateway_input_owner("accepted-input", "owner-2")


def test_display_history_filters_model_only_rows_before_paging(state_db):
    from agent.context_compressor import MODEL_ONLY_DISPLAY_METADATA_KEY

    state_db.create_session("display-page", "cli")
    for content, metadata in (
        ("First question", None),
        ("Model context", {MODEL_ONLY_DISPLAY_METADATA_KEY: True}),
        ("Visible answer", {MODEL_ONLY_DISPLAY_METADATA_KEY: False}),
        ("Next question", None),
    ):
        state_db.append_message("display-page", "user", content, display_metadata=metadata)

    def displayed(**kwargs):
        return [row["content"] for row in state_db.get_messages(
            "display-page", include_compacted=True, **kwargs
        )]

    assert displayed() == ["First question", "Visible answer", "Next question"]
    assert displayed(offset=1) == ["Visible answer", "Next question"]
    assert displayed(limit=1, offset=1) == ["Visible answer"]
    assert displayed(limit=2, latest=True) == ["Visible answer", "Next question"]
    assert "Model context" in [row["content"] for row in state_db.get_messages("display-page")]


def test_verified_delete_rejects_changed_display_history(state_db):
    state_db.create_session("verified-delete", "cli")
    state_db.append_message("verified-delete", "user", "Exported question")
    snapshot = state_db.get_messages("verified-delete", include_compacted=True)
    state_db.append_message("verified-delete", "assistant", "New answer")

    assert not state_db.delete_session(
        "verified-delete", expected_display_messages={"verified-delete": snapshot}
    )
    assert state_db.get_session("verified-delete") is not None
    snapshot = state_db.get_messages("verified-delete", include_compacted=True)
    assert state_db.delete_session(
        "verified-delete", expected_display_messages={"verified-delete": snapshot}
    )
    assert state_db.get_session("verified-delete") is None


def test_gateway_routes_remain_isolated_by_scope(state_db):
    first = json.dumps({"session_id": "first"})
    second = json.dumps({"session_id": "second"})
    state_db.save_gateway_routing_entry("shared-key", first, scope="gateway-a")
    state_db.save_gateway_routing_entry("shared-key", second, scope="gateway-b")

    assert state_db.load_gateway_routing_entries(scope="gateway-a") == {"shared-key": first}
    assert state_db.load_gateway_routing_entries(scope="gateway-b") == {"shared-key": second}
    state_db.set_meta("route-version", "updated")
    assert state_db.get_meta("route-version") == "updated"


def test_compaction_preserves_messages_arriving_after_watermark(state_db):
    state_db.create_session("watermark", "cli")
    state_db.append_message("watermark", "user", "Earlier question")
    state_db.append_message("watermark", "assistant", "Earlier answer")
    watermark = state_db.get_active_message_watermark("watermark")
    original_id = state_db.append_message(
        "watermark", "user", "Arrived during compression", platform_message_id="late-platform-id",
        timestamp=1_800_000_000.125, display_metadata={"gateway_input_owner": "late-owner"},
    )

    state_db.archive_and_compact(
        "watermark", [{"role": "user", "content": "Summary"}], watermark=watermark
    )

    active = state_db.get_messages("watermark")
    assert [message["content"] for message in active] == ["Summary", "Arrived during compression"]
    assert active[-1]["id"] > original_id
    assert active[-1]["platform_message_id"] == "late-platform-id"
    assert active[-1]["timestamp"] == 1_800_000_000.125
    assert active[-1]["display_metadata"] == {"gateway_input_owner": "late-owner"}
    assert state_db.get_session("watermark")["message_count"] == len(active)


def test_existing_export_import_preserves_content_and_skips_duplicates(state_db):
    state_db.create_session("exported", "cli", model="test-model")
    state_db.append_message("exported", "user", "Portable history 世界")
    payload = state_db.export_session("exported")
    payload["id"] = "imported"
    result = state_db.import_sessions([payload])

    assert result["imported"] == 1
    assert state_db.get_messages_as_conversation("imported") == state_db.get_messages_as_conversation("exported")
    assert state_db.import_sessions([payload])["skipped"] == 1


def test_revision_changes_only_when_the_writer_commits(postgres_db):
    postgres_db.create_session("revision", "cli")
    writing = threading.Event()
    release = threading.Event()

    def update_title(connection):
        connection.execute("UPDATE sessions SET title = ? WHERE id = ?", ("Committed", "revision"))
        writing.set()
        assert release.wait(timeout=15)

    with SessionDB(postgres_db.db_path, read_only=True) as reader:
        original_revision = reader.get_change_revision()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(postgres_db._execute_write, update_title)
            try:
                assert writing.wait(timeout=10)
                assert reader.get_change_revision() == original_revision
                assert reader.get_session("revision")["title"] is None
            finally:
                release.set()
            future.result(timeout=10)

        committed_revision = reader.get_change_revision()
        assert committed_revision > original_revision
        assert reader.get_session("revision")["title"] == "Committed"

        def interrupted_write(connection):
            connection.execute("UPDATE sessions SET title = ? WHERE id = ?", ("Rolled back", "revision"))
            assert reader.get_change_revision() == committed_revision
            raise RuntimeError("interrupted transaction")

        with pytest.raises(RuntimeError, match="interrupted transaction"):
            postgres_db._execute_write(interrupted_write)
        assert reader.get_change_revision() == committed_revision
        assert reader.get_session("revision")["title"] == "Committed"


def test_live_foreign_pid_leases_can_only_be_reclaimed_after_expiry(postgres_db):
    import psutil

    assert not psutil.pid_exists(2147483647)
    postgres_db.create_session("remote-owner", "cli")
    holder = "host=another-machine:pid=2147483647:token=test"
    assert postgres_db.try_acquire_compression_lock("remote-owner", holder)
    assert postgres_db.try_acquire_session_turn_lease("remote-owner", holder)

    with SessionDB(postgres_db.db_path) as other:
        assert not other.try_acquire_compression_lock("remote-owner", "local-worker")
        assert not other.try_acquire_session_turn_lease("remote-owner", "local-worker")
        assert other.get_compression_lock_holder("remote-owner") == holder
        with pytest.raises(SessionTurnLeaseLostError):
            other.append_message("remote-owner", "user", "Unowned write", turn_lease_holder="local-worker")

        def expire_leases(connection):
            connection.execute("UPDATE compression_locks SET expires_at = 0 WHERE session_id = ?", ("remote-owner",))
            connection.execute("UPDATE session_turn_leases SET expires_at = 0 WHERE conversation_id = ?", ("remote-owner",))

        postgres_db._execute_write(expire_leases)
        assert other.try_acquire_compression_lock("remote-owner", "local-worker")
        assert other.try_acquire_session_turn_lease("remote-owner", "local-worker")


def test_readonly_open_does_not_initialize_and_concurrent_writers_initialize_once(postgres_home):
    import psycopg

    from hermes_state_backend import resolve_database_settings

    path = postgres_home / "state.db"
    settings = resolve_database_settings(path)
    with psycopg.connect(settings.database_url, autocommit=True) as connection:
        def schema_exists():
            return connection.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (settings.schema,)).fetchone()

        assert schema_exists() is None
        with pytest.raises(RuntimeError, match="PostgreSQL session database could not be opened"):
            SessionDB(path, read_only=True)
        assert schema_exists() is None

        barrier = threading.Barrier(4)

        def open_and_write(worker):
            barrier.wait(timeout=15)
            with SessionDB(path) as database:
                database.create_session(f"startup-{worker}", "cli")

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(open_and_write, range(4)))

        with SessionDB(path, read_only=True) as reader:
            assert reader.session_count() == 4
            assert {reader.get_session(f"startup-{worker}")["id"] for worker in range(4)} == {
                f"startup-{worker}" for worker in range(4)
            }
            assert reader._read_one("SELECT COUNT(*) FROM postgres_schema_version")[0] == 1


def test_registry_ignores_local_file_replacement_and_closes_after_final_release(postgres_home):
    import sqlite3
    from contextlib import closing

    from hermes_state_registry import acquire, close_all_under, release

    path = postgres_home / "state.db"
    replacement = postgres_home / "replacement.db"
    for local_path in (path, replacement):
        with closing(sqlite3.connect(local_path)) as connection:
            connection.execute("CREATE TABLE local_history (id INTEGER PRIMARY KEY)")

    try:
        first = acquire(path)
        first.create_session("shared", "cli")
        second = acquire(path)
        assert second is first
        replacement.replace(path)
        third = acquire(path)
        assert third is first

        first.close()
        assert release(second)
        assert third.get_session("shared")["id"] == "shared"
        third.close()
        assert first._pool.closed
        with pytest.raises(RuntimeError, match="connection is closed"):
            first.get_session("shared")

        with acquire(path) as reopened:
            assert reopened is not first
            assert reopened.get_session("shared")["id"] == "shared"
        assert reopened._pool.closed
    finally:
        close_all_under(postgres_home)


def test_postgres_bindings_preserve_literals_and_postgres_casts(postgres_db):
    value = "Bound ? :name 100% 'quoted' text"
    positional = postgres_db._read_one(
        "SELECT ?::text AS bound, '?' AS question, ':name' AS named_literal, "
        "'100%' AS percent, ?::bigint AS number /* ? :ignored 20% */ -- :ignored ?\n",
        (value, 17),
    )
    assert dict(positional) == {
        "bound": value, "question": "?", "named_literal": ":name", "percent": "100%", "number": 17,
    }
    named = postgres_db._read_one(
        "SELECT :value::text AS bound, ':value' AS literal, '50%' AS percent, "
        ":number::integer AS number, :value::text AS repeated",
        {"value": value, "number": 23},
    )
    assert dict(named) == {
        "bound": value, "literal": ":value", "percent": "50%", "number": 23, "repeated": value,
    }
