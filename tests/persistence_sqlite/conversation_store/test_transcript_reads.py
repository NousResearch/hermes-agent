"""Transcript reads in the conversation store on a backend store's StoreHolder: ``get_messages`` pagination and the resume and export size guards (the export guard through the maintenance repository, limits as the callers read them from ``storage_limits``), lone surrogates and display metadata across every read path and an export/import round trip, unknown BLOB columns, ancestor reads along a compression lineage, and handoff timestamps carried through compaction."""

import json

import pytest

import storage_limits
from hermes_cli.session_export import to_legacy_export_dict
from persistence.core.errors import ExportTooLarge
from persistence.sqlite.conversation._store.api_content import set_latest_user_api_content
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.core import SessionResumeTooLargeError
from persistence.sqlite.conversation._store.messages import (
    _resume_lineage_ids,
    append_message,
    append_messages_batch,
    archive_and_compact,
    assert_resume_safe,
    get_ancestor_display_prefix,
    get_messages,
    get_messages_around,
    get_messages_as_conversation,
    get_resume_conversations,
    replace_messages,
)
from persistence.sqlite.conversation._store.search import get_anchored_view
from persistence.sqlite.conversation._store.sessions import create_session, end_session


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def _resume_count(db, session_id, *, tip_only=False):
    """The rows a resume would load: the resume guard's count under a limit no test reaches."""
    return assert_resume_safe(db, session_id, max_messages=1_000_000, tip_only=tip_only)


# =========================================================================
# get_messages pagination (salvage follow-up for #60347)
# =========================================================================

class TestGetMessagesPagination:
    """get_messages(limit=, offset=) pages in insertion order; the default
    (limit=None) returns the full transcript unchanged."""

    def _seed(self, db, n=10):
        create_session(db, session_id="s1", source="cli")
        # One write transaction for the whole seed: per-row append_message
        # pays a commit (and, off WAL, an fsync) per message, which at
        # n=3000 was ~10s of pure seeding before the query under test ran.
        append_messages_batch(db,
            "s1",
            [
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"msg-{i}",
                }
                for i in range(n)
            ],
        )

    def test_default_returns_all_messages(self, db):
        self._seed(db)
        messages = get_messages(db, "s1")
        assert [m["content"] for m in messages] == [f"msg-{i}" for i in range(10)]

    def test_window_query_bounded_work(self, db):
        """Perf contract: get_messages_around must seek by index, not scan
        the session's whole message history. Measured behaviorally via
        SQLite progress-handler steps (behavior contracts over snapshots,
        AGENTS.md — no EXPLAIN text). Calibrated on this seed (3000
        messages): indexed = ~12 handler calls, unindexed full-session
        scan = ~855. Threshold 300: >25x headroom above the indexed path,
        ~3x below the scan. Same pattern as the loader call-count pins in
        tests/tools/test_approval_config_readonly.py."""
        self._seed(db, n=3000)
        mid = get_messages(db, "s1", limit=1, offset=1500)[0]["id"]
        steps = [0]

        def progress():
            steps[0] += 1
            return 0

        db._conn.set_progress_handler(progress, 100)
        try:
            get_messages_around(db, "s1", mid, window=20)
        finally:
            db._conn.set_progress_handler(None, 0)
        assert steps[0] < 300, (
            f"get_messages_around executed {steps[0]}x100 VM steps — the "
            "session-history scan is back (idx_messages_session_id missing "
            "or unused)")

    def test_window_results_identical_with_and_without_index(self, db):
        """The index must not change results: identical windows at probe
        points across the session, with and without it."""
        self._seed(db, n=500)
        ids = [m["id"] for m in get_messages(db, "s1")]
        probes = (ids[0], ids[len(ids) // 2], ids[-1])
        with_index = [get_messages_around(db, "s1", mid, window=5)
                      for mid in probes]
        db._conn.execute("DROP INDEX idx_messages_session_id")
        without_index = [get_messages_around(db, "s1", mid, window=5)
                         for mid in probes]
        assert with_index == without_index

    def test_limit_pages_in_insertion_order(self, db):
        self._seed(db)
        page1 = get_messages(db, "s1", limit=4, offset=0)
        page2 = get_messages(db, "s1", limit=4, offset=4)
        page3 = get_messages(db, "s1", limit=4, offset=8)
        assert [m["content"] for m in page1] == ["msg-0", "msg-1", "msg-2", "msg-3"]
        assert [m["content"] for m in page2] == ["msg-4", "msg-5", "msg-6", "msg-7"]
        assert [m["content"] for m in page3] == ["msg-8", "msg-9"]

    def test_latest_pages_count_back_from_newest_but_remain_chronological(self, db):
        self._seed(db)
        page1 = get_messages(db, "s1", limit=4, offset=0, latest=True)
        page2 = get_messages(db, "s1", limit=4, offset=4, latest=True)
        page3 = get_messages(db, "s1", limit=4, offset=8, latest=True)
        assert [m["content"] for m in page1] == ["msg-6", "msg-7", "msg-8", "msg-9"]
        assert [m["content"] for m in page2] == ["msg-2", "msg-3", "msg-4", "msg-5"]
        assert [m["content"] for m in page3] == ["msg-0", "msg-1"]

    def test_after_id_keyset_pages_forward_in_insertion_order(self, db):
        self._seed(db)
        page1 = get_messages(db, "s1", limit=4, after_id=0)
        assert [m["content"] for m in page1] == ["msg-0", "msg-1", "msg-2", "msg-3"]
        page2 = get_messages(db, "s1", limit=4, after_id=page1[-1]["id"])
        assert [m["content"] for m in page2] == ["msg-4", "msg-5", "msg-6", "msg-7"]
        page3 = get_messages(db, "s1", limit=4, after_id=page2[-1]["id"])
        assert [m["content"] for m in page3] == ["msg-8", "msg-9"]
        with pytest.raises(ValueError):
            get_messages(db, "s1", limit=4, after_id=0, latest=True)
        with pytest.raises(ValueError):
            get_messages(db, "s1", limit=4, after_id=0, offset=2)

    def test_resume_safety_counts_active_rows_across_lineage(self, db):
        create_session(db, session_id="root", source="cli")
        append_messages_batch(db,
            "root",
            [{"role": "user", "content": f"root-{i}"} for i in range(3)],
        )
        # A real rotation stamps the parent end_reason='compression' (publish_compression_child).
        end_session(db, "root", "compression")
        create_session(db,
            session_id="tip",
            source="compression",
            parent_session_id="root",
        )
        append_messages_batch(db,
            "tip",
            [{"role": "assistant", "content": f"tip-{i}"} for i in range(2)],
        )

        assert _resume_count(db, "tip") == 5
        with pytest.raises(SessionResumeTooLargeError) as exc_info:
            assert_resume_safe(db, "tip", max_messages=4)
        assert exc_info.value.message_count == 5
        assert exc_info.value.limit == 4

    def test_resume_safety_tip_only_counts_the_tip_segment(self, db):
        """A deep compression lineage behind a small tip resumes tip-only.

        The Desktop Bot Chat shape: many compaction segments (~29k rows of
        lineage) and a small live tip. Callers that never materialize the
        ancestors (deferred / omit_messages / lazy resume, tip-only model
        restore) must be bounded by the tip alone, and the message must name
        the scope it counted.
        """
        prev = None
        for i in range(6):
            sid = f"seg-{i}"
            kwargs = {"parent_session_id": prev} if prev else {}
            create_session(db, session_id=sid, source="tui", **kwargs)
            append_messages_batch(db,
                sid,
                [{"role": "user", "content": f"{sid}-{j}"} for j in range(4)],
            )
            if i < 5:
                end_session(db, sid, "compression")
            prev = sid

        assert _resume_count(db, "seg-5") == 24
        assert _resume_count(db, "seg-5", tip_only=True) == 4
        with pytest.raises(SessionResumeTooLargeError) as full:
            assert_resume_safe(db, "seg-5", max_messages=10)
        assert full.value.scope == "across its lineage"
        assert assert_resume_safe(db, "seg-5", max_messages=10, tip_only=True) == 4
        with pytest.raises(SessionResumeTooLargeError) as tip:
            assert_resume_safe(db, "seg-5", max_messages=3, tip_only=True)
        assert tip.value.message_count == 4
        assert tip.value.scope == "in its tip segment"

    def test_resume_guard_counts_exactly_what_a_branch_resume_loads(self, db):
        """An explicit /branch copy owns its transcript: the guard and the
        resume readers must agree that its lineage is itself alone."""
        create_session(db, session_id="parent", source="tui")
        append_messages_batch(db,
            "parent",
            [{"role": "user", "content": f"parent-{i}"} for i in range(6)],
        )
        create_session(db,
            session_id="branch",
            source="tui",
            parent_session_id="parent",
            model_config={"_branched_from": "parent"},
        )
        append_messages_batch(db,
            "branch",
            [{"role": "user", "content": f"branch-{i}"} for i in range(2)],
        )

        _, display = get_resume_conversations(db, "branch")
        assert len(display) == 2
        assert get_ancestor_display_prefix(db, "branch") == []
        # Before: the guard walked parent_session_id and counted 8, so a branch
        # could be refused for rows a resume would never load.
        assert _resume_count(db, "branch") == 2
        assert assert_resume_safe(db, "branch", max_messages=5) == 2

    def test_export_safety_is_bounded_to_the_requested_active_segment(self, db, store):
        create_session(db, session_id="root", source="cli")
        append_messages_batch(db,
            "root",
            [{"role": "user", "content": f"root-{i}"} for i in range(3)],
        )
        create_session(db,
            session_id="tip",
            source="compression",
            parent_session_id="root",
        )
        append_messages_batch(db,
            "tip",
            [{"role": "assistant", "content": f"tip-{i}"} for i in range(2)],
        )

        assert store.maintenance.assert_export_safe("tip", max_messages=2) == 2
        with pytest.raises(ExportTooLarge) as exc_info:
            store.maintenance.assert_export_safe("root", max_messages=2)
        assert exc_info.value.session_id == "root"
        assert exc_info.value.message_count == 3
        assert exc_info.value.limit == 2

    def test_zero_limit_disables_resume_and_export_guards(self, db, store, monkeypatch):
        """sessions.max_*_messages: 0 disables the guard entirely."""
        create_session(db, session_id="big", source="cli")
        append_messages_batch(db,
            "big",
            [{"role": "user", "content": f"msg-{i}"} for i in range(5)],
        )

        # A small explicit limit rejects...
        with pytest.raises(SessionResumeTooLargeError):
            assert_resume_safe(db, "big", max_messages=2)
        with pytest.raises(ExportTooLarge):
            store.maintenance.assert_export_safe("big", max_messages=2)

        # ...but a config-resolved limit of 0 disables both guards: no raise,
        # and no counting work at all (returns 0 — callers use the raise side
        # effect only).
        monkeypatch.setattr(storage_limits, "resolved_max_resume_messages", lambda: 0)
        monkeypatch.setattr(storage_limits, "resolved_max_export_messages", lambda: 0)
        assert assert_resume_safe(db, "big", max_messages=storage_limits.resolved_max_resume_messages()) == 0
        assert store.maintenance.assert_export_safe("big", max_messages=storage_limits.resolved_max_export_messages()) == 0
        # An explicit 0 disables too, independent of config.
        assert assert_resume_safe(db, "big", max_messages=0) == 0
        assert store.maintenance.assert_export_safe("big", max_messages=0) == 0

    def test_guard_limits_resolve_from_config_at_call_time(self, db, store, monkeypatch):
        create_session(db, session_id="cfg", source="cli")
        append_messages_batch(db,
            "cfg",
            [{"role": "user", "content": f"msg-{i}"} for i in range(4)],
        )

        monkeypatch.setattr(storage_limits, "resolved_max_resume_messages", lambda: 3)
        monkeypatch.setattr(storage_limits, "resolved_max_export_messages", lambda: 3)
        with pytest.raises(SessionResumeTooLargeError) as resume_exc:
            assert_resume_safe(db, "cfg", max_messages=storage_limits.resolved_max_resume_messages())
        assert resume_exc.value.limit == 3
        with pytest.raises(ExportTooLarge) as export_exc:
            store.maintenance.assert_export_safe("cfg", max_messages=storage_limits.resolved_max_export_messages())
        assert export_exc.value.limit == 3


# =========================================================================
# Lone-surrogate persistence
# =========================================================================

class TestLoneSurrogatePersistence:
    """sqlite3 encodes bound str params as UTF-8 and raises UnicodeEncodeError
    on lone surrogates (U+D800..U+DFFF). Tool results scraped from the web can
    carry them, so a single such code point aborted the whole message write —
    and because run_agent swallows the failure with a warning, the session then
    silently stopped persisting for the rest of its life.
    """

    DIRTY = "scraped \ud835 price"

    def test_append_message_survives_lone_surrogate_content(self, db):
        create_session(db, "s1", source="cli")
        append_message(db, "s1", "assistant", "hello world")
        append_message(db, "s1", "tool", self.DIRTY, tool_name="web_search")

        rows = get_messages(db, "s1")
        assert len(rows) == 2
        # Surrogate replaced with U+FFFD; the surrounding text is intact.
        assert rows[1]["content"] == "scraped � price"

    # -- sibling raw-str bind sites (follow-up widening of the same bug class)

    def test_set_latest_user_api_content_survives_lone_surrogate(self, db):
        create_session(db, "s1", source="cli")
        append_message(db, "s1", "user", "turn text")
        assert set_latest_user_api_content(db, "s1", "turn text", self.DIRTY) == 1


class TestDisplayMetadataPersistence:
    """Round-trip display_kind/display_metadata through every write path."""

    def test_append_message_round_trips_display_fields(self, db):
        create_session(db, "s1", source="cli")
        meta = {"task_count": 2, "delegation_id": "del-1"}
        append_message(db,
            "s1", "user", "event text",
            display_kind="async_delegation_complete",
            display_metadata=meta,
        )
        conv = get_messages_as_conversation(db, "s1")
        assert conv[0]["display_kind"] == "async_delegation_complete"
        assert conv[0]["display_metadata"] == meta

    def test_replace_messages_preserves_display_metadata(self, db):
        create_session(db, "s1", source="cli")
        meta = {"task_count": 3, "delegation_id": "del-2", "duration_seconds": 12.5}
        append_message(db,
            "s1", "user", "event",
            display_kind="async_delegation_complete",
            display_metadata=meta,
        )
        # Reload via get_messages_as_conversation (which decodes display fields)
        # then replace_messages (which re-inserts via _insert_message_rows).
        conv = get_messages_as_conversation(db, "s1")
        replace_messages(db, "s1", conv)
        reloaded = get_messages_as_conversation(db, "s1")
        assert reloaded[0]["display_kind"] == "async_delegation_complete"
        assert reloaded[0]["display_metadata"] == meta


class TestDisplayMetadataReadPaths:
    """Every message read path must hand back the decoded dict.

    Returning the raw column instead reaches the desktop as a string, where
    ``'task_count' in meta`` throws and fails the whole session resume.
    """

    META = {
        "delegation_id": "deleg_0d84d484",
        "task_count": 1,
        "completed_count": 1,
        "failed_count": 0,
        "duration_seconds": 193.55,
    }

    @staticmethod
    def _seed(db):
        create_session(db, "s1", source="desktop")
        message_id = append_message(db,
            "s1", "user", "event",
            display_kind="async_delegation_complete",
            display_metadata=TestDisplayMetadataReadPaths.META,
        )
        return message_id, append_message(db, "s1", "assistant", "anchor")

    @staticmethod
    def _read(db, reader, message_id, anchor_id):
        if reader == "get_messages":
            return get_messages(db, "s1")[0]
        if reader == "get_messages_around":
            return get_messages_around(db, "s1", message_id, window=0)["window"][0]
        if reader == "get_anchored_view":
            view = get_anchored_view(db, "s1", anchor_id, window=0, bookend=1)
            return view["bookend_start"][0]
        return get_messages_as_conversation(db, "s1")[0]

    READERS = ("get_messages", "get_messages_around", "get_anchored_view", "conversation")

    @pytest.mark.parametrize("reader", READERS)
    def test_every_reader_decodes_display_metadata(self, db, reader):
        message_id, anchor_id = self._seed(db)
        assert self._read(db, reader, message_id, anchor_id)["display_metadata"] == self.META

    @pytest.mark.parametrize("reader", READERS)
    @pytest.mark.parametrize("raw", ["", "{not-json", "[]", '"text"', "0"])
    def test_every_reader_drops_unusable_display_metadata(self, db, reader, raw):
        """Bad presentation metadata must not take the message down with it."""
        message_id, anchor_id = self._seed(db)

        def _corrupt(conn):
            conn.execute(
                "UPDATE messages SET display_metadata = ? WHERE id = ?",
                (raw, message_id),
            )

        db._execute_write(_corrupt)
        message = self._read(db, reader, message_id, anchor_id)
        assert message.get("display_metadata") is None
        assert message["content"] == "event"

    def test_export_import_round_trip_keeps_metadata_decodable(self, db, store, backend, scope_b):
        """The read leak used to write a permanently double-encoded row here.

        ``export_session`` reads through ``get_messages``, so an undecoded
        string went back through ``_insert_message_rows`` and got re-dumped.
        """
        self._seed(db)
        blob = to_legacy_export_dict(store.maintenance.export_session("s1"))
        assert isinstance(blob["messages"][0]["display_metadata"], dict)

        target = backend.open(scope_b)
        target.maintenance.import_sessions([json.loads(json.dumps(blob))])
        assert get_messages_as_conversation(target.holder(), "s1")[0]["display_metadata"] == self.META
        assert get_messages(target.holder(), "s1")[0]["display_metadata"] == self.META


class TestUnknownBlobColumnSurvivesRead:
    """A `messages` column added by a future migration must not take every reader down with it.

    Every reader here does ``SELECT *``, so a BLOB column reaches the dict unfiltered. FastAPI's
    response encoder calls ``.decode()`` on any raw ``bytes`` value and dies with
    ``UnicodeDecodeError`` the moment the bytes are not valid utf-8 — this already happened for
    the ``display_identity BLOB`` column before it got an explicit pop;
    the next binary column would repeat it with no reader-side defense. See #116510.
    """

    @staticmethod
    def _seed_with_future_blob(db):
        create_session(db, "s1", source="desktop")
        message_id = append_message(db, "s1", "user", "hello")

        def _migrate(conn):
            conn.execute("ALTER TABLE messages ADD COLUMN future_blob BLOB")
            conn.execute(
                "UPDATE messages SET future_blob = ? WHERE id = ?", (b"\xff\xfe not utf-8", message_id))

        db._execute_write(_migrate)
        return message_id

    def test_get_messages_drops_unknown_blob_and_stays_json_safe(self, db):
        self._seed_with_future_blob(db)
        messages = get_messages(db, "s1")
        assert messages[0]["content"] == "hello"
        assert "future_blob" not in messages[0]
        json.dumps(messages)  # raises TypeError on a raw bytes value, same class of failure as FastAPI's encoder

    def test_get_messages_around_drops_unknown_blob_and_stays_json_safe(self, db):
        message_id = self._seed_with_future_blob(db)
        window = get_messages_around(db, "s1", message_id)["window"]
        assert "future_blob" not in window[0]
        json.dumps(window)

    def test_schema_column_holding_bytes_keeps_its_key(self, db):
        """The bytes pop is for columns this module does not know. A schema column such as
        ``content`` must never vanish from the dict: every resume/compaction reader indexes
        ``msg["content"]`` and a KeyError there is worse than the raw value it replaced."""
        create_session(db, "s1", source="cli")
        message_id = append_message(db, "s1", "user", "hello")
        db._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET content = X'FFFE' WHERE id = ?", (message_id,)))
        (message,) = get_messages(db, "s1")
        assert message["content"] == b"\xff\xfe"
        assert message["role"] == "user"


class TestGetMessagesAncestors:
    """get_messages(include_ancestors=True) spans the compression lineage (#51058)."""

    def test_include_ancestors_returns_lineage_messages(self, db):
        """After a compression rotation, include_ancestors=True returns the
        parent's rows plus the child continuation in insertion order."""
        create_session(db, "parent", source="tui")
        append_message(db, "parent", role="user", content="early ask")
        append_message(db, "parent", role="assistant", content="early answer")
        # A real rotation stamps the parent end_reason='compression' (publish_compression_child).
        end_session(db, "parent", "compression")

        create_session(db, "child", source="tui", parent_session_id="parent")
        append_message(db, "child", role="user", content="continuation ask")
        append_message(db, "child", role="assistant", content="continuation answer")

        # Without the flag: only the child continuation (legacy behaviour).
        assert [m["content"] for m in get_messages(db, "child")] == [
            "continuation ask",
            "continuation answer",
        ]
        # With the flag: the full root→tip transcript, insertion order.
        assert [m["content"] for m in get_messages(db, "child", include_ancestors=True)] == [
            "early ask",
            "early answer",
            "continuation ask",
            "continuation answer",
        ]

    def test_include_ancestors_explicit_branch_stays_single_session(self, db):
        """Explicit /branch copies own their copied transcript: the lineage
        expansion must not pull the parent's rows in (parity with
        get_messages_as_conversation)."""
        create_session(db, "parent", source="tui")
        append_message(db, "parent", role="user", content="before branch")
        create_session(db,
            "branch",
            source="tui",
            parent_session_id="parent",
            model_config={"_branched_from": "parent"},
        )
        append_message(db, "branch", role="user", content="copied turn")

        assert [m["content"] for m in get_messages(db, "branch", include_ancestors=True)] == [
            "copied turn"
        ]

    def test_include_ancestors_api_fork_stays_single_session(self, db):
        """An API fork (parent ended 'branched', transcript copied, NO _branched_from
        marker — what _handle_fork_session produced before the marker fix) must not be
        mistaken for a compression lineage: ancestor expansion returns the fork's own
        copied transcript only, never the parent's rows re-prepended (which duplicated
        every turn and produced the [assistant, user, assistant] order ehz0ah hit)."""
        create_session(db, "source", source="tui")
        append_message(db, "source", role="user", content="question")
        append_message(db, "source", role="assistant", content="answer")
        end_session(db, "source", "branched")
        # Exactly what the API fork handler does, minus the _branched_from marker.
        create_session(db, "fork", source="api_server", parent_session_id="source")
        replace_messages(db, "fork", get_messages(db, "source"))

        assert [m["content"] for m in get_messages(db, "fork", include_ancestors=True)] == [
            "question",
            "answer",
        ]
        conversation = get_messages_as_conversation(db, "fork", include_ancestors=True)
        assert [m["role"] for m in conversation] == ["user", "assistant"]
        assert [m["content"] for m in conversation] == ["question", "answer"]

    def test_include_ancestors_reset_child_stays_single_session(self, db):
        """A /reset continuation (_reset_from marker, parent ended 'session_reset') is a
        fresh conversation, not a compression lineage: ancestor expansion must not cross
        the reset boundary."""
        create_session(db, "parent", source="tui")
        append_message(db, "parent", role="user", content="old conversation")
        end_session(db, "parent", "session_reset")
        create_session(db,
            "child",
            source="tui",
            parent_session_id="parent",
            model_config={"_reset_from": "parent"},
        )
        append_message(db, "child", role="user", content="fresh start")

        assert [m["content"] for m in get_messages(db, "child", include_ancestors=True)] == [
            "fresh start"
        ]
        conversation = get_messages_as_conversation(db, "child", include_ancestors=True)
        assert [m["content"] for m in conversation] == ["fresh start"]

    def test_include_ancestors_deep_lineage_verified_chain(self, db):
        """A multi-hop compression lineage merges root→tip, and a plain (non-compression)
        grandparent above the root is not crossed: only verified compression hops count."""
        create_session(db, "grandparent", source="tui")
        append_message(db, "grandparent", role="user", content="plain old turn")
        end_session(db, "grandparent", "stopped")
        create_session(db, "root", source="tui", parent_session_id="grandparent")
        append_message(db, "root", role="user", content="root turn")
        end_session(db, "root", "compression")
        create_session(db, "mid", source="tui", parent_session_id="root")
        append_message(db, "mid", role="user", content="mid turn")
        end_session(db, "mid", "compression")
        create_session(db, "tip", source="tui", parent_session_id="mid")
        append_message(db, "tip", role="user", content="tip turn")

        assert _resume_lineage_ids(db, "tip") == ["root", "mid", "tip"]
        assert [m["content"] for m in get_messages(db, "tip", include_ancestors=True)] == [
            "root turn",
            "mid turn",
            "tip turn",
        ]
        conversation = get_messages_as_conversation(db, "tip", include_ancestors=True)
        assert [m["content"] for m in conversation] == ["root turn", "mid turn", "tip turn"]

    def test_include_ancestors_paging_still_applies(self, db):
        """Paging applies to the merged lineage set, not per-session."""
        create_session(db, "parent", source="tui")
        append_message(db, "parent", role="user", content="p1")
        # A real rotation stamps the parent end_reason='compression' (publish_compression_child).
        end_session(db, "parent", "compression")
        create_session(db, "child", source="tui", parent_session_id="parent")
        for i in range(2, 6):
            append_message(db, "child", role="user", content=f"c{i}")

        rows = get_messages(db, "child", include_ancestors=True, limit=3, offset=0)
        assert [m["content"] for m in rows] == ["p1", "c2", "c3"]
        rows = get_messages(db, "child", include_ancestors=True, limit=3, offset=3)
        assert [m["content"] for m in rows] == ["c4", "c5"]


class TestGetMessagesAncestorsGuards:
    """Paging guards for the merged-lineage read (get_messages)."""

    def test_after_id_rejected_with_include_ancestors(self, db):
        """after_id keyset paging must not silently drop the cursor on a merged
        lineage read (the multi-segment set is deduped before paging, so the
        per-row cursor is not meaningful across segments)."""
        create_session(db, "parent", source="tui")
        append_message(db, "parent", role="user", content="p1")
        end_session(db, "parent", "compression")
        create_session(db, "child", source="tui", parent_session_id="parent")
        append_message(db, "child", role="user", content="c1")
        anchor = get_messages(db, "child")[0]

        with pytest.raises(ValueError, match="after_id is incompatible with include_ancestors"):
            get_messages(db, "child", include_ancestors=True, after_id=anchor["id"])


class TestHandoffTimestampCarry:
    """#59661: carried handoff rows keep the durable parent row's timestamp so
    ``_display_dedupe_key`` collapses each logical message once in the lineage read."""

    @staticmethod
    def _rotate(db, parent_id, child_id, handoff):
        publish_compression_child(db,
            parent_session_id=parent_id, child_session_id=child_id, source="test",
            messages=handoff, require_compression_lease=False)

    def test_rotation_handoff_tail_appears_once_across_the_lineage(self, db):
        """The probe shape: parent tail rows re-inserted as child rows without a timestamp must
        not duplicate in the display lineage read."""
        create_session(db, "root", source="test")
        for role, text in (("user", "u0"), ("assistant", "a0"), ("user", "u4"), ("assistant", "a4"),
                           ("user", "u5"), ("assistant", "a5")):
            append_message(db, "root", role=role, content=text)
        handoff = [{"role": "assistant", "content": "SUMMARY"},
                   {"role": "user", "content": "u4"},
                   {"role": "assistant", "content": "a4"},
                   {"role": "user", "content": "u5"},
                   {"role": "assistant", "content": "a5"}]

        self._rotate(db, "root", "child", handoff)

        lineage = get_messages(db, "child", include_ancestors=True)
        contents = [(m["role"], m["content"]) for m in lineage]
        assert len(contents) == len(set(contents)) == 7  # each logical message exactly once
        assert [m["content"] for m in lineage] == ["u0", "a0", "u4", "a4", "u5", "a5", "SUMMARY"]
        # The carry is observable on the caller's dicts: carried rows hold the parent's durable ts.
        parent_ts = {m["content"]: m["timestamp"] for m in get_messages(db, "root")}
        for carried in handoff[1:]:
            assert carried["timestamp"] == parent_ts[carried["content"]]

    def test_handoff_dedupe_across_three_rotation_boundaries(self, db):
        """The same tail dicts carried root -> seg1 -> seg2 -> seg3 (reused across publishes, as
        production does) collapse to exactly one copy of each logical message in seg3's lineage."""
        create_session(db, "root", source="test")
        for role, text in (("user", "u0"), ("assistant", "a0"),
                           ("user", "tail-u"), ("assistant", "tail-a")):
            append_message(db, "root", role=role, content=text)
        tail = [{"role": "user", "content": "tail-u"}, {"role": "assistant", "content": "tail-a"}]
        for child_id, parent_id in (("seg1", "root"), ("seg2", "seg1"), ("seg3", "seg2")):
            handoff = [{"role": "assistant", "content": f"SUMMARY-{child_id}"}, *tail]
            self._rotate(db, parent_id, child_id, handoff)

        lineage = get_messages(db, "seg3", include_ancestors=True)
        contents = [(m["role"], m["content"]) for m in lineage]
        assert len(contents) == len(set(contents)) == 7
        assert [m["content"] for m in lineage] == [
            "u0", "a0", "tail-u", "tail-a", "SUMMARY-seg1", "SUMMARY-seg2", "SUMMARY-seg3"]

    def test_distinct_same_content_turns_are_not_collapsed(self, db):
        """Repeated identical turns are DISTINCT logical messages: the identity collapse must never
        merge two turns that merely share content."""
        create_session(db, "s", source="test")
        append_messages_batch(db, "s", [{"role": "user", "content": "ok"}, {"role": "user", "content": "ok"}])
        # One batch: the second row's timestamp is bumped past the first (:540 monotonic bump).
        assert [m["content"] for m in get_messages(db, "s", include_compacted=True)] == ["ok", "ok"]

        # Cross-boundary: a carried "ok" and a fresh post-rotation "ok" both display.
        create_session(db, "root", source="test")
        append_message(db, "root", role="user", content="ok")
        self._rotate(db, "root", "seg", [
            {"role": "assistant", "content": "SUMMARY"},
            {"role": "user", "content": "ok"},  # carried: adopts the root row's timestamp
        ])
        append_message(db, "seg", role="user", content="ok")  # fresh turn: its own newer timestamp
        oks = [m for m in get_messages(db, "seg", include_ancestors=True) if m["content"] == "ok"]
        assert len(oks) == 2

    def test_carry_parent_timestamps_does_not_overwrite_carried_timestamp(self, db):
        """Handoff dicts that already carry a timestamp are never clobbered by the carry."""
        create_session(db, "root", source="test")
        append_message(db, "root", role="user", content="u1")
        append_message(db, "root", role="assistant", content="a1")
        stored = {m["content"]: m["timestamp"] for m in get_messages(db, "root")}
        handoff = [{"role": "assistant", "content": "SUMMARY"},
                   {"role": "user", "content": "u1", "timestamp": stored["u1"]},
                   {"role": "assistant", "content": "a1", "timestamp": stored["a1"] + 500.0},
                   {"role": "user", "content": "fresh", "timestamp": 1234.5}]

        self._rotate(db, "root", "child", handoff)

        assert handoff[1]["timestamp"] == stored["u1"]
        assert handoff[2]["timestamp"] == stored["a1"] + 500.0
        assert handoff[3]["timestamp"] == 1234.5
        contents = [(m["role"], m["content"])
                    for m in get_messages(db, "child", include_ancestors=True)]
        # u1 collapses (carried the durable ts); the offset a1 does not (carried ts is authoritative).
        assert contents.count(("user", "u1")) == 1
        assert contents.count(("assistant", "a1")) == 2
        assert contents.count(("user", "fresh")) == 1

    def test_archive_and_compact_carried_rows_keep_identity(self, db):
        """In-place double compaction: a row carried through both compactions keeps a stable display
        identity (same durable timestamp) and never duplicates in the display read."""
        create_session(db, "s", source="test")
        for role, text in (("user", "u0"), ("assistant", "a0"),
                           ("user", "keep me"), ("assistant", "tail-a")):
            append_message(db, "s", role=role, content=text)
        carried = [{"role": "user", "content": "keep me"}, {"role": "assistant", "content": "tail-a"}]
        archive_and_compact(db, "s", [{"role": "assistant", "content": "S1"}, *carried], tail_count=2)
        first_ts = {m["content"]: m["timestamp"] for m in get_messages(db, "s")}

        append_messages_batch(db,
            "s", [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}])
        archive_and_compact(db,
            "s",
            [{"role": "assistant", "content": "S2"}, *carried,
             {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}],
            tail_count=4)

        visible = get_messages(db, "s", include_compacted=True)
        contents = [(m["role"], m["content"]) for m in visible]
        assert len(contents) == len(set(contents))
        assert [m["content"] for m in visible] == ["u0", "a0", "S1", "S2", "keep me", "tail-a", "u2", "a2"]
        second_ts = {m["content"]: m["timestamp"] for m in get_messages(db, "s")}
        assert second_ts["keep me"] == first_ts["keep me"]
        assert second_ts["tail-a"] == first_ts["tail-a"]
