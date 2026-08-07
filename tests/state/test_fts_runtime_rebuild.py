"""Runtime FTS-corruption self-heal on the SessionDB write path (#65637 class).

A corrupted FTS5 shadow table (``messages_fts_data``) makes every message
write raise ``sqlite3.DatabaseError: database disk image is malformed``
through the FTS sync triggers, while the canonical ``messages`` rows stay
intact. Before this fix the gateway swallowed the failure at debug level and
the in-memory session advanced while disk silently fell behind — surfacing
later as "Persisted transcript lagged live cached history" amnesia.

The fix: ``_execute_write`` detects the malformed-image class, performs a
one-shot in-place FTS rebuild (FTS5 ``'rebuild'`` command — index rewritten
from canonical rows, no messages touched), and retries the failed write.
"""

import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    try:
        d.close()
    except Exception:
        pass


def _corrupt_fts(db_path):
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()


def _corrupt_trigram_fts(db_path):
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "UPDATE messages_fts_trigram_data "
        "SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()


def _message_contents(db_path):
    raw = sqlite3.connect(str(db_path))
    rows = raw.execute("SELECT content FROM messages ORDER BY id").fetchall()
    raw.close()
    return [r[0] for r in rows]


class TestRuntimeFtsRebuild:
    def test_corruption_error_classification_covers_both_sqlite_messages(self):
        """SQLite's message for a corrupt FTS index varies by version: older
        builds raise the generic malformed-image error, newer builds raise an
        FTS5-specific one. Both must trigger the self-heal."""
        assert SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError("database disk image is malformed")
        )
        assert SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError(
                'fts5: corrupt structure record for table "messages_fts"'
            )
        )
        assert not SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError("no such table: nothing_fts_related")
        )

    def test_append_self_heals_after_fts_corruption(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "hello world")

        _corrupt_fts(tmp_path / "state.db")

        # Before the fix this raised DatabaseError and the row was lost.
        msg_id = db.append_message("s1", "user", "healed append")
        assert msg_id is not None
        assert _message_contents(tmp_path / "state.db") == [
            "hello world",
            "healed append",
        ]

    def test_search_works_after_self_heal(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "before corruption")
        _corrupt_fts(tmp_path / "state.db")
        db.append_message("s1", "user", "searchable needle text")

        raw = sqlite3.connect(str(tmp_path / "state.db"))
        hits = raw.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'needle'"
        ).fetchall()
        raw.close()
        assert len(hits) == 1

    def test_search_messages_self_heals_after_fts_corruption(self, db, tmp_path):
        """A read-only session that only SEARCHES (no write after corruption)
        must self-heal too. The MATCH read raises the corruption class
        (DatabaseError / 'fts5: corrupt structure record'), NOT the
        OperationalError that search_messages caught — so before this fix the
        search crashed until a write or restart rebuilt the index.
        """
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "a searchable needle here")

        _corrupt_fts(tmp_path / "state.db")
        # Injected via a raw connection, so no write on THIS instance has
        # consumed the one-shot rebuild yet.
        assert db._fts_runtime_rebuild_attempted is False

        results = db.search_messages("needle")

        assert db._fts_runtime_rebuild_attempted is True  # the search rebuilt it
        assert results  # non-empty: the rebuilt index matched the query
        assert any("needle" in (r.get("snippet") or "") for r in results)

    def test_trigram_search_self_heals_after_fts_corruption(self, db, tmp_path):
        """The CJK/trigram MATCH branch has the same read-corruption exposure
        as the main FTS5 branch: it caught only OperationalError (query
        syntax), so a corrupt trigram shadow table raised DatabaseError
        straight out of search_messages. It must self-heal via the shared
        one-shot rebuild and answer from the rebuilt trigram index.
        """
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        if not db._trigram_available:
            pytest.skip("trigram tokenizer unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "关于大别山项目的进展报告")

        _corrupt_trigram_fts(tmp_path / "state.db")
        assert db._fts_runtime_rebuild_attempted is False

        # >=3 CJK chars per token → routed to the trigram branch.
        results = db.search_messages("大别山项目")

        assert db._fts_runtime_rebuild_attempted is True  # search rebuilt it
        assert results
        # The rebuilt trigram index answered (trigram snippets use >>> <<<),
        # i.e. we did not silently degrade to the LIKE fallback.
        assert any(">>>" in (r.get("snippet") or "") for r in results)


    def test_rebuild_is_one_shot_per_instance(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(tmp_path / "state.db")
        db.append_message("s1", "user", "first heal")  # consumes the one shot
        assert db._fts_runtime_rebuild_attempted is True

        # Corrupt again: the guard must NOT loop — the write now propagates.
        _corrupt_fts(tmp_path / "state.db")
        with pytest.raises(sqlite3.DatabaseError):
            db.append_message("s1", "user", "second corruption")
        # #78182: a rebuild that already ran, then a still-broken write path,
        # must mark recovery failure (not claim silent success).
        assert db._fts_write_path_broken is True


class TestPendingCapSpoolsOverflow:
    """Gateway pending-cap must not discard transcripts (#78182)."""

    def test_overflow_message_is_spooled_not_only_dropped(self, tmp_path, monkeypatch):
        import threading

        from gateway.session import SessionStore
        from gateway import shutdown_flush

        monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: tmp_path / "pending")
        (tmp_path / "pending").mkdir()

        class FakeDb:
            def rebuild_fts(self):
                return 0

            def append_message(self, **kwargs):
                raise RuntimeError("database disk image is malformed")

        store = object.__new__(SessionStore)
        store._db = FakeDb()
        store._transcript_retry_lock = threading.Lock()
        store._dirty_transcripts = {}
        store._transcript_append_failures = {}
        store._fts_rebuild_attempted = True

        for i in range(store._MAX_PENDING_PER_SESSION + 3):
            store.append_to_transcript("s1", {"role": "user", "content": f"msg{i}"})

        pending = store._dirty_transcripts.get("s1", [])
        assert len(pending) <= store._MAX_PENDING_PER_SESSION
        # Overflow was written under pending_messages/
        spools = list((tmp_path / "pending").glob("*.json"))
        assert len(spools) >= 3
        # Oldest messages should be among the spools
        texts = []
        for path in spools:
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
            texts.append(payload["data"]["text"])
        assert "msg0" in texts

    def test_spool_filenames_sort_in_write_order(self, tmp_path, monkeypatch):
        """#78323: recovery must re-insert burst spools in write order."""
        from gateway import shutdown_flush

        monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: tmp_path / "pending")
        (tmp_path / "pending").mkdir()

        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        assert shutdown_flush.spool_transcript_messages("sess-order", msgs) is True
        names = sorted(p.name for p in (tmp_path / "pending").glob("*.json"))
        # Lexicographic order of filenames must match write order.
        texts = []
        for name in names:
            payload = __import__("json").loads(
                (tmp_path / "pending" / name).read_text(encoding="utf-8")
            )
            texts.append(payload["data"]["text"])
        assert texts == ["first", "second", "third"]

    def test_tool_call_row_spools_and_recovers_without_text(
        self, tmp_path, monkeypatch
    ):
        """#78323: assistant tool_calls with content=None must recover intact."""
        from gateway import shutdown_flush

        monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: tmp_path / "pending")
        (tmp_path / "pending").mkdir()

        tool_row = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }
            ],
            "timestamp": 1_700_000_000,
        }
        assert (
            shutdown_flush.spool_transcript_messages("sess-tools", [tool_row]) is True
        )

        calls = []

        class FakeDb:
            def append_message(self, **kwargs):
                calls.append(kwargs)
                return 1

        recovered = shutdown_flush.recover_pending_to_db(FakeDb())
        assert recovered == 1
        assert len(calls) == 1
        assert calls[0]["session_id"] == "sess-tools"
        assert calls[0]["role"] == "assistant"
        assert calls[0]["content"] is None
        assert calls[0]["tool_calls"] == tool_row["tool_calls"]
        assert calls[0]["timestamp"] == 1_700_000_000
        # File removed on success — no perpetual startup warning.
        assert list((tmp_path / "pending").glob("*.json")) == []


