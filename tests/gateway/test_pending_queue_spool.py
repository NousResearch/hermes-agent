"""Regression tests for runtime spool-on-drop of the pending transcript queue.

When the per-session pending cap (``_MAX_PENDING_PER_SESSION``) forces the
gateway to evict the oldest queued transcript message during live operation,
the message must be spooled to the on-disk pending spool (the same machinery
``flush_pending_to_file`` uses at shutdown) and replayed on the next
successful transcript flush — not silently discarded (#78182, #82616).
"""
import json
import logging
import threading

import pytest

from gateway import shutdown_flush
from gateway.session import SessionStore


def _make_store(db):
    store = object.__new__(SessionStore)
    store._db = db
    store._transcript_retry_lock = threading.Lock()
    store._dirty_transcripts = {}
    store._transcript_append_failures = {}
    store._fts_rebuild_attempted = True
    return store


class BrokenThenHealedDb:
    """append_message fails while ``broken`` is True, then records rows."""

    def __init__(self):
        self.broken = True
        self.rows = []

    def append_message(self, **kwargs):
        if self.broken:
            raise RuntimeError("db unavailable")
        self.rows.append(kwargs)


@pytest.fixture()
def spool_home(tmp_path, monkeypatch):
    """Point the pending spool at an isolated HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants
    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: tmp_path, raising=True
    )
    return tmp_path


def _spool_files(home):
    d = home / "pending_messages"
    return sorted(d.glob("pending-*.json")) if d.exists() else []


class TestSpoolOnDrop:
    def test_drop_spool_drain_roundtrip(self, spool_home, caplog, monkeypatch):
        # Small cap so the test stays fast.
        monkeypatch.setattr(SessionStore, "_MAX_PENDING_PER_SESSION", 5)
        db = BrokenThenHealedDb()
        store = _make_store(db)

        n_extra = 3
        with caplog.at_level(logging.WARNING, logger="gateway.session"):
            for i in range(SessionStore._MAX_PENDING_PER_SESSION + n_extra):
                store.append_to_transcript(
                    "sess-1", {"role": "user", "content": f"msg{i}"}
                )

        # The oldest n_extra messages were evicted — and spooled, not lost.
        files = _spool_files(spool_home)
        assert len(files) == n_extra
        payloads = [json.loads(p.read_text()) for p in files]
        assert all(
            p["reason"] == shutdown_flush.TRANSCRIPT_CAP_DROP_REASON
            for p in payloads
        )
        spooled_contents = sorted(
            p["data"]["message"]["content"] for p in payloads
        )
        assert spooled_contents == ["msg0", "msg1", "msg2"]

        # Drop log escalated to WARNING and includes the spool path.
        drop_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "spooled oldest message" in r.getMessage()
        ]
        assert len(drop_warnings) == n_extra
        assert str(spool_home / "pending_messages") in drop_warnings[0].getMessage()

        # DB heals; the next successful flush drains the backlog AND
        # replays the spooled messages in drop order.
        db.broken = False
        store.append_to_transcript(
            "sess-1", {"role": "assistant", "content": "recovered"}
        )

        contents = [r["content"] for r in db.rows]
        # All surviving in-memory messages plus the recovery trigger...
        for i in range(n_extra, SessionStore._MAX_PENDING_PER_SESSION + n_extra):
            assert f"msg{i}" in contents
        assert "recovered" in contents
        # ...and the previously dropped messages, replayed in drop order.
        replayed = [c for c in contents if c in ("msg0", "msg1", "msg2")]
        assert replayed == ["msg0", "msg1", "msg2"]
        # Spool files consumed after successful replay.
        assert _spool_files(spool_home) == []
        # Nothing pending in memory.
        assert "sess-1" not in store._dirty_transcripts

    def test_drain_only_touches_own_session(self, spool_home, monkeypatch):
        monkeypatch.setattr(SessionStore, "_MAX_PENDING_PER_SESSION", 3)
        db = BrokenThenHealedDb()
        store = _make_store(db)

        for i in range(SessionStore._MAX_PENDING_PER_SESSION + 1):
            store.append_to_transcript("sess-a", {"role": "user", "content": f"a{i}"})
            store.append_to_transcript("sess-b", {"role": "user", "content": f"b{i}"})

        assert len(_spool_files(spool_home)) == 2  # one drop per session

        db.broken = False
        store.append_to_transcript("sess-a", {"role": "user", "content": "go-a"})

        # Only sess-a's spooled drop was replayed; sess-b's remains on disk.
        remaining = [
            json.loads(p.read_text()) for p in _spool_files(spool_home)
        ]
        assert len(remaining) == 1
        assert remaining[0]["session_key"] == "sess-b"
        a_rows = [r["content"] for r in db.rows if r["session_id"] == "sess-a"]
        assert "a0" in a_rows

    def test_spool_failure_degrades_to_plain_drop(
        self, spool_home, caplog, monkeypatch
    ):
        """If the spool cannot be written, behave exactly like the old
        drop-oldest path: cap enforced, WARNING logged, no crash."""
        monkeypatch.setattr(SessionStore, "_MAX_PENDING_PER_SESSION", 4)

        def _boom():
            raise OSError("disk full")

        monkeypatch.setattr(shutdown_flush, "_get_flush_dir", _boom)

        db = BrokenThenHealedDb()
        store = _make_store(db)

        with caplog.at_level(logging.WARNING, logger="gateway.session"):
            for i in range(SessionStore._MAX_PENDING_PER_SESSION + 5):
                store.append_to_transcript(
                    "sess-x", {"role": "user", "content": f"msg{i}"}
                )

        pending = store._dirty_transcripts.get("sess-x", [])
        assert len(pending) <= SessionStore._MAX_PENDING_PER_SESSION
        assert _spool_files(spool_home) == []
        degraded = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "on-disk spool unavailable" in r.getMessage()
        ]
        assert len(degraded) == 5
        # No spool bookkeeping means recovery must not attempt a drain.
        db.broken = False
        store.append_to_transcript("sess-x", {"role": "user", "content": "fin"})
        assert [r["content"] for r in db.rows][-1] == "fin"

    def test_replay_failure_keeps_spool_files(self, spool_home, monkeypatch):
        """A failed replay must preserve the spool files for a later retry."""
        monkeypatch.setattr(SessionStore, "_MAX_PENDING_PER_SESSION", 3)
        db = BrokenThenHealedDb()
        store = _make_store(db)

        for i in range(SessionStore._MAX_PENDING_PER_SESSION + 2):
            store.append_to_transcript("sess-r", {"role": "user", "content": f"m{i}"})
        assert len(_spool_files(spool_home)) == 2

        # DB heals only for live writes; replayed (spooled) rows still fail.
        class FlakyDb(BrokenThenHealedDb):
            def append_message(self, **kwargs):
                if kwargs["content"] in ("m0", "m1", "m2"):
                    raise RuntimeError("still broken for replays")
                self.rows.append(kwargs)

        flaky = FlakyDb()
        flaky.broken = False
        store._db = flaky
        # This append pushes pending over the cap again (dropping/spooling
        # m2) before the successful flush triggers the drain.
        store.append_to_transcript("sess-r", {"role": "user", "content": "go"})

        # Spool files survive the failed replay for the next attempt.
        assert len(_spool_files(spool_home)) == 3
        assert "sess-r" in getattr(store, "_spooled_drop_sessions", set())


class TestSpoolPrimitives:
    def test_drain_skips_other_reasons(self, spool_home):
        # A shutdown-format flush file must not be consumed by the drain.
        shutdown_flush.flush_pending_to_file({"key1": "hello"}, reason="shutdown")
        assert len(_spool_files(spool_home)) == 1
        replayed, remaining = shutdown_flush.drain_transcript_spool(
            "key1", lambda m: None
        )
        assert replayed == 0
        assert len(_spool_files(spool_home)) == 1

    def test_roundtrip_order(self, spool_home):
        for i in range(3):
            shutdown_flush.spool_dropped_transcript_message(
                "s", {"role": "user", "content": f"c{i}"}
            )
        seen = []
        replayed, remaining = shutdown_flush.drain_transcript_spool(
            "s", lambda m: seen.append(m["content"])
        )
        assert replayed == 3
        assert remaining == 0
        assert seen == ["c0", "c1", "c2"]
        assert _spool_files(spool_home) == []
