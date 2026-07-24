"""Tests for People message store / identity / iMessage adapter / writer (#12323)."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from plugins.people.adapters.imessage import IMessageAdapter, _apple_ns_to_unix
from plugins.people.identity import IdentityResolver, normalize_identity, slugify_name
from plugins.people.store import IngestMessage, PeopleMessageStore
from plugins.people.writer import write_people_markdown


@pytest.fixture()
def store(tmp_path: Path):
    db = tmp_path / "messages.db"
    s = PeopleMessageStore(db)
    yield s
    s.close()


class TestNormalize:
    def test_phone_us_ten_digit(self):
        assert normalize_identity("phone", "(415) 555-1212") == "+14155551212"

    def test_email_lower(self):
        assert normalize_identity("email", "A@B.Com") == "a@b.com"

    def test_handle_strip_at(self):
        assert normalize_identity("handle", "@Alice") == "alice"

    def test_slugify(self):
        assert "alice" in slugify_name("Alice Smith")


class TestStoreDedupe:
    def test_incremental_dedupe(self, store: PeopleMessageStore):
        m = IngestMessage(
            channel="imessage",
            ext_msg_id="guid-1",
            direction="in",
            ts=time.time(),
            body="hi",
            raw_handle="+14155551212",
        )
        ins, skip = store.ingest_many([m])
        assert ins == 1 and skip == 0
        ins2, skip2 = store.ingest_many([m])
        assert ins2 == 0 and skip2 == 1
        assert store.message_count() == 1


class TestIdentityResolver:
    def test_merge_same_phone(self, store: PeopleMessageStore):
        r = IdentityResolver(store)
        a = r.resolve("415-555-1212", kind="phone", display_name="Alice")
        b = r.resolve("+14155551212", kind="phone")
        assert a == b

    def test_manual_override(self, store: PeopleMessageStore):
        r = IdentityResolver(store)
        p1 = r.resolve("a@example.com", kind="email", display_name="A")
        store.set_manual_override("email", "b@example.com", p1)
        p2 = r.resolve("b@example.com", kind="email")
        assert p2 == p1


class TestIMessageAdapter:
    def test_apple_ts(self):
        # known: cocoa 0 → 2001-01-01
        assert abs(_apple_ns_to_unix(0) - 978307200.0) < 0.1

    def test_sync_fixture_db(self, store: PeopleMessageStore, tmp_path: Path):
        chat = tmp_path / "chat.db"
        conn = sqlite3.connect(str(chat))
        conn.executescript(
            """
            CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                guid TEXT,
                text TEXT,
                is_from_me INTEGER,
                date INTEGER,
                cache_roomnames TEXT,
                handle_id INTEGER
            );
            INSERT INTO handle(ROWID, id) VALUES (1, '+14155550100');
            INSERT INTO message(ROWID, guid, text, is_from_me, date, cache_roomnames, handle_id)
            VALUES (1, 'G-1', 'hello from alice', 0, 0, NULL, 1);
            INSERT INTO message(ROWID, guid, text, is_from_me, date, cache_roomnames, handle_id)
            VALUES (2, 'G-2', 'hi alice', 1, 10, NULL, 1);
            """
        )
        conn.commit()
        conn.close()

        adapter = IMessageAdapter(chat)
        ins, skip = adapter.sync_to_store(store)
        assert ins == 2 and skip == 0
        # second sync dedupes
        ins2, skip2 = adapter.sync_to_store(store)
        assert ins2 == 0 and skip2 == 2

        people_dir = tmp_path / "people_md"
        n = write_people_markdown(store, people_dir=people_dir)
        assert n >= 1
        files = list(people_dir.glob("*.md"))
        assert files
        text = files[0].read_text()
        assert "type: person" in text
        assert "hello from alice" in text
        assert "notes-extract:begin facts" in text
