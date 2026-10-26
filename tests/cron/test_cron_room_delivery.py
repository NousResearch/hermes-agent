"""Cron ``deliver: room:<id>`` posts into a real hosted room store under a temp HERMES_HOME."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from cron import scheduler_delivery as delivery
from gateway import hosted_rooms
from hermes_constants import get_hermes_home

ROOM_ID = "room-cron"
MEMBERS = [{"member_id": f"m-{p}", "profile": p, "handle": p} for p in ("alpha", "beta")]


def _create(db):
    hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Feed", members=MEMBERS,
        authority_gateway_id=hosted_rooms.local_authority_gateway_id())
    return db


def _posts(db) -> list[dict]:
    return [e for e in hosted_rooms.read_events(db, room_id=ROOM_ID)["events"] if e["kind"] == "message.user"]


@pytest.fixture
def room_store():
    return _create(hosted_rooms.default_db_path())


def _job(**extra) -> dict:
    return {"id": "j1", "name": "Digest", "deliver": f"room:{ROOM_ID}", "execution_id": "exec-1", **extra}


def test_room_target_resolves_and_posts_job_output(room_store, monkeypatch):
    monkeypatch.setattr(delivery._sched, "load_config", lambda: {"cron": {"wrap_response": True}})

    assert delivery._resolve_delivery_targets(_job()) == [
        {"platform": "room", "chat_id": ROOM_ID, "thread_id": None}]
    assert delivery._deliver_result(_job(), "Three new issues today") is None
    assert delivery._deliver_result(_job(), "Three new issues today") is None  # same run: one post

    posts = _posts(room_store)
    assert len(posts) == 1
    assert posts[0]["payload"]["text"] == "Three new issues today"
    assert posts[0]["actor"] == {"kind": "user", "id": "cron:j1", "display_name": "Digest"}


def test_unknown_room_reports_a_delivery_error(room_store):
    error = delivery._deliver_result(_job(deliver="room:nope"), "x")

    assert error and "no live Group Chat" in error


def test_malformed_room_token_resolves_to_nothing():
    assert delivery._resolve_delivery_targets({"id": "j", "deliver": "room:"}) == []


def test_post_skips_the_unserved_session_store_copy_left_by_the_shared_store_import(room_store):
    leftover = _create(get_hermes_home() / "state.db")  # pre-isolation copy: no gateway owner entry

    assert delivery._deliver_result(_job(), "Digest ready") is None

    assert [p["payload"]["text"] for p in _posts(room_store)] == ["Digest ready"]
    assert _posts(leftover) == []


def test_post_goes_to_the_session_store_when_its_gateway_owns_the_room(room_store):
    from gateway.session_hosted_service import _OWNER
    owned = _create(get_hermes_home() / "state.db")
    with closing(sqlite3.connect(owned)) as conn, conn:
        conn.execute("CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO state_meta(key, value) VALUES (?, 'owner')", (_OWNER + ROOM_ID,))

    assert delivery._deliver_result(_job(), "Digest ready") is None

    assert [p["payload"]["text"] for p in _posts(owned)] == ["Digest ready"]
    assert _posts(room_store) == []
