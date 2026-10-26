"""Cron ``deliver: room:<id>`` posts into a real hosted room store under a temp HERMES_HOME."""

from __future__ import annotations

import pytest

from cron import scheduler_delivery as delivery
from gateway import hosted_rooms
from hermes_constants import get_hermes_home

ROOM_ID = "room-cron"
MEMBERS = [{"member_id": f"m-{p}", "profile": p, "handle": p} for p in ("alpha", "beta")]


@pytest.fixture
def room_store():
    db = get_hermes_home() / "state.db"
    hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Feed", members=MEMBERS,
        authority_gateway_id=hosted_rooms.local_authority_gateway_id())
    return db


def _job(**extra) -> dict:
    return {"id": "j1", "name": "Digest", "deliver": f"room:{ROOM_ID}", "execution_id": "exec-1", **extra}


def test_room_target_resolves_and_posts_job_output(room_store, monkeypatch):
    monkeypatch.setattr(delivery._sched, "load_config", lambda: {"cron": {"wrap_response": True}})

    assert delivery._resolve_delivery_targets(_job()) == [
        {"platform": "room", "chat_id": ROOM_ID, "thread_id": None}]
    assert delivery._deliver_result(_job(), "Three new issues today") is None
    assert delivery._deliver_result(_job(), "Three new issues today") is None  # same run: one post

    posts = [e for e in hosted_rooms.read_events(room_store, room_id=ROOM_ID)["events"] if e["kind"] == "message.user"]
    assert len(posts) == 1
    assert posts[0]["payload"]["text"] == "Three new issues today"
    assert posts[0]["actor"] == {"kind": "user", "id": "cron:j1", "display_name": "Digest"}


def test_unknown_room_reports_a_delivery_error(room_store):
    error = delivery._deliver_result(_job(deliver="room:nope"), "x")

    assert error and "no live Group Chat" in error


def test_malformed_room_token_resolves_to_nothing():
    assert delivery._resolve_delivery_targets({"id": "j", "deliver": "room:"}) == []
