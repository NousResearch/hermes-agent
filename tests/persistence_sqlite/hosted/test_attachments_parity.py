"""The SQLite ``HostedRoomAttachmentRepository`` against main's attachment store (gateway/hosted_room_attachments.py).

The placement's file and its twin start from one room seeded through the room verbs; each step builds main's store on
the twin (as main's callers build one per operation) and calls its method, then calls the backend verb, and compares
the outcomes, both attachment tables and the blob folder beside each file. A room change between steps runs main's
``gateway.hosted_rooms`` (read from the pinned main checkout) on the twin and the room verb on the placement's file. ``secrets.token_hex`` is replaced by a counter reset to
the same value before each side, so both mint the same attachment and blob ids; each module's first-construction set
starts empty, so both run their first-touch reconciliation and prune on the first step.
"""

from __future__ import annotations

import itertools
import secrets
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.persistence_sqlite._main_callers import hosted_room_attachments as m1
from persistence.gateway.hosted_rooms import AttachmentNotFoundError
from persistence.sqlite.gateway import hosted_room_attachments as backend_module
from persistence.sqlite.gateway.hosted_room_attachments import SQLiteHostedRoomAttachmentRepository
from tests.persistence_sqlite._legacy_store import legacy_import, legacy_store
from tests.persistence_sqlite.hosted._task_fixtures import (
    ATTACHMENT_TABLES, GATEWAY, ROOM, Hosted, blobs, dump, m1_event, make_hosted, outcome, schema, seed_event,
    seed_room, status, twin,
)

__all__ = ["legacy_store"]

FIELDS = ("attachment_id", "kind", "name", "size", "mime")
TTL_S = 60 * 60


@pytest.fixture
def hosted(monkeypatch) -> Hosted:
    monkeypatch.setattr(m1, "_STARTED", set())
    monkeypatch.setattr(backend_module, "_STARTED", set())
    hosted = make_hosted()
    yield hosted
    hosted.backend.close()


class Tokens:
    """A ``secrets.token_hex`` that counts from a base the runner resets before each side."""

    def __init__(self) -> None:
        self.counter = itertools.count()

    def reset(self, base: int) -> None:
        self.counter = itertools.count(base)

    def __call__(self, nbytes: int = 16) -> str:
        return format(next(self.counter), f"0{nbytes * 2}x")


@pytest.fixture
def tokens(monkeypatch) -> Tokens:
    fake = Tokens()
    monkeypatch.setattr(secrets, "token_hex", fake)
    return fake


@dataclass
class Step:
    verb: str
    kwargs: dict = field(default_factory=dict)
    now: float = 0.0
    expect: str = "ok"
    save: str | None = None


@dataclass
class Seed:
    """One room change on both files: ``m1(path, saved)`` with main's functions on the twin, ``verb(hosted, placement,
    saved)`` through the room verbs on the placement's file."""

    m1: Callable[[Path, dict], Any]
    verb: Callable[[Hosted, str, dict], Any]


def entry(name: str) -> Callable[[dict], dict]:
    return lambda saved: {key: saved[name][key] for key in FIELDS}


def ident(name: str) -> Callable[[dict], str]:
    return lambda saved: saved[name]["attachment_id"]


def _resolve(value: Any, saved: dict) -> Any:
    if callable(value):
        return value(saved)
    if isinstance(value, list):
        return [_resolve(item, saved) for item in value]
    return value


def run_chain(hosted: Hosted, tmp_path: Path, tokens: Tokens, placement: str, steps: list[Step | Seed]) -> list:
    path = hosted.prepare(placement)
    for room_id in (ROOM, "room-2"):
        seed_room(hosted, placement, room_id=room_id)
    legacy_path = twin(path, tmp_path / "m1")
    repo = SQLiteHostedRoomAttachmentRepository(hosted.store)
    saved: dict[str, Any] = {}
    results = []
    for index, step in enumerate(steps):
        if isinstance(step, Seed):
            step.m1(legacy_path, saved)
            step.verb(hosted, placement, saved)
            continue
        kwargs = {key: _resolve(value, saved) for key, value in step.kwargs.items()}
        tokens.reset(index * 100)
        legacy = outcome(lambda: getattr(m1.HostedRoomAttachmentStore(legacy_path, clock=lambda: step.now),
                                         step.verb)(**kwargs))
        tokens.reset(index * 100)
        backend = outcome(lambda: getattr(repo, step.verb)(placement=placement, now=step.now, **kwargs))
        label = f"step {index} {step.verb}"
        assert backend == legacy, label
        assert status(backend) == step.expect, (label, backend)
        assert dump(path, ATTACHMENT_TABLES) == dump(legacy_path, ATTACHMENT_TABLES), label
        assert blobs(path) == blobs(legacy_path), label
        if step.save is not None:
            saved[step.save] = backend[1]
        results.append(backend)
    return results


def _put(upload_id: str, data: bytes = b"hello", *, now: float, room_id: str = ROOM, expect: str = "ok",
         save: str | None = None) -> Step:
    return Step("put", dict(room_id=room_id, upload_id=upload_id, kind="file", name="a.txt", mime="text/plain",
                            data=data), now=now, expect=expect, save=save)


def _commit(name: str, event_id: str, *, now: float, viewer: bool = False, hold: bool = False,
            retention: float | None = None, recipients: tuple = ("m1", "m2"), expect: str = "ok") -> Step:
    return Step("commit_message_with_receipt", dict(
        room_id=ROOM, event_id=event_id, manifest=[entry(name)], recipient_member_ids=list(recipients),
        viewer_access=viewer, retention_seconds=retention, hold_until_event=hold), now=now, expect=expect)


def _read(name: str, *, now: float, member: str = "m1", event_id: str | None = "e1", expect: str = "ok") -> Step:
    return Step("read", dict(room_id=ROOM, attachment_id=ident(name), recipient_member_id=member, event_id=event_id),
                now=now, expect=expect)


def _publish(name: str, event_id: str, now: float) -> Seed:
    def payload(saved: dict) -> dict:
        return {"text": "hi", "attachments": [entry(name)(saved)]}

    return Seed(
        lambda path, saved: m1_event(path, event_id=event_id, kind="message.user", payload=payload(saved), now=now),
        lambda hosted, placement, saved: seed_event(
            hosted, placement, event_id=event_id, kind="message.user", payload=payload(saved), now=now))


def _disband(room_id: str, now: float) -> Seed:
    return Seed(
        lambda path, saved: legacy_import("gateway.hosted_rooms").disband_room(
            path, room_id=room_id, expected_gateway_id=GATEWAY, expected_epoch=1, now=now),
        lambda hosted, placement, saved: hosted.store.hosted_rooms.disband_room(
            placement=placement, room_id=room_id, expected_gateway_id=GATEWAY, expected_epoch=1, now=now))


@pytest.mark.parametrize("placement", ["profile", "shared"])
def test_upload_commit_read_publish_abort_prune_and_disband(hosted, tmp_path, tokens, placement, legacy_store):
    results = run_chain(hosted, tmp_path, tokens, placement, [
        _put("up-1", now=110.0, save="a1"),
        _put("up-1", now=500.0),
        _put("up-1", b"other", now=111.0, expect="AttachmentConflictError"),
        _put("up-2", now=112.0, save="a2"),
        _put("up-3", b"world", now=113.0, save="a3"),
        _commit("a1", "e1", now=120.0, retention=600.0),
        _commit("a1", "e1", now=121.0, retention=600.0),
        _commit("a1", "e2", now=122.0, expect="AttachmentConflictError"),
        _commit("a2", "e-v", now=123.0, viewer=True, hold=True),
        _read("a1", now=124.0),
        _read("a1", now=124.0, member="m3", expect="AttachmentNotFoundError"),
        _read("a1", now=124.0, event_id="e9", expect="AttachmentNotFoundError"),
        _read("a3", now=124.0, expect="AttachmentNotFoundError"),
        Step("describe", dict(room_id=ROOM, attachment_id=ident("a1"), recipient_member_id="m2", event_id="e1"),
             now=125.0),
        Step("read_range", dict(room_id=ROOM, attachment_id=ident("a1"), recipient_member_id="m1", event_id="e1",
                                offset=1, length=3), now=126.0),
        Step("read_range", dict(room_id=ROOM, attachment_id=ident("a1"), recipient_member_id="m1", event_id="e1",
                                offset=5, length=1), now=126.0, expect="AttachmentError"),
        _read("a2", now=127.0, event_id="e-v", expect="AttachmentNotFoundError"),
        Step("read_viewer", dict(room_id=ROOM, attachment_id=ident("a2"), event_id="e-v", authority_gateway_id=GATEWAY,
                                 authority_epoch=1), now=127.0, expect="AttachmentNotFoundError"),
        _publish("a2", "e-v", 128.0),
        Step("read_viewer", dict(room_id=ROOM, attachment_id=ident("a2"), event_id="e-v", authority_gateway_id=GATEWAY,
                                 authority_epoch=1), now=129.0),
        Step("read_viewer", dict(room_id=ROOM, attachment_id=ident("a2"), event_id="e-v", authority_gateway_id=GATEWAY,
                                 authority_epoch=2), now=129.0, expect="AttachmentNotFoundError"),
        _read("a2", now=129.0, event_id="e-v"),
        Step("abort_message_commit", dict(room_id=ROOM, event_id="e-v", attachment_ids=[ident("a2")]), now=130.0),
        Step("abort_message_commit", dict(room_id=ROOM, event_id="e1", attachment_ids=[ident("a1"), ident("a3")]),
             now=131.0),
        Step("abort_message_commit", dict(room_id=ROOM, event_id="e1", attachment_ids=[]), now=131.0),
        _commit("a1", "e1", now=132.0, retention=600.0),
        _read("a1", now=133.0 + 600.0, expect="AttachmentNotFoundError"),
        # The prune of the next upload deletes the expired commitment and the uploads past their TTL.
        _put("up-4", b"four", now=133.0 + TTL_S, save="a4"),
        _commit("a3", "e3", now=134.0 + TTL_S, expect="AttachmentNotFoundError"),
        _disband(ROOM, 200.0 + TTL_S),
        _put("up-5", b"five", now=201.0 + TTL_S, room_id="room-2"),
        _read("a2", now=202.0 + TTL_S, event_id="e-v", expect="AttachmentNotFoundError"),
        _put("up-6", b"six", now=202.0 + TTL_S + 15 * 60, room_id="room-2"),
        _commit("a4", "e4", now=203.0 + TTL_S + 15 * 60),
    ])
    assert results[1][1]["idempotent"] is True
    assert results[21][1] == 0 and results[22][1] == 1


def test_retain_message_attachments_matches_m1_inside_the_append_transaction(hosted, tmp_path, tokens):
    path = hosted.prepare("profile")
    seed_room(hosted, "profile")
    repo = SQLiteHostedRoomAttachmentRepository(hosted.store)
    meta = repo.put(placement="profile", room_id=ROOM, upload_id="up-1", kind="file", name="a.txt", mime="text/plain",
                    data=b"hello", now=110.0)
    held = repo.put(placement="profile", room_id=ROOM, upload_id="up-2", kind="file", name="b.txt",
                    mime="text/plain", data=b"bye", now=110.0)
    manifest = [{key: meta[key] for key in FIELDS}]
    repo.commit_message_with_receipt(placement="profile", room_id=ROOM, event_id="e1", manifest=manifest,
                                     recipient_member_ids=["m1"], viewer_access=True, hold_until_event=True, now=120.0)
    legacy_path = twin(path, tmp_path / "m1")

    def legacy(**kwargs: Any) -> Any:
        conn = sqlite3.connect(legacy_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            m1.retain_message_attachments(conn, **kwargs)
            conn.commit()
        finally:
            conn.close()

    def backend(**kwargs: Any) -> Any:
        hosted.files._hosted_write(
            "profile", lambda conn: backend_module.retain_message_attachments(conn, **kwargs), immediate=True)

    cases = [
        (dict(room_id=ROOM, event_id="e1", manifest=[], now=125.0), "ok"),
        (dict(room_id=ROOM, event_id="e1", manifest=[{**manifest[0], "mime": "TEXT/plain"}], now=125.0),
         "AttachmentError"),
        (dict(room_id=ROOM, event_id="e1", manifest=[{**manifest[0], "size": 0}], now=125.0), "AttachmentError"),
        (dict(room_id=ROOM, event_id="e2", manifest=manifest, now=125.0), "EventAttachmentConflictError"),
        (dict(room_id=ROOM, event_id="e1", manifest=[{key: held[key] for key in FIELDS}], now=125.0),
         "EventAttachmentConflictError"),
        (dict(room_id=ROOM, event_id="e1", manifest=manifest, now=120.0 + TTL_S), "EventAttachmentConflictError"),
        (dict(room_id=ROOM, event_id="e1", manifest=manifest, now=125.0), "ok"),
    ]
    for kwargs, expected in cases:
        legacy_result, backend_result = outcome(lambda: legacy(**kwargs)), outcome(lambda: backend(**kwargs))
        assert backend_result == legacy_result and status(backend_result) == expected, kwargs
        assert dump(path, ATTACHMENT_TABLES) == dump(legacy_path, ATTACHMENT_TABLES), kwargs


def test_attachment_schema_matches_m1(hosted, tmp_path):
    # CONTRACT: the attachment tables appear with the first attachment verb on the file, as main creates them when it
    # first constructs its attachment store for the file; the room verbs never create them.
    path = hosted.prepare("profile")
    seed_room(hosted, "profile")
    hosted.store.hosted_rooms.room_state(placement="profile", room_id=ROOM)
    assert schema(path, "*hosted_room_attachment*") == []
    legacy = twin(path, tmp_path / "m1")
    m1.HostedRoomAttachmentStore(legacy)
    with pytest.raises(AttachmentNotFoundError):
        SQLiteHostedRoomAttachmentRepository(hosted.store).describe(
            placement="profile", room_id=ROOM, attachment_id="att_" + "0" * 32, recipient_member_id="m1",
            event_id="e1", now=125.0)
    assert schema(path, "*hosted_room_attachment*") == schema(legacy, "*hosted_room_attachment*")
    assert schema(path, "*hosted_room_attachment*")
