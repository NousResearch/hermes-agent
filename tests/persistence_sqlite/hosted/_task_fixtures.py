"""Seeding and comparison helpers for the hosted-room driver, policy, attachment and work-record backend tests.

``make_hosted`` builds a SQLite backend over a real profile home under the test's HERMES_HOME (the contract scopes
live under a fake root, so their ``'shared'`` file is not writable here), with that profile's ``state.db`` as the
``'profile'`` file and the install root's ``shared-state.db`` as the ``'shared'`` file. A parity case seeds the
placement's file through the room verbs (``seed_room``, ``seed_event``), copies it (with its attachment directory) to a
twin path in another folder, runs main's function on the twin and the backend verb on the original, and compares the
outcomes and what each side left in its tables and blob files. ``m1_room`` and ``m1_event`` run main's own functions,
read from the pinned main checkout.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home
from persistence import ProfileScope
from persistence.sqlite import SQLiteBackend
from persistence.sqlite._backend import SQLiteStore
from persistence.sqlite.gateway._hosted_rooms_schema import ROOMS_STEP
from persistence.sqlite.gateway._hosted_store import HostedFiles
from tests.persistence_sqlite._legacy_store import legacy_import, legacy_scope

MEMBERS = [{"member_id": "m1", "profile": "p1"}, {"member_id": "m2", "profile": "p2"}]
ROOM = "room"
GATEWAY = "gw"
ATTACHMENT_DIR = "hosted-room-attachments"
DRIVER_TABLES = ("hosted_room_driver_leases", "hosted_room_driver_tasks")
WORK_TABLES = ("hosted_room_work_records_source", "hosted_room_work_records_pending", "hosted_room_work_records_invalid")
POLICY_TABLES = (
    "hosted_room_policy_cursors", "hosted_room_policy_threads", "hosted_room_policy_events",
    "hosted_room_policy_watermarks", "hosted_room_policy_publications", "hosted_room_policy_transcript",
    "hosted_room_policy_transcript_state")
ATTACHMENT_TABLES = ("hosted_room_attachments", "hosted_room_attachment_blobs")
ROOM_TABLES = ("hosted_rooms", "hosted_room_events", "hosted_room_retired_ids")


@dataclass
class Hosted:
    """One backend handle on a real profile home and its two hosted-room files."""

    backend: SQLiteBackend
    scope: ProfileScope
    store: SQLiteStore
    files: HostedFiles

    def path(self, placement: str) -> Path:
        return self.files.path(placement)

    def prepare(self, placement: str) -> Path:
        """Open the placement's file as main's room verbs do (the rooms step only) and return its path."""
        HostedFiles(self.store, (ROOMS_STEP,))._hosted_write(placement, lambda conn: None, immediate=True)
        return self.path(placement)


def make_hosted(profile: str | None = "profile-a") -> Hosted:
    """A backend over ``<HERMES_HOME>/profiles/<profile>``, or over the install root itself (the default profile,
    whose ``state.db`` sits beside ``shared-state.db``) when ``profile`` is None."""
    home = get_hermes_home() if profile is None else get_hermes_home() / "profiles" / profile
    home.mkdir(parents=True, exist_ok=True)
    scope = ProfileScope(profile_dir=home, name=profile or "default")
    backend = SQLiteBackend({scope.hermes_home_key: home / "state.db"})
    store = backend.open(scope)
    store.ready()
    return Hosted(backend=backend, scope=scope, store=store, files=HostedFiles(store))


def _actor(kind: str) -> dict:
    """The actor of a seeded event of ``kind``: the desktop user, member ``m1`` or the gateway."""
    return ({"kind": "user", "id": "desktop"} if kind == "message.user"
            else {"kind": "member", "id": "m1"} if kind == "message.member" else {"kind": "gateway", "id": GATEWAY})


def m1_room(path: Path, *, room_id: str = ROOM, members: list | None = None, now: float = 100.0) -> dict:
    """Create a room in ``path`` with main's own ``create_room``, from the pinned checkout, inside its scope."""
    with legacy_scope():
        return legacy_import("gateway.hosted_rooms").create_room(
            path, room_id=room_id, name="Room", members=MEMBERS if members is None else members,
            authority_gateway_id=GATEWAY, now=now)


def m1_event(path: Path, *, event_id: str, kind: str, payload: dict, actor: dict | None = None,
             room_id: str = ROOM, now: float) -> dict:
    """Append one room event in ``path`` with main's own ``append_event`` (epoch 1, gateway ``gw``), from the pinned
    checkout, inside its scope."""
    with legacy_scope():
        return legacy_import("gateway.hosted_rooms").append_event(
            path, room_id=room_id, event_id=event_id, kind=kind, actor=_actor(kind) if actor is None else actor,
            payload=payload, authority_gateway_id=GATEWAY, authority_epoch=1, now=now)


def seed_room(hosted: Hosted, placement: str, *, room_id: str = ROOM, members: list | None = None,
              now: float = 100.0) -> dict:
    """Create the room ``m1_room`` creates, through the room verb on the placement's file."""
    return hosted.store.hosted_rooms.create_room(
        placement=placement, room_id=room_id, name="Room", members=MEMBERS if members is None else members,
        authority_gateway_id=GATEWAY, now=now)


def seed_event(hosted: Hosted, placement: str, *, event_id: str, kind: str, payload: dict, actor: dict | None = None,
               room_id: str = ROOM, now: float) -> dict:
    """Append the event ``m1_event`` appends, through the room verb on the placement's file."""
    return hosted.store.hosted_rooms.append_event(
        placement=placement, room_id=room_id, event_id=event_id, kind=kind,
        actor=_actor(kind) if actor is None else actor, payload=payload, authority_gateway_id=GATEWAY,
        authority_epoch=1, now=now)


def raw(path: Path, sql: str, params: tuple = ()) -> None:
    """One raw statement on ``path``: for states no public verb produces (a tampered room or task row)."""
    with sqlite3.connect(path) as conn:
        conn.execute(sql, params)


def twin(path: Path, folder: Path) -> Path:
    """Copy ``path`` and its attachment directory into ``folder``; returns the copy's path."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / path.name
    with sqlite3.connect(path) as source, sqlite3.connect(target) as copy:
        source.backup(copy)
    attachments = path.parent / ATTACHMENT_DIR
    if attachments.is_dir():
        shutil.copytree(attachments, folder / ATTACHMENT_DIR, dirs_exist_ok=True)
    return target


def dump(path: Path, tables: tuple[str, ...]) -> dict[str, Any]:
    """Every row of each table (sorted, so row order never matters), or None for an absent table."""
    with sqlite3.connect(path) as conn:
        present = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {table: (sorted(conn.execute(f"SELECT * FROM {table}").fetchall(), key=repr)
                        if table in present else None) for table in tables}


def schema(path: Path, pattern: str) -> list[tuple]:
    """The sqlite_master entries whose name matches the GLOB ``pattern``."""
    with sqlite3.connect(path) as conn:
        return sorted(conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name GLOB ? AND name NOT GLOB 'sqlite_*'",
            (pattern,)).fetchall())


def work_records(path: Path) -> list[dict]:
    """The captured source records of ``path``, decoded, each with its row's ``disposition``."""
    with sqlite3.connect(path) as conn:
        return [{"disposition": disposition, **json.loads(record)} for record, disposition in conn.execute(
            "SELECT record_json, disposition FROM hosted_room_work_records_source ORDER BY producer_epoch")]


def blob_dir(path: Path) -> Path:
    return path.parent / ATTACHMENT_DIR / "blobs"


def blobs(path: Path) -> dict[str, bytes]:
    """The blob directory beside ``path``: file name to bytes."""
    folder = blob_dir(path)
    return {item.name: item.read_bytes() for item in folder.iterdir()} if folder.is_dir() else {}


def plain(value: Any) -> Any:
    """``value`` with every dataclass turned into a dict of its fields, so main's and the contract's records compare."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def outcome(call: Callable[[], Any]) -> tuple:
    """``('ok', plain value)`` or ``('error', class name, message)``."""
    try:
        return "ok", plain(call())
    except Exception as exc:  # compared by class name and message
        return "error", type(exc).__name__, str(exc)


def status(result: tuple) -> str:
    """``'ok'`` or the error class name of one outcome."""
    return result[0] if result[0] == "ok" else result[1]
