"""The blob files of the SQLite ``HostedRoomAttachmentRepository``: where each file step runs relative to its write
transaction, what a crash leaves for the orphan sweep, the once-per-file first-touch work, and main's sweep across the
blob folder the root ``state.db`` and ``shared-state.db`` share.

The transaction boundaries are observed by wrapping the repository's ``HostedFiles._hosted_write`` so the blob folder
is listed after each transaction body returns and before its commit.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.persistence_sqlite._main_callers import hosted_room_attachments as m1
from persistence.gateway.hosted_rooms import AttachmentIntegrityError, AttachmentNotFoundError
from persistence.sqlite.gateway import hosted_room_attachments as backend_module
from persistence.sqlite.gateway.hosted_room_attachments import SQLiteHostedRoomAttachmentRepository
from tests.persistence_sqlite.hosted._task_fixtures import (
    ATTACHMENT_TABLES, ROOM, Hosted, blob_dir, blobs, dump, m1_event, m1_room, make_hosted, raw, twin,
)

FIELDS = ("attachment_id", "kind", "name", "size", "mime")
TTL_S = 60 * 60


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch) -> None:
    monkeypatch.setattr(m1, "_STARTED", set())
    monkeypatch.setattr(backend_module, "_STARTED", set())


@pytest.fixture
def hosted() -> Hosted:
    hosted = make_hosted()
    yield hosted
    hosted.backend.close()


def _repo(hosted: Hosted, placement: str = "profile") -> tuple[SQLiteHostedRoomAttachmentRepository, Path]:
    path = hosted.prepare(placement)
    m1_room(path)
    return SQLiteHostedRoomAttachmentRepository(hosted.store), path


def _put(repo, upload_id: str = "up-1", data: bytes = b"hello", *, placement: str = "profile", now: float = 110.0):
    return repo.put(placement=placement, room_id=ROOM, upload_id=upload_id, kind="file", name="a.txt",
                    mime="text/plain", data=data, now=now)


def _blob_names(path: Path) -> set[str]:
    return {name for name in blobs(path) if name.startswith("blob_")}


def _blob_rows(path: Path) -> list[str]:
    with sqlite3.connect(path) as conn:
        return [row[0] for row in conn.execute("SELECT blob_id FROM hosted_room_attachment_blobs")]


def _record_transactions(monkeypatch, repo, path: Path) -> list[tuple[Any, set[str]]]:
    """Record each write transaction's result and the blob folder listing taken inside it, after its body ran."""
    seen: list[tuple[Any, set[str]]] = []
    original = repo._files._hosted_write

    def recording(placement, fn, **kwargs):
        def body(conn):
            result = fn(conn)
            seen.append((result, set(blobs(path))))
            return result
        return original(placement, body, **kwargs)
    monkeypatch.setattr(repo._files, "_hosted_write", recording)
    return seen


def test_put_writes_its_blob_inside_the_transaction_and_a_refused_put_leaves_it_to_the_sweep(hosted, monkeypatch):
    repo, path = _repo(hosted)
    fold = backend_module.fold_catalog_text

    def refuse(_name: str) -> str:
        raise RuntimeError("row insert failed")

    monkeypatch.setattr(backend_module, "fold_catalog_text", refuse)
    with pytest.raises(RuntimeError, match="^row insert failed$"):
        _put(repo)
    # CONTRACT: the blob is written before its row inside the write; the rollback leaves the file and no row.
    (orphan,) = _blob_names(path)
    assert _blob_rows(path) == []
    monkeypatch.setattr(backend_module, "fold_catalog_text", fold)
    meta = _put(repo)
    (live,) = _blob_names(path)
    assert live != orphan and _blob_rows(path) == [live] and meta["idempotent"] is False


def test_released_blobs_are_unlinked_only_after_the_prune_commits(hosted, monkeypatch):
    repo, path = _repo(hosted)
    _put(repo)
    (blob,) = _blob_names(path)
    seen = _record_transactions(monkeypatch, repo, path)
    _put(repo, "up-2", b"other", now=110.0 + TTL_S)
    (prune,) = [listing for result, listing in seen if isinstance(result, tuple) and result[1] == [blob]]
    # CONTRACT: the prune deleted the last row of the blob inside its write while the file stayed; it went after the commit.
    assert blob in prune
    assert blob not in _blob_names(path) and blob not in _blob_rows(path)


def test_crash_leftovers_are_renamed_inside_the_sweep_and_unlinked_after_it(hosted, monkeypatch):
    repo, path = _repo(hosted)
    _put(repo)
    (live,) = _blob_names(path)
    planted = {".tmp-1": b"x", ".prune-2": b"x", "blob_" + "a" * 32: b"x", "README": b"x"}
    for name, data in planted.items():
        (blob_dir(path) / name).write_bytes(data)
    seen = _record_transactions(monkeypatch, repo, path)
    _put(repo, "up-2", now=120.0)
    (inside,) = [listing for result, listing in seen if isinstance(result, list) and len(result) == 3]
    # CONTRACT: the sweep renamed the three leftovers to private quarantine names inside its write and unlinked them after it.
    assert not inside & {".tmp-1", ".prune-2", "blob_" + "a" * 32}
    assert len({name for name in inside if name.startswith(".prune-")}) == 3
    assert set(blobs(path)) == {live, "README"}


def test_first_touch_reconciles_and_prunes_once_per_file_as_m1_construction_does(hosted, tmp_path):
    repo, path = _repo(hosted)
    meta = _put(repo)
    entry = {key: meta[key] for key in FIELDS}
    repo.commit_message_with_receipt(placement="profile", room_id=ROOM, event_id="e1", manifest=[entry],
                                     recipient_member_ids=["m1"], viewer_access=True, hold_until_event=True, now=120.0)
    m1_event(path, event_id="e1", kind="message.user", payload={"text": "hi", "attachments": [entry]}, now=121.0)
    _put(repo, "up-2", b"stale", now=122.0)
    # A row a store without the in-append retain left held, and an upload past its TTL.
    raw(path, "UPDATE hosted_room_attachments SET expires_at=? WHERE event_id='e1'", (122.0 + 2 * TTL_S,))
    legacy_path = twin(path, tmp_path / "m1")
    backend_module._STARTED.clear()
    now = 123.0 + TTL_S
    m1.HostedRoomAttachmentStore(legacy_path, clock=lambda: now)
    with pytest.raises(AttachmentNotFoundError):
        repo.describe(placement="profile", room_id=ROOM, attachment_id="att_" + "0" * 32, recipient_member_id="m1",
                      event_id="e1", now=now)
    assert dump(path, ATTACHMENT_TABLES) == dump(legacy_path, ATTACHMENT_TABLES)
    assert blobs(path) == blobs(legacy_path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT upload_id, expires_at FROM hosted_room_attachments").fetchall() == [("up-1", None)]
    # The second verb in the same process does not run it again.
    raw(path, "UPDATE hosted_room_attachments SET expires_at=1.0")
    repo.describe(placement="profile", room_id=ROOM, attachment_id=meta["attachment_id"], recipient_member_id="m1",
                  event_id="e1", now=0.5)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT expires_at FROM hosted_room_attachments").fetchall() == [(1.0,)]


@pytest.fixture
def root_hosted() -> Hosted:
    hosted = make_hosted(None)
    yield hosted
    hosted.backend.close()


def test_shared_put_sweeps_the_root_profiles_live_blobs_as_m1_does(root_hosted):
    repo = SQLiteHostedRoomAttachmentRepository(root_hosted.store)
    profile_path, shared_path = root_hosted.prepare("profile"), root_hosted.prepare("shared")
    for path in (profile_path, shared_path):
        m1_room(path)
    meta = _put(repo, data=b"live attachment")
    repo.commit_message_with_receipt(placement="profile", room_id=ROOM, event_id="e1",
                                     manifest=[{key: meta[key] for key in FIELDS}], recipient_member_ids=["m1"],
                                     now=120.0)
    (live,) = _blob_names(profile_path)
    assert blob_dir(profile_path) == blob_dir(shared_path) and _blob_rows(profile_path) == [live]
    _put(repo, "up-s", b"shared upload", placement="shared", now=130.0)
    # QUIRK(m1-hosted-attachment-shared-sweep): the shared file's sweep deletes the profile's live blob in the shared folder.
    assert live not in _blob_names(profile_path) and _blob_rows(profile_path) == [live]
    # QUIRK(m1-hosted-attachment-shared-sweep): the profile's committed attachment no longer reads; the fix is a separate upstream report.
    with pytest.raises(AttachmentIntegrityError, match="^canonical attachment blob is unavailable$"):
        repo.read(placement="profile", room_id=ROOM, attachment_id=meta["attachment_id"], recipient_member_id="m1",
                  event_id="e1", now=140.0)


def test_first_touch_of_the_shared_file_sweeps_the_root_profiles_live_blobs_as_m1_construction_does(root_hosted):
    repo = SQLiteHostedRoomAttachmentRepository(root_hosted.store)
    profile_path, shared_path = root_hosted.prepare("profile"), root_hosted.prepare("shared")
    for path in (profile_path, shared_path):
        m1_room(path)
    _put(repo, data=b"live attachment")
    (live,) = _blob_names(profile_path)
    with pytest.raises(AttachmentNotFoundError):
        repo.describe(placement="shared", room_id=ROOM, attachment_id="att_" + "0" * 32, recipient_member_id="m1",
                      event_id="e1", now=130.0)
    # QUIRK(m1-hosted-attachment-shared-sweep): a read verb's first touch of shared-state.db prunes and sweeps the shared folder; the fix is a separate upstream report.
    assert live not in _blob_names(profile_path) and _blob_rows(profile_path) == [live]
