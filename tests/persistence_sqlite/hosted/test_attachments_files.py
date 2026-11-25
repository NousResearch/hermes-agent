"""The blob files of the SQLite ``HostedRoomAttachmentRepository``: where each file step runs relative to its write
transaction, and what a crash leaves for the orphan sweep.

The transaction boundaries are observed by wrapping the repository's ``HostedFiles._hosted_write`` so the blob folder
is listed after each transaction body returns and before its commit.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from persistence.sqlite.gateway import hosted_room_attachments as backend_module
from persistence.sqlite.gateway.hosted_room_attachments import SQLiteHostedRoomAttachmentRepository
from tests.persistence_sqlite.hosted._task_fixtures import ATTACHMENT_DIR, ROOM, Hosted, make_hosted, seed_room

TTL_S = 60 * 60


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch) -> None:
    monkeypatch.setattr(backend_module, "_STARTED", set())


@pytest.fixture
def hosted() -> Hosted:
    hosted = make_hosted()
    yield hosted
    hosted.backend.close()


def _repo(hosted: Hosted, placement: str = "profile") -> tuple[SQLiteHostedRoomAttachmentRepository, Path]:
    path = hosted.prepare(placement)
    seed_room(hosted, placement)
    return SQLiteHostedRoomAttachmentRepository(hosted.store), path


def _put(repo, upload_id: str = "up-1", data: bytes = b"hello", *, placement: str = "profile", now: float = 110.0):
    return repo.put(placement=placement, room_id=ROOM, upload_id=upload_id, kind="file", name="a.txt",
                    mime="text/plain", data=data, now=now)


def blob_dir(path: Path) -> Path:
    return path.parent / ATTACHMENT_DIR / "blobs"


def blobs(path: Path) -> dict[str, bytes]:
    """The blob directory beside ``path``: file name to bytes."""
    folder = blob_dir(path)
    return {item.name: item.read_bytes() for item in folder.iterdir()} if folder.is_dir() else {}


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


def test_orphan_sweep_serializes_with_concurrent_canonical_blob_commit(hosted, monkeypatch):
    repo, path = _repo(hosted)
    root = repo._blob_root("profile")
    entered_sweep, release_sweep, writer_done = threading.Event(), threading.Event(), threading.Event()
    errors: list[BaseException] = []
    original_iterdir = type(root).iterdir

    def blocking_iterdir(folder):
        if folder == root and threading.current_thread().name == "stale-sweep":
            entered_sweep.set()
            assert release_sweep.wait(5)
        return original_iterdir(folder)

    monkeypatch.setattr(type(root), "iterdir", blocking_iterdir)
    sweep_thread = threading.Thread(target=repo._sweep_orphans, args=("profile",), name="stale-sweep")
    sweep_thread.start()
    assert entered_sweep.wait(5)

    def write_blob():
        try:
            _put(repo)
        except BaseException as exc:
            errors.append(exc)
        finally:
            writer_done.set()

    writer_thread = threading.Thread(target=write_blob, name="blob-writer")
    writer_thread.start()
    # CONTRACT: the sweep lists the blob folder inside its write, so a put waits for it instead of losing its blob.
    assert not writer_done.wait(0.1)
    release_sweep.set()
    sweep_thread.join(5)
    writer_thread.join(5)

    assert not errors
    assert writer_done.is_set()
    assert _put(repo)["idempotent"] is True
    assert len(_blob_names(path)) == 1 and _blob_rows(path) == list(_blob_names(path))
