import json
import os
import sqlite3
import time
import pytest

from hermes_cli.kanban_generation import (
    GenerationError, acquire_fence, acquire_reader_lease, activate_first_write,
    build_ready, gc_generations, initialize_generation_db, publish_prepared,
    release_fence, release_reader_lease, sha256_file,
)


def test_gc_retains_invalid_ready_for_recovery(tmp_path):
    board = tmp_path / "board"
    root = board / "generations"
    stale = root / "stale"
    stale.mkdir(parents=True)
    (stale / "READY.json").write_text("{}")
    with pytest.raises(GenerationError, match="READY_INVALID"):
        gc_generations(board)
    assert stale.exists()


def test_gc_retains_journaled_generation_even_without_pointer(tmp_path):
    board = tmp_path / "board"
    generation = board / "generations" / "journaled"
    generation.mkdir(parents=True)
    (generation / "import-journal.json").write_text('{"state":"prepared"}')
    with pytest.raises(GenerationError, match="READY_INVALID"):
        gc_generations(board)
    assert generation.exists()


def _active_blob_generation(tmp_path):
    board = tmp_path / "board"
    generation = board / "generations" / "g1"
    generation.mkdir(parents=True)
    db = generation / "kanban.db"
    initialize_generation_db(db, gid="g1", nonce="n1", control_root=board)
    referenced = generation / "blobs" / "sha256" / "aa" / ("a" * 64)
    orphan = generation / "blobs" / "sha256" / "bb" / ("b" * 64)
    referenced.parent.mkdir(parents=True)
    referenced.write_bytes(b"referenced")
    digest = sha256_file(referenced)
    canonical = generation / "blobs" / "sha256" / digest[:2] / digest
    canonical.parent.mkdir(parents=True, exist_ok=True)
    os.replace(referenced, canonical)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE recovery_attachment_resolution ("
            "generation_id TEXT, attachment_id INTEGER PRIMARY KEY, "
            "original_stored_path TEXT, source_row_sha256 TEXT, "
            "blob_sha256 TEXT, size INTEGER, custody_relpath TEXT, disposition TEXT)"
        )
        conn.execute(
            "INSERT INTO recovery_attachment_resolution VALUES "
            "('g1',1,'old','row',?,?,?,'legacy-custodied')",
            (digest, len(b"referenced"), str(canonical.relative_to(generation))),
        )
    ready = build_ready(generation, gid="g1", nonce="n1")
    fence = acquire_fence(board, generation="g1", owner="gc-test")
    publish_prepared(
        generation, expected_manifest=ready["manifest_digest"],
        fence_token=fence["token"],
    )
    activate_first_write(generation, fence_epoch=fence["epoch"])
    release_fence(board, token=fence["token"])
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    old = time.time() - 3600
    os.utime(orphan, (old, old))
    return board, generation, canonical, orphan


def test_gc_sweeps_old_unreferenced_blob_but_keeps_current_resolution(tmp_path):
    board, _generation, referenced, orphan = _active_blob_generation(tmp_path)
    result = gc_generations(board, grace_seconds=0)
    assert referenced.is_file()
    assert not orphan.exists()
    assert any(str(orphan) in path for path in result["blob_sweep"])


def test_gc_marks_every_blob_while_generation_has_live_lease(tmp_path):
    board, generation, referenced, orphan = _active_blob_generation(tmp_path)
    lease = acquire_reader_lease(
        board, consumer="test-reader", operation_id="old-snapshot"
    )
    result = gc_generations(board, grace_seconds=0)
    assert referenced.is_file() and orphan.is_file()
    assert result["blob_sweep"] == []
    release_reader_lease(generation, lease["lease_id"])
    gc_generations(board, grace_seconds=0)
    assert referenced.is_file() and not orphan.exists()
