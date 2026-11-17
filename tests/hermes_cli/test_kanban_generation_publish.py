import json
import sqlite3

import pytest

from hermes_cli.kanban_generation import (
    GenerationError, acquire_fence, acquire_reader_lease, build_ready,
    initialize_generation_db, open_generation, publish_prepared, release_reader_lease,
    write_pointer,
)


def _published(tmp_path):
    board = tmp_path / "board"
    generation = board / "generations" / "g1"
    generation.mkdir(parents=True)
    db = generation / "kanban.db"
    initialize_generation_db(db, gid="g1", nonce="nonce", control_root=board)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute("INSERT INTO tasks VALUES ('t1', 'one')")
        conn.commit()
    ready = build_ready(generation, gid="g1", nonce="nonce")
    fence = acquire_fence(board, generation="g1", owner="publisher")
    publish_prepared(generation, expected_manifest=ready["manifest_digest"], fence_token=fence["token"])
    activate = __import__("hermes_cli.kanban_generation", fromlist=["activate_first_write"])
    activate.activate_first_write(generation, fence_epoch=fence["epoch"])
    return board, generation


def test_reader_lease_binds_database_to_one_validated_generation(tmp_path):
    board, generation = _published(tmp_path)
    lease = acquire_reader_lease(board, consumer="gateway", operation_id="op-1")
    assert lease["generation_id"] == "g1"
    assert lease["generation_nonce"] == "nonce"
    assert lease["lease_keeper_pid"] == lease["consumer_pid"]
    assert (generation / "leases" / f"{lease['lease_id']}.json").exists()
    release_reader_lease(generation, lease["lease_id"])


def test_reader_lease_reclaims_stale_pid_fingerprint(tmp_path):
    board, generation = _published(tmp_path)
    lease = acquire_reader_lease(board, consumer="gateway", operation_id="op-1")
    path = generation / "leases" / f"{lease['lease_id']}.json"
    data = json.loads(path.read_text())
    data["lease_keeper_pid"] = 99999999
    data["lease_keeper_start_fingerprint"] = "dead"
    path.write_text(json.dumps(data))
    replacement = acquire_reader_lease(board, consumer="gateway", operation_id="op-2")
    assert replacement["lease_id"] != lease["lease_id"]
