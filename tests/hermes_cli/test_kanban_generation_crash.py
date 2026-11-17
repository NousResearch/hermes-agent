import json
import sqlite3

import pytest

from hermes_cli.kanban_generation import (
    DurabilityAdapter, GenerationError, activate_first_write, build_ready,
    generation_id, initialize_generation_db, publish_prepared, read_pointer,
    rollback_pre_activation, write_pointer,
)
from hermes_cli.kanban_boards import _publish_status


def _prepared(tmp_path, board=None):
    board = board or tmp_path / "board"
    generation = board / "generations" / "g1"
    generation.mkdir(parents=True)
    db = generation / "kanban.db"
    initialize_generation_db(db, gid="g1", nonce="n1", control_root=board)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE values_table (value TEXT)")
        conn.execute("INSERT INTO values_table VALUES ('before')")
        conn.commit()
    ready = build_ready(generation, gid="g1", nonce="n1")
    return board, generation, ready


def test_activation_is_single_durable_first_write_protocol(tmp_path):
    board, generation, _ready = _prepared(tmp_path)
    fence = __import__("hermes_cli.kanban_generation", fromlist=["acquire_fence"]).acquire_fence(board, generation="g1", owner="writer")
    marker = activate_first_write(generation, fence_epoch=fence["epoch"])
    assert marker["ready_sha256"]
    assert activate_first_write(generation, fence_epoch=fence["epoch"] + 1) == marker


def test_activation_failpoint_leaves_no_false_activation(tmp_path):
    board, generation, _ready = _prepared(tmp_path)
    adapter = DurabilityAdapter(failpoint="activation-before-rename")
    fence = __import__("hermes_cli.kanban_generation", fromlist=["acquire_fence"]).acquire_fence(board, generation="g1", owner="writer")
    with pytest.raises(OSError, match="activation-before-rename"):
        activate_first_write(generation, fence_epoch=fence["epoch"], durability=adapter)
    assert not (generation / "ACTIVATED.json").exists()


def test_pre_activation_rollback_restores_pointer_byte_for_byte(tmp_path):
    board, old, old_ready = _prepared(tmp_path)
    write_pointer(board, {"format_version": 1, "generation_id": "g1", "generation_nonce": "n1", "ready_sha256": __import__("hermes_cli.kanban_generation", fromlist=["sha256_file"]).sha256_file(old / "READY.json")})
    token = __import__("hermes_cli.kanban_generation", fromlist=["acquire_fence"]).acquire_fence(board, generation="g2", owner="writer")["token"]
    with pytest.raises(GenerationError, match="ROLLBACK_NOT_AVAILABLE"):
        rollback_pre_activation(board, expected_active_manifest=old_ready["manifest_digest"], fence_token=token)


def test_reader_status_is_immutable_no_replace(tmp_path):
    directory = tmp_path / "status"
    _publish_status(directory, "ready.json", {"type": "ready", "operation_id": "op"})
    with pytest.raises(FileExistsError):
        _publish_status(directory, "ready.json", {"type": "ready", "operation_id": "op"})
