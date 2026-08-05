"""S2: Bridge — soft FK, native write forbidden, board mirror, parent bind."""

import pytest
import os
import sys
import tempfile
import hashlib
import sqlite3
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_bridge import (
    BridgeError,
    NativeTaskRef,
    OrchBridge,
    init_sidecar_db,
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture
def bridge_env():
    """Create temp native + sidecar DBs and return a bridge."""
    tmpdir = tempfile.mkdtemp()
    native_path = os.path.join(tmpdir, "native.db")
    sidecar_path = os.path.join(tmpdir, "orch_v4.db")

    # Create minimal native DB with tasks table
    nconn = sqlite3.connect(native_path)
    nconn.execute("PRAGMA foreign_keys=ON")
    nconn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT NOT NULL, created_at INTEGER NOT NULL)")
    nconn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('task-1', 'Test', 'pending', 1)")
    nconn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('task-2', 'Done Task', 'done', 2)")
    nconn.commit()
    nconn.close()

    # Create sidecar
    init_sidecar_db(sidecar_path)

    sha_before = _sha256_file(native_path)

    bridge = OrchBridge(native_path, sidecar_path)
    yield bridge, native_path, sidecar_path, sha_before, tmpdir
    bridge.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_soft_fk_accepts_existing_task(bridge_env):
    """Soft FK: existing native task is accepted."""
    bridge, native_path, sidecar_path, sha_before, _ = bridge_env
    ref = bridge.assert_task_exists("task-1")
    assert ref.task_id == "task-1"
    assert ref.title == "Test"
    assert ref.status == "pending"


def test_soft_fk_rejects_missing_task(bridge_env):
    """Soft FK: missing native task raises BridgeError."""
    bridge, *_ = bridge_env
    with pytest.raises(BridgeError, match="soft_fk_violation"):
        bridge.assert_task_exists("nonexistent-task")


def test_bridge_never_writes_native(bridge_env):
    """Bridge must never write to native DB (SHA-256 unchanged)."""
    bridge, native_path, sidecar_path, sha_before, _ = bridge_env

    # Perform sidecar operations
    bridge.ensure_board_mirror("board_0123456789abcdef", "default")
    bridge.bind_parent_task("board_0123456789abcdef", "", "orch-1", "task-1")

    # Verify native DB bytes unchanged
    sha_after = _sha256_file(native_path)
    assert sha_before == sha_after, "Native DB was mutated by bridge!"


def test_native_write_forbidden_by_default():
    """OrchBridge with native_writable=True must raise."""
    tmpdir = tempfile.mkdtemp()
    try:
        native_path = os.path.join(tmpdir, "native.db")
        sidecar_path = os.path.join(tmpdir, "orch_v4.db")
        sqlite3.connect(native_path).close()  # create empty
        init_sidecar_db(sidecar_path)

        with pytest.raises(BridgeError, match="native_write_forbidden_by_sidecar_decision"):
            OrchBridge(native_path, sidecar_path, native_writable=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_board_mirror_roundtrip(bridge_env):
    """Board mirror: upsert into sidecar, read back."""
    bridge, native_path, sidecar_path, sha_before, _ = bridge_env
    bridge.ensure_board_mirror("board_0123456789abcdef", "default")

    # Verify mirror exists in sidecar
    row = bridge._sidecar.execute(
        "SELECT board_instance_id, canonical_board_key FROM kanban_board_identity WHERE board_instance_id = ?",
        ("board_0123456789abcdef",)
    ).fetchone()
    assert row is not None
    assert row["board_instance_id"] == "board_0123456789abcdef"
    assert row["canonical_board_key"] == "default"

    # Idempotent: second call should not error
    bridge.ensure_board_mirror("board_0123456789abcdef", "default")


def test_bind_parent_task_sidecar_only(bridge_env):
    """bind_parent_task writes only to sidecar, not native.

    The key invariant: soft FK is enforced (native task must exist),
    and native DB bytes are unchanged after the operation.
    """
    bridge, native_path, sidecar_path, sha_before, _ = bridge_env

    # Bind task-1 (exists in native)
    bridge.bind_parent_task("board_0123456789abcdef", "", "orch-1", "task-1")

    # Verify sidecar was written to (board mirror exists)
    rows = bridge._sidecar.execute(
        "SELECT count(*) as cnt FROM kanban_board_identity WHERE board_instance_id = ?",
        ("board_0123456789abcdef",)
    ).fetchone()
    assert rows["cnt"] >= 1, "Sidecar should have board mirror entry after bind"

    # Verify native unchanged
    sha_after = _sha256_file(native_path)
    assert sha_before == sha_after, "Native DB was mutated by bind_parent_task!"


def test_bind_parent_rejects_missing_native_task(bridge_env):
    """bind_parent_task with non-existent native task raises soft FK error."""
    bridge, *_ = bridge_env
    with pytest.raises(BridgeError, match="soft_fk_violation"):
        bridge.bind_parent_task("board_0123456789abcdef", "", "orch-2", "nonexistent")


def test_live_paths_not_used(bridge_env):
    """Bridge fixture paths must not be live paths."""
    bridge, native_path, sidecar_path, sha_before, tmpdir = bridge_env
    LIVE_PATHS = {"/home/claw/.hermes/kanban.db", "/home/claw/.hermes/orch_v4.db"}
    assert native_path not in LIVE_PATHS
    assert sidecar_path not in LIVE_PATHS
    assert tmpdir not in LIVE_PATHS


def test_read_native_task(bridge_env):
    """read_native_task returns NativeTaskRef for existing task."""
    bridge, *_ = bridge_env
    ref = bridge.read_native_task("task-2")
    assert ref is not None
    assert ref.task_id == "task-2"
    assert ref.title == "Done Task"
    assert ref.status == "done"

    # Missing task returns None
    assert bridge.read_native_task("nonexistent") is None
