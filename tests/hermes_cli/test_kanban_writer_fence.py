import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli.kanban_generation import GenerationError, acquire_fence, adopt_fence, board_write_guard, release_fence


def test_fence_is_exclusive_and_live_owner_cannot_be_adopted(tmp_path):
    first = acquire_fence(tmp_path, generation="g1", owner="builder")
    entered = tmp_path / "entered"
    probe = subprocess.Popen(
        [sys.executable, "-c", "from pathlib import Path; import sys; from hermes_cli.kanban_generation import acquire_fence, release_fence; b=Path(sys.argv[1]); f=acquire_fence(b, generation='g1', owner='other'); Path(sys.argv[2]).write_text('ok'); release_fence(b, token=f['token'])", str(tmp_path), str(entered)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    from tests.waiting import wait_until
    with pytest.raises(AssertionError):
        wait_until(lambda: entered.exists(), timeout=2.0, what="fence waiter before release")
    with pytest.raises(GenerationError, match="WRITER_FENCE_OWNER_LIVE"):
        adopt_fence(
            tmp_path,
            token=first["token"],
            owner="recovered",
            expected_epoch=first["epoch"],
            expected_generation="g1",
        )
    release_fence(tmp_path, token=first["token"])
    assert probe.wait(timeout=10) == 0
    assert not (tmp_path / "generation.writer-fence").exists()


def _crashed_fence(board: Path) -> dict:
    script = (
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "from hermes_cli.kanban_generation import acquire_fence\n"
        "f=acquire_fence(Path(sys.argv[1]),generation='g1',owner='crashed')\n"
        "Path(sys.argv[2]).write_text(json.dumps(f))\n"
        "os._exit(0)\n"
    )
    receipt = board / "fence-receipt.json"
    result = subprocess.run(
        [sys.executable, "-c", script, str(board), str(receipt)], check=False
    )
    assert result.returncode == 0
    return json.loads(receipt.read_text())


def test_fence_adoption_requires_exact_stale_identity_epoch_and_generation(tmp_path):
    first = _crashed_fence(tmp_path)
    with pytest.raises(GenerationError, match="WRITER_FENCE_EPOCH_MISMATCH"):
        adopt_fence(
            tmp_path, token=first["token"], owner="recovered",
            expected_epoch=first["epoch"] + 1, expected_generation="g1",
        )
    with pytest.raises(GenerationError, match="WRITER_FENCE_GENERATION_MISMATCH"):
        adopt_fence(
            tmp_path, token=first["token"], owner="recovered",
            expected_epoch=first["epoch"], expected_generation="g2",
        )
    adopted = adopt_fence(
        tmp_path, token=first["token"], owner="recovered",
        expected_epoch=first["epoch"], expected_generation="g1",
    )
    assert adopted["token"] == first["token"]
    assert adopted["epoch"] > first["epoch"]
    assert adopted["owner_pid"] == os.getpid()
    release_fence(tmp_path, token=adopted["token"])


def test_fence_adoption_treats_reused_pid_fingerprint_as_stale(tmp_path):
    first = _crashed_fence(tmp_path)
    record_path = tmp_path / "generation.writer-fence"
    record = json.loads(record_path.read_text())
    record["owner_pid"] = os.getpid()
    record["owner_start_fingerprint"] = "different-incarnation"
    record_path.write_text(json.dumps(record))
    adopted = adopt_fence(
        tmp_path, token=first["token"], owner="recovered",
        expected_epoch=first["epoch"], expected_generation="g1",
    )
    assert adopted["owner_pid"] == os.getpid()
    release_fence(tmp_path, token=adopted["token"])


def test_shared_mutation_guard_requires_current_fence_token(tmp_path):
    fence = acquire_fence(tmp_path, generation="g1", owner="builder")
    with pytest.raises(GenerationError, match="WRITER_FENCE_TOKEN_INVALID"):
        with board_write_guard(tmp_path, owner="writer"):
            pass
    with board_write_guard(tmp_path, owner="writer", token=fence["token"]):
        (tmp_path / "mutation.marker").write_text("ok", encoding="utf-8")


def test_guard_is_legacy_compatible_without_a_generation_fence(tmp_path):
    with board_write_guard(tmp_path, owner="legacy"):
        (tmp_path / "legacy.marker").write_text("ok", encoding="utf-8")


def test_connection_writer_rejects_pointer_replacement_after_open(tmp_path):
    from hermes_cli.kanban_generation import activate_first_write, build_ready, initialize_generation_db, publish_prepared, write_pointer
    from hermes_cli.kanban_db_connect import connect, write_txn

    board = tmp_path / "board"
    generation = board / "generations" / "g1"
    generation.mkdir(parents=True)
    initialize_generation_db(generation / "kanban.db", gid="g1", nonce="n1", control_root=board)
    with sqlite3.connect(generation / "kanban.db") as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT)")
    ready = build_ready(generation, gid="g1", nonce="n1")
    fence = acquire_fence(board, generation="g1", owner="publisher")
    publish_prepared(generation, expected_manifest=ready["manifest_digest"], fence_token=fence["token"])
    activate_first_write(generation, fence_epoch=fence["epoch"])
    release_fence(board, token=fence["token"])

    conn = connect(db_path=generation / "kanban.db")
    pointer = json.loads((board / "generation.current").read_text())
    write_pointer(board, {**pointer, "generation_id": "replacement", "generation_nonce": "replacement"})
    with pytest.raises(GenerationError, match="GENERATION_POINTER_CHANGED"):
        with write_txn(conn):
            conn.execute("INSERT INTO tasks VALUES ('stale', 'must not land')")
    conn.close()


def test_publication_rejects_fence_for_different_generation(tmp_path):
    from tests.hermes_cli.test_kanban_corrections import _prepared
    from hermes_cli.kanban_generation import publish_prepared

    board, generation, ready = _prepared(tmp_path)
    fence = acquire_fence(board, generation="different-generation", owner="wrong")
    with pytest.raises(GenerationError, match="WRITER_FENCE_GENERATION_MISMATCH"):
        publish_prepared(
            generation,
            expected_manifest=ready["manifest_digest"],
            fence_token=fence["token"],
        )
    assert not (board / "generation.accepted.json").exists()
    assert not (board / "generation.current").exists()
    release_fence(board, token=fence["token"])
