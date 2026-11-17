import json
import sqlite3
import subprocess
import sys

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
    assert activate_first_write(generation, fence_epoch=fence["epoch"]) == marker
    with pytest.raises(GenerationError, match="ACTIVATION_POISONED"):
        activate_first_write(generation, fence_epoch=fence["epoch"] + 1)


def test_activation_failpoint_leaves_no_false_activation(tmp_path):
    board, generation, _ready = _prepared(tmp_path)
    adapter = DurabilityAdapter(failpoint="activation-before-rename")
    fence = __import__("hermes_cli.kanban_generation", fromlist=["acquire_fence"]).acquire_fence(board, generation="g1", owner="writer")
    with pytest.raises(OSError, match="activation-before-rename"):
        activate_first_write(generation, fence_epoch=fence["epoch"], durability=adapter)
    assert not (generation / "ACTIVATED.json").exists()
    assert not (generation / "ACTIVATION.poison.json").exists()
    marker = activate_first_write(generation, fence_epoch=fence["epoch"])
    assert marker["fence_epoch"] == fence["epoch"]


def test_reported_post_rename_failure_is_durably_retryable(tmp_path):
    board, generation, _ready = _prepared(tmp_path)
    from hermes_cli.kanban_generation import acquire_fence
    fence = acquire_fence(board, generation="g1", owner="writer")
    with pytest.raises(OSError, match="g1-after-fsync-dir"):
        activate_first_write(
            generation, fence_epoch=fence["epoch"],
            durability=DurabilityAdapter(failpoint="g1-after-fsync-dir"),
        )
    assert not (generation / "ACTIVATED.json").exists()
    assert not (generation / "ACTIVATION.poison.json").exists()
    assert activate_first_write(generation, fence_epoch=fence["epoch"])["fence_epoch"] == fence["epoch"]


def test_invalid_surviving_marker_is_poisoned(tmp_path):
    board, generation, _ready = _prepared(tmp_path)
    from hermes_cli.kanban_generation import acquire_fence
    fence = acquire_fence(board, generation="g1", owner="writer")
    (generation / "ACTIVATED.json").write_text("{}")
    with pytest.raises(GenerationError, match="ACTIVATION_POISONED"):
        activate_first_write(generation, fence_epoch=fence["epoch"])
    assert (generation / "ACTIVATION.poison.json").is_file()


def test_ambiguous_activation_cleanup_is_poisoned(tmp_path, monkeypatch):
    board, generation, _ready = _prepared(tmp_path)
    from hermes_cli.kanban_generation import acquire_fence
    fence = acquire_fence(board, generation="g1", owner="writer")
    final = generation / "ACTIVATED.json"
    original_unlink = type(final).unlink

    def refuse_marker_unlink(path, *args, **kwargs):
        if path == final:
            raise OSError("cleanup blocked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(final), "unlink", refuse_marker_unlink)
    with pytest.raises(GenerationError, match="ACTIVATION_POISONED"):
        activate_first_write(
            generation, fence_epoch=fence["epoch"],
            durability=DurabilityAdapter(failpoint="g1-after-fsync-dir"),
        )
    assert (generation / "ACTIVATION.poison.json").is_file()


def test_poison_fallback_blocks_fresh_reads_writes_and_rollback(tmp_path, monkeypatch):
    board, generation, ready = _prepared(tmp_path)
    from hermes_cli import kanban_generation
    from hermes_cli.kanban_generation import acquire_fence, release_fence
    fence = acquire_fence(board, generation="g1", owner="writer")
    publish_prepared(
        generation, expected_manifest=ready["manifest_digest"],
        fence_token=fence["token"],
    )
    final = generation / "ACTIVATED.json"
    original_unlink = type(final).unlink
    original_atomic = kanban_generation._atomic_json

    def refuse_marker_unlink(path, *args, **kwargs):
        if path == final:
            raise OSError("cleanup blocked")
        return original_unlink(path, *args, **kwargs)

    def fail_json_poison(path, value, **kwargs):
        if path.name == "ACTIVATION.poison.json":
            raise OSError("poison json blocked")
        return original_atomic(path, value, **kwargs)

    monkeypatch.setattr(type(final), "unlink", refuse_marker_unlink)
    monkeypatch.setattr(kanban_generation, "_atomic_json", fail_json_poison)
    with pytest.raises(GenerationError, match="ACTIVATION_POISONED"):
        activate_first_write(
            generation, fence_epoch=fence["epoch"],
            durability=DurabilityAdapter(failpoint="g1-after-fsync-dir"),
        )
    assert (generation / "ACTIVATION.poison").is_file()
    release_fence(board, token=fence["token"])
    probe = (
        "import json,sys\n"
        "from pathlib import Path\n"
        "from hermes_cli.kanban_db_connect import connect\n"
        "from hermes_cli.kanban_generation import acquire_fence,open_generation,rollback_pre_activation\n"
        "b=Path(sys.argv[1]); g=Path(sys.argv[2]); out=[]\n"
        "for op in ('read','write','rollback'):\n"
        " try:\n"
        "  if op=='read': open_generation(b,require_activation=True)\n"
        "  elif op=='write': connect(db_path=g/'kanban.db')\n"
        "  else:\n"
        "   f=acquire_fence(b,generation='g1',owner='probe'); rollback_pre_activation(b,expected_active_manifest=sys.argv[3],fence_token=f['token'])\n"
        " except Exception as e: out.append(str(e))\n"
        "print(json.dumps(out))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, str(board), str(generation), ready["manifest_digest"]],
        text=True, capture_output=True, check=True,
    )
    assert json.loads(result.stdout) == ["ACTIVATION_POISONED"] * 3


@pytest.mark.parametrize("damage", ["integrity", "foreign-key"])
def test_activated_generation_validates_sqlite_state(tmp_path, damage):
    board, generation, _ready = _prepared(tmp_path)
    from hermes_cli.kanban_generation import acquire_fence, validate_generation
    fence = acquire_fence(board, generation="g1", owner="writer")
    activate_first_write(generation, fence_epoch=fence["epoch"])
    with sqlite3.connect(generation / "kanban.db") as conn:
        if damage == "integrity":
            conn.execute("PRAGMA writable_schema=ON")
            conn.execute("UPDATE sqlite_schema SET rootpage=999999 WHERE name='values_table'")
            conn.execute("PRAGMA writable_schema=RESET")
        else:
            conn.executescript(
                "CREATE TABLE fk_parent(id INTEGER PRIMARY KEY);"
                "CREATE TABLE fk_child(id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES fk_parent(id));"
                "INSERT INTO fk_child VALUES(1, 999);"
            )
        conn.commit()
    with pytest.raises(GenerationError, match="GENERATION_META_INVALID"):
        validate_generation(generation, activated=True)


def test_published_activation_uses_exact_accepted_fence_epoch(tmp_path):
    board, generation, ready = _prepared(tmp_path)
    from hermes_cli.kanban_generation import acquire_fence
    fence = acquire_fence(board, generation="g1", owner="writer")
    publish_prepared(
        generation,
        expected_manifest=ready["manifest_digest"],
        fence_token=fence["token"],
    )
    with pytest.raises(GenerationError, match="ACTIVATION_FENCE_MISMATCH"):
        activate_first_write(generation, fence_epoch=fence["epoch"] + 1)
    assert not (generation / "ACTIVATED.json").exists()
    marker = activate_first_write(generation)
    assert marker["fence_epoch"] == fence["epoch"]
    assert not (board / "generation.accepted.json").exists()


@pytest.mark.parametrize(
    "checkpoint",
    [
        "ACTIVATED.json-before-fsync",
        "ACTIVATED.json-after-fsync",
        "ACTIVATED.json-before-rename",
        "ACTIVATED.json-after-rename",
        "g1-before-fsync-dir",
        "g1-after-fsync-dir",
    ],
)
def test_sigkill_activation_failpoints_reopen_without_false_effect(tmp_path, checkpoint):
    board, generation, ready = _prepared(tmp_path)
    from hermes_cli.kanban_generation import acquire_fence, release_fence
    fence = acquire_fence(board, generation="g1", owner="publisher")
    publish_prepared(
        generation,
        expected_manifest=ready["manifest_digest"],
        fence_token=fence["token"],
    )
    release_fence(board, token=fence["token"])
    reached = tmp_path / "checkpoint"
    script = (
        "import os,sys\n"
        "from pathlib import Path\n"
        "from hermes_cli.kanban_generation import DurabilityAdapter,activate_first_write\n"
        "class Stop(DurabilityAdapter):\n"
        "  def checkpoint(self,name):\n"
        "    if name==sys.argv[3]:\n"
        "      Path(sys.argv[2]).write_text(name); os.kill(os.getpid(), 19)\n"
        "activate_first_write(Path(sys.argv[1]), durability=Stop())\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(generation), str(reached), checkpoint]
    )
    from tests.waiting import wait_until
    wait_until(reached.exists, timeout=10, what=f"activation checkpoint {checkpoint}")
    child.kill()
    assert child.wait(timeout=10) < 0
    marker = activate_first_write(generation)
    assert marker["fence_epoch"] == fence["epoch"]
    assert not (generation / "ACTIVATION.poison.json").exists()
    with sqlite3.connect(generation / "kanban.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM values_table").fetchone()[0] == 1


def test_reader_recovery_takes_exclusive_catalog_before_shared_selection(tmp_path):
    board, generation, ready = _prepared(tmp_path)
    from hermes_cli.kanban_generation import acquire_fence, release_fence
    fence = acquire_fence(board, generation="g1", owner="publisher")
    publish_prepared(
        generation,
        expected_manifest=ready["manifest_digest"],
        fence_token=fence["token"],
    )
    accepted = (board / "generation.accepted.json").read_bytes()
    activate_first_write(generation)
    release_fence(board, token=fence["token"])
    (board / "generation.accepted.json").write_bytes(accepted)
    (board / "generation.current").unlink()

    import fcntl
    lock = (board / "generation.catalog-lock").open("a+b")
    fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
    done = tmp_path / "reader-done"
    script = (
        "from pathlib import Path\n"
        "import sys\n"
        "from hermes_cli.kanban_generation import acquire_reader_lease,release_reader_lease\n"
        "b=Path(sys.argv[1]); r=acquire_reader_lease(b,consumer='hermes-office',operation_id='op'); "
        "release_reader_lease(Path(r['database_path']).parent,r['lease_id']); Path(sys.argv[2]).write_text('done')\n"
    )
    child = subprocess.Popen([sys.executable, "-c", script, str(board), str(done)])
    from tests.waiting import wait_until
    with pytest.raises(AssertionError):
        wait_until(done.exists, timeout=1, what="reader bypassing exclusive recovery")
    assert not (board / "generation.current").exists()
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()
    assert child.wait(timeout=10) == 0
    assert done.exists()
    assert read_pointer(board)["generation_id"] == "g1"


@pytest.mark.parametrize("kill_writer_a", [False, True])
def test_writers_and_rollback_have_deterministic_effect_barriers(tmp_path, kill_writer_a):
    board, generation, ready = _prepared(tmp_path)
    from hermes_cli.kanban_generation import acquire_fence, release_fence
    fence = acquire_fence(board, generation="g1", owner="publisher")
    publish_prepared(
        generation,
        expected_manifest=ready["manifest_digest"],
        fence_token=fence["token"],
    )
    release_fence(board, token=fence["token"])

    writer_a_script = (
        "from pathlib import Path\n"
        "import sys,time\n"
        "import hermes_cli.kanban_generation as kg\n"
        "from hermes_cli.kanban_db_connect import connect,write_txn\n"
        "class Pause(kg.DurabilityAdapter):\n"
        " def checkpoint(self,name):\n"
        "  if name=='g1-before-fsync-dir':\n"
        "   Path(sys.argv[2]).write_text('paused')\n"
        "   while not Path(sys.argv[3]).exists(): time.sleep(.01)\n"
        "original=kg.activate_first_write\n"
        "kg.activate_first_write=lambda path: original(path,durability=Pause())\n"
        "c=connect(db_path=Path(sys.argv[1]))\n"
        "with write_txn(c): c.execute('INSERT INTO values_table VALUES (?)',('writer-a',))\n"
        "c.close()\n"
    )
    writer_b_script = (
        "from pathlib import Path\n"
        "import sys\n"
        "from hermes_cli.kanban_db_connect import connect,write_txn\n"
        "from hermes_cli.kanban_generation import board_write_guard\n"
        "db=Path(sys.argv[1])\n"
        "with board_write_guard(db.parents[2],expected_generation='g1'):\n"
        " Path(sys.argv[2]).write_text('admitted')\n"
        " c=connect(db_path=db)\n"
        " with write_txn(c): c.execute('INSERT INTO values_table VALUES (?)',('writer-b',))\n"
        " c.close()\n"
        "Path(sys.argv[3]).write_text('effect')\n"
    )
    rollback_script = (
        "from pathlib import Path\n"
        "import sys\n"
        "from hermes_cli.kanban_generation import acquire_fence,release_fence,rollback_pre_activation\n"
        "b=Path(sys.argv[1]); f=acquire_fence(b,generation='g1',owner='rollback'); "
        "Path(sys.argv[2]).write_text('acquired')\n"
        "try:\n"
        " rollback_pre_activation(b,expected_active_manifest=sys.argv[4],fence_token=f['token'])\n"
        "except Exception as e:\n"
        " Path(sys.argv[3]).write_text(str(e))\n"
        "finally:\n"
        " release_fence(b,token=f['token'])\n"
    )
    a_paused, a_release = tmp_path / "a-paused", tmp_path / "a-release"
    b_admitted, b_effect = tmp_path / "b-admitted", tmp_path / "b-effect"
    rollback_acquired, rollback_result = tmp_path / "rollback-acquired", tmp_path / "rollback-result"
    writer_a = subprocess.Popen([
        sys.executable, "-c", writer_a_script, str(generation / "kanban.db"),
        str(a_paused), str(a_release),
    ])
    from tests.waiting import wait_until
    wait_until(a_paused.exists, timeout=10, what="writer A post-rename barrier")
    writer_b = subprocess.Popen([
        sys.executable, "-c", writer_b_script, str(generation / "kanban.db"),
        str(b_admitted), str(b_effect),
    ])
    wait_until(b_admitted.exists, timeout=10, what="writer B first-write admission")
    rollback = subprocess.Popen([
        sys.executable, "-c", rollback_script, str(board), str(rollback_acquired),
        str(rollback_result), ready["manifest_digest"],
    ])
    with pytest.raises(AssertionError):
        wait_until(b_effect.exists, timeout=1, what="writer B effect before activation durability")
    assert not rollback_acquired.exists()
    with sqlite3.connect(generation / "kanban.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM values_table WHERE value LIKE 'writer-%'"
        ).fetchone()[0] == 0
    if kill_writer_a:
        writer_a.kill()
        assert writer_a.wait(timeout=10) < 0
    else:
        a_release.touch()
        assert writer_a.wait(timeout=10) == 0
    assert writer_b.wait(timeout=10) == 0
    assert rollback.wait(timeout=10) == 0
    assert rollback_result.read_text() == "ROLLBACK_CLOSED"
    with sqlite3.connect(generation / "kanban.db") as conn:
        effects = [row[0] for row in conn.execute(
            "SELECT value FROM values_table WHERE value LIKE 'writer-%' ORDER BY value"
        )]
    assert effects == (["writer-b"] if kill_writer_a else ["writer-a", "writer-b"])


def test_pre_activation_rollback_restores_pointer_byte_for_byte(tmp_path):
    board, old, old_ready = _prepared(tmp_path)
    write_pointer(board, {"format_version": 1, "generation_id": "g1", "generation_nonce": "n1", "ready_sha256": __import__("hermes_cli.kanban_generation", fromlist=["sha256_file"]).sha256_file(old / "READY.json")})
    token = __import__("hermes_cli.kanban_generation", fromlist=["acquire_fence"]).acquire_fence(board, generation="g2", owner="writer")["token"]
    with pytest.raises(GenerationError, match="WRITER_FENCE_GENERATION_MISMATCH"):
        rollback_pre_activation(board, expected_active_manifest=old_ready["manifest_digest"], fence_token=token)


def test_reader_status_is_immutable_no_replace(tmp_path):
    directory = tmp_path / "status"
    directory.mkdir(mode=0o700)
    _publish_status(directory, "ready.json", {"type": "ready", "operation_id": "op"})
    with pytest.raises(FileExistsError):
        _publish_status(directory, "ready.json", {"type": "ready", "operation_id": "op"})
