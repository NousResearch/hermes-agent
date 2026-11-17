import json
import io
import multiprocessing
import os
import sys
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli.kanban_generation import (
    DurabilityAdapter,
    GenerationError,
    acquire_fence,
    acquire_reader_lease,
    activate_first_write,
    build_ready,
    bind_reader_consumer,
    canonical_json,
    gc_generations,
    initialize_generation_db,
    publish_prepared,
    read_pointer,
    release_fence,
    release_reader_lease,
    manifest_digest,
    row_manifest,
    sha256_file,
    encode_reader_frame,
    decode_reader_frame,
    discard_created_blobs,
)
from hermes_cli.kanban_reconcile import reconcile


def _db(path: Path, task: str, *, child_body: str = "history") -> None:
    with sqlite3.connect(path) as conn:
        from hermes_cli.kanban_db import SCHEMA_SQL
        conn.executescript(SCHEMA_SQL)
        conn.execute("ALTER TABLE tasks ADD COLUMN completion_contract TEXT")
        for sql in (
            "CREATE INDEX idx_tasks_assignee_status ON tasks(assignee, status)",
            "CREATE INDEX idx_tasks_tenant ON tasks(tenant)",
            "CREATE INDEX idx_tasks_idempotency ON tasks(idempotency_key)",
            "CREATE INDEX idx_tasks_session_id ON tasks(session_id)",
            "CREATE INDEX idx_events_run ON task_events(run_id, id)",
        ):
            conn.execute(sql)
        conn.execute("INSERT INTO tasks(id, title, status, created_at, workspace_kind, block_kind, block_recurrences) VALUES (?, ?, 'blocked', 1, 'scratch', 'incident', 0)", (task, task))
        conn.execute("INSERT INTO task_comments(task_id, author, body, created_at) VALUES (?, 'fixture', ?, 1)", (task, child_body))
        conn.execute("INSERT INTO task_runs(task_id, status, started_at) VALUES (?, 'released', 1)", (task,))
        conn.execute("INSERT INTO task_events(task_id, kind, run_id, created_at) VALUES (?, 'history', 1, 1)", (task,))


def test_target_wins_does_not_append_source_children(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    _db(source, "same", child_body="source")
    _db(target, "same", child_body="target")
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({"task_ids": ["other"]}))
    with sqlite3.connect(source) as conn:
        conn.execute("INSERT INTO tasks(id, title, status, created_at, workspace_kind, block_kind, block_recurrences) VALUES ('other', 'other', 'blocked', 1, 'scratch', 'incident', 0)")
        conn.execute("INSERT INTO task_links VALUES ('other', 'same')")
    generation = tmp_path / "generation"
    reconcile(source, target, seeds, generation, apply=True)
    with sqlite3.connect(generation / "kanban.db") as conn:
        assert [r[0] for r in conn.execute("SELECT body FROM task_comments WHERE task_id='same'")] == ["target"]
        assert conn.execute("SELECT disposition FROM recovery_row_provenance WHERE table_name='task_comments'").fetchone()[0] == "target-wins"


@pytest.mark.parametrize("shape", ["empty", "multiple", "unrelated"])
def test_target_wins_leaves_ambiguous_child_provenance_unresolved(tmp_path, shape):
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    _db(source, "same", child_body="same-content")
    _db(target, "same", child_body="different-content")
    with sqlite3.connect(target) as conn:
        if shape == "multiple":
            conn.execute("INSERT INTO task_comments(task_id, author, body, created_at) VALUES ('same', 'fixture', 'different-content', 2)")
        elif shape == "unrelated":
            conn.execute("INSERT INTO task_comments(task_id, author, body, created_at) VALUES ('same', 'other', 'same-content', 2)")
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({"task_ids": ["other"]}))
    with sqlite3.connect(source) as conn:
        conn.execute("INSERT INTO tasks(id, title, status, created_at, workspace_kind, block_kind, block_recurrences) VALUES ('other', 'other', 'blocked', 1, 'scratch', 'incident', 0)")
        conn.execute("INSERT INTO task_links VALUES ('other', 'same')")
    generation = tmp_path / "generation"
    reconcile(source, target, seeds, generation, apply=True)
    with sqlite3.connect(generation / "kanban.db") as conn:
        assert conn.execute(
            "SELECT target_pk FROM recovery_row_provenance WHERE table_name='task_comments'"
        ).fetchone()[0] == ""


def test_target_wins_records_only_content_grounded_child_match(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    _db(source, "same", child_body="same-content")
    _db(target, "same", child_body="same-content")
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({"task_ids": ["other"]}))
    with sqlite3.connect(source) as conn:
        conn.execute("INSERT INTO tasks(id, title, status, created_at, workspace_kind, block_kind, block_recurrences) VALUES ('other', 'other', 'blocked', 1, 'scratch', 'incident', 0)")
        conn.execute("INSERT INTO task_links VALUES ('other', 'same')")
    generation = tmp_path / "generation"
    reconcile(source, target, seeds, generation, apply=True)
    with sqlite3.connect(generation / "kanban.db") as conn:
        assert conn.execute(
            "SELECT target_pk FROM recovery_row_provenance WHERE table_name='task_comments'"
        ).fetchone()[0] == "1"


def test_activated_generation_is_writable_and_survives_reopen(tmp_path):
    board = tmp_path / "board"
    generation = board / "generations" / "g1"
    generation.mkdir(parents=True)
    db = generation / "kanban.db"
    initialize_generation_db(db, gid="g1", nonce="n", control_root=board)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT)")
        conn.commit()
    ready = build_ready(generation, gid="g1", nonce="n")
    fence = acquire_fence(board, generation="g1", owner="test")
    publish_prepared(generation, expected_manifest=ready["manifest_digest"], fence_token=fence["token"])
    activate_first_write(generation, fence_epoch=fence["epoch"])
    release_fence(board, token=fence["token"])
    from hermes_cli.kanban_db_connect import write_txn
    conn = sqlite3.connect(db)
    with write_txn(conn):
        conn.execute("INSERT INTO tasks VALUES ('after', 'write')")
    conn.close()
    with sqlite3.connect(db) as reopened:
        assert reopened.execute("SELECT title FROM tasks WHERE id='after'").fetchone()[0] == "write"


def test_publish_authenticates_actual_active_manifest_after_first_write(tmp_path):
    board, generation, ready = _prepared(tmp_path)
    fence = acquire_fence(board, generation="g1", owner="publisher")
    publish_prepared(generation, expected_manifest=ready["manifest_digest"], fence_token=fence["token"])
    activate_first_write(generation, fence_epoch=fence["epoch"])
    with sqlite3.connect(generation / "kanban.db") as conn:
        conn.execute("INSERT INTO tasks VALUES ('written', 'after publish')")
        conn.commit()
        active_manifest = manifest_digest(row_manifest(conn))
    release_fence(board, token=fence["token"])

    candidate = board / "generations" / "g2"
    candidate.mkdir(parents=True)
    initialize_generation_db(candidate / "kanban.db", gid="g2", nonce="n2", control_root=board)
    with sqlite3.connect(candidate / "kanban.db"):
        pass
    build_ready(candidate, gid="g2", nonce="n2")
    fence = acquire_fence(board, generation="g2", owner="republisher")
    publish_prepared(candidate, expected_target_manifest=active_manifest, fence_token=fence["token"])
    release_fence(board, token=fence["token"])

    candidate = board / "generations" / "g3"
    candidate.mkdir(parents=True)
    initialize_generation_db(candidate / "kanban.db", gid="g3", nonce="n3", control_root=board)
    with sqlite3.connect(candidate / "kanban.db"):
        pass
    stale = build_ready(candidate, gid="g3", nonce="n3")
    fence = acquire_fence(board, generation="g3", owner="stale")
    with pytest.raises(GenerationError, match="TARGET_MANIFEST_MISMATCH"):
        publish_prepared(candidate, expected_target_manifest=stale["manifest_digest"], fence_token=fence["token"])
    release_fence(board, token=fence["token"])


@pytest.mark.parametrize("kind", ["directory", "symlink", "bad-json", "bad-shape"])
def test_gc_refuses_malformed_retained_marker_without_mutation(tmp_path, kind):
    board, _generation = _published(tmp_path)
    marker = board / "generation.retained.json"
    if kind == "directory":
        marker.mkdir()
    elif kind == "symlink":
        target = tmp_path / "retained-target"
        target.write_text("[]")
        marker.symlink_to(target)
    elif kind == "bad-json":
        marker.write_text("not-json")
    else:
        marker.write_text(json.dumps({"generation_ids": [1]}))
    with pytest.raises(GenerationError, match="RETAINED_INVALID"):
        gc_generations(board, dry_run=True)
    with pytest.raises(GenerationError, match="RETAINED_INVALID"):
        gc_generations(board, dry_run=False)
    assert marker.exists() or marker.is_symlink()


def test_public_first_write_activates_prepared_generation_and_reopens(tmp_path, monkeypatch):
    board, generation, ready = _prepared(tmp_path)
    fence = acquire_fence(board, generation="g1", owner="publisher")
    publish_prepared(generation, expected_manifest=ready["manifest_digest"], fence_token=fence["token"])
    release_fence(board, token=fence["token"])
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
    from hermes_cli.kanban_db_connect import connect, write_txn
    conn = connect(db_path=generation / "kanban.db")
    assert not (generation / "ACTIVATED.json").exists()
    with write_txn(conn):
        conn.execute("INSERT INTO tasks VALUES ('first', 'write')")
    conn.close()
    assert (generation / "ACTIVATED.json").exists()
    reopened = connect(db_path=generation / "kanban.db")
    assert reopened.execute("SELECT title FROM tasks WHERE id='first'").fetchone()[0] == "write"
    reopened.close()


def test_lease_stale_reclaim_requires_both_identities(tmp_path):
    board, generation = _published(tmp_path)
    lease = acquire_reader_lease(board, consumer="hermes-office", operation_id="op")
    path = generation / "leases" / f"{lease['lease_id']}.json"
    record = json.loads(path.read_text())
    record["lease_keeper_pid"] = 99999999
    record["lease_keeper_start_fingerprint"] = "dead"
    path.write_text(json.dumps(record))
    replacement = acquire_reader_lease(board, consumer="hermes-office", operation_id="next")
    assert path.exists()
    release_reader_lease(generation, replacement["lease_id"])
    release_reader_lease(generation, lease["lease_id"])


def test_reader_lease_persists_only_closed_v1_fields(tmp_path):
    board, generation = _published(tmp_path)
    lease = acquire_reader_lease(
        board, consumer="hermes-office", operation_id="not-persisted"
    )
    path = generation / "leases" / f"{lease['lease_id']}.json"
    expected_fields = {
        "lease_format_version", "lease_id", "board_slug", "generation_id",
        "generation_nonce", "ready_sha256", "consumer", "lease_keeper_pid",
        "lease_keeper_start_fingerprint", "consumer_pid",
        "consumer_start_fingerprint", "consumer_kind", "created_at",
    }
    initial = json.loads(path.read_text())
    assert set(initial) == expected_fields
    assert path.read_bytes() == canonical_json(initial) + b"\n"

    bind_reader_consumer(
        generation, lease["lease_id"], pid=os.getpid(), start_token="private-token"
    )
    bound = json.loads(path.read_text())
    assert set(bound) == expected_fields
    assert path.read_bytes() == canonical_json(bound) + b"\n"
    assert "private-token" not in path.read_text()
    release_reader_lease(generation, lease["lease_id"])


def test_reader_selection_retries_changed_pointer_after_durable_lease(
    tmp_path, monkeypatch
):
    board, generation = _published(tmp_path)
    other = board / "generations" / "g2"
    other.mkdir()
    initialize_generation_db(
        other / "kanban.db", gid="g2", nonce="n2", control_root=board
    )
    with sqlite3.connect(other / "kanban.db") as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT)")
    build_ready(other, gid="g2", nonce="n2")
    activate_first_write(other, fence_epoch=2)
    changed_pointer = {
        "format_version": 1,
        "generation_id": "g2",
        "generation_nonce": "n2",
        "ready_sha256": sha256_file(other / "READY.json"),
    }

    from hermes_cli import kanban_generation as kg
    real_read_pointer = kg.read_pointer
    reads = 0

    def changed_once(path):
        nonlocal reads
        reads += 1
        if reads == 2:
            return changed_pointer
        return real_read_pointer(path)

    monkeypatch.setattr(kg, "read_pointer", changed_once)
    lease = acquire_reader_lease(
        board, consumer="hermes-office", operation_id="retry"
    )
    assert Path(lease["database_path"]).parent == generation
    assert reads >= 4
    assert [path.name for path in (generation / "leases").glob("*.json")] == [
        f"{lease['lease_id']}.json"
    ]
    release_reader_lease(generation, lease["lease_id"])


def test_reader_revalidation_failure_durably_discards_lease(tmp_path, monkeypatch):
    board, generation = _published(tmp_path)
    from hermes_cli import kanban_generation as kg
    real_read_pointer = kg.read_pointer
    reads = 0

    def malformed_on_revalidation(path):
        nonlocal reads
        reads += 1
        if reads == 2:
            raise GenerationError("POINTER_INVALID")
        return real_read_pointer(path)

    monkeypatch.setattr(kg, "read_pointer", malformed_on_revalidation)
    with pytest.raises(GenerationError, match="LEASE_REVALIDATION_FAILED"):
        acquire_reader_lease(
            board, consumer="hermes-office", operation_id="malformed"
        )
    assert list((generation / "leases").glob("*.json")) == []


@pytest.mark.parametrize(
    "code",
    ["LEASE_CREATE_FAILED", "LEASE_DURABILITY_FAILED", "LEASE_REVALIDATION_FAILED"],
)
def test_reader_closed_lease_failures_are_retryable_after_cleanup(code):
    from hermes_cli.kanban_boards import _reader_is_retryable

    assert _reader_is_retryable(code) is True


def _published(tmp_path):
    board = tmp_path / "board"
    generation = board / "generations" / "g1"
    generation.mkdir(parents=True)
    db = generation / "kanban.db"
    initialize_generation_db(db, gid="g1", nonce="n", control_root=board)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT)")
        conn.commit()
    ready = build_ready(generation, gid="g1", nonce="n")
    fence = acquire_fence(board, generation="g1", owner="test")
    publish_prepared(generation, expected_manifest=ready["manifest_digest"], fence_token=fence["token"])
    activate_first_write(generation, fence_epoch=fence["epoch"])
    release_fence(board, token=fence["token"])
    return board, generation


def _prepared(tmp_path):
    board = tmp_path / "prepared-board"
    generation = board / "generations" / "g1"
    generation.mkdir(parents=True)
    db = generation / "kanban.db"
    initialize_generation_db(db, gid="g1", nonce="n", control_root=board)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT)")
    ready = build_ready(generation, gid="g1", nonce="n")
    return board, generation, ready


def test_gc_keeps_generation_with_malformed_lease(tmp_path):
    board, _generation = _published(tmp_path)
    old = board / "generations" / "old"
    old.mkdir()
    (old / "leases").mkdir()
    (old / "leases" / "bad.json").write_text("not-json")
    with pytest.raises(GenerationError, match="LEASE_INVALID"):
        gc_generations(board, dry_run=True)
    assert old.exists()


def test_reader_command_uses_length_prefixed_capability_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    legacy = tmp_path / "home" / "kanban.db"
    legacy.parent.mkdir(parents=True)
    sqlite3.connect(legacy).close()
    script = tmp_path / "child.py"
    script.write_text("import json, os, struct, sys; print('started', flush=True); assert 'HERMES_READER_ACK_FD' not in os.environ; fd=int(os.environ['HERMES_READER_BOOTSTRAP_FD']); h=os.read(fd,4); cap=json.loads(os.read(fd, struct.unpack('!I', h)[0])); assert cap['protocol_version']==1 and cap['start_token']; sys.stdin.buffer.read(); print('opened', flush=True)")
    from hermes_cli.kanban_boards import _cmd_with_reader_lease
    from argparse import Namespace
    result = _cmd_with_reader_lease(Namespace(protocol_version=1, operation_id="op", board="default", consumer="hermes-office", json_status_dir=str(tmp_path / "status"), command=[sys.executable, str(script)]))
    assert result == 0


def test_real_node_consumer_opens_authenticated_generation_over_v1(tmp_path, monkeypatch):
    board, generation = _published(tmp_path)
    monkeypatch.setattr("hermes_cli.kanban_boards.kb.board_dir", lambda _board: board)
    import shutil
    node = shutil.which("node")
    assert node, "managed Node runtime is required for the v1 qualification"
    proof = tmp_path / "node-proof.json"
    script = tmp_path / "consumer.cjs"
    script.write_text(
        "const fs=require('fs'); const {DatabaseSync}=require('node:sqlite');\n"
        "console.log('started');\n"
        "const fd=Number(process.env.HERMES_READER_BOOTSTRAP_FD);\n"
        "const h=Buffer.alloc(4); fs.readSync(fd,h,0,4,null);\n"
        "const n=h.readUInt32BE(0), p=Buffer.alloc(n); fs.readSync(fd,p,0,n,null);\n"
        "const cap=JSON.parse(p.toString('utf8'));\n"
        "if(cap.protocol_version!==1||cap.storage_mode!=='generation'||!cap.lease_id) process.exit(71);\n"
        "const db=new DatabaseSync(cap.database_path,{readOnly:true});\n"
        "const row=db.prepare('SELECT COUNT(*) AS n FROM tasks').get(); db.close();\n"
        "fs.writeFileSync(process.argv[2],JSON.stringify({generation_id:cap.generation_id,rows:Number(row.n)}));\n"
    )
    from argparse import Namespace
    from hermes_cli.kanban_boards import _cmd_with_reader_lease
    result = _cmd_with_reader_lease(Namespace(
        protocol_version=1, operation_id="node-v1", board="default",
        consumer="hermes-office", json_status_dir=str(tmp_path / "node-status"),
        command=[node, str(script), str(proof)],
    ))
    assert result == 0
    assert json.loads(proof.read_text()) == {"generation_id": "g1", "rows": 0}
    assert not list((generation / "leases").glob("*.json"))


def test_reader_session_uses_closed_newline_json_contract(tmp_path, monkeypatch, capsys):
    board, _generation = _published(tmp_path)
    monkeypatch.setattr("hermes_cli.kanban_boards.kb.board_dir", lambda _board: board)
    from argparse import Namespace
    from hermes_cli.kanban_boards import _cmd_reader_session
    class _EmptyInput:
        buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", _EmptyInput())

    result = _cmd_reader_session(Namespace(
        protocol_version=1, operation_id="op", board="default",
        consumer="hermes-office",
    ))
    assert result == 0
    lines = capsys.readouterr().out.splitlines()
    ready = json.loads(lines[0])
    assert set(ready) == {
        "protocol_version", "type", "operation_id", "board_slug", "generation_id",
        "generation_nonce", "database_path", "storage_mode", "ready_sha256",
        "activated_sha256", "lease_id", "lease_keeper_pid",
        "lease_keeper_start_fingerprint", "consumer_pid", "consumer_start_fingerprint",
        "selected_at",
    }
    released = json.loads(lines[-1])
    assert released == {"protocol_version": 1, "type": "released",
                        "lease_id": ready["lease_id"]}
    assert all(not line.startswith("{") or not line.endswith("}") or b"\x00" not in line.encode() for line in lines)


def test_reader_operations_close_missing_legacy_and_internal_errors(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    from hermes_cli import kanban_boards

    board = tmp_path / "missing"
    board_dir = board
    monkeypatch.setattr(kanban_boards.kb, "board_dir", lambda _board: board)
    monkeypatch.setattr(kanban_boards.kb, "kanban_db_path", lambda board=None: board_dir / "kanban.db")
    local = kanban_boards._cmd_reader_session(Namespace(
        protocol_version=1, operation_id="missing-local", board="default",
        consumer="hermes-office",
    ))
    assert local == 1
    assert json.loads(capsys.readouterr().out)["code"] == "BOARD_NOT_FOUND"
    bounded = kanban_boards._cmd_with_reader_lease(Namespace(
        protocol_version=1, operation_id="missing-child", board="default",
        consumer="hermes-office", json_status_dir=str(tmp_path / "status"),
        command=[sys.executable, "-c", "raise SystemExit(99)"],
    ))
    assert bounded == 1
    assert json.loads(capsys.readouterr().out)["code"] == "BOARD_NOT_FOUND"

    board.mkdir()
    (board / "generation.current").write_text("{}")
    monkeypatch.setattr(
        "hermes_cli.kanban_generation.acquire_reader_lease",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("internal path detail")),
    )
    local = kanban_boards._cmd_reader_session(Namespace(
        protocol_version=1, operation_id="error-local", board="default",
        consumer="hermes-office",
    ))
    assert local == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "LEASE_CREATE_FAILED"
    assert "internal path detail" not in payload["message"]


def test_legacy_reader_helper_remains_live_until_release(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    sqlite3.connect(home / "kanban.db").close()
    env = os.environ.copy()
    env.update({"HERMES_HOME": str(home), "HERMES_KANBAN_HOME": str(home)})
    env.pop("HERMES_KANBAN_DB", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.main", "kanban", "boards", "reader-session",
         "--board", "default", "--protocol-version", "1", "--consumer", "hermes-office",
         "--operation-id", "legacy-live", "--json-lines"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    assert proc.stdout is not None and proc.stdin is not None
    ready = json.loads(proc.stdout.readline())
    assert ready["type"] == "ready" and ready["lease_id"] is None
    assert proc.poll() is None
    proc.stdin.write(json.dumps({"protocol_version": 1, "op": "release", "lease_id": None}) + "\n")
    proc.stdin.flush()
    assert json.loads(proc.stdout.readline()) == {
        "protocol_version": 1, "type": "released", "lease_id": None,
    }
    assert proc.wait(timeout=10) == 0


@pytest.mark.parametrize(
    ("code", "retryable"),
    [("GENERATION_NOT_ACTIVATED", True), ("ACTIVATION_BUSY", True),
     ("CATALOG_BUSY", True), ("POINTER_INVALID", False),
     ("READY_INVALID", False), ("ACTIVATION_INVALID", False),
     ("GENERATION_META_INVALID", False)],
)
def test_reader_errors_have_closed_retry_semantics(tmp_path, monkeypatch, capsys, code, retryable):
    from argparse import Namespace
    from hermes_cli import kanban_boards
    from hermes_cli.kanban_generation import GenerationError

    board = tmp_path / "board"
    board.mkdir()
    (board / "generation.current").write_text("{}")
    monkeypatch.setattr(kanban_boards.kb, "board_dir", lambda _board: board)
    monkeypatch.setattr(
        "hermes_cli.kanban_generation.acquire_reader_lease",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(GenerationError(code)),
    )
    result = kanban_boards._cmd_reader_session(Namespace(
        protocol_version=1, operation_id=f"local-{code}", board="default",
        consumer="hermes-office",
    ))
    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == code and payload["retryable"] is retryable
    assert payload["message"] == code
    bounded = kanban_boards._cmd_with_reader_lease(Namespace(
        protocol_version=1, operation_id=f"bounded-{code}", board="default",
        consumer="hermes-office", json_status_dir=str(tmp_path / f"status-{code}"),
        command=[sys.executable, "-c", "raise SystemExit(99)"],
    ))
    assert bounded == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == code and payload["retryable"] is retryable
    assert payload["message"] == code


@pytest.mark.parametrize("consumer", ["gateway", "dashboard", "kanban-reader", "other"])
def test_reader_v1_rejects_non_office_consumers(tmp_path, monkeypatch, capsys, consumer):
    from argparse import Namespace
    from hermes_cli import kanban_boards

    monkeypatch.setattr(kanban_boards.kb, "board_dir", lambda _board: tmp_path)
    result = kanban_boards._cmd_reader_session(Namespace(
        protocol_version=1, operation_id="bad-consumer", board="default", consumer=consumer,
    ))
    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "LEASE_CREATE_FAILED"
    assert payload["retryable"] is False


def test_bounded_reader_exclusively_claims_status_directory(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    from hermes_cli import kanban_boards

    status = tmp_path / "status-race"
    original_mkdir = Path.mkdir

    def competing_mkdir(path, *args, **kwargs):
        if path == status:
            original_mkdir(path, *args, **kwargs)
            raise FileExistsError(path)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", competing_mkdir)
    result = kanban_boards._cmd_with_reader_lease(Namespace(
        protocol_version=1, operation_id="status-race", board="default",
        consumer="hermes-office", json_status_dir=str(status),
        command=[sys.executable, "-c", "raise SystemExit(99)"],
    ))
    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "STATUS_ID_CONFLICT"
    assert list(status.iterdir()) == []


def test_reader_status_short_write_never_publishes_partial_final(tmp_path, monkeypatch):
    from hermes_cli.kanban_boards import _publish_status
    status = tmp_path / "status"
    status.mkdir()
    real_write = os.write
    calls = 0

    def short_then_stop(fd, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, payload[:2])
        return 0

    monkeypatch.setattr(os, "write", short_then_stop)
    with pytest.raises(OSError, match="short status write"):
        _publish_status(status, "ready.json", {"type": "ready"})
    assert not (status / "ready.json").exists()
    assert not list(status.iterdir())


def test_imported_attachment_resolution_authenticates_final_target_id(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    _db(source, "imported")
    _db(target, "target")
    attachment = tmp_path / "source.txt"
    attachment.write_bytes(b"attachment")
    with sqlite3.connect(source) as conn:
        conn.execute("INSERT INTO task_attachments(task_id, filename, stored_path, size, created_at) VALUES ('imported', 'x', ?, 10, 1)", (str(attachment),))
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({"task_ids": ["imported"]}))
    generation = tmp_path / "generation"
    reconcile(source, target, seeds, generation, apply=True)
    with sqlite3.connect(generation / "kanban.db") as conn:
        resolution = conn.execute("SELECT source_row_sha256 FROM recovery_attachment_resolution").fetchone()[0]
        row = conn.execute("SELECT * FROM task_attachments WHERE task_id='imported'").fetchone()
        from hermes_cli.kanban_generation import manifest_digest
        columns = [d[1] for d in conn.execute("PRAGMA table_info(task_attachments)")]
        expected = manifest_digest(dict(zip(columns, row)))
    assert resolution == expected


@pytest.mark.parametrize(
    ("mode", "expected", "exit_code"),
    [
        ("handshake", "CHILD_HANDSHAKE_FAILED", 1),
        ("nonzero", "CHILD_FAILED", 1),
    ],
)
def test_reader_failure_matrix_finalizes_process_and_lease_once(
    tmp_path, monkeypatch, mode, expected, exit_code,
):
    board, generation = _published(tmp_path)
    monkeypatch.setattr("hermes_cli.kanban_boards.kb.board_dir", lambda _board: board)
    script = tmp_path / "reader.py"
    script.write_text(
        "import json, os, struct, sys, time\n"
        "mode = sys.argv[1]\n"
        "print('started' if mode != 'handshake' else 'wrong', flush=True)\n"
        "if mode == 'handshake': time.sleep(30)\n"
        "else:\n"
        "  capfd = int(os.environ['HERMES_READER_BOOTSTRAP_FD']); h=os.read(capfd,4); cap=json.loads(os.read(capfd, struct.unpack('!I', h)[0]))\n"
        "  assert set(cap) == {'protocol_version','operation_id','board_slug','storage_mode','database_path','generation_id','generation_nonce','ready_sha256','activated_sha256','lease_id','start_token'}\n"
        "  if mode == 'nonzero': sys.stdin.buffer.read(); sys.exit(7)\n"
        "  sys.stdin.buffer.read()\n",
    )
    status = tmp_path / "status"
    from argparse import Namespace
    result = __import__("hermes_cli.kanban_boards", fromlist=["_cmd_with_reader_lease"])._cmd_with_reader_lease(
        Namespace(protocol_version=1, operation_id=f"op-{mode}", board="default",
                  consumer="hermes-office", json_status_dir=str(status),
                  command=[sys.executable, str(script), mode]),
    )
    assert result == exit_code
    terminal = json.loads((status / "terminal.json").read_text())
    assert terminal["type"] == "child_failed"
    assert terminal["code"] == expected
    assert len(list((generation / "leases").glob("*.json"))) == 0


def test_reader_silent_handshake_is_deadline_bounded_and_finalized(tmp_path, monkeypatch):
    board, generation = _published(tmp_path)
    monkeypatch.setattr("hermes_cli.kanban_boards.kb.board_dir", lambda _board: board)
    monkeypatch.setattr("hermes_cli.kanban_boards._READER_HANDSHAKE_TIMEOUT_SECONDS", 0.1)
    script = tmp_path / "silent-reader.py"
    script.write_text("import time; time.sleep(30)\n")
    status = tmp_path / "status"
    from argparse import Namespace
    result = __import__("hermes_cli.kanban_boards", fromlist=["_cmd_with_reader_lease"])._cmd_with_reader_lease(
        Namespace(protocol_version=1, operation_id="op-silent", board="default",
                  consumer="hermes-office", json_status_dir=str(status),
                  command=[sys.executable, str(script)]),
    )
    assert result == 1
    terminal = json.loads((status / "terminal.json").read_text())
    assert terminal["type"] == "child_failed"
    assert terminal["code"] == "CHILD_HANDSHAKE_FAILED"
    assert not list((generation / "leases").glob("*.json"))


def test_failed_cas_cleanup_keeps_a_blob_adopted_by_public_producer(tmp_path, monkeypatch):
    _board, generation = _published(tmp_path)
    db = generation / "kanban.db"
    relative = Path("blobs/sha256/aa/adopted")
    blob = generation / relative
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"adopted")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE task_attachments (id INTEGER PRIMARY KEY, stored_path TEXT)")
        conn.execute("INSERT INTO task_attachments(stored_path) VALUES (?)", (str(relative),))
        conn.commit()
        monkeypatch.setattr(
            "hermes_cli.kanban_db_connect._main_db_file", lambda _conn: str(db),
        )
        discard_created_blobs(conn, [blob])
    assert blob.exists()


def test_failed_cas_cleanup_removes_only_owned_orphan(tmp_path, monkeypatch):
    _board, generation = _published(tmp_path)
    db = generation / "kanban.db"
    relative = Path("blobs/sha256/bb/orphan")
    blob = generation / relative
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"orphan")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE task_attachments (id INTEGER PRIMARY KEY, stored_path TEXT)")
        conn.commit()
        monkeypatch.setattr(
            "hermes_cli.kanban_db_connect._main_db_file", lambda _conn: str(db),
        )
        discard_created_blobs(conn, [blob])
    assert not blob.exists()


def _public_attachment_producer(db_path, entered, release):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    import hermes_cli.kanban_generation as generation
    original = generation.install_content_addressed_blob_owned

    def held_install(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result

    generation.install_content_addressed_blob_owned = held_install
    conn = kbc.connect(db_path=db_path)
    try:
        kb.store_attachment_bytes(conn, "producer-task", "adopted.txt", b"adopted")
    finally:
        conn.close()


def _cleanup_created_blob(db_path, blob, finished):
    from hermes_cli.kanban_generation import discard_created_blobs
    conn = sqlite3.connect(db_path)
    try:
        discard_created_blobs(conn, [Path(blob)])
    finally:
        conn.close()
        finished.set()


def test_public_producer_adoption_blocks_cleanup_and_preserves_blob(tmp_path):
    board = tmp_path / "board"
    generation = board / "generations" / "g1"
    generation.mkdir(parents=True)
    db = generation / "kanban.db"
    initialize_generation_db(db, gid="g1", nonce="n", control_root=board)
    with sqlite3.connect(db) as conn:
        from hermes_cli.kanban_db import SCHEMA_SQL
        conn.executescript(SCHEMA_SQL)
        conn.execute("ALTER TABLE tasks ADD COLUMN completion_contract TEXT")
        conn.execute(
            "INSERT INTO tasks(id, title, status, created_at, workspace_kind, block_kind, block_recurrences) "
            "VALUES ('producer-task', 'producer', 'blocked', 1, 'scratch', 'incident', 0)"
        )
        conn.commit()
    ready = build_ready(generation, gid="g1", nonce="n")
    fence = acquire_fence(board, generation="g1", owner="publisher")
    publish_prepared(generation, expected_manifest=ready["manifest_digest"], fence_token=fence["token"])
    release_fence(board, token=fence["token"])
    digest = __import__("hashlib").sha256(b"adopted").hexdigest()
    blob = generation / "blobs" / "sha256" / digest[:2] / digest
    ctx = multiprocessing.get_context("fork")
    entered, release, finished = ctx.Event(), ctx.Event(), ctx.Event()
    producer = ctx.Process(target=_public_attachment_producer, args=(str(db), entered, release))
    producer.start()
    assert entered.wait(10)
    cleaner = ctx.Process(target=_cleanup_created_blob, args=(str(db), str(blob), finished))
    cleaner.start()
    assert not finished.wait(0.25)
    release.set()
    producer.join(10)
    cleaner.join(10)
    assert producer.exitcode == 0
    assert cleaner.exitcode == 0
    assert blob.exists()


def test_reader_session_uses_golden_newline_json_and_exact_eof_terminal(monkeypatch, tmp_path, capsys):
    from hermes_cli import kanban_boards

    class _BinaryInput:
        def __init__(self, data):
            self.buffer = io.BytesIO(data)

    class _BinaryOutput:
        def __init__(self):
            self.buffer = io.BytesIO()

    output = _BinaryOutput()
    monkeypatch.setattr(sys, "stdin", _BinaryInput(b""))
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(kanban_boards.kb, "board_dir", lambda board=None: tmp_path / "board")
    monkeypatch.setattr(kanban_boards.kb, "kanban_db_path", lambda board=None: tmp_path / "board" / "kanban.db")
    monkeypatch.setattr(kanban_boards, "_reader_timestamp", lambda: "2026-11-17T00:00:00Z")
    (tmp_path / "board").mkdir()
    (tmp_path / "board" / "kanban.db").touch()
    result = kanban_boards._cmd_reader_session(__import__("argparse").Namespace(
        protocol_version=1, consumer="hermes-office", operation_id="op", board="default"))
    assert result == 0
    raw = output.buffer.getvalue()
    expected = (json.dumps({
        "protocol_version": 1, "type": "ready", "operation_id": "op", "board_slug": "default",
        "database_path": str(tmp_path / "board" / "kanban.db"), "storage_mode": "legacy",
        "generation_id": None, "generation_nonce": None, "ready_sha256": None,
        "activated_sha256": None, "lease_id": None, "lease_keeper_pid": None,
        "lease_keeper_start_fingerprint": None, "consumer_pid": None,
        "consumer_start_fingerprint": None, "selected_at": "2026-11-17T00:00:00Z",
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" +
               json.dumps({"protocol_version": 1, "type": "released", "lease_id": None},
                          sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert raw == expected
    assert [json.loads(line) for line in raw.splitlines()] == [json.loads(line) for line in expected.splitlines()]


def test_reader_session_distinguishes_wrong_well_formed_lease_id(tmp_path, monkeypatch, capsys):
    board, generation = _published(tmp_path)
    monkeypatch.setattr("hermes_cli.kanban_boards.kb.board_dir", lambda _board: board)
    class _Input:
        buffer = io.BytesIO(b'{"protocol_version":1,"op":"release","lease_id":"wrong"}\n')
    monkeypatch.setattr(sys, "stdin", _Input())
    from argparse import Namespace
    result = __import__("hermes_cli.kanban_boards", fromlist=["_cmd_reader_session"])._cmd_reader_session(
        Namespace(protocol_version=1, operation_id="op-mismatch", board="default", consumer="hermes-office"))
    assert result == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[-1]["code"] == "LEASE_ID_MISMATCH"
    assert lines[-1]["generation_id"] == lines[0]["generation_id"]
    assert (generation / "leases" / f"{lines[0]['lease_id']}.json").is_file()


def test_reconcile_clears_every_runtime_identity_and_rejects_unknown_schema(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    _db(source, "runtime")
    _db(target, "target")
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE tasks SET claim_lock='lock', worker_pid=99, session_id='s', last_heartbeat_at=7")
        conn.execute("UPDATE task_runs SET claim_lock='run-lock', worker_pid=99, last_heartbeat_at=8")
    seed = tmp_path / "seed.json"
    seed.write_text('["runtime"]')
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE unknown_persistent (id INTEGER)")
    with pytest.raises(GenerationError, match="unsupported persistent table"):
        reconcile(source, target, seed, tmp_path / "generation", apply=True)


def test_fence_custody_blocks_second_process_until_release(tmp_path):
    board = tmp_path / "board"
    board.mkdir()
    first = acquire_fence(board, generation="g1", owner="first")
    entered = board / "second-entered"
    probe = subprocess.Popen(
        [sys.executable, "-c", "from pathlib import Path; import sys; from hermes_cli.kanban_generation import acquire_fence, release_fence; board=Path(sys.argv[1]); fence=acquire_fence(board, generation='g2', owner='second'); Path(sys.argv[2]).write_text('entered'); release_fence(board, token=fence['token'])", str(board), str(entered)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    from tests.waiting import wait_until
    with pytest.raises(AssertionError):
        wait_until(lambda: entered.exists(), timeout=2.0, what="fence waiter before release")
    release_fence(board, token=first["token"])
    assert probe.wait(timeout=10) == 0, probe.stderr.read()
    assert entered.exists()


def test_apply_reconcile_uses_one_snapshot_pair(tmp_path, monkeypatch):
    from hermes_cli import kanban_reconcile
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    _db(source, "source")
    _db(target, "target")
    seed = tmp_path / "seed.json"
    seed.write_text('["source"]')
    calls = []
    original = kanban_reconcile._snapshot_db
    monkeypatch.setattr(kanban_reconcile, "_snapshot_db", lambda *args: (calls.append(args), original(*args))[1])
    kanban_reconcile.reconcile(source, target, seed, tmp_path / "generation", apply=True)
    assert len(calls) == 2


def test_canonical_native_schema_accepts_all_migrated_indexes(tmp_path, monkeypatch):
    disposable_home = tmp_path / "home"
    disposable_board = tmp_path / "board"
    monkeypatch.setenv("HERMES_HOME", str(disposable_home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(disposable_home))
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    from hermes_cli.kanban_db_connect import init_db
    init_db(source)
    init_db(target)
    with sqlite3.connect(source) as conn:
        conn.execute("INSERT INTO tasks(id, title, status, created_at) VALUES ('source', 'source', 'blocked', 1)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status ON tasks(assignee, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON task_events(run_id, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_idempotency ON tasks(idempotency_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant)")
    seed = tmp_path / "seed.json"
    seed.write_text('["source"]')
    result = reconcile(source, target, seed, disposable_board / "generation", apply=False)
    assert result["dry_run"] is True
    assert not (disposable_home / "kanban.db").exists()


def test_generated_connection_holds_and_releases_reader_lease(tmp_path, monkeypatch):
    _board, generation = _published(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.kanban_db.board_dir", lambda board=None: board and Path(board) or tmp_path / "missing")
    from hermes_cli.kanban_db_connect import connect
    conn = connect(db_path=generation / "kanban.db")
    try:
        leases = list((generation / "leases").glob("*.json"))
        assert leases
    finally:
        conn.close()
    assert not list((generation / "leases").glob("*.json"))


def test_reconcile_rejects_wrong_definition_for_migrated_index(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    from hermes_cli.kanban_db_connect import init_db
    init_db(source)
    init_db(target)
    with sqlite3.connect(source) as conn:
        conn.execute("INSERT INTO tasks(id, title, status, created_at) VALUES ('source', 'source', 'blocked', 1)")
        conn.execute("DROP INDEX IF EXISTS idx_tasks_tenant")
        conn.execute("CREATE INDEX idx_tasks_tenant ON tasks(title)")
    seed = tmp_path / "seed.json"
    seed.write_text('["source"]')
    with pytest.raises(GenerationError, match="canonical schema mismatch: index idx_tasks_tenant"):
        reconcile(source, target, seed, tmp_path / "generation", apply=False)


def test_reconcile_rejects_skeletal_schema(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    _db(source, "source")
    _db(target, "target")
    seed = tmp_path / "seed.json"
    seed.write_text('["source"]')
    with sqlite3.connect(source) as conn:
        conn.execute("DROP INDEX idx_tasks_tenant")
    with pytest.raises(GenerationError, match="canonical schema mismatch"):
        reconcile(source, target, seed, tmp_path / "generation", apply=False)


def test_publication_recovery_finishes_acceptance_before_pointer(tmp_path):
    board, generation, ready = _prepared(tmp_path)
    fence = acquire_fence(board, generation="g1", owner="publisher")
    publish_prepared(generation, expected_manifest=ready["manifest_digest"], fence_token=fence["token"])
    (board / "generation.current").unlink()
    from hermes_cli.kanban_generation import recover_publication
    recover_publication(board)
    assert read_pointer(board)["generation_id"] == "g1"


def test_activation_clean_failure_is_retryable_without_poison(tmp_path):
    board, generation, _ready = _prepared(tmp_path)
    fence = acquire_fence(board, generation="g1", owner="writer")
    with pytest.raises(OSError):
        activate_first_write(generation, fence_epoch=fence["epoch"], durability=DurabilityAdapter(failpoint="activation-before-rename"))
    assert not (generation / "ACTIVATION.poison.json").exists()
    assert activate_first_write(generation, fence_epoch=fence["epoch"])["generation_id"] == "g1"


def test_public_add_attachment_refuses_uncustodied_generated_metadata(tmp_path):
    _board, generation = _published(tmp_path)
    from hermes_cli.kanban_db import add_attachment
    conn = sqlite3.connect(generation / "kanban.db")
    with pytest.raises(ValueError, match="store_attachment_bytes"):
        add_attachment(conn, "t1", filename="x.txt", stored_path=str(tmp_path / "x.txt"))
    conn.close()
