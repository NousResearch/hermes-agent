import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli.kanban_generation import resolve_attachment_blob


def test_dashboard_style_attachment_read_uses_generation_custody(tmp_path):
    from tests.hermes_cli.test_kanban_recovery_round_trip import _db
    from hermes_cli.kanban_reconcile import reconcile

    source, target = tmp_path / "source.db", tmp_path / "target.db"
    old = tmp_path / "legacy.txt"
    old.write_bytes(b"dashboard custody")
    _db(source, "source-task")
    _db(target, "target-task", old)
    seed = tmp_path / "seed.json"
    seed.write_text('["source-task"]')
    generation = tmp_path / "board" / "generations" / "g1"
    generation.parent.mkdir(parents=True)
    reconcile(source, target, seed, generation, apply=True)
    old.unlink()
    with sqlite3.connect(generation / "kanban.db") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM task_attachments WHERE id=10").fetchone()
        resolved = resolve_attachment_blob(conn, row, board=generation)
        assert resolved.read_bytes() == b"dashboard custody"
        assert resolved != old


def _published_attachment_board(root: Path):
    from tests.hermes_cli.test_kanban_recovery_round_trip import _db
    from hermes_cli.kanban_generation import (
        acquire_fence, activate_first_write, publish_prepared, release_fence,
    )
    from hermes_cli.kanban_reconcile import reconcile

    root.mkdir(parents=True)
    home = root / "home"
    source, target = root / "source.db", root / "target.db"
    source_blob = root / "imported.txt"
    source_blob.write_bytes(b"imported custody")
    _db(source, "imported", source_blob)
    _db(target, "target")
    seed = root / "seed.json"
    seed.write_text(json.dumps(["imported"]))
    board = home / "kanban" / "boards" / "default"
    generation = board / "generations" / "g1"
    generation.parent.mkdir(parents=True)
    report = reconcile(source, target, seed, generation, apply=True)
    fence = acquire_fence(board, generation=report["generation_id"], owner="reader-test")
    publish_prepared(generation, expected_manifest=report["manifest_digest"], fence_token=fence["token"])
    activate_first_write(generation, fence_epoch=fence["epoch"])
    release_fence(board, token=fence["token"])
    with sqlite3.connect(generation / "kanban.db") as conn:
        attachment_id, custody_relpath = conn.execute(
            "SELECT attachment_id, custody_relpath FROM recovery_attachment_resolution"
        ).fetchone()
    source_blob.unlink()
    return home, generation, int(attachment_id), generation / custody_relpath, source_blob


def _attachment_cli(home: Path):
    env = os.environ.copy()
    env.update(HERMES_HOME=str(home))
    env.pop("HERMES_KANBAN_DB", None)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", "attachments", "imported", "--json"],
        env=env, capture_output=True, text=True, check=False,
    )


def test_imported_attachment_public_readers_use_production_selection_and_fail_closed(tmp_path, monkeypatch):
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_db import build_worker_context
    from tools.kanban_tools import _handle_attachments
    from plugins.kanban.dashboard.plugin_api import (
        download_attachment, get_task, list_task_attachments,
    )

    home, generation, attachment_id, _blob, _source_blob = _published_attachment_board(tmp_path / "clean")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)

    cli = _attachment_cli(home)
    assert cli.returncode == 0, cli.stderr
    assert Path(json.loads(cli.stdout)[0]["stored_path"]).read_bytes() == b"imported custody"
    tool = json.loads(_handle_attachments({"task_id": "imported", "board": "default"}))
    assert Path(tool["attachments"][0]["stored_path"]).read_bytes() == b"imported custody"
    listed = list_task_attachments("imported", board="default")
    assert Path(listed["attachments"][0]["stored_path"]).read_bytes() == b"imported custody"
    task = get_task("imported", board="default", run_state_type=None, run_state_name=None)
    assert Path(task["attachments"][0]["stored_path"]).read_bytes() == b"imported custody"
    downloaded = download_attachment(attachment_id, board="default")
    assert Path(downloaded.path).read_bytes() == b"imported custody"
    with kbc.connect_closing(board="default") as conn:
        context = build_worker_context(conn, "imported")
    assert "a.txt" in context and "imported custody" not in context
    assert str(generation) in cli.stdout

    for tamper in ("row", "blob", "generation"):
        bad_home, bad_generation, bad_attachment_id, bad_blob, bad_source = _published_attachment_board(
            tmp_path / tamper
        )
        if tamper == "row":
            with sqlite3.connect(bad_generation / "kanban.db") as conn:
                conn.execute("UPDATE task_attachments SET size = size + 1 WHERE task_id='imported'")
        elif tamper == "blob":
            bad_blob.write_bytes(b"tampered custody")
        else:
            with sqlite3.connect(bad_generation / "kanban.db") as conn:
                conn.execute(
                    "UPDATE recovery_attachment_resolution SET generation_id='other-generation'"
                )
        monkeypatch.setenv("HERMES_HOME", str(bad_home))

        cli = _attachment_cli(bad_home)
        assert cli.returncode != 0
        assert str(bad_source) not in cli.stdout + cli.stderr
        tool = _handle_attachments({"task_id": "imported", "board": "default"})
        assert str(bad_source) not in tool
        tool_payload = json.loads(tool)
        assert "attachments" not in tool_payload
        assert tool_payload.get("error")
        with pytest.raises(Exception):
            list_task_attachments("imported", board="default")
        with pytest.raises(Exception):
            get_task("imported", board="default", run_state_type=None, run_state_name=None)
        with pytest.raises(Exception):
            download_attachment(bad_attachment_id, board="default")
        with pytest.raises(Exception):
            with kbc.connect_closing(board="default") as conn:
                build_worker_context(conn, "imported")
        assert not bad_source.exists()


def test_deleted_dynamic_attachment_blob_is_collected_after_last_reference_and_lease(tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_generation import (
        acquire_reader_lease, gc_generations, release_reader_lease,
    )

    _home, generation, _attachment_id, baseline_blob, _source = (
        _published_attachment_board(tmp_path / "delete-gc")
    )
    board = generation.parents[1]
    with kbc.connect_closing(db_path=generation / "kanban.db") as conn:
        first = kb.store_attachment_bytes(
            conn, "imported", "dynamic-a.txt", b"shared dynamic"
        )
        second = kb.store_attachment_bytes(
            conn, "imported", "dynamic-b.txt", b"shared dynamic"
        )
        dynamic_relpath = conn.execute(
            "SELECT custody_relpath FROM recovery_attachment_resolution "
            "WHERE attachment_id = ?", (first,)
        ).fetchone()[0]
        dynamic_blob = generation / dynamic_relpath
        kb.delete_attachment(conn, first)
        assert dynamic_blob.is_file()
        gc_generations(board, grace_seconds=0)
        assert dynamic_blob.is_file()
        lease = acquire_reader_lease(
            board, consumer="test-reader", operation_id="old-snapshot"
        )
        kb.delete_attachment(conn, second)
    old = 1
    os.utime(dynamic_blob, (old, old))
    gc_generations(board, grace_seconds=0)
    assert dynamic_blob.is_file()
    release_reader_lease(generation, lease["lease_id"])
    gc_generations(board, grace_seconds=0)
    assert not dynamic_blob.exists()
    assert baseline_blob.is_file()
    with sqlite3.connect(generation / "kanban.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM recovery_attachment_resolution "
            "WHERE attachment_id IN (?, ?)", (first, second)
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    "tamper", ["missing-resolution", "row", "digest", "size", "missing-blob", "symlink", "escape"]
)
def test_generated_attachment_delete_refuses_invalid_custody_without_mutation(tmp_path, tamper):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_generation import GenerationError

    _home, generation, attachment_id, blob, source = _published_attachment_board(
        tmp_path / tamper
    )
    with kbc.connect_closing(db_path=generation / "kanban.db") as conn:
        match tamper:
            case "missing-resolution":
                conn.execute(
                    "DELETE FROM recovery_attachment_resolution WHERE attachment_id = ?",
                    (attachment_id,),
                )
            case "row":
                conn.execute(
                    "UPDATE task_attachments SET size = size + 1 WHERE id = ?",
                    (attachment_id,),
                )
            case "digest":
                conn.execute(
                    "UPDATE recovery_attachment_resolution SET blob_sha256 = ? WHERE attachment_id = ?",
                    ("f" * 64, attachment_id),
                )
            case "size":
                conn.execute(
                    "UPDATE recovery_attachment_resolution SET size = size + 1 WHERE attachment_id = ?",
                    (attachment_id,),
                )
            case "missing-blob":
                blob.unlink()
            case "symlink":
                blob.unlink()
                blob.symlink_to(source)
            case "escape":
                conn.execute(
                    "UPDATE recovery_attachment_resolution SET custody_relpath = '../escape' WHERE attachment_id = ?",
                    (attachment_id,),
                )
        conn.commit()
        before_attachment = conn.execute(
            "SELECT COUNT(*) FROM task_attachments WHERE id = ?", (attachment_id,)
        ).fetchone()[0]
        before_resolution = conn.execute(
            "SELECT COUNT(*) FROM recovery_attachment_resolution WHERE attachment_id = ?",
            (attachment_id,),
        ).fetchone()[0]
        with pytest.raises(GenerationError):
            kb.delete_attachment(conn, attachment_id)
        assert conn.execute(
            "SELECT COUNT(*) FROM task_attachments WHERE id = ?", (attachment_id,)
        ).fetchone()[0] == before_attachment
        assert conn.execute(
            "SELECT COUNT(*) FROM recovery_attachment_resolution WHERE attachment_id = ?",
            (attachment_id,),
        ).fetchone()[0] == before_resolution
