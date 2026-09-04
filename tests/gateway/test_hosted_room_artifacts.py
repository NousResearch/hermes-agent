"""Security and durability tests for Bot-published Group Chat files."""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_artifacts as artifacts
from gateway.hosted_room_artifacts import (
    ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS,
    GENERATION_FENCE_RETENTION_SECONDS,
    UNACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS,
    RoomArtifactError,
    RoomArtifactOutbox,
    RoomArtifactScope,
    bind_room_artifact_scope,
    reset_room_artifact_scope,
    terminal_artifact_manifest,
    validate_terminal_artifact_manifest,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import hosted_room_artifact as room_artifact_tool
from tools.hosted_room_artifact import (
    _read_backend_file_bytes_nofollow,
    ensure_share_group_file_tool,
    share_group_file,
)
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


def _scope(**overrides) -> RoomArtifactScope:
    value = {
        "room_id": "room-1",
        "task_id": "dtask:abc",
        "execution_generation": 1,
        "member_id": "member-build",
        "target_profile": "build",
        "home_install_id": "install-home",
        "target_install_id": "install-target",
        "authority_gateway_id": "gateway-home",
        "authority_epoch": 1,
    }
    value.update(overrides)
    return RoomArtifactScope.from_mapping(value)


def _ack(outbox, scope, artifact_id):
    return outbox.acknowledge(
        scope,
        [artifact_id],
        message_event_id=f"dmessage:{scope.task_id.removeprefix('dtask:')}",
    )


def _insert_legacy_artifact(
    conn: sqlite3.Connection,
    scope: RoomArtifactScope,
    *,
    suffix: str,
) -> None:
    conn.execute(
        """INSERT INTO hosted_room_output_artifacts
           (artifact_id, scope_key, scope_json, name, kind, mime, size,
            sha256, blob_name, created_at, acknowledged_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
        (
            f"rart_legacy_{suffix}",
            scope.key,
            json.dumps(scope.as_mapping(), sort_keys=True, separators=(",", ":")),
            f"legacy-{suffix}.bin",
            "file",
            "application/octet-stream",
            1,
            suffix.ljust(64, "0")[:64],
            f"blob_legacy_{suffix}",
            time.time(),
        ),
    )


def test_outbox_is_idempotent_scoped_and_acknowledged(tmp_path: Path):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("# Handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    scope = _scope()

    first = outbox.put_path(scope=scope, path=path)
    replay = outbox.put_path(scope=scope, path=path)
    assert replay == first
    metadata, data = outbox.read(scope, first["artifact_id"])
    assert metadata == first
    assert data == b"# Handoff\n"
    with pytest.raises(RoomArtifactError, match="not found"):
        outbox.read(_scope(task_id="dtask:other"), first["artifact_id"])

    manifest = terminal_artifact_manifest(db, scope)
    assert validate_terminal_artifact_manifest(manifest) == [first]
    assert _ack(outbox, scope, first["artifact_id"]) == 1
    assert _ack(outbox, scope, first["artifact_id"]) == 0
    with pytest.raises(RoomArtifactError, match="not found"):
        outbox.read(scope, first["artifact_id"])


def test_named_profiles_share_one_gateway_artifact_quota(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    default_db = home / "state.db"
    profile_db = home / "profiles" / "writer" / "state.db"
    profile_db.parent.mkdir(parents=True)
    first_path = tmp_path / "first.md"
    second_path = tmp_path / "second.md"
    first_path.write_text("first payload\n", encoding="utf-8")
    second_path.write_text("second payload\n", encoding="utf-8")
    default = RoomArtifactOutbox(default_db)
    profile = RoomArtifactOutbox(profile_db)
    assert profile.db_path == default_db
    first = default.put_path(scope=_scope(), path=first_path)
    monkeypatch.setattr(artifacts, "MAX_GATEWAY_BLOB_BYTES", first["size"])
    with pytest.raises(RoomArtifactError, match="gateway room artifact quota"):
        profile.put_path(
            scope=_scope(task_id="dtask:profile", target_profile="writer"),
            path=second_path,
        )


def test_repeated_acknowledge_uses_receipt_without_reunlinking(
    tmp_path: Path,
    monkeypatch,
):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    scope = _scope()
    stored = outbox.put_path(scope=scope, path=path)

    assert _ack(outbox, scope, stored["artifact_id"]) == 1
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            """SELECT acknowledged_at, blob_reclaimed_at
                 FROM hosted_room_output_artifacts
                WHERE artifact_id=?""",
            (stored["artifact_id"],),
        ).fetchone()
    assert row is not None and row[0] is not None and row[1] is not None

    unlink_calls = []
    original_unlink = Path.unlink

    def tracked_unlink(candidate, *args, **kwargs):
        unlink_calls.append(candidate)
        return original_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", tracked_unlink)
    assert _ack(outbox, scope, stored["artifact_id"]) == 0
    assert unlink_calls == []


def test_acknowledged_receipts_prune_in_indexed_batches(
    tmp_path: Path,
    monkeypatch,
):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    monkeypatch.setattr(artifacts, "ACKNOWLEDGED_ARTIFACT_PRUNE_BATCH", 2)

    for index in range(3):
        scope = _scope(task_id=f"dtask:{index}")
        stored = outbox.put_path(scope=scope, path=path)
        assert _ack(outbox, scope, stored["artifact_id"]) == 1
    with sqlite3.connect(db) as conn:
        conn.execute(
            """UPDATE hosted_room_output_artifacts
                  SET acknowledged_at=0, receipt_expires_at=0,
                      blob_reclaimed_at=1"""
        )
        conn.commit()
        indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(hosted_room_output_artifacts)")
        }
        expiry_plan = " ".join(
            str(row[3])
            for row in conn.execute(
                """EXPLAIN QUERY PLAN
                   SELECT artifact_id, blob_name, blob_reclaimed_at
                     FROM hosted_room_output_artifacts
                    WHERE acknowledged_at IS NOT NULL
                      AND acknowledged_at<=?
                    ORDER BY acknowledged_at, artifact_id
                    LIMIT ?""",
                (1, 2),
            )
        )
        cleanup_plan = " ".join(
            str(row[3])
            for row in conn.execute(
                """EXPLAIN QUERY PLAN
                   SELECT artifact_id, blob_name
                     FROM hosted_room_output_artifacts
                    WHERE acknowledged_at IS NOT NULL
                      AND blob_reclaimed_at IS NULL
                    ORDER BY acknowledged_at, artifact_id
                    LIMIT ?""",
                (2,),
            )
        )
    assert "idx_hosted_room_output_ack_expiry" in indexes
    assert "idx_hosted_room_output_ack_cleanup" in indexes
    assert "idx_hosted_room_output_ack_expiry" in expiry_plan
    assert "idx_hosted_room_output_ack_cleanup" in cleanup_plan

    now = ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS + 1
    assert outbox.prune_acknowledged_receipts(now=now) == 2
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM hosted_room_output_artifacts"
            ).fetchone()[0]
            == 1
        )
    assert outbox.prune_acknowledged_receipts(now=now) == 1
    assert outbox.prune_acknowledged_receipts(now=now) == 0


def test_abandoned_unacknowledged_output_expires_without_leaking_blob(tmp_path: Path):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("private handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    scope = _scope()
    stored = outbox.put_path(scope=scope, path=path)
    with sqlite3.connect(db) as conn:
        blob_name = conn.execute(
            """SELECT blob_name FROM hosted_room_output_artifacts
                WHERE artifact_id=?""",
            (stored["artifact_id"],),
        ).fetchone()[0]
        conn.execute(
            """UPDATE hosted_room_output_artifacts SET created_at=0
                WHERE artifact_id=?""",
            (stored["artifact_id"],),
        )
    blob = outbox.blob_root / str(blob_name)
    assert blob.is_file()
    assert outbox.prune_unacknowledged_artifacts(
        now=UNACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS + 1
    ) == 1
    assert not blob.exists()
    assert outbox.list(scope) == []
    assert _ack(outbox, scope, stored["artifact_id"]) == 0


def test_failed_cleanup_retries_from_durable_scope_on_restart(
    tmp_path: Path, monkeypatch
):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("private handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    scope = _scope()
    outbox.put_path(scope=scope, path=path)
    original = RoomArtifactOutbox.discard
    attempts = 0

    def fail_once(instance, artifact_scope):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary cleanup fault")
        return original(instance, artifact_scope)

    monkeypatch.setattr(RoomArtifactOutbox, "discard", fail_once)
    with pytest.raises(OSError, match="temporary cleanup fault"):
        outbox.discard_durably(scope)
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            """SELECT cleanup_required_at FROM hosted_room_output_artifacts
                WHERE scope_key=?""",
            (scope.key,),
        ).fetchone()[0] is not None

    recovered = RoomArtifactOutbox(db)
    assert attempts == 2
    assert recovered.list(scope) == []


def test_unlink_failure_keeps_cleanup_obligation_until_restart(
    tmp_path: Path, monkeypatch
):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("private handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    scope = _scope()
    stored = outbox.put_path(scope=scope, path=path)
    original_unlink = Path.unlink
    attempts = 0

    def fail_once(candidate, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary unlink fault")
        return original_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    with pytest.raises(OSError, match="temporary unlink fault"):
        outbox.discard_durably(scope)
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            """SELECT cleanup_required_at FROM hosted_room_output_artifacts
                WHERE artifact_id=?""",
            (stored["artifact_id"],),
        ).fetchone()
    assert row is not None and row[0] is not None

    recovered = RoomArtifactOutbox(db)
    assert attempts == 2
    assert recovered.list(scope) == []


def test_constructor_does_not_unlink_historical_ack_receipts(
    tmp_path: Path,
    monkeypatch,
):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    for index in range(10):
        scope = _scope(task_id=f"dtask:{index}")
        stored = outbox.put_path(scope=scope, path=path)
        assert _ack(outbox, scope, stored["artifact_id"]) == 1

    unlink_calls = []
    original_unlink = Path.unlink

    def tracked_unlink(candidate, *args, **kwargs):
        unlink_calls.append(candidate)
        return original_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", tracked_unlink)
    RoomArtifactOutbox(db)
    assert unlink_calls == []


def test_constructor_reclaims_blob_after_ack_commit_crash(tmp_path: Path):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    scope = _scope()
    stored = outbox.put_path(scope=scope, path=path)
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            """SELECT blob_name FROM hosted_room_output_artifacts
                WHERE artifact_id=?""",
            (stored["artifact_id"],),
        ).fetchone()
        assert row is not None
        blob = outbox.blob_root / str(row[0])
        conn.execute(
            """UPDATE hosted_room_output_artifacts
                  SET acknowledged_at=?, blob_reclaimed_at=NULL
                WHERE artifact_id=?""",
            (time.time(), stored["artifact_id"]),
        )
        conn.commit()
    assert blob.is_file()

    recovered = RoomArtifactOutbox(db)
    assert not blob.exists()
    with sqlite3.connect(db) as conn:
        reclaimed_at = conn.execute(
            """SELECT blob_reclaimed_at FROM hosted_room_output_artifacts
                WHERE artifact_id=?""",
            (stored["artifact_id"],),
        ).fetchone()[0]
    assert reclaimed_at is not None
    assert _ack(recovered, scope, stored["artifact_id"]) == 0


def test_retirement_retries_acknowledged_blob_unlink_before_metadata_delete(
    tmp_path: Path,
    monkeypatch,
):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    scope = _scope()
    stored = outbox.put_path(scope=scope, path=path)
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT blob_name FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (stored["artifact_id"],),
        ).fetchone()
        blob = outbox.blob_root / str(row[0])
        conn.execute(
            """UPDATE hosted_room_output_artifacts
                  SET acknowledged_at=?, blob_reclaimed_at=NULL
                WHERE artifact_id=?""",
            (time.time(), stored["artifact_id"]),
        )
        conn.commit()
    original_unlink = Path.unlink
    fail_once = [True]

    def flaky_unlink(candidate, *args, **kwargs):
        if candidate == blob and fail_once[0]:
            fail_once[0] = False
            raise OSError("unlink failed")
        return original_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    with pytest.raises(OSError, match="unlink failed"):
        outbox.discard_durably(scope)
    assert blob.is_file()
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            """SELECT cleanup_required_at FROM hosted_room_output_artifacts
                WHERE artifact_id=?""",
            (stored["artifact_id"],),
        ).fetchone()[0] is not None

    assert outbox.retry_scheduled_cleanups() == 1
    assert not blob.exists()
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (stored["artifact_id"],),
        ).fetchone()[0] == 0


def test_existing_ack_receipt_schema_migrates_into_bounded_cleanup(tmp_path: Path):
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE hosted_room_output_artifacts (
                artifact_id TEXT PRIMARY KEY,
                scope_key TEXT NOT NULL,
                scope_json TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                mime TEXT NOT NULL,
                size INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                blob_name TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL,
                acknowledged_at REAL,
                UNIQUE(scope_key, sha256, name)
            )"""
        )
        conn.execute(
            """INSERT INTO hosted_room_output_artifacts
               (artifact_id, scope_key, scope_json, name, kind, mime, size,
                sha256, blob_name, created_at, acknowledged_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "rart_" + "a" * 32,
                "scope-key",
                "{}",
                "handoff.md",
                "file",
                "text/markdown",
                1,
                "b" * 64,
                "blob_missing_after_prior_ack",
                time.time(),
                time.time(),
            ),
        )
        conn.commit()

    RoomArtifactOutbox(db)
    with sqlite3.connect(db) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(hosted_room_output_artifacts)")
        }
        reclaimed_at = conn.execute(
            """SELECT blob_reclaimed_at FROM hosted_room_output_artifacts
                WHERE artifact_id=?""",
            ("rart_" + "a" * 32,),
        ).fetchone()[0]
    assert "blob_reclaimed_at" in columns
    assert reclaimed_at is not None


def test_share_group_file_is_visible_only_inside_bound_room_turn(tmp_path: Path):
    path = tmp_path / "review.md"
    path.write_text("Review this.\n", encoding="utf-8")
    assert json.loads(share_group_file(str(path)))["ok"] is False

    home_token = set_hermes_home_override(tmp_path)
    scope_token = bind_room_artifact_scope(_scope())
    try:
        result = json.loads(share_group_file(str(path), name="handoff.md"))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)

    assert result["ok"] is True
    assert result["name"] == "handoff.md"
    assert str(path) not in json.dumps(result)
    assert RoomArtifactOutbox(tmp_path / "state.db").list(_scope())[0][
        "artifact_id"
    ] == result["artifact_id"]


def test_bot_room_turn_injects_tool_without_broadening_other_sessions():
    room_agent = SimpleNamespace(platform="bot_room", tools=[], valid_tool_names=set())
    other_agent = SimpleNamespace(platform="telegram", tools=[], valid_tool_names=set())

    assert ensure_share_group_file_tool(room_agent) is True
    assert [tool["function"]["name"] for tool in room_agent.tools] == [
        "share_group_file"
    ]
    assert ensure_share_group_file_tool(room_agent) is True
    assert len(room_agent.tools) == 1
    assert ensure_share_group_file_tool(other_agent) is False
    assert other_agent.tools == []


def test_share_group_file_uses_the_active_hosted_session_scope(tmp_path: Path):
    from tui_gateway.server import _current_runtime_session_record

    path = tmp_path / "review.md"
    path.write_text("Review this.\n", encoding="utf-8")
    home_token = set_hermes_home_override(tmp_path)
    session_token = _current_runtime_session_record.set({
        "_hosted_room_task": _scope().as_mapping()
    })
    try:
        result = json.loads(share_group_file(str(path)))
    finally:
        _current_runtime_session_record.reset(session_token)
        reset_hermes_home_override(home_token)

    assert result["ok"] is True
    assert RoomArtifactOutbox(tmp_path / "state.db").list(_scope())[0][
        "artifact_id"
    ] == result["artifact_id"]


def test_share_group_file_still_rejects_relative_path(
    tmp_path: Path,
    monkeypatch,
):
    path = tmp_path / "review.md"
    path.write_text("Review this.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    home_token = set_hermes_home_override(tmp_path)
    scope_token = bind_room_artifact_scope(_scope())
    try:
        result = json.loads(share_group_file(path.name))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)

    assert result == {
        "ok": False,
        "error": (
            "That file cannot be shared. Move it to the workspace or a Hermes "
            "media folder and try again."
        ),
    }


def test_share_group_file_redacts_unexpected_filesystem_errors(
    tmp_path: Path,
    monkeypatch,
):
    path = tmp_path / "review.md"
    path.write_text("Review this.\n", encoding="utf-8")
    monkeypatch.setattr(
        RoomArtifactOutbox,
        "put_open_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("/Users/alice/private/review.md")
        ),
    )
    home_token = set_hermes_home_override(tmp_path)
    scope_token = bind_room_artifact_scope(_scope())
    try:
        result = json.loads(share_group_file(str(path)))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)

    assert result == {
        "ok": False,
        "error": "That file could not be shared. Check the file and try again.",
    }
    assert "/Users/alice" not in json.dumps(result)


def test_share_group_file_rejects_direct_symlink(tmp_path: Path):
    target = tmp_path / "target.md"
    target.write_text("secret-ish\n", encoding="utf-8")
    link = tmp_path / "link.md"
    link.symlink_to(target)
    home_token = set_hermes_home_override(tmp_path)
    scope_token = bind_room_artifact_scope(_scope())
    try:
        result = json.loads(share_group_file(str(link)))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)
    assert result == {"ok": False, "error": "Symbolic links cannot be shared."}


def test_share_group_file_rejects_ancestor_symlink(tmp_path: Path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "handoff.md").write_text("secret-ish\n", encoding="utf-8")
    link_dir = tmp_path / "linkdir"
    link_dir.symlink_to(target_dir, target_is_directory=True)
    home_token = set_hermes_home_override(tmp_path)
    scope_token = bind_room_artifact_scope(_scope())
    try:
        result = json.loads(share_group_file(str(link_dir / "handoff.md")))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)
    assert result == {"ok": False, "error": "Symbolic links cannot be shared."}


def test_share_group_file_rejects_another_rooms_private_outbox_blob(tmp_path: Path):
    home = tmp_path / ".hermes"
    home.mkdir()
    source = tmp_path / "private.md"
    source.write_text("room A only\n", encoding="utf-8")
    first_scope = _scope(room_id="room-a", task_id="dtask:room-a")
    second_scope = _scope(room_id="room-b", task_id="dtask:room-b")
    outbox = RoomArtifactOutbox(home / "state.db")
    stored = outbox.put_path(scope=first_scope, path=source)
    with sqlite3.connect(home / "state.db") as conn:
        blob_name = conn.execute(
            "SELECT blob_name FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (stored["artifact_id"],),
        ).fetchone()[0]
    private_blob = outbox.blob_root / str(blob_name)

    home_token = set_hermes_home_override(home)
    scope_token = bind_room_artifact_scope(second_scope)
    try:
        result = json.loads(share_group_file(str(private_blob)))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)

    assert result == {
        "ok": False,
        "error": "Private Group Chat storage cannot be shared.",
    }
    assert outbox.list(second_scope) == []


@pytest.mark.parametrize(
    "storage_name",
    [
        "hosted-room-artifact-outbox",
        "hosted-room-attachments",
        "roomlink-attachment-spool",
    ],
)
def test_share_group_file_rejects_private_room_storage_roots(
    tmp_path: Path,
    storage_name: str,
):
    home = tmp_path / ".hermes"
    private_file = home / storage_name / "private.bin"
    private_file.parent.mkdir(parents=True)
    private_file.write_bytes(b"private bytes")
    home_token = set_hermes_home_override(home)
    scope_token = bind_room_artifact_scope(_scope())
    try:
        result = json.loads(share_group_file(str(private_file)))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)

    assert result == {
        "ok": False,
        "error": "Private Group Chat storage cannot be shared.",
    }


def test_private_storage_guard_does_not_use_string_prefix(tmp_path: Path):
    home = tmp_path / ".hermes"
    allowed = home / "hosted-room-artifact-outbox-safe" / "handoff.md"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("safe workspace file\n", encoding="utf-8")
    home_token = set_hermes_home_override(home)
    scope_token = bind_room_artifact_scope(_scope())
    try:
        result = json.loads(share_group_file(str(allowed)))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)
    assert result["ok"] is True


def test_share_group_file_reads_bytes_from_remote_execution_backend(
    tmp_path: Path,
    monkeypatch,
):
    from tools import file_tools, file_tools_paths

    home = tmp_path / ".hermes"
    home.mkdir()
    payload = b"remote handoff\n"
    file_ops = SimpleNamespace(
        _has_command=lambda command: command == "python3",
        _escape_shell_arg=lambda value: repr(value),
        _exec=lambda command, timeout: SimpleNamespace(
            exit_code=0,
            stdout=(
                "HERMES_ROOM_FILE_V1:"
                + json.dumps({
                    "ok": True,
                    "data": base64.b64encode(payload).decode("ascii"),
                })
                + "\n"
            ),
        ),
    )
    monkeypatch.setattr(
        file_tools_paths,
        "_terminal_env_type_for_task",
        lambda task_id: "ssh",
    )
    monkeypatch.setattr(
        file_tools_paths,
        "_resolve_path_for_task",
        lambda path, task_id: Path(path),
    )
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id: file_ops)
    scope = _scope()
    home_token = set_hermes_home_override(home)
    scope_token = bind_room_artifact_scope(scope)
    try:
        result = json.loads(
            share_group_file(
                "/remote/workspace/handoff.md",
                task_id="room-session",
            )
        )
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)

    assert result["ok"] is True
    _metadata, copied = RoomArtifactOutbox(home / "state.db").read(
        scope,
        result["artifact_id"],
    )
    assert copied == payload


@pytest.mark.parametrize("ancestor", [False, True])
def test_remote_backend_reader_rejects_symlink_components(
    tmp_path: Path,
    ancestor: bool,
):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "private.md"
    target.write_text("do not follow\n", encoding="utf-8")
    if ancestor:
        link = tmp_path / "linked-dir"
        link.symlink_to(target_dir, target_is_directory=True)
        candidate = link / target.name
    else:
        candidate = tmp_path / "linked.md"
        candidate.symlink_to(target)
    file_ops = ShellFileOperations(
        LocalEnvironment(cwd=str(tmp_path), timeout=10),
        cwd=str(tmp_path),
    )

    with pytest.raises(RoomArtifactError, match="active execution environment"):
        _read_backend_file_bytes_nofollow(file_ops, str(candidate))


def test_remote_backend_reader_returns_exact_bounded_regular_bytes(tmp_path: Path):
    candidate = tmp_path / "handoff.bin"
    candidate.write_bytes(b"\x00exact remote bytes\xff")
    file_ops = ShellFileOperations(
        LocalEnvironment(cwd=str(tmp_path), timeout=10),
        cwd=str(tmp_path),
    )

    assert _read_backend_file_bytes_nofollow(file_ops, str(candidate)) == (
        b"\x00exact remote bytes\xff"
    )


def test_remote_backend_reader_rejects_oversized_regular_file(tmp_path: Path):
    candidate = tmp_path / "oversized.bin"
    with candidate.open("wb") as handle:
        handle.truncate(artifacts.MAX_ATTACHMENT_BYTES + 1)
    file_ops = ShellFileOperations(
        LocalEnvironment(cwd=str(tmp_path), timeout=10),
        cwd=str(tmp_path),
    )

    with pytest.raises(RoomArtifactError, match="active execution environment"):
        _read_backend_file_bytes_nofollow(file_ops, str(candidate))


def test_macos_tmp_alias_keeps_no_follow_policy_without_rejecting_tmp(tmp_path: Path):
    if os.name == "nt" or Path("/tmp").resolve() != Path("/private/tmp"):
        pytest.skip("macOS /tmp alias only")
    descriptor, raw_path = tempfile.mkstemp(prefix="hermes-room-file-", suffix=".md")
    os.close(descriptor)
    candidate = Path(raw_path)
    candidate.write_text("temporary handoff\n", encoding="utf-8")
    home_token = set_hermes_home_override(tmp_path)
    scope_token = bind_room_artifact_scope(_scope())
    try:
        result = json.loads(share_group_file(str(candidate)))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)
        candidate.unlink(missing_ok=True)
    assert result["ok"] is True


def test_linux_tmp_path_is_never_rewritten_to_private_tmp(monkeypatch):
    monkeypatch.setattr(room_artifact_tool.sys, "platform", "linux")
    candidate = Path("/tmp/group-file.md")
    assert room_artifact_tool._canonical_macos_alias_path(candidate) == candidate


def test_macos_tmp_alias_does_not_allow_user_symlink(tmp_path: Path):
    if os.name == "nt" or Path("/tmp").resolve() != Path("/private/tmp"):
        pytest.skip("macOS /tmp alias only")
    test_root = Path("/tmp") / f"hermes-room-symlink-{os.getpid()}-{time.time_ns()}"
    test_root.mkdir()
    target = test_root / "target.md"
    target.write_text("do not follow\n", encoding="utf-8")
    link = test_root / "link.md"
    link.symlink_to(target)
    home_token = set_hermes_home_override(tmp_path)
    scope_token = bind_room_artifact_scope(_scope())
    try:
        result = json.loads(share_group_file(str(link)))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)
        link.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        test_root.rmdir()
    assert result == {"ok": False, "error": "Symbolic links cannot be shared."}


def test_share_group_file_copies_opened_bytes_after_path_swap(
    tmp_path: Path,
    monkeypatch,
):
    source_dir = tmp_path / "safe"
    source_dir.mkdir()
    source = source_dir / "handoff.md"
    source.write_bytes(b"SAFE BYTES\n")
    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    (hidden_dir / source.name).write_bytes(b"TOP SECRET\n")
    home = tmp_path / "home"
    home.mkdir()
    original_put = RoomArtifactOutbox.put_open_file

    def swap_path_then_put(outbox, *args, **kwargs):
        source_dir.rename(tmp_path / "original-safe")
        source_dir.symlink_to(hidden_dir, target_is_directory=True)
        return original_put(outbox, *args, **kwargs)

    monkeypatch.setattr(RoomArtifactOutbox, "put_open_file", swap_path_then_put)
    scope = _scope()
    home_token = set_hermes_home_override(home)
    scope_token = bind_room_artifact_scope(scope)
    try:
        result = json.loads(share_group_file(str(source)))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)

    assert result["ok"] is True
    metadata, data = RoomArtifactOutbox(home / "state.db").read(
        scope,
        result["artifact_id"],
    )
    assert metadata["sha256"] == result["sha256"]
    assert data == b"SAFE BYTES\n"


def test_terminal_manifest_rejects_tampered_digest(tmp_path: Path):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("# Handoff\n", encoding="utf-8")
    scope = _scope()
    RoomArtifactOutbox(db).put_path(scope=scope, path=path)
    manifest = terminal_artifact_manifest(db, scope)
    manifest["items"][0]["name"] = "changed.md"

    with pytest.raises(RoomArtifactError, match="digest changed"):
        validate_terminal_artifact_manifest(manifest)


def test_exact_scope_cancel_and_grant_revoke_cleanup(tmp_path: Path):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    first = _scope()
    second = _scope(task_id="dtask:second")
    other_member = _scope(
        task_id="dtask:third",
        member_id="member-other",
        target_profile="other",
    )
    outbox.put_path(scope=first, path=path)
    outbox.put_path(scope=second, path=path)
    outbox.put_path(scope=other_member, path=path)

    assert outbox.discard(first) == 1
    assert outbox.list(first) == []
    assert len(outbox.list(second)) == 1
    assert outbox.discard_claims(second.as_mapping()) == 1
    assert outbox.list(second) == []
    assert len(outbox.list(other_member)) == 1


def test_new_execution_generation_reclaims_crash_stranded_output(tmp_path: Path):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    crashed = _scope(execution_generation=1)
    retry = _scope(execution_generation=2)
    other_task = _scope(task_id="dtask:other", execution_generation=1)
    outbox.put_path(scope=crashed, path=path)
    outbox.put_path(scope=other_task, path=path)

    assert outbox.discard_superseded(retry) == 1
    assert outbox.list(crashed) == []
    assert len(outbox.list(other_task)) == 1


def test_superseded_generation_cannot_recreate_private_output(tmp_path: Path):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    first = _scope(execution_generation=1)
    second = _scope(execution_generation=2)

    outbox.put_path(scope=first, path=path)
    outbox.put_path(scope=second, path=path)

    with pytest.raises(RoomArtifactError, match="generation is stale"):
        outbox.put_path(scope=first, path=path)
    assert outbox.list(first) == []
    assert len(outbox.list(second)) == 1


def test_retired_generation_cannot_recreate_private_output(tmp_path: Path):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    scope = _scope()

    outbox.put_path(scope=scope, path=path)
    assert outbox.discard_durably(scope) == 1

    with pytest.raises(RoomArtifactError, match="generation is stale"):
        outbox.put_path(scope=scope, path=path)


def test_upgrade_backfills_existing_generation_before_installing_guard(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    latest = _scope(execution_generation=2)
    stale = _scope(execution_generation=1)
    outbox = RoomArtifactOutbox(db)
    outbox.put_path(scope=latest, path=path)
    with sqlite3.connect(db) as conn:
        for trigger in (
            "hosted_room_output_generation_guard_insert",
            "hosted_room_output_generation_guard_update",
            "hosted_room_output_generation_track_insert",
            "hosted_room_output_generation_track_update",
            "hosted_room_output_generation_track_terminal",
            "hosted_room_output_generation_track_delete",
        ):
            conn.execute(f"DROP TRIGGER {trigger}")
        conn.execute("DELETE FROM hosted_room_output_generation_fences")

    recovered = RoomArtifactOutbox(db)
    with pytest.raises(RoomArtifactError, match="generation is stale"):
        recovered.put_path(scope=stale, path=path)


def test_database_trigger_rejects_stale_insert_from_pre_upgrade_writer(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    retired = _scope(execution_generation=2)
    outbox.put_path(scope=retired, path=path)
    outbox.discard_durably(retired)

    with sqlite3.connect(db) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="stale room artifact"):
            _insert_legacy_artifact(
                conn,
                _scope(execution_generation=1),
                suffix="stale",
            )


def test_database_trigger_tracks_newer_insert_from_pre_upgrade_writer(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    latest = _scope(execution_generation=2)
    with sqlite3.connect(db) as conn:
        _insert_legacy_artifact(conn, latest, suffix="latest")

    with pytest.raises(RoomArtifactError, match="generation is stale"):
        outbox.put_path(scope=_scope(execution_generation=1), path=path)


def test_database_trigger_rejects_stale_scope_update_from_old_writer(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    RoomArtifactOutbox(db)
    latest = _scope(execution_generation=2)
    stale = _scope(execution_generation=1)
    with sqlite3.connect(db) as conn:
        _insert_legacy_artifact(conn, latest, suffix="updated")
        with pytest.raises(sqlite3.IntegrityError, match="stale room artifact"):
            conn.execute(
                """UPDATE hosted_room_output_artifacts SET scope_json=?
                    WHERE artifact_id='rart_legacy_updated'""",
                (
                    json.dumps(
                        stale.as_mapping(), sort_keys=True, separators=(",", ":")
                    ),
                ),
            )


def test_generation_fence_pruning_waits_for_artifact_rows(tmp_path: Path):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    retired = _scope(task_id="dtask:retired")
    active = _scope(task_id="dtask:active")
    outbox.put_path(scope=retired, path=path)
    outbox.discard_durably(retired)
    outbox.put_path(scope=active, path=path)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hosted_room_output_generation_fences SET updated_at=0"
        )

    assert outbox.prune_generation_fences(
        now=GENERATION_FENCE_RETENTION_SECONDS + 1
    ) == 1
    with sqlite3.connect(db) as conn:
        identities = {
            row[0]
            for row in conn.execute(
                "SELECT lineage_identity FROM hosted_room_output_generation_fences"
            )
        }
    assert retired.lineage_json not in identities
    assert active.lineage_json in identities


def test_supersede_unlink_failure_replays_cleanup_without_losing_fence(
    tmp_path: Path,
    monkeypatch,
):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    first = _scope(execution_generation=1)
    second = _scope(execution_generation=2)
    stored = outbox.put_path(scope=first, path=path)
    with sqlite3.connect(db) as conn:
        blob_name = conn.execute(
            "SELECT blob_name FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (stored["artifact_id"],),
        ).fetchone()[0]
    blob = outbox.blob_root / str(blob_name)
    original_unlink = Path.unlink
    failed = False

    def fail_blob_once(candidate, *args, **kwargs):
        nonlocal failed
        if candidate == blob and not failed:
            failed = True
            raise OSError("temporary supersede unlink fault")
        return original_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_blob_once)
    with pytest.raises(OSError, match="temporary supersede unlink fault"):
        outbox.put_path(scope=second, path=path)
    with sqlite3.connect(db) as conn:
        obligation = conn.execute(
            """SELECT cleanup_required_at FROM hosted_room_output_artifacts
                WHERE artifact_id=?""",
            (stored["artifact_id"],),
        ).fetchone()
    assert obligation is not None and obligation[0] is not None
    with pytest.raises(RoomArtifactError, match="generation is stale"):
        outbox.put_path(scope=first, path=path)

    monkeypatch.setattr(Path, "unlink", original_unlink)
    recovered = RoomArtifactOutbox(db)
    assert not blob.exists()
    assert recovered.list(first) == []
    assert recovered.put_path(scope=second, path=path)["name"] == "handoff.md"


def test_grant_revoke_unlink_failure_replays_exact_cleanup(
    tmp_path: Path,
    monkeypatch,
):
    db = tmp_path / "state.db"
    path = tmp_path / "handoff.md"
    path.write_text("handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    scope = _scope()
    stored = outbox.put_path(scope=scope, path=path)
    with sqlite3.connect(db) as conn:
        blob_name = conn.execute(
            "SELECT blob_name FROM hosted_room_output_artifacts WHERE artifact_id=?",
            (stored["artifact_id"],),
        ).fetchone()[0]
    blob = outbox.blob_root / str(blob_name)
    original_unlink = Path.unlink
    failed = False

    def fail_blob_once(candidate, *args, **kwargs):
        nonlocal failed
        if candidate == blob and not failed:
            failed = True
            raise OSError("temporary revoke unlink fault")
        return original_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_blob_once)
    with pytest.raises(OSError, match="temporary revoke unlink fault"):
        outbox.discard_claims(scope.as_mapping())
    with sqlite3.connect(db) as conn:
        obligation = conn.execute(
            """SELECT cleanup_required_at FROM hosted_room_output_artifacts
                WHERE artifact_id=?""",
            (stored["artifact_id"],),
        ).fetchone()
    assert obligation is not None and obligation[0] is not None

    monkeypatch.setattr(Path, "unlink", original_unlink)
    recovered = RoomArtifactOutbox(db)
    assert not blob.exists()
    assert recovered.list(scope) == []
    with pytest.raises(RoomArtifactError, match="generation is stale"):
        recovered.put_path(scope=scope, path=path)


@pytest.mark.parametrize(
    ("name", "data", "kind", "mime"),
    [
        ("diagram.png", b"\x89PNG\r\n\x1a\nimage", "image", "image/png"),
        ("brief.pdf", b"%PDF-1.7\nbody", "pdf", "application/pdf"),
        ("archive.bin", b"\x00\x01\x02", "file", "application/octet-stream"),
    ],
)
def test_output_artifact_kinds_keep_verified_bytes(
    tmp_path: Path,
    name: str,
    data: bytes,
    kind: str,
    mime: str,
):
    path = tmp_path / name
    path.write_bytes(data)
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    stored = outbox.put_path(scope=_scope(), path=path)
    metadata, copied = outbox.read(_scope(), stored["artifact_id"])
    assert metadata["kind"] == kind
    assert metadata["mime"] == mime
    assert copied == data


def test_output_artifact_rejects_mislabeled_image(tmp_path: Path):
    path = tmp_path / "not-an-image.png"
    path.write_text("plain text", encoding="utf-8")
    with pytest.raises(RoomArtifactError, match="image bytes"):
        RoomArtifactOutbox(tmp_path / "state.db").put_path(
            scope=_scope(),
            path=path,
        )


@pytest.mark.parametrize("name", [".env", "auth.json", "config.yaml"])
def test_share_group_file_rejects_sibling_profile_state(
    tmp_path: Path,
    monkeypatch,
    name: str,
):
    root = tmp_path / "hermes"
    alpha = root / "profiles" / "alpha"
    beta = root / "profiles" / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    candidate = beta / name
    candidate.write_text("do not share\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    home_token = set_hermes_home_override(alpha)
    scope_token = bind_room_artifact_scope(_scope(target_profile="alpha"))
    try:
        result = json.loads(share_group_file(str(candidate)))
    finally:
        reset_room_artifact_scope(scope_token)
        reset_hermes_home_override(home_token)
    assert result == {
        "ok": False,
        "error": "Files owned by another Hermes profile cannot be shared.",
    }
