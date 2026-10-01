"""The private per-turn outbox behind Group Chat file sharing: scope, bytes, fences, cleanup."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from gateway import hosted_room_artifacts as artifacts
from gateway.hosted_room_artifacts import (
    ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS,
    GENERATION_FENCE_RETENTION_SECONDS,
    UNACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS,
    RoomArtifactError,
    RoomArtifactOutbox,
    RoomArtifactScope,
    open_room_artifact_path,
    terminal_artifact_manifest,
    validate_terminal_artifact_manifest,
)


def _scope(**overrides) -> RoomArtifactScope:
    value = {
        "room_id": "room-1",
        "task_id": "dtask:abc",
        "execution_generation": 1,
        "member_id": "member-build",
        "target_profile": "build",
        "home_install_id": "install-home",
        "target_install_id": "install-home",
        "authority_gateway_id": "install-home",
        "authority_epoch": 1,
    }
    value.update(overrides)
    return RoomArtifactScope.from_mapping(value)


def _put(outbox, scope, path, name=None):
    with open_room_artifact_path(path) as (opened, descriptor):
        return outbox.put_open_file(scope=scope, descriptor=descriptor, source_name=opened.name, name=name)


def _ack(outbox, scope, *artifact_ids):
    return outbox.acknowledge(
        scope, list(artifact_ids), message_event_id=f"dmessage:{scope.task_id.removeprefix('dtask:')}")


def _file(tmp_path: Path, name="handoff.md", data=b"# Handoff\n") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _blob(outbox, artifact_id) -> Path:
    with sqlite3.connect(outbox.db_path) as conn:
        name = conn.execute("SELECT blob_name FROM hosted_room_output_artifacts WHERE artifact_id=?",
                            (artifact_id,)).fetchone()[0]
    return outbox.blob_root / name


def test_outbox_is_idempotent_scoped_and_acknowledged_once(tmp_path: Path):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    scope = _scope()
    path = _file(tmp_path)

    first = _put(outbox, scope, path)
    assert _put(outbox, scope, path) == first
    metadata, data = outbox.read(scope, first["artifact_id"])
    assert (metadata, data) == (first, b"# Handoff\n")
    with pytest.raises(RoomArtifactError, match="not found"):
        outbox.read(_scope(task_id="dtask:other"), first["artifact_id"])

    manifest = terminal_artifact_manifest(outbox.list(scope))
    assert validate_terminal_artifact_manifest(manifest) == [first]
    assert _ack(outbox, scope, first["artifact_id"]) == 1
    assert _ack(outbox, scope, first["artifact_id"]) == 0
    assert outbox.scope_manifest(scope) == [first] and outbox.list(scope) == []
    with pytest.raises(RoomArtifactError, match="not found"):
        outbox.read(scope, first["artifact_id"])
    with pytest.raises(RoomArtifactError, match="commitment changed"):
        outbox.acknowledge(scope, [first["artifact_id"]], message_event_id="dmessage:other")


def test_read_range_serves_bounded_slices_of_open_output(tmp_path: Path):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    scope = _scope()
    stored = _put(outbox, scope, _file(tmp_path, data=b"0123456789"))

    assert outbox.read_range(scope, stored["artifact_id"], offset=3, length=4)[1] == b"3456"
    assert outbox.read_range(scope, stored["artifact_id"], offset=8, length=100)[1] == b"89"
    for offset, length in ((10, 1), (-1, 1), (0, 0)):
        with pytest.raises(RoomArtifactError, match="range is invalid"):
            outbox.read_range(scope, stored["artifact_id"], offset=offset, length=length)
    _blob(outbox, stored["artifact_id"]).unlink()
    with pytest.raises(RoomArtifactError, match="bytes are missing"):
        outbox.read(scope, stored["artifact_id"])


def test_terminal_manifest_rejects_tampered_digest(tmp_path: Path):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    _put(outbox, _scope(), _file(tmp_path))
    manifest = terminal_artifact_manifest(outbox.list(_scope()))
    manifest["items"][0]["name"] = "changed.md"
    with pytest.raises(RoomArtifactError, match="digest changed"):
        validate_terminal_artifact_manifest(manifest)
    assert terminal_artifact_manifest([]) is None


@pytest.mark.parametrize(
    ("name", "data", "kind", "mime"),
    [
        ("diagram.png", b"\x89PNG\r\n\x1a\nimage", "image", "image/png"),
        ("chart.gif", b"GIF89a....", "image", "image/gif"),
        ("brief.pdf", b"%PDF-1.7\nbody", "pdf", "application/pdf"),
        ("archive.bin", b"\x00\x01\x02", "file", "application/octet-stream"),
        ("notes.txt", b"plain notes\n", "file", "text/plain"),
        # A text name over bytes the room store would not call text is shared opaque.
        ("bmw-report.txt", b"BMW report\n", "file", "application/octet-stream"),
        ("broken.txt", b"\xff\xfe\x00broken", "file", "application/octet-stream"),
    ],
)
def test_output_kinds_follow_the_room_store_classification(tmp_path, name, data, kind, mime):
    from gateway.hosted_room_attachments import _validate_mime_and_kind

    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    stored = _put(outbox, _scope(), _file(tmp_path, name, data))
    metadata, copied = outbox.read(_scope(), stored["artifact_id"])
    assert (metadata["kind"], metadata["mime"], copied) == (kind, mime, data)
    _validate_mime_and_kind(copied, kind=kind, mime=mime)  # the room store will accept it


@pytest.mark.parametrize(("name", "error"), [("not-an-image.png", "image bytes"), ("fake.pdf", "PDF bytes")])
def test_output_refuses_mislabeled_images_and_pdfs(tmp_path, name, error):
    with pytest.raises(RoomArtifactError, match=error):
        _put(RoomArtifactOutbox(tmp_path / "state.db"), _scope(), _file(tmp_path, name, b"plain text"))


def test_turn_and_outbox_quotas_are_enforced(tmp_path: Path, monkeypatch):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    scope = _scope()
    for index in range(artifacts.MAX_ATTACHMENTS_PER_MESSAGE):
        _put(outbox, scope, _file(tmp_path, f"file-{index}.txt", f"payload {index}\n".encode()))
    with pytest.raises(RoomArtifactError, match="turn artifact quota"):
        _put(outbox, scope, _file(tmp_path, "one-more.txt", b"one more\n"))
    monkeypatch.setattr(artifacts, "MAX_GATEWAY_BLOB_BYTES", 1)
    with pytest.raises(RoomArtifactError, match="gateway room artifact quota"):
        _put(outbox, _scope(task_id="dtask:other"), _file(tmp_path, "other.txt", b"other\n"))


def test_repeated_acknowledge_uses_receipt_without_reunlinking(tmp_path: Path, monkeypatch):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    scope = _scope()
    stored = _put(outbox, scope, _file(tmp_path))
    assert _ack(outbox, scope, stored["artifact_id"]) == 1
    with sqlite3.connect(outbox.db_path) as conn:
        row = conn.execute("SELECT acknowledged_at, blob_reclaimed_at FROM hosted_room_output_artifacts").fetchone()
    assert row[0] is not None and row[1] is not None

    unlinked = []
    original = Path.unlink
    monkeypatch.setattr(Path, "unlink", lambda candidate, *a, **k: (unlinked.append(candidate),
                                                                     original(candidate, *a, **k))[1])
    assert _ack(outbox, scope, stored["artifact_id"]) == 0
    assert unlinked == []


def test_acknowledged_receipts_prune_in_indexed_batches(tmp_path: Path, monkeypatch):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    path = _file(tmp_path)
    monkeypatch.setattr(artifacts, "ACKNOWLEDGED_ARTIFACT_PRUNE_BATCH", 2)
    for index in range(3):
        scope = _scope(task_id=f"dtask:{index}")
        assert _ack(outbox, scope, _put(outbox, scope, path)["artifact_id"]) == 1
    with sqlite3.connect(outbox.db_path) as conn:
        conn.execute("UPDATE hosted_room_output_artifacts SET acknowledged_at=0, receipt_expires_at=0")
        conn.commit()
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(hosted_room_output_artifacts)")}
        plan = " ".join(str(row[3]) for row in conn.execute(
            """EXPLAIN QUERY PLAN SELECT artifact_id FROM hosted_room_output_artifacts
               WHERE acknowledged_at IS NOT NULL AND acknowledged_at<=? ORDER BY acknowledged_at, artifact_id
               LIMIT ?""", (1, 2)))
    assert {"idx_hosted_room_output_ack_expiry", "idx_hosted_room_output_ack_cleanup"} <= indexes
    assert "idx_hosted_room_output_ack_expiry" in plan

    now = ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS + 1
    assert [outbox.prune_acknowledged_receipts(now=now) for _ in range(3)] == [2, 1, 0]


def test_abandoned_open_output_expires_without_leaking_bytes(tmp_path: Path):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    scope = _scope()
    stored = _put(outbox, scope, _file(tmp_path))
    blob = _blob(outbox, stored["artifact_id"])
    with sqlite3.connect(outbox.db_path) as conn:
        conn.execute("UPDATE hosted_room_output_artifacts SET created_at=0")
    assert outbox.prune_unacknowledged_artifacts(now=UNACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS + 1) == 1
    assert not blob.exists()
    assert outbox.list(scope) == []
    # The tombstone answers a very late ACK replay without bytes.
    assert _ack(outbox, scope, stored["artifact_id"]) == 0
    with pytest.raises(RoomArtifactError, match="generation is stale"):
        _put(outbox, scope, _file(tmp_path))


def test_durable_discard_replays_after_a_failed_removal(tmp_path: Path, monkeypatch):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    scope = _scope()
    stored = _put(outbox, scope, _file(tmp_path))
    blob = _blob(outbox, stored["artifact_id"])
    original = Path.unlink
    failures = [True]

    def flaky(candidate, *args, **kwargs):
        if candidate == blob and failures:
            failures.pop()
            raise OSError("temporary unlink fault")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky)
    with pytest.raises(OSError, match="temporary unlink fault"):
        outbox.discard_durably(scope)
    with sqlite3.connect(outbox.db_path) as conn:
        assert conn.execute("SELECT cleanup_required_at FROM hosted_room_output_artifacts").fetchone()[0]
    # The intent alone already refuses publication and new writes of this attempt.
    assert outbox.list(scope) == []
    with pytest.raises(RoomArtifactError, match="already discarded"):
        _ack(outbox, scope, stored["artifact_id"])
    with pytest.raises(RoomArtifactError, match="generation is stale"):
        _put(outbox, scope, _file(tmp_path))

    RoomArtifactOutbox(outbox.db_path)  # reopening replays the intent
    assert not blob.exists()
    with sqlite3.connect(outbox.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_output_artifacts").fetchone()[0] == 0


def test_discard_never_touches_a_published_receipt(tmp_path: Path):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    scope = _scope()
    stored = _put(outbox, scope, _file(tmp_path))
    assert _ack(outbox, scope, stored["artifact_id"]) == 1
    assert outbox.discard_durably(scope) == 0
    assert outbox.discard_attempt(room_id=scope.room_id, task_id=scope.task_id, execution_generation=1,
                                  member_id=scope.member_id, target_profile=scope.target_profile) == 0
    assert _ack(outbox, scope, stored["artifact_id"]) == 0  # the receipt still replays


def test_constructor_reclaims_bytes_after_an_ack_commit_crash(tmp_path: Path):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    scope = _scope()
    stored = _put(outbox, scope, _file(tmp_path))
    blob = _blob(outbox, stored["artifact_id"])
    with sqlite3.connect(outbox.db_path) as conn:
        conn.execute("UPDATE hosted_room_output_artifacts SET acknowledged_at=?, ack_message_event_id=?",
                     (time.time(), "dmessage:abc"))
    assert blob.is_file()
    recovered = RoomArtifactOutbox(outbox.db_path)
    assert not blob.exists()
    assert _ack(recovered, scope, stored["artifact_id"]) == 0


def test_constructor_does_not_unlink_historical_receipts(tmp_path: Path, monkeypatch):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    path = _file(tmp_path)
    for index in range(5):
        scope = _scope(task_id=f"dtask:{index}")
        _ack(outbox, scope, _put(outbox, scope, path)["artifact_id"])
    unlinked = []
    original = Path.unlink
    monkeypatch.setattr(Path, "unlink", lambda candidate, *a, **k: (unlinked.append(candidate),
                                                                     original(candidate, *a, **k))[1])
    RoomArtifactOutbox(outbox.db_path)
    assert unlinked == []


def test_discard_attempt_retires_only_the_named_attempt(tmp_path: Path):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    path = _file(tmp_path)
    target = _scope()
    other_task = _scope(task_id="dtask:other")
    other_member = _scope(member_id="member-other", target_profile="other")
    for scope in (target, other_task, other_member):
        _put(outbox, scope, path)
    assert outbox.discard_attempt(room_id="room-1", task_id="dtask:abc", execution_generation=1,
                                  member_id="member-build", target_profile="build") == 1
    assert outbox.list(target) == []
    assert len(outbox.list(other_task)) == len(outbox.list(other_member)) == 1


def test_new_generation_reclaims_older_output_and_fences_it(tmp_path: Path):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    path = _file(tmp_path)
    first, second = _scope(execution_generation=1), _scope(execution_generation=2)
    other = _scope(task_id="dtask:other")
    _put(outbox, first, path)
    _put(outbox, other, path)
    _put(outbox, second, path)
    assert outbox.list(first) == []
    assert len(outbox.list(other)) == len(outbox.list(second)) == 1
    with pytest.raises(RoomArtifactError, match="generation is stale"):
        _put(outbox, first, path)


def test_retired_generation_cannot_write_again(tmp_path: Path):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    scope = _scope()
    _put(outbox, scope, _file(tmp_path))
    assert outbox.discard_durably(scope) == 1
    with pytest.raises(RoomArtifactError, match="generation is stale"):
        _put(outbox, scope, _file(tmp_path))


def _insert_directly(conn, scope, suffix):
    conn.execute(
        """INSERT INTO hosted_room_output_artifacts
           (artifact_id, scope_key, scope_json, name, kind, mime, size, sha256, blob_name, created_at)
           VALUES (?, ?, ?, ?, 'file', 'application/octet-stream', 1, ?, ?, ?)""",
        (f"rart_direct_{suffix}", scope.key, json.dumps(scope.as_mapping(), sort_keys=True, separators=(",", ":")),
         f"direct-{suffix}.bin", suffix.ljust(64, "0")[:64], f"blob_direct_{suffix}", time.time()))


def test_database_triggers_fence_writers_that_skip_the_python_checks(tmp_path: Path):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    retired = _scope(execution_generation=2)
    _put(outbox, retired, _file(tmp_path))
    outbox.discard_durably(retired)
    with sqlite3.connect(outbox.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="stale room artifact"):
            _insert_directly(conn, _scope(execution_generation=1), "stale")
        newer = _scope(task_id="dtask:newer", execution_generation=2)
        _insert_directly(conn, newer, "newer")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="stale room artifact"):
            conn.execute("UPDATE hosted_room_output_artifacts SET scope_json=? WHERE artifact_id='rart_direct_newer'",
                         (json.dumps(_scope(task_id="dtask:newer").as_mapping(), sort_keys=True,
                                     separators=(",", ":")),))
    # The trigger tracked the direct newer insert, so the Python path sees it too.
    with pytest.raises(RoomArtifactError, match="generation is stale"):
        _put(outbox, _scope(task_id="dtask:newer"), _file(tmp_path))


def test_generation_fence_pruning_waits_for_artifact_rows(tmp_path: Path):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    retired, active = _scope(task_id="dtask:retired"), _scope(task_id="dtask:active")
    _put(outbox, retired, _file(tmp_path))
    outbox.discard_durably(retired)
    _put(outbox, active, _file(tmp_path))
    with sqlite3.connect(outbox.db_path) as conn:
        conn.execute("UPDATE hosted_room_output_generation_fences SET updated_at=0")
    assert outbox.prune_generation_fences(now=GENERATION_FENCE_RETENTION_SECONDS + 1) == 1
    with sqlite3.connect(outbox.db_path) as conn:
        identities = {row[0] for row in conn.execute("SELECT lineage_identity FROM hosted_room_output_generation_fences")}
    assert retired.lineage_json not in identities and active.lineage_json in identities


def test_supersede_unlink_failure_replays_without_losing_the_fence(tmp_path: Path, monkeypatch):
    outbox = RoomArtifactOutbox(tmp_path / "state.db")
    path = _file(tmp_path)
    first, second = _scope(execution_generation=1), _scope(execution_generation=2)
    blob = _blob(outbox, _put(outbox, first, path)["artifact_id"])
    original = Path.unlink
    failures = [True]

    def flaky(candidate, *args, **kwargs):
        if candidate == blob and failures:
            failures.pop()
            raise OSError("temporary supersede unlink fault")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky)
    with pytest.raises(OSError, match="temporary supersede unlink fault"):
        _put(outbox, second, path)
    with pytest.raises(RoomArtifactError, match="generation is stale"):
        _put(outbox, first, path)
    recovered = RoomArtifactOutbox(outbox.db_path)
    assert not blob.exists()
    assert recovered.list(first) == []
    assert _put(recovered, second, path)["name"] == "handoff.md"
