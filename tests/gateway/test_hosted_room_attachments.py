import os
import sqlite3
import threading

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_attachments import (
    AttachmentConflictError,
    AttachmentError,
    AttachmentIntegrityError,
    AttachmentNotFoundError,
    AttachmentQuotaError,
    HostedRoomAttachmentStore,
    MAX_TASK_ATTACHMENT_BYTES,
    MAX_TASK_ATTACHMENTS,
    UNCOMMITTED_TTL_SECONDS,
    decode_content_base64,
    validate_task_manifest,
)


PNG = b"\x89PNG\r\n\x1a\n" + b"image-payload"
PDF = b"%PDF-1.7\nattachment\n%%EOF\n"


def _put(store, *, upload_id="upload-1", data=PNG, **overrides):
    params = {
        "room_id": "room-1",
        "upload_id": upload_id,
        "kind": "image",
        "name": "diagram.png",
        "mime": "image/png",
        "data": data,
    }
    params.update(overrides)
    return store.put(**params)


def _manifest(*attachments):
    return [
        {
            "attachment_id": item["attachment_id"],
            "kind": item["kind"],
            "name": item["name"],
            "size": item["size"],
            "mime": item["mime"],
        }
        for item in attachments
    ]


def test_store_is_private_atomic_deduplicated_and_upload_idempotent(tmp_path):
    store = HostedRoomAttachmentStore(tmp_path / "state.db")

    first = _put(store)
    repeated = _put(store)
    second = _put(store, upload_id="upload-2")

    assert repeated["attachment_id"] == first["attachment_id"]
    assert repeated["idempotent"] is True
    assert second["attachment_id"] != first["attachment_id"]
    assert store.stats() == {
        "attachments": 2,
        "logical_bytes": len(PNG) * 2,
        "blobs": 1,
        "physical_bytes": len(PNG),
    }
    if os.name != "nt":
        assert store.root.stat().st_mode & 0o777 == 0o700
        assert store.blob_root.stat().st_mode & 0o777 == 0o700
        blob = next(path for path in store.blob_root.iterdir() if path.name.startswith("blob_"))
        assert blob.stat().st_mode & 0o777 == 0o600
    assert not list(store.blob_root.glob(".tmp-*"))


def test_orphan_sweep_serializes_with_concurrent_canonical_blob_commit(
    tmp_path,
    monkeypatch,
):
    store = HostedRoomAttachmentStore(tmp_path / "state.db")
    entered_sweep = threading.Event()
    release_sweep = threading.Event()
    writer_done = threading.Event()
    errors: list[BaseException] = []
    original_iterdir = type(store.blob_root).iterdir

    def blocking_iterdir(path):
        if (
            path == store.blob_root
            and threading.current_thread().name == "stale-sweep"
        ):
            entered_sweep.set()
            assert release_sweep.wait(5)
        return original_iterdir(path)

    monkeypatch.setattr(type(store.blob_root), "iterdir", blocking_iterdir)
    sweep_thread = threading.Thread(
        target=store._sweep_orphans,
        name="stale-sweep",
    )
    sweep_thread.start()
    assert entered_sweep.wait(5)

    def write_blob():
        try:
            _put(store)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            writer_done.set()

    writer_thread = threading.Thread(target=write_blob, name="blob-writer")
    writer_thread.start()
    assert not writer_done.wait(0.1)
    release_sweep.set()
    sweep_thread.join(5)
    writer_thread.join(5)

    assert not errors
    assert writer_done.is_set()
    assert _put(store)["idempotent"] is True
    assert len(list(store.blob_root.iterdir())) == 1


def test_upload_id_conflict_and_manifest_metadata_conflict_fail_closed(tmp_path):
    store = HostedRoomAttachmentStore(tmp_path / "state.db")
    attachment = _put(store)

    with pytest.raises(AttachmentConflictError):
        _put(store, data=PNG + b"different")

    manifest = _manifest(attachment)
    manifest[0]["name"] = "renamed.png"
    with pytest.raises(AttachmentConflictError):
        store.commit_message(
            room_id="room-1",
            event_id="event-1",
            manifest=manifest,
            recipient_member_ids=("research", "builder"),
        )


def test_failed_replay_rolls_back_only_attachments_transitioned_by_that_attempt(
    tmp_path,
):
    db = tmp_path / "state.db"
    store = HostedRoomAttachmentStore(db)
    committed = _put(store, upload_id="already-committed")
    new_upload = _put(store, upload_id="new-upload", data=PNG + b"new")
    store.commit_message(
        room_id="room-1",
        event_id="event-1",
        manifest=_manifest(committed),
        recipient_member_ids=("research",),
        hold_until_event=True,
    )

    _normalized, transitioned = store.commit_message_with_receipt(
        room_id="room-1",
        event_id="event-1",
        manifest=_manifest(committed, new_upload),
        recipient_member_ids=("research",),
        hold_until_event=True,
    )
    assert transitioned == (new_upload["attachment_id"],)

    assert store.abort_message_commit(
        room_id="room-1",
        event_id="event-1",
        attachment_ids=transitioned,
    ) == 1
    assert store.read(
        room_id="room-1",
        attachment_id=committed["attachment_id"],
        recipient_member_id="research",
    ).data == PNG
    conn = sqlite3.connect(db)
    states = dict(
        conn.execute(
            "SELECT upload_id, state FROM hosted_room_attachments ORDER BY upload_id"
        )
    )
    conn.close()
    assert states == {
        "already-committed": "committed",
        "new-upload": "uploaded",
    }


def test_image_pdf_text_and_binary_round_trip_with_frozen_recipients(tmp_path):
    store = HostedRoomAttachmentStore(tmp_path / "state.db")
    image = _put(store)
    pdf = _put(
        store,
        upload_id="upload-pdf",
        data=PDF,
        kind="pdf",
        name="brief.pdf",
        mime="application/pdf",
    )
    text = _put(
        store,
        upload_id="upload-text",
        data=b"release notes\n",
        kind="file",
        name="notes.txt",
        mime="text/plain",
    )
    binary = _put(
        store,
        upload_id="upload-bin",
        data=b"\x00\x01\x02\xff",
        kind="file",
        name="payload.bin",
        mime="application/octet-stream",
    )
    manifest = _manifest(image, pdf, text, binary)

    committed = store.commit_message(
        room_id="room-1",
        event_id="event-1",
        manifest=manifest,
        recipient_member_ids=("research", "builder"),
    )
    repeated = store.commit_message(
        room_id="room-1",
        event_id="event-1",
        manifest=manifest,
        recipient_member_ids=("research", "builder"),
    )

    assert committed == repeated == manifest
    assert store.read(
        room_id="room-1",
        attachment_id=image["attachment_id"],
        recipient_member_id="research",
        event_id="event-1",
    ).data == PNG
    assert store.read(
        room_id="room-1",
        attachment_id=pdf["attachment_id"],
        recipient_member_id="builder",
    ).data == PDF
    with pytest.raises(AttachmentNotFoundError):
        store.read(
            room_id="room-1",
            attachment_id=image["attachment_id"],
            recipient_member_id="late-member",
        )


@pytest.mark.parametrize(
    ("kind", "mime", "data"),
    [
        ("image", "image/png", b"not an image"),
        ("pdf", "application/pdf", b"not a pdf"),
        ("file", "text/plain", b"text\x00binary"),
        ("file", "image/png", PNG),
    ],
)
def test_mime_mismatches_are_rejected(tmp_path, kind, mime, data):
    store = HostedRoomAttachmentStore(tmp_path / "state.db")

    with pytest.raises(AttachmentError, match="MIME|mime|kind|bytes"):
        _put(store, data=data, kind=kind, mime=mime, name="claimed.dat")


@pytest.mark.parametrize(
    ("name", "mime"),
    [
        (
            "brief.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (
            "budget.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        (
            "deck.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
        ("book.epub", "application/epub+zip"),
        ("library.jar", "application/java-archive"),
        ("mobile.apk", "application/vnd.android.package-archive"),
    ],
)
def test_generic_zip_container_formats_are_accepted_as_opaque_files(
    tmp_path, name, mime
):
    store = HostedRoomAttachmentStore(tmp_path / "state.db")
    container = b"PK\x03\x04" + b"opaque-container-payload"

    attachment = _put(
        store,
        data=container,
        kind="file",
        name=name,
        mime=mime,
    )

    assert attachment["name"] == name
    assert attachment["mime"] == mime


def test_task_manifest_has_independent_count_and_byte_budgets():
    entry = {
        "attachment_id": "att_00000000000000000000000000000000",
        "kind": "file",
        "name": "part.bin",
        "size": 1,
        "mime": "application/octet-stream",
    }
    too_many = [
        {
            **entry,
            "attachment_id": f"att_{index:032x}",
            "name": f"part-{index}.bin",
        }
        for index in range(MAX_TASK_ATTACHMENTS + 1)
    ]
    with pytest.raises(AttachmentError, match="at most"):
        validate_task_manifest(too_many)

    too_large = [
        {
            **entry,
            "attachment_id": f"att_{index + 1:032x}",
            "name": f"chunk-{index}.bin",
            "size": 13_000_000,
        }
        for index in range(4)
    ]
    with pytest.raises(AttachmentError, match="aggregate byte limit"):
        validate_task_manifest(too_large)


def test_uncommitted_expiry_and_disband_grace_release_deduplicated_blob(tmp_path):
    now = [100.0]
    store = HostedRoomAttachmentStore(tmp_path / "state.db", clock=lambda: now[0])
    first = _put(store)
    second = _put(store, upload_id="upload-2")
    store.commit_message(
        room_id="room-1",
        event_id="event-1",
        manifest=_manifest(first),
        recipient_member_ids=("research",),
    )

    now[0] += UNCOMMITTED_TTL_SECONDS + 1
    assert store.prune() == 1
    assert store.stats()["blobs"] == 1
    assert store.stats()["attachments"] == 1

    assert store.mark_room_disbanded("room-1") == 1
    now[0] += 15 * 60 + 1
    assert store.prune() == 1
    assert store.stats()["blobs"] == 0
    assert not any(store.blob_root.iterdir())
    with pytest.raises(AttachmentNotFoundError):
        store.read(
            room_id="room-1",
            attachment_id=second["attachment_id"],
            recipient_member_id="research",
        )


def test_new_upload_reclaims_expired_uncommitted_quota_without_restart(tmp_path):
    now = [100.0]
    store = HostedRoomAttachmentStore(
        tmp_path / "state.db",
        clock=lambda: now[0],
        room_quota_bytes=len(PNG),
    )
    _put(store, upload_id="expired")
    now[0] += UNCOMMITTED_TTL_SECONDS + 1

    current = _put(store, upload_id="current")

    assert current["attachment_id"].startswith("att_")
    assert store.stats()["attachments"] == 1


def test_integrity_check_rejects_tampering_and_symlink_replacement(tmp_path):
    store = HostedRoomAttachmentStore(tmp_path / "state.db")
    attachment = _put(store)
    store.commit_message(
        room_id="room-1",
        event_id="event-1",
        manifest=_manifest(attachment),
        recipient_member_ids=("research",),
    )
    blob = next(path for path in store.blob_root.iterdir() if path.name.startswith("blob_"))
    blob.write_bytes(PNG + b"tampered")
    with pytest.raises(AttachmentIntegrityError):
        store.read(
            room_id="room-1",
            attachment_id=attachment["attachment_id"],
            recipient_member_id="research",
        )

    blob.unlink()
    target = tmp_path / "outside"
    target.write_bytes(PNG)
    try:
        blob.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(AttachmentIntegrityError):
        store.read(
            room_id="room-1",
            attachment_id=attachment["attachment_id"],
            recipient_member_id="research",
        )


def test_room_and_gateway_quotas_are_bounded(tmp_path):
    room_limited = HostedRoomAttachmentStore(
        tmp_path / "room.db", room_quota_bytes=len(PNG)
    )
    _put(room_limited)
    with pytest.raises(AttachmentQuotaError, match="room"):
        _put(room_limited, upload_id="upload-2")

    gateway_limited = HostedRoomAttachmentStore(
        tmp_path / "gateway.db", gateway_quota_bytes=len(PNG)
    )
    _put(gateway_limited)
    with pytest.raises(AttachmentQuotaError, match="gateway"):
        _put(gateway_limited, upload_id="upload-2", data=PNG + b"other")


def test_base64_decoder_is_bounded_and_rejects_data_urls():
    with pytest.raises(AttachmentError, match="data URL"):
        decode_content_base64("data:image/png;base64,AAAA")
    with pytest.raises(AttachmentError, match="valid base64"):
        decode_content_base64("not-base64")


def test_metadata_database_never_contains_blob_bytes_or_paths(tmp_path):
    db = tmp_path / "state.db"
    store = HostedRoomAttachmentStore(db)
    attachment = _put(store)
    store.commit_message(
        room_id="room-1",
        event_id="event-1",
        manifest=_manifest(attachment),
        recipient_member_ids=("research",),
    )

    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT * FROM hosted_room_attachments WHERE attachment_id=?",
        (attachment["attachment_id"],),
    ).fetchone()
    conn.close()
    encoded = repr(row)
    assert repr(PNG) not in encoded
    assert str(store.root) not in encoded


def test_restart_reconciles_event_commit_and_expires_abandoned_commit(tmp_path):
    now = [100.0]
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Release",
        members=[{"member_id": "research"}],
        authority_gateway_id="gateway-a",
        now=now[0],
    )
    store = HostedRoomAttachmentStore(db, clock=lambda: now[0])
    retained = _put(store, upload_id="retained")
    abandoned = _put(store, upload_id="abandoned", data=PNG + b"other")
    store.commit_message(
        room_id="room-1",
        event_id="event-retained",
        manifest=_manifest(retained),
        recipient_member_ids=("research",),
        viewer_access=True,
        hold_until_event=True,
    )
    store.commit_message(
        room_id="room-1",
        event_id="event-abandoned",
        manifest=_manifest(abandoned),
        recipient_member_ids=("research",),
        hold_until_event=True,
    )
    hosted_rooms.append_event(
        db,
        room_id="room-1",
        event_id="event-retained",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        payload={
            "text": "inspect",
            "thread_id": "thread-1",
            "attachments": _manifest(retained),
        },
        now=now[0],
    )

    now[0] += UNCOMMITTED_TTL_SECONDS + 1
    restarted = HostedRoomAttachmentStore(db, clock=lambda: now[0])

    assert restarted.read(
        room_id="room-1",
        attachment_id=retained["attachment_id"],
        recipient_member_id="research",
    ).data == PNG
    with pytest.raises(AttachmentNotFoundError):
        restarted.read(
            room_id="room-1",
            attachment_id=abandoned["attachment_id"],
            recipient_member_id="research",
        )
