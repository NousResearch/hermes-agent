"""Files stage privately under an exact task/attempt id and publish only with its settled reply.
Other profiles on this host use attested owner-transport chunks. Unpublished files lose their
hold at terminal publication and expire as abandoned uploads."""

from __future__ import annotations

import hashlib
import mimetypes
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping, Sequence

from gateway.hosted_room_attachments import (
    MAX_ATTACHMENTS_PER_MESSAGE,
    MAX_MESSAGE_ATTACHMENT_BYTES,
    OUTPUT_UPLOAD_PREFIX,
    AttachmentConflictError,
    AttachmentError,
    HostedRoomAttachmentStore,
)

_MEDIA_IMAGES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
# An attempt stages only on the profile process that runs it; serializing the per-attempt limit
# check with the put keeps concurrent tool calls of one turn inside one message's limits.
_STAGE_LOCK = threading.Lock()


class GroupFileError(ValueError):
    """A file the current turn cannot share; the message is safe to show the bot."""


@dataclass(frozen=True)
class HostedTurn:
    """The executing attempt and its store; transport routes another local profile to the owner."""

    room_id: str
    task_id: str
    execution_generation: int
    db_path: Path
    task: Mapping[str, Any] | None = None
    transport: Mapping[str, Any] | None = None


_TURN: ContextVar[HostedTurn | None] = ContextVar("hosted_room_turn", default=None)


def current_turn() -> HostedTurn | None:
    return _TURN.get()


def hosted_attempt(request_id: Any) -> tuple[dict[str, Any], int] | None:
    """``(task identity fields, execution_generation)`` of a hosted admission's request id, else None."""
    if not isinstance(request_id, str) or not request_id.startswith("hosted:"):
        return None
    try:
        task, generation = json.loads(request_id[len("hosted:"):])
        room_id, task_id = task["room_id"], task["task_id"]
    except (TypeError, ValueError, KeyError):
        return None
    if (not isinstance(room_id, str) or not room_id or not isinstance(task_id, str) or not task_id
            or type(generation) is not int or generation < 1):
        return None
    return task, generation


def hosted_turn_for(authority, ref, row) -> HostedTurn | None:
    """The hosted attempt an admission executes on this profile, with its owner transport if any."""
    attempt = hosted_attempt(row["request_id"])
    if attempt is None:
        return None
    task, generation = attempt
    from gateway.session_hosted_transport import _BINDING
    with authority.db._read_ctx() as conn:
        stored = conn.execute("SELECT value FROM state_meta WHERE key=?", (_BINDING + ref.session_id,)).fetchone()
    return HostedTurn(task["room_id"], task["task_id"], generation, Path(authority.db.db_path), task=task,
                      transport=json.loads(stored[0]) if stored is not None else None)


@contextmanager
def hosted_turn_scope(turn: HostedTurn | None) -> Iterator[HostedTurn | None]:
    """Expose the hosted attempt to this execution only (tools see it through copied contexts)."""
    token = _TURN.set(turn)
    try:
        yield turn
    finally:
        _TURN.reset(token)


def task_prefix(task_id: str) -> str:
    """The upload-id prefix of every attempt's shared files for one task."""
    return f"{OUTPUT_UPLOAD_PREFIX}{hashlib.sha256(task_id.encode()).hexdigest()[:32]}:"


def output_prefix(task_id: str, execution_generation: int) -> str:
    """The upload-id prefix of one attempt's shared files."""
    return f"{task_prefix(task_id)}{execution_generation}:"


def _upload_id(turn: HostedTurn, name: str) -> str:
    return output_prefix(turn.task_id, turn.execution_generation) + hashlib.sha256(
        name.encode("utf-8")).hexdigest()[:16]


def classify(name: str, data: bytes) -> tuple[str, str]:
    """Kind and MIME by content, exactly as the room's attachment store validates them."""
    from gateway.hosted_room_attachments import _sniff_known_mime, _validate_mime_and_kind
    sniffed = _sniff_known_mime(data)
    guessed = (mimetypes.guess_type(name)[0] or "application/octet-stream").lower()
    if sniffed == "application/pdf":
        kind, mime = "pdf", sniffed
    elif sniffed in _MEDIA_IMAGES or (sniffed == "image/bmp" and guessed == "image/bmp"):
        kind, mime = "image", sniffed
    elif guessed.startswith("image/") or guessed == "application/pdf":
        raise GroupFileError("That file's contents do not match its type.")
    else:
        kind, mime = "file", guessed
        if mime.startswith("text/") and sniffed != "text/plain":
            mime = "application/octet-stream"  # not clean UTF-8: opaque bytes, never text
    try:
        _validate_mime_and_kind(data, kind=kind, mime=mime)
    except AttachmentError as exc:
        raise GroupFileError(str(exc)) from exc
    return kind, mime


def _limit_refusal(staged: Sequence[Mapping[str, Any]], upload_id: str, size: int,
                   arriving: Sequence[int] = ()) -> str | None:
    """Why this attempt may not stage one more file, counting files still ``arriving`` in chunks."""
    if any(item["upload_id"] == upload_id for item in staged):
        return None  # the same file again
    if len(staged) + len(arriving) >= MAX_ATTACHMENTS_PER_MESSAGE:
        return f"A turn can share at most {MAX_ATTACHMENTS_PER_MESSAGE} files."
    if sum(item["size"] for item in staged) + sum(arriving) + size > MAX_MESSAGE_ATTACHMENT_BYTES:
        return f"A turn can share at most {MAX_MESSAGE_ATTACHMENT_BYTES // 1_000_000} MB of files."
    return None


def stage(turn: HostedTurn, *, name: str, data: bytes) -> dict[str, Any]:
    """Stage one file for this attempt; the same name and bytes again is the same file."""
    kind, mime = classify(name, data)
    if turn.transport is not None:
        return _push(turn, name=name, data=data)
    store = HostedRoomAttachmentStore(turn.db_path)
    upload_id = _upload_id(turn, name)
    with _STAGE_LOCK:
        staged = store.staged_uploads(
            room_id=turn.room_id, upload_prefix=output_prefix(turn.task_id, turn.execution_generation))
        if (refusal := _limit_refusal(staged, upload_id, len(data))) is not None:
            raise GroupFileError(refusal)
        try:
            return store.put(room_id=turn.room_id, upload_id=upload_id, kind=kind, name=name, mime=mime,
                             data=data, output=True)
        except AttachmentConflictError as exc:
            raise GroupFileError(f"A different file named {name} was already shared in this turn.") from exc
        except AttachmentError as exc:
            raise GroupFileError(str(exc)) from exc


def manifest(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: item[key] for key in ("attachment_id", "kind", "name", "size", "mime")} for item in items]


def bind_outputs(service, room: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    """Commit readable files before publication; name unreadable ones without stalling the reply.

    If the message was already logged, its manifest wins over later changes to staged bytes."""
    identity = task["identity"]
    prefix = output_prefix(identity.task_id, int(task["execution_generation"]))
    message_id = "dmessage:" + identity.task_id.removeprefix("dtask:")
    store = HostedRoomAttachmentStore(service.db_path)
    staged = manifest(store.staged_uploads(room_id=identity.room_id, upload_prefix=prefix, event_id=message_id))
    kept = store.logged_manifest(room_id=identity.room_id, event_id=message_id)
    if kept is not None:
        logged = {entry["attachment_id"] for entry in kept}
        unshared = [item["name"] for item in staged if item["attachment_id"] not in logged]
    else:
        author = task["payload"]["target_member_id"]
        recipients = [member["member_id"] for member in room["members"] if member["member_id"] != author]
        kept, unshared = [], []
        for item in staged:
            try:
                store.commit_message(room_id=identity.room_id, event_id=message_id, manifest=[item],
                                     recipient_member_ids=recipients, viewer_access=True)
            except AttachmentError:
                unshared.append(item["name"])
            else:
                kept.append(item)
    return {**({"attachments": kept} if kept else {}), **({"unshared": unshared} if unshared else {})}


def release_outputs(service, room: Mapping[str, Any], task: Mapping[str, Any]) -> None:
    """Release every unused attempt's hold after its task outcome is logged."""
    store = HostedRoomAttachmentStore(service.db_path)
    prefix = task_prefix(task["identity"].task_id)
    store.release_staged(room_id=room["room_id"], upload_prefix=prefix)
    # A transfer from another profile still arriving for this task can no longer be published.
    with _INCOMING_LOCK:
        for part in store.incoming_root.glob(_part_prefix(prefix) + "*.part"):
            _PARTS.pop(part, None)
            part.unlink(missing_ok=True)


# Raw bytes per owner-transport request: base64 plus the attested envelope stays under the
# private socket's 64 KiB request line (gateway.session_hosted_transport.owner_request).
_PUSH_CHUNK_BYTES = 40 * 1024
_SHA256_HEX = frozenset("0123456789abcdef")
_INCOMING_LOCK = threading.Lock()
# (device, inode) of each incoming part file this process created, by path: a later chunk appends
# only to that very file, never to a link or file swapped in under its name between chunks.
_PARTS: dict[Path, tuple[int, int]] = {}


def _push(turn: HostedTurn, *, name: str, data: bytes) -> dict[str, Any]:
    """Send chunks attested against the exact running attempt, member and profile."""
    import base64
    from gateway.session_hosted_transport import _attest
    from hermes_state_runtime import RuntimeStoreError
    digest = hashlib.sha256(data).hexdigest()
    params = {"task": dict(turn.task or {}), "execution_generation": turn.execution_generation,
              "name": name, "size": len(data), "sha256": digest}
    result: Mapping[str, Any] = {}
    try:
        for offset in range(0, len(data), _PUSH_CHUNK_BYTES):
            result = _attest(turn.transport, "output", {
                **params, "offset": offset,
                "data_base64": base64.b64encode(data[offset:offset + _PUSH_CHUNK_BYTES]).decode("ascii")})
            if result.get("refused"):
                raise GroupFileError(str(result["refused"]))
    except RuntimeStoreError as exc:
        raise GroupFileError("The group chat could not receive that file. Try again.") from exc
    stored = result.get("stored")
    if not isinstance(stored, Mapping) or stored.get("sha256") != digest:
        raise GroupFileError("The group chat could not receive that file. Try again.")
    return dict(stored)


def receive_output_chunk(service, task: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Append an attested chunk; the final chunk stages the file or returns a refusal.

    CanonicalHostedRoomService.attest has verified the running attempt, member and profile."""
    import base64
    import binascii
    import os
    from hermes_dir_acl import make_user_dir
    from hermes_state_runtime import RuntimeStoreError
    name, size, digest, offset = (params.get(key) for key in ("name", "size", "sha256", "offset"))
    try:
        chunk = base64.b64decode(params.get("data_base64") or "", validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise RuntimeStoreError("invalid_params") from exc
    from gateway.hosted_room_attachments import MAX_ATTACHMENT_BYTES
    if (not isinstance(name, str) or type(size) is not int or not 0 < size <= MAX_ATTACHMENT_BYTES
            or not isinstance(digest, str) or len(digest) != 64 or not set(digest) <= _SHA256_HEX
            or type(offset) is not int or offset < 0 or not chunk or len(chunk) > _PUSH_CHUNK_BYTES
            or offset + len(chunk) > size):
        raise RuntimeStoreError("invalid_params")
    identity = task["identity"]
    turn = HostedTurn(identity.room_id, identity.task_id, int(task["execution_generation"]), Path(service.db_path))
    upload_id = _upload_id(turn, name)
    store = HostedRoomAttachmentStore(service.db_path)
    make_user_dir(store.incoming_root, parents=True, exist_ok=True, mode=0o700)
    part = store.incoming_root / f"{_part_prefix(upload_id)}-{digest[:16]}.part"
    flags = os.O_RDWR | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    with _INCOMING_LOCK:
        if offset == 0:
            if (refusal := _arrival_refusal(store, turn, upload_id, size, part)) is not None:
                return {"refused": refusal}
            part.unlink(missing_ok=True)
            _PARTS.pop(part, None)
            flags |= os.O_CREAT | os.O_EXCL
        elif part not in _PARTS:
            raise RuntimeStoreError("invalid_params")
        try:
            descriptor = os.open(part, flags, 0o600)
        except OSError as exc:  # a link or another file swapped in under the part's name
            _PARTS.pop(part, None)
            raise RuntimeStoreError("invalid_params") from exc
        with os.fdopen(descriptor, "r+b") as handle:
            info = os.fstat(descriptor)
            created = _PARTS.setdefault(part, (info.st_dev, info.st_ino))
            # Append only to the file this transfer's first chunk created, exactly as far as it got.
            if (info.st_dev, info.st_ino) != created or info.st_nlink != 1 or info.st_size != offset:
                _PARTS.pop(part, None)
                raise RuntimeStoreError("invalid_params")
            handle.write(chunk)
            if offset + len(chunk) < size:
                return {"received": offset + len(chunk)}
            handle.flush()
            handle.seek(0)
            data = handle.read(size + 1)
        _PARTS.pop(part, None)
        part.unlink(missing_ok=True)
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeStoreError("invalid_params")
    try:
        return {"stored": stage(turn, name=name, data=data)}
    except GroupFileError as exc:
        return {"refused": str(exc)}


def _arrival_refusal(store, turn: HostedTurn, upload_id: str, size: int, part: Path) -> str | None:
    """The attempt's limits at a new transfer's first chunk, counting its other transfers in flight."""
    prefix = output_prefix(turn.task_id, turn.execution_generation)
    arriving = []
    for other in store.incoming_root.glob(_part_prefix(prefix) + "*.part"):
        try:
            if other != part:
                arriving.append(other.stat().st_size)
        except OSError:
            continue
    return _limit_refusal(store.staged_uploads(room_id=turn.room_id, upload_prefix=prefix), upload_id, size,
                          arriving)


def _part_prefix(upload_id_prefix: str) -> str:
    """A chunked transfer's file name starts with its upload id, with ``:`` made filename-safe."""
    return upload_id_prefix.replace(":", "-")
