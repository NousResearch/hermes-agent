#!/usr/bin/env python3
"""Private crash-safe publication boundary for owner-authored source artifacts.

The two supported artifacts have fixed production destinations.  A complete
candidate is first published inside an owner-only transaction directory, then
hard-linked into the final name without replacement.  The final name is
therefore always absent or complete, and a process death after collection can
resume from the immutable candidate without repeating live collection.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn


_JOURNAL_DIRECTORY = ".source-artifact-transactions"
_HISTORY_DIRECTORY = ".source-artifact-history"
_DIRECT_KIND = "direct-iam-v1"
_HOST_KIND = "owner-gate-host-v2"
_HOST_V3_KIND = "owner-gate-host-v3"
_DIRECT_RELATIVE = Path(
    ".hermes/trusted/owner-gate-direct-iam-identity-authority-v1.json"
)
_HOST_RELATIVE = Path(
    ".hermes/trusted/owner-gate-iap-host-identity-v2.json"
)
_HOST_V3_RELATIVE = Path(
    ".hermes/trusted/owner-gate-iap-host-identity-v3.json"
)
_DIRECT_MODE = 0o400
_HOST_MODE = 0o600
_JOURNAL_MODE = 0o700
_TRANSITION_MODE = 0o600
_MAX_TRANSITION_BYTES = 256 * 1024
_TRANSACTION_ID = re.compile(r"^[0-9a-f]{64}$")
_HISTORY_ENTRY = re.compile(r"^([0-9a-f]{64})\.bin$")
_SCRATCH = re.compile(
    r"^\.(intent|candidate|success)\.([0-9a-f]{32})\.scratch$"
)
_FIXED_ENTRIES = frozenset({"intent.json", "candidate.bin", "success.json"})
_ROTATION_INTENT_SCHEMA = "muncho-source-artifact-publication-intent.v2"
_ROTATION_SUCCESS_SCHEMA = "muncho-source-artifact-publication-success.v2"
_ROTATION_SCHEMA = "muncho-source-artifact-rotation.v1"


class _SourceArtifactPublicationError(RuntimeError):
    pass


def _error(code: str, exc: BaseException | None = None) -> NoReturn:
    del exc
    raise _SourceArtifactPublicationError(code) from None


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        _error("source_artifact_publication_json_invalid", exc)


def _decode_canonical(raw: bytes) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for name, value in items:
            if not isinstance(name, str) or name in result:
                raise ValueError("duplicate key")
            result[name] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        _error("source_artifact_publication_json_invalid", exc)
    if not isinstance(value, Mapping) or _canonical(value) != raw:
        _error("source_artifact_publication_json_invalid")
    return dict(value)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        _error("source_artifact_publication_directory_fsync_failed", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_directory(path: Path) -> tuple[int, int]:
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
    except OSError as exc:
        _error("source_artifact_publication_directory_invalid", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        or before.st_uid != os.geteuid()  # windows-footgun: ok — POSIX owner boundary
        or before.st_gid != os.getegid()  # windows-footgun: ok — POSIX owner boundary
        or stat.S_IMODE(before.st_mode) != _JOURNAL_MODE
    ):
        _error("source_artifact_publication_directory_invalid")
    return before.st_dev, before.st_ino


def _ensure_directory(path: Path, *, parent: Path) -> None:
    _validate_directory(parent)
    if path.parent != parent or not path.is_absolute() or ".." in path.parts:
        _error("source_artifact_publication_directory_invalid")
    try:
        os.mkdir(path, _JOURNAL_MODE)
    except FileExistsError:
        pass
    except OSError as exc:
        _error("source_artifact_publication_directory_invalid", exc)
    _validate_directory(path)
    _fsync_directory(parent)


def _read_file(
    path: Path,
    *,
    maximum: int,
    mode: int,
    links: frozenset[int] = frozenset({1}),
) -> tuple[bytes, tuple[int, ...]]:
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_gid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.geteuid()  # windows-footgun: ok — POSIX owner boundary
            or before.st_gid != os.getegid()  # windows-footgun: ok — POSIX owner boundary
            or stat.S_IMODE(before.st_mode) != mode
            or before.st_nlink not in links
            or not 0 < before.st_size <= maximum
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_uid,
                opened.st_gid,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            != identity
        ):
            _error("source_artifact_publication_file_invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _error("source_artifact_publication_file_invalid")
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) != before.st_size or os.read(descriptor, 1):
            _error("source_artifact_publication_file_invalid")
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != identity:
            _error("source_artifact_publication_file_changed")
        return raw, identity
    except _SourceArtifactPublicationError:
        raise
    except OSError as exc:
        _error("source_artifact_publication_file_invalid", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_complete_scratch(
    path: Path,
    raw: bytes,
    *,
    mode: int,
    checkpoint: _Checkpoint | None = None,
    checkpoint_name: str,
) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        os.fchmod(descriptor, mode)
        if checkpoint is not None:
            checkpoint(checkpoint_name)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _error("source_artifact_publication_write_stalled")
            offset += written
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()  # windows-footgun: ok — POSIX owner boundary
            or opened.st_gid != os.getegid()  # windows-footgun: ok — POSIX owner boundary
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != mode
            or opened.st_size != len(raw)
        ):
            _error("source_artifact_publication_scratch_invalid")
    except _SourceArtifactPublicationError:
        raise
    except OSError as exc:
        _error("source_artifact_publication_write_failed", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_no_replace(
    directory: Path,
    name: str,
    raw: bytes,
    *,
    mode: int,
    checkpoint: _Checkpoint | None = None,
) -> None:
    final = directory / name
    if os.path.lexists(final):
        existing, _identity = _read_file(
            final,
            maximum=max(len(raw), _MAX_TRANSITION_BYTES),
            mode=mode,
        )
        if existing != raw:
            _error("source_artifact_publication_diverged")
        return
    scratch = directory / f".{name.split('.')[0]}.{secrets.token_hex(16)}.scratch"
    _write_complete_scratch(
        scratch,
        raw,
        mode=mode,
        checkpoint=checkpoint,
        checkpoint_name=f"after_{name.split('.')[0]}_scratch_open",
    )
    _fsync_directory(directory)
    try:
        os.link(scratch, final, follow_symlinks=False)
        _fsync_directory(directory)
        linked_raw, linked_identity = _read_file(
            final,
            maximum=max(len(raw), _MAX_TRANSITION_BYTES),
            mode=mode,
            links=frozenset({2}),
        )
        scratch_raw, scratch_identity = _read_file(
            scratch,
            maximum=max(len(raw), _MAX_TRANSITION_BYTES),
            mode=mode,
            links=frozenset({2}),
        )
        if (
            linked_raw != raw
            or scratch_raw != raw
            or linked_identity[:2] != scratch_identity[:2]
        ):
            _error("source_artifact_publication_diverged")
        os.unlink(scratch)
        _fsync_directory(directory)
        reread, _identity = _read_file(
            final,
            maximum=max(len(raw), _MAX_TRANSITION_BYTES),
            mode=mode,
        )
        if reread != raw:
            _error("source_artifact_publication_diverged")
    except _SourceArtifactPublicationError:
        raise
    except FileExistsError as exc:
        _error("source_artifact_publication_diverged", exc)
    except OSError as exc:
        _error("source_artifact_publication_write_failed", exc)


def _scratch_mode(name: str, *, artifact_mode: int) -> int:
    match = _SCRATCH.fullmatch(name)
    if match is None:
        _error("source_artifact_publication_inventory_invalid")
    return artifact_mode if match.group(1) == "candidate" else _TRANSITION_MODE


def _validate_inventory(directory: Path) -> tuple[str, ...]:
    try:
        entries = tuple(sorted(os.listdir(directory)))
    except OSError as exc:
        _error("source_artifact_publication_inventory_invalid", exc)
    if (
        len(entries) > 16
        or any(
            name not in _FIXED_ENTRIES and _SCRATCH.fullmatch(name) is None
            for name in entries
        )
    ):
        _error("source_artifact_publication_inventory_invalid")
    return entries


def _validate_journal_inventory(journal_root: Path, kind_root: Path) -> None:
    try:
        purposes = tuple(sorted(os.listdir(journal_root)))
        transactions = tuple(sorted(os.listdir(kind_root)))
    except OSError as exc:
        _error("source_artifact_publication_inventory_invalid", exc)
    if (
        len(purposes) > 3
        or any(
            name not in {_DIRECT_KIND, _HOST_KIND, _HOST_V3_KIND}
            for name in purposes
        )
        or len(transactions) > 128
        or any(_TRANSACTION_ID.fullmatch(name) is None for name in transactions)
    ):
        _error("source_artifact_publication_inventory_invalid")
    for name in purposes:
        _validate_directory(journal_root / name)
    for name in transactions:
        transaction = kind_root / name
        _validate_directory(transaction)
        _validate_inventory(transaction)


def _clean_scratch(
    directory: Path,
    *,
    artifact_mode: int,
    maximum: int,
) -> None:
    changed = False
    for name in _validate_inventory(directory):
        if _SCRATCH.fullmatch(name) is None:
            continue
        scratch = directory / name
        mode = _scratch_mode(name, artifact_mode=artifact_mode)
        try:
            before = os.lstat(scratch)
            descriptor = os.open(
                scratch,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened = os.fstat(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            _error("source_artifact_publication_scratch_invalid", exc)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.geteuid()  # windows-footgun: ok — POSIX owner boundary
        or before.st_gid != os.getegid()  # windows-footgun: ok — POSIX owner boundary
            or stat.S_IMODE(before.st_mode) != mode
            or before.st_nlink not in {1, 2}
            or (before.st_dev, before.st_ino, before.st_mode, before.st_nlink)
            != (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink)
            or before.st_size
            > (maximum if mode == artifact_mode else _MAX_TRANSITION_BYTES)
        ):
            _error("source_artifact_publication_scratch_invalid")
        base = name.split(".")[1]
        peer_name = {
            "intent": "intent.json",
            "candidate": "candidate.bin",
            "success": "success.json",
        }[base]
        peer = directory / peer_name
        if before.st_nlink == 2:
            if not os.path.lexists(peer):
                _error("source_artifact_publication_scratch_link_invalid")
            raw, identity = _read_file(
                scratch,
                maximum=(
                    maximum if mode == artifact_mode else _MAX_TRANSITION_BYTES
                ),
                mode=mode,
                links=frozenset({2}),
            )
            peer_raw, peer_identity = _read_file(
                peer,
                maximum=maximum if base == "candidate" else _MAX_TRANSITION_BYTES,
                mode=mode,
                links=frozenset({2}),
            )
            if peer_raw != raw or peer_identity[:2] != identity[:2]:
                _error("source_artifact_publication_scratch_link_invalid")
        try:
            os.unlink(scratch)
        except OSError as exc:
            _error("source_artifact_publication_scratch_cleanup_failed", exc)
        changed = True
    if changed:
        _fsync_directory(directory)


@dataclass(frozen=True)
class _ValidatedArtifact:
    value: Any
    logical_sha256: str


@dataclass(frozen=True)
class _PublicationResult:
    value: Any
    raw: bytes
    logical_sha256: str
    file_sha256: str
    path: str
    replayed: bool


_Validator = Callable[[bytes], _ValidatedArtifact]
_Collector = Callable[[], bytes]
_Checkpoint = Callable[[str], None]


def _validate_artifact(
    validator: _Validator,
    raw: bytes,
) -> _ValidatedArtifact:
    try:
        value = validator(raw)
    except _SourceArtifactPublicationError:
        raise
    except Exception as exc:
        _error("source_artifact_publication_artifact_invalid", exc)
    if (
        type(value) is not _ValidatedArtifact
        or _TRANSACTION_ID.fullmatch(value.logical_sha256 or "") is None
    ):
        _error("source_artifact_publication_artifact_invalid")
    return value


def _transaction_id(kind: str, chain: Mapping[str, Any]) -> str:
    return _sha256(_canonical({"kind": kind, "chain": dict(chain)}))


@dataclass(frozen=True)
class _CompletedPublication:
    transaction_id: str
    chain_sha256: str
    artifact_logical_sha256: str
    artifact_file_sha256: str


def _rotation_archive_path(
    final_parent: Path,
    *,
    artifact_file_sha256: str,
) -> Path:
    if _TRANSACTION_ID.fullmatch(artifact_file_sha256 or "") is None:
        _error("source_artifact_rotation_predecessor_invalid")
    return (
        final_parent
        / _HISTORY_DIRECTORY
        / _DIRECT_KIND
        / f"{artifact_file_sha256}.bin"
    )


def _validate_rotation_record(
    value: Any,
    *,
    final_parent: Path,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "predecessor_transaction_id",
        "predecessor_chain_sha256",
        "predecessor_artifact_logical_sha256",
        "predecessor_artifact_file_sha256",
        "predecessor_archive_path",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _error("source_artifact_rotation_record_invalid")
    checked = dict(value)
    digests = (
        checked.get("predecessor_transaction_id"),
        checked.get("predecessor_chain_sha256"),
        checked.get("predecessor_artifact_logical_sha256"),
        checked.get("predecessor_artifact_file_sha256"),
    )
    archive = _rotation_archive_path(
        final_parent,
        artifact_file_sha256=str(
            checked.get("predecessor_artifact_file_sha256", "")
        ),
    )
    if (
        checked.get("schema") != _ROTATION_SCHEMA
        or any(_TRANSACTION_ID.fullmatch(str(item or "")) is None for item in digests)
        or checked.get("predecessor_archive_path") != str(archive)
    ):
        _error("source_artifact_rotation_record_invalid")
    return checked


def _completed_publication(
    transaction_root: Path,
    *,
    kind: str,
    final: Path,
    artifact_mode: int,
) -> _CompletedPublication | None:
    intent_path = transaction_root / "intent.json"
    success_path = transaction_root / "success.json"
    if not os.path.lexists(intent_path) or not os.path.lexists(success_path):
        return None
    intent_raw, _intent_identity = _read_file(
        intent_path,
        maximum=_MAX_TRANSITION_BYTES,
        mode=_TRANSITION_MODE,
    )
    success_raw, _success_identity = _read_file(
        success_path,
        maximum=_MAX_TRANSITION_BYTES,
        mode=_TRANSITION_MODE,
    )
    intent = _decode_canonical(intent_raw)
    success = _decode_canonical(success_raw)
    base_intent_fields = {
        "schema",
        "purpose",
        "transaction_id",
        "final_path",
        "artifact_mode",
        "chain",
        "chain_sha256",
    }
    rotation: Mapping[str, Any] | None = None
    if intent.get("schema") == _ROTATION_INTENT_SCHEMA:
        if set(intent) != base_intent_fields | {"rotation"}:
            _error("source_artifact_publication_intent_diverged")
        rotation = _validate_rotation_record(
            intent.get("rotation"),
            final_parent=final.parent,
        )
    elif (
        intent.get("schema") != "muncho-source-artifact-publication-intent.v1"
        or set(intent) != base_intent_fields
    ):
        _error("source_artifact_publication_intent_diverged")
    chain = intent.get("chain")
    if not isinstance(chain, Mapping) or not chain:
        _error("source_artifact_publication_intent_diverged")
    transaction_id = _transaction_id(kind, chain)
    if (
        transaction_root.name != transaction_id
        or intent.get("purpose") != kind
        or intent.get("transaction_id") != transaction_id
        or intent.get("final_path") != str(final)
        or intent.get("artifact_mode") != artifact_mode
        or intent.get("chain_sha256") != _sha256(_canonical(chain))
    ):
        _error("source_artifact_publication_intent_diverged")
    base_success = {
        "schema",
        "purpose",
        "transaction_id",
        "intent_sha256",
        "final_path",
        "artifact_logical_sha256",
        "artifact_file_sha256",
    }
    if rotation is None:
        if (
            success.get("schema")
            != "muncho-source-artifact-publication-success.v1"
            or set(success) != base_success
        ):
            _error("source_artifact_publication_success_invalid")
    elif (
        success.get("schema") != _ROTATION_SUCCESS_SCHEMA
        or set(success) != base_success | {"rotation"}
        or success.get("rotation") != rotation
    ):
        _error("source_artifact_publication_success_invalid")
    logical_sha256 = str(success.get("artifact_logical_sha256", ""))
    file_sha256 = str(success.get("artifact_file_sha256", ""))
    if (
        success.get("purpose") != kind
        or success.get("transaction_id") != transaction_id
        or success.get("intent_sha256") != _sha256(intent_raw)
        or success.get("final_path") != str(final)
        or _TRANSACTION_ID.fullmatch(logical_sha256) is None
        or _TRANSACTION_ID.fullmatch(file_sha256) is None
    ):
        _error("source_artifact_publication_success_invalid")
    return _CompletedPublication(
        transaction_id=transaction_id,
        chain_sha256=str(intent["chain_sha256"]),
        artifact_logical_sha256=logical_sha256,
        artifact_file_sha256=file_sha256,
    )


def _discover_rotation_predecessor(
    *,
    kind_root: Path,
    current_transaction_id: str,
    final: Path,
    artifact_mode: int,
    maximum: int,
    expected_file_sha256: str,
) -> tuple[_CompletedPublication, bytes, tuple[int, ...]]:
    if _TRANSACTION_ID.fullmatch(expected_file_sha256 or "") is None:
        _error("source_artifact_rotation_predecessor_invalid")
    final_raw, final_identity = _read_file(
        final,
        maximum=maximum,
        mode=artifact_mode,
    )
    if _sha256(final_raw) != expected_file_sha256:
        _error("source_artifact_rotation_predecessor_mismatch")
    matches: list[_CompletedPublication] = []
    try:
        names = tuple(sorted(os.listdir(kind_root)))
    except OSError as exc:
        _error("source_artifact_publication_inventory_invalid", exc)
    for name in names:
        if name == current_transaction_id:
            continue
        completed = _completed_publication(
            kind_root / name,
            kind=_DIRECT_KIND,
            final=final,
            artifact_mode=artifact_mode,
        )
        if (
            completed is not None
            and completed.artifact_file_sha256 == expected_file_sha256
        ):
            matches.append(completed)
    if len(matches) != 1:
        _error("source_artifact_rotation_predecessor_unproven")
    return matches[0], final_raw, final_identity


def _rotation_record(
    completed: _CompletedPublication,
    *,
    final_parent: Path,
) -> Mapping[str, Any]:
    archive = _rotation_archive_path(
        final_parent,
        artifact_file_sha256=completed.artifact_file_sha256,
    )
    return {
        "schema": _ROTATION_SCHEMA,
        "predecessor_transaction_id": completed.transaction_id,
        "predecessor_chain_sha256": completed.chain_sha256,
        "predecessor_artifact_logical_sha256": (
            completed.artifact_logical_sha256
        ),
        "predecessor_artifact_file_sha256": completed.artifact_file_sha256,
        "predecessor_archive_path": str(archive),
    }


def _validate_history_inventory(kind_history: Path) -> None:
    try:
        names = tuple(sorted(os.listdir(kind_history)))
    except OSError as exc:
        _error("source_artifact_history_inventory_invalid", exc)
    if (
        len(names) > 128
        or any(_HISTORY_ENTRY.fullmatch(name) is None for name in names)
    ):
        _error("source_artifact_history_inventory_invalid")


def _run(
    *,
    kind: str,
    owner_home: Path,
    relative: Path,
    artifact_mode: int,
    maximum: int,
    chain: Mapping[str, Any],
    validator: _Validator,
    collector: _Collector,
    checkpoint: _Checkpoint | None,
    recovery_only: bool,
) -> _PublicationResult:
    if (
        kind not in {_DIRECT_KIND, _HOST_KIND, _HOST_V3_KIND}
        or not isinstance(owner_home, Path)
        or not owner_home.is_absolute()
        or os.path.realpath(owner_home) != str(owner_home)
        or relative
        not in {_DIRECT_RELATIVE, _HOST_RELATIVE, _HOST_V3_RELATIVE}
        or artifact_mode not in {_DIRECT_MODE, _HOST_MODE}
        or type(maximum) is not int
        or not 0 < maximum <= 16 * 1024 * 1024
        or not isinstance(chain, Mapping)
        or not chain
        or not callable(validator)
        or not callable(collector)
        or (checkpoint is not None and not callable(checkpoint))
        or type(recovery_only) is not bool
    ):
        _error("source_artifact_publication_contract_invalid")
    expected_relative = (
        _DIRECT_RELATIVE
        if kind == _DIRECT_KIND
        else _HOST_V3_RELATIVE
        if kind == _HOST_V3_KIND
        else _HOST_RELATIVE
    )
    expected_mode = _DIRECT_MODE if kind == _DIRECT_KIND else _HOST_MODE
    if relative != expected_relative or artifact_mode != expected_mode:
        _error("source_artifact_publication_contract_invalid")
    final = owner_home / relative
    final_parent = final.parent
    if os.path.realpath(final_parent) != str(final_parent):
        _error("source_artifact_publication_directory_invalid")
    _validate_directory(final_parent)
    transaction_id = _transaction_id(kind, chain)
    if _TRANSACTION_ID.fullmatch(transaction_id) is None:
        _error("source_artifact_publication_transaction_invalid")
    journal_root = final_parent / _JOURNAL_DIRECTORY
    _ensure_directory(journal_root, parent=final_parent)
    kind_root = journal_root / kind
    _ensure_directory(kind_root, parent=journal_root)
    transaction_root = kind_root / transaction_id
    _ensure_directory(transaction_root, parent=kind_root)
    lock_descriptor: int | None = None
    try:
        lock_descriptor = os.open(
            kind_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        _validate_directory(kind_root)
        _validate_directory(transaction_root)
        _validate_journal_inventory(journal_root, kind_root)
        _clean_scratch(
            transaction_root,
            artifact_mode=artifact_mode,
            maximum=maximum,
        )
        intent_path = transaction_root / "intent.json"
        candidate_path = transaction_root / "candidate.bin"
        success_path = transaction_root / "success.json"
        final_exists = os.path.lexists(final)
        intent_exists = os.path.lexists(intent_path)
        if final_exists and not intent_exists:
            _error("source_artifact_publication_final_without_intent")
        if recovery_only and (
            not intent_exists
            or not any(
                os.path.lexists(path)
                for path in (candidate_path, final, success_path)
            )
        ):
            _error("source_artifact_publication_recovery_unavailable")
        intent = {
            "schema": "muncho-source-artifact-publication-intent.v1",
            "purpose": kind,
            "transaction_id": transaction_id,
            "final_path": str(final),
            "artifact_mode": artifact_mode,
            "chain": dict(chain),
            "chain_sha256": _sha256(_canonical(chain)),
        }
        intent_raw = _canonical(intent)
        if intent_exists:
            existing_intent, _identity = _read_file(
                intent_path,
                maximum=_MAX_TRANSITION_BYTES,
                mode=_TRANSITION_MODE,
            )
            if existing_intent != intent_raw or _decode_canonical(existing_intent) != intent:
                _error("source_artifact_publication_intent_diverged")
        else:
            _publish_no_replace(
                transaction_root,
                "intent.json",
                intent_raw,
                mode=_TRANSITION_MODE,
                checkpoint=checkpoint,
            )
        if checkpoint is not None:
            checkpoint("after_intent")

        candidate: _ValidatedArtifact | None = None
        candidate_raw: bytes | None = None
        if os.path.lexists(candidate_path):
            candidate_raw, _identity = _read_file(
                candidate_path,
                maximum=maximum,
                mode=artifact_mode,
                links=frozenset({1, 2}),
            )
            candidate = _validate_artifact(validator, candidate_raw)
        final_value: _ValidatedArtifact | None = None
        final_raw: bytes | None = None
        if os.path.lexists(final):
            final_raw, final_identity = _read_file(
                final,
                maximum=maximum,
                mode=artifact_mode,
                links=frozenset({1, 2}),
            )
            final_value = _validate_artifact(validator, final_raw)
            if candidate_raw is not None:
                _candidate_raw, candidate_identity = _read_file(
                    candidate_path,
                    maximum=maximum,
                    mode=artifact_mode,
                    links=frozenset({2}),
                )
                if (
                    final_raw != candidate_raw
                    or final_identity[:2] != candidate_identity[:2]
                ):
                    _error("source_artifact_publication_final_diverged")
        success: Mapping[str, Any] | None = None
        if os.path.lexists(success_path):
            success_raw, _identity = _read_file(
                success_path,
                maximum=_MAX_TRANSITION_BYTES,
                mode=_TRANSITION_MODE,
            )
            success = _decode_canonical(success_raw)
        if success is not None:
            if final_value is None or final_raw is None or candidate_raw is not None:
                _error("source_artifact_publication_success_invalid")
            final_raw, _identity = _read_file(
                final,
                maximum=maximum,
                mode=artifact_mode,
            )
            final_value = _validate_artifact(validator, final_raw)
            expected_success = {
                "schema": "muncho-source-artifact-publication-success.v1",
                "purpose": kind,
                "transaction_id": transaction_id,
                "intent_sha256": _sha256(intent_raw),
                "final_path": str(final),
                "artifact_logical_sha256": final_value.logical_sha256,
                "artifact_file_sha256": _sha256(final_raw),
            }
            if success != expected_success:
                _error("source_artifact_publication_success_invalid")
            return _PublicationResult(
                value=final_value.value,
                raw=final_raw,
                logical_sha256=final_value.logical_sha256,
                file_sha256=_sha256(final_raw),
                path=str(final),
                replayed=True,
            )

        replayed = candidate is not None or final_value is not None
        if final_value is None:
            if candidate is None or candidate_raw is None:
                if recovery_only:
                    _error("source_artifact_publication_recovery_unavailable")
                collected_raw = collector()
                if (
                    type(collected_raw) is not bytes
                    or not 0 < len(collected_raw) <= maximum
                ):
                    _error("source_artifact_publication_collector_invalid")
                candidate = _validate_artifact(validator, collected_raw)
                candidate_raw = collected_raw
                if checkpoint is not None:
                    checkpoint("after_collection")
                _publish_no_replace(
                    transaction_root,
                    "candidate.bin",
                    candidate_raw,
                    mode=artifact_mode,
                    checkpoint=checkpoint,
                )
                if checkpoint is not None:
                    checkpoint("after_candidate")
            try:
                os.link(candidate_path, final, follow_symlinks=False)
            except FileExistsError:
                final_raw, final_identity = _read_file(
                    final,
                    maximum=maximum,
                    mode=artifact_mode,
                    links=frozenset({1, 2}),
                )
                final_value = _validate_artifact(validator, final_raw)
                _candidate_raw, candidate_identity = _read_file(
                    candidate_path,
                    maximum=maximum,
                    mode=artifact_mode,
                    links=frozenset({1, 2}),
                )
                if (
                    final_raw != candidate_raw
                    or final_identity[:2] != candidate_identity[:2]
                ):
                    _error("source_artifact_publication_final_diverged")
            except OSError as exc:
                _error("source_artifact_publication_final_write_failed", exc)
            else:
                _fsync_directory(final_parent)
                final_raw, final_identity = _read_file(
                    final,
                    maximum=maximum,
                    mode=artifact_mode,
                    links=frozenset({2}),
                )
                final_value = _validate_artifact(validator, final_raw)
                _candidate_raw, candidate_identity = _read_file(
                    candidate_path,
                    maximum=maximum,
                    mode=artifact_mode,
                    links=frozenset({2}),
                )
                if (
                    final_raw != candidate_raw
                    or final_identity[:2] != candidate_identity[:2]
                ):
                    _error("source_artifact_publication_final_diverged")
            if checkpoint is not None:
                checkpoint("after_final_link")
        if final_value is None or final_raw is None:
            _error("source_artifact_publication_final_invalid")
        if os.path.lexists(candidate_path):
            try:
                os.unlink(candidate_path)
            except OSError as exc:
                _error("source_artifact_publication_candidate_cleanup_failed", exc)
            _fsync_directory(transaction_root)
        final_raw, _identity = _read_file(
            final,
            maximum=maximum,
            mode=artifact_mode,
        )
        final_value = _validate_artifact(validator, final_raw)
        success_value = {
            "schema": "muncho-source-artifact-publication-success.v1",
            "purpose": kind,
            "transaction_id": transaction_id,
            "intent_sha256": _sha256(intent_raw),
            "final_path": str(final),
            "artifact_logical_sha256": final_value.logical_sha256,
            "artifact_file_sha256": _sha256(final_raw),
        }
        _publish_no_replace(
            transaction_root,
            "success.json",
            _canonical(success_value),
            mode=_TRANSITION_MODE,
            checkpoint=checkpoint,
        )
        if checkpoint is not None:
            checkpoint("after_success")
        return _PublicationResult(
            value=final_value.value,
            raw=final_raw,
            logical_sha256=final_value.logical_sha256,
            file_sha256=_sha256(final_raw),
            path=str(final),
            replayed=replayed,
        )
    except _SourceArtifactPublicationError:
        raise
    except OSError as exc:
        _error("source_artifact_publication_transaction_failed", exc)
    finally:
        if lock_descriptor is not None:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lock_descriptor)


def _run_direct_iam_rotation(
    *,
    owner_home: Path,
    chain: Mapping[str, Any],
    predecessor_file_sha256: str,
    maximum: int,
    validator: _Validator,
    collector: _Collector,
    _checkpoint: _Checkpoint | None = None,
    _recovery_only: bool = False,
) -> _PublicationResult:
    """Explicitly replace one proven direct-IAM source with its successor."""

    if (
        not isinstance(owner_home, Path)
        or not owner_home.is_absolute()
        or os.path.realpath(owner_home) != str(owner_home)
        or not isinstance(chain, Mapping)
        or not chain
        or _TRANSACTION_ID.fullmatch(predecessor_file_sha256 or "") is None
        or type(maximum) is not int
        or not 0 < maximum <= 16 * 1024 * 1024
        or not callable(validator)
        or not callable(collector)
        or (_checkpoint is not None and not callable(_checkpoint))
        or type(_recovery_only) is not bool
    ):
        _error("source_artifact_rotation_contract_invalid")
    final = owner_home / _DIRECT_RELATIVE
    final_parent = final.parent
    if os.path.realpath(final_parent) != str(final_parent):
        _error("source_artifact_publication_directory_invalid")
    _validate_directory(final_parent)
    transaction_id = _transaction_id(_DIRECT_KIND, chain)
    journal_root = final_parent / _JOURNAL_DIRECTORY
    _ensure_directory(journal_root, parent=final_parent)
    kind_root = journal_root / _DIRECT_KIND
    _ensure_directory(kind_root, parent=journal_root)
    transaction_root = kind_root / transaction_id
    _ensure_directory(transaction_root, parent=kind_root)
    lock_descriptor: int | None = None
    try:
        lock_descriptor = os.open(
            kind_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        _validate_directory(kind_root)
        _validate_directory(transaction_root)
        _validate_journal_inventory(journal_root, kind_root)
        _clean_scratch(
            transaction_root,
            artifact_mode=_DIRECT_MODE,
            maximum=maximum,
        )
        history_root = final_parent / _HISTORY_DIRECTORY
        _ensure_directory(history_root, parent=final_parent)
        history_kind = history_root / _DIRECT_KIND
        _ensure_directory(history_kind, parent=history_root)
        _validate_history_inventory(history_kind)

        intent_path = transaction_root / "intent.json"
        candidate_path = transaction_root / "candidate.bin"
        success_path = transaction_root / "success.json"
        intent_preexisted = os.path.lexists(intent_path)
        if intent_preexisted:
            intent_raw, _intent_identity = _read_file(
                intent_path,
                maximum=_MAX_TRANSITION_BYTES,
                mode=_TRANSITION_MODE,
            )
            intent_value = _decode_canonical(intent_raw)
            rotation = _validate_rotation_record(
                intent_value.get("rotation"),
                final_parent=final_parent,
            )
            expected_intent = {
                "schema": _ROTATION_INTENT_SCHEMA,
                "purpose": _DIRECT_KIND,
                "transaction_id": transaction_id,
                "final_path": str(final),
                "artifact_mode": _DIRECT_MODE,
                "chain": dict(chain),
                "chain_sha256": _sha256(_canonical(chain)),
                "rotation": dict(rotation),
            }
            if (
                intent_value != expected_intent
                or rotation["predecessor_artifact_file_sha256"]
                != predecessor_file_sha256
                or rotation["predecessor_transaction_id"] == transaction_id
            ):
                _error("source_artifact_rotation_intent_diverged")
            predecessor_root = kind_root / str(
                rotation["predecessor_transaction_id"]
            )
            completed = _completed_publication(
                predecessor_root,
                kind=_DIRECT_KIND,
                final=final,
                artifact_mode=_DIRECT_MODE,
            )
            if completed is None or dict(rotation) != _rotation_record(
                completed,
                final_parent=final_parent,
            ):
                _error("source_artifact_rotation_predecessor_unproven")
        else:
            if _recovery_only:
                _error("source_artifact_publication_recovery_unavailable")
            if not os.path.lexists(final):
                _error("source_artifact_rotation_predecessor_unavailable")
            completed, _predecessor_raw, _predecessor_identity = (
                _discover_rotation_predecessor(
                    kind_root=kind_root,
                    current_transaction_id=transaction_id,
                    final=final,
                    artifact_mode=_DIRECT_MODE,
                    maximum=maximum,
                    expected_file_sha256=predecessor_file_sha256,
                )
            )
            rotation = _rotation_record(
                completed,
                final_parent=final_parent,
            )
            intent_value = {
                "schema": _ROTATION_INTENT_SCHEMA,
                "purpose": _DIRECT_KIND,
                "transaction_id": transaction_id,
                "final_path": str(final),
                "artifact_mode": _DIRECT_MODE,
                "chain": dict(chain),
                "chain_sha256": _sha256(_canonical(chain)),
                "rotation": dict(rotation),
            }
            intent_raw = _canonical(intent_value)
            _publish_no_replace(
                transaction_root,
                "intent.json",
                intent_raw,
                mode=_TRANSITION_MODE,
                checkpoint=_checkpoint,
            )
        if _checkpoint is not None:
            _checkpoint("after_intent")

        archive = Path(str(rotation["predecessor_archive_path"]))
        archive_raw: bytes | None = None
        archive_identity: tuple[int, ...] | None = None
        if os.path.lexists(archive):
            archive_raw, archive_identity = _read_file(
                archive,
                maximum=maximum,
                mode=_DIRECT_MODE,
                links=frozenset({1, 2}),
            )
            if _sha256(archive_raw) != predecessor_file_sha256:
                _error("source_artifact_rotation_archive_diverged")

        candidate: _ValidatedArtifact | None = None
        candidate_raw: bytes | None = None
        if os.path.lexists(candidate_path):
            candidate_raw, _candidate_identity = _read_file(
                candidate_path,
                maximum=maximum,
                mode=_DIRECT_MODE,
            )
            candidate = _validate_artifact(validator, candidate_raw)

        final_state = "absent"
        final_value: _ValidatedArtifact | None = None
        final_raw: bytes | None = None
        final_identity: tuple[int, ...] | None = None
        if os.path.lexists(final):
            final_raw, final_identity = _read_file(
                final,
                maximum=maximum,
                mode=_DIRECT_MODE,
                links=frozenset({1, 2}),
            )
            if _sha256(final_raw) == predecessor_file_sha256:
                final_state = "predecessor"
            else:
                final_value = _validate_artifact(validator, final_raw)
                final_state = "successor"
        if final_state == "absent":
            _error("source_artifact_rotation_final_absent")
        if archive_identity is not None and archive_identity[5] == 2:
            if (
                final_state != "predecessor"
                or final_identity is None
                or archive_identity[:2] != final_identity[:2]
            ):
                _error("source_artifact_rotation_archive_diverged")

        success: Mapping[str, Any] | None = None
        if os.path.lexists(success_path):
            success_raw, _success_identity = _read_file(
                success_path,
                maximum=_MAX_TRANSITION_BYTES,
                mode=_TRANSITION_MODE,
            )
            success = _decode_canonical(success_raw)
        if success is not None:
            if (
                final_state != "successor"
                or final_value is None
                or final_raw is None
                or candidate_raw is not None
                or archive_raw is None
                or archive_identity is None
                or archive_identity[5] != 1
            ):
                _error("source_artifact_publication_success_invalid")
            expected_success = {
                "schema": _ROTATION_SUCCESS_SCHEMA,
                "purpose": _DIRECT_KIND,
                "transaction_id": transaction_id,
                "intent_sha256": _sha256(intent_raw),
                "final_path": str(final),
                "artifact_logical_sha256": final_value.logical_sha256,
                "artifact_file_sha256": _sha256(final_raw),
                "rotation": dict(rotation),
            }
            if success != expected_success:
                _error("source_artifact_publication_success_invalid")
            return _PublicationResult(
                value=final_value.value,
                raw=final_raw,
                logical_sha256=final_value.logical_sha256,
                file_sha256=_sha256(final_raw),
                path=str(final),
                replayed=True,
            )

        replayed = (
            intent_preexisted
            or candidate is not None
            or archive_raw is not None
            or final_state == "successor"
        )
        if final_state == "predecessor" and candidate is None:
            if _recovery_only:
                _error("source_artifact_publication_recovery_unavailable")
            collected_raw = collector()
            if (
                type(collected_raw) is not bytes
                or not 0 < len(collected_raw) <= maximum
            ):
                _error("source_artifact_publication_collector_invalid")
            candidate = _validate_artifact(validator, collected_raw)
            candidate_raw = collected_raw
            if _checkpoint is not None:
                _checkpoint("after_collection")
            _publish_no_replace(
                transaction_root,
                "candidate.bin",
                candidate_raw,
                mode=_DIRECT_MODE,
                checkpoint=_checkpoint,
            )
            if _checkpoint is not None:
                _checkpoint("after_candidate")
        if final_state == "predecessor":
            if candidate is None or candidate_raw is None:
                _error("source_artifact_publication_candidate_invalid")
            predecessor_raw, predecessor_identity = _read_file(
                final,
                maximum=maximum,
                mode=_DIRECT_MODE,
                links=frozenset({1, 2}),
            )
            if _sha256(predecessor_raw) != predecessor_file_sha256:
                _error("source_artifact_rotation_predecessor_mismatch")
            if archive_raw is None:
                try:
                    os.link(final, archive, follow_symlinks=False)
                except FileExistsError:
                    pass
                except OSError as exc:
                    _error("source_artifact_rotation_archive_write_failed", exc)
                _fsync_directory(history_kind)
                archive_raw, archive_identity = _read_file(
                    archive,
                    maximum=maximum,
                    mode=_DIRECT_MODE,
                    links=frozenset({1, 2}),
                )
            if (
                archive_raw != predecessor_raw
                or archive_identity is None
                or _sha256(archive_raw) != predecessor_file_sha256
                or (
                    archive_identity[5] == 2
                    and archive_identity[:2] != predecessor_identity[:2]
                )
            ):
                _error("source_artifact_rotation_archive_diverged")
            if _checkpoint is not None:
                _checkpoint("after_predecessor_archive")
            candidate_raw, _candidate_identity = _read_file(
                candidate_path,
                maximum=maximum,
                mode=_DIRECT_MODE,
            )
            _validate_artifact(validator, candidate_raw)
            predecessor_raw, _predecessor_identity = _read_file(
                final,
                maximum=maximum,
                mode=_DIRECT_MODE,
                links=frozenset({1, 2}),
            )
            if _sha256(predecessor_raw) != predecessor_file_sha256:
                _error("source_artifact_rotation_predecessor_mismatch")
            try:
                os.replace(candidate_path, final)
            except OSError as exc:
                _error("source_artifact_rotation_final_replace_failed", exc)
            _fsync_directory(transaction_root)
            _fsync_directory(final_parent)
            if _checkpoint is not None:
                _checkpoint("after_final_replace")
        elif candidate_raw is not None:
            _error("source_artifact_rotation_candidate_diverged")

        final_raw, final_identity = _read_file(
            final,
            maximum=maximum,
            mode=_DIRECT_MODE,
        )
        final_value = _validate_artifact(validator, final_raw)
        archive_raw, archive_identity = _read_file(
            archive,
            maximum=maximum,
            mode=_DIRECT_MODE,
        )
        if _sha256(archive_raw) != predecessor_file_sha256:
            _error("source_artifact_rotation_archive_diverged")
        success_value = {
            "schema": _ROTATION_SUCCESS_SCHEMA,
            "purpose": _DIRECT_KIND,
            "transaction_id": transaction_id,
            "intent_sha256": _sha256(intent_raw),
            "final_path": str(final),
            "artifact_logical_sha256": final_value.logical_sha256,
            "artifact_file_sha256": _sha256(final_raw),
            "rotation": dict(rotation),
        }
        _publish_no_replace(
            transaction_root,
            "success.json",
            _canonical(success_value),
            mode=_TRANSITION_MODE,
            checkpoint=_checkpoint,
        )
        if _checkpoint is not None:
            _checkpoint("after_success")
        return _PublicationResult(
            value=final_value.value,
            raw=final_raw,
            logical_sha256=final_value.logical_sha256,
            file_sha256=_sha256(final_raw),
            path=str(final),
            replayed=replayed,
        )
    except _SourceArtifactPublicationError:
        raise
    except OSError as exc:
        _error("source_artifact_rotation_transaction_failed", exc)
    finally:
        if lock_descriptor is not None:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lock_descriptor)


def _run_direct_iam(
    *,
    owner_home: Path,
    chain: Mapping[str, Any],
    maximum: int,
    validator: _Validator,
    collector: _Collector,
    _checkpoint: _Checkpoint | None = None,
    _recovery_only: bool = False,
) -> _PublicationResult:
    return _run(
        kind=_DIRECT_KIND,
        owner_home=owner_home,
        relative=_DIRECT_RELATIVE,
        artifact_mode=_DIRECT_MODE,
        maximum=maximum,
        chain=chain,
        validator=validator,
        collector=collector,
        checkpoint=_checkpoint,
        recovery_only=_recovery_only,
    )


def _run_host_identity(
    *,
    owner_home: Path,
    chain: Mapping[str, Any],
    maximum: int,
    validator: _Validator,
    collector: _Collector,
    _checkpoint: _Checkpoint | None = None,
    _recovery_only: bool = False,
) -> _PublicationResult:
    return _run(
        kind=_HOST_KIND,
        owner_home=owner_home,
        relative=_HOST_RELATIVE,
        artifact_mode=_HOST_MODE,
        maximum=maximum,
        chain=chain,
        validator=validator,
        collector=collector,
        checkpoint=_checkpoint,
        recovery_only=_recovery_only,
    )


def _run_host_identity_v3(
    *,
    owner_home: Path,
    chain: Mapping[str, Any],
    maximum: int,
    validator: _Validator,
    collector: _Collector,
    _checkpoint: _Checkpoint | None = None,
    _recovery_only: bool = False,
) -> _PublicationResult:
    return _run(
        kind=_HOST_V3_KIND,
        owner_home=owner_home,
        relative=_HOST_V3_RELATIVE,
        artifact_mode=_HOST_MODE,
        maximum=maximum,
        chain=chain,
        validator=validator,
        collector=collector,
        checkpoint=_checkpoint,
        recovery_only=_recovery_only,
    )


__all__: list[str] = []
