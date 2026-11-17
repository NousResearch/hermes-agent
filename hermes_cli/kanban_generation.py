"""Fail-closed custody and lifecycle primitives for prepared Kanban generations."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import struct
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


class GenerationError(ValueError):
    """A generation is malformed, inconsistent, or unsafe to consume."""


CHILD_TABLES = ("task_comments", "task_events", "task_runs", "task_attachments")
POINTER_FIELDS = {"format_version", "generation_id", "generation_nonce", "ready_sha256"}
ACCEPTED_FIELDS = {"format_version", "generation_id", "generation_nonce", "ready_sha256", "fence_epoch", "accepted_at"}
ACTIVATED_FIELDS = {"format_version", "generation_id", "generation_nonce", "ready_sha256", "fence_epoch", "activated_at"}
_ACTIVE_FENCE_TOKEN: ContextVar[str | None] = ContextVar("kanban_fence_token", default=None)
LEASE_RECLAIM_GRACE_SECONDS = 60
_HELD_FENCE_HANDLES: dict[str, tuple[Any, Any]] = {}


class DurabilityAdapter:
    """Small injectable durability surface used by crash-boundary tests."""

    def __init__(self, *, failpoint: str | None = None):
        self.failpoint = failpoint

    def checkpoint(self, name: str) -> None:
        if self.failpoint == name or (
            self.failpoint == "activation-before-rename" and name == "ACTIVATED.json-before-rename"
        ):
            raise OSError(self.failpoint or name)

    def fsync_file(self, path: Path) -> None:
        self.checkpoint(f"{path.name}-before-fsync")
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        self.checkpoint(f"{path.name}-after-fsync")

    def fsync_dir(self, path: Path) -> None:
        self.checkpoint(f"{path.name}-before-fsync-dir")
        _fsync_dir(path)
        self.checkpoint(f"{path.name}-after-fsync-dir")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def encode_reader_frame(value: dict[str, Any]) -> bytes:
    """The sole v1 reader wire encoder: canonical UTF-8 JSON with u32 length."""
    payload = canonical_json(value)
    if len(payload) > 16 * 1024 * 1024:
        raise GenerationError("READER_FRAME_TOO_LARGE")
    return struct.pack("!I", len(payload)) + payload


def decode_reader_frame(data: bytes) -> dict[str, Any]:
    if len(data) < 4:
        raise GenerationError("READER_FRAME_MALFORMED")
    size = struct.unpack("!I", data[:4])[0]
    if size != len(data) - 4 or size > 16 * 1024 * 1024:
        raise GenerationError("READER_FRAME_MALFORMED")
    try:
        value = json.loads(data[4:])
    except (TypeError, ValueError) as exc:
        raise GenerationError("READER_FRAME_MALFORMED") from exc
    if not isinstance(value, dict):
        raise GenerationError("READER_FRAME_MALFORMED")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_regular_file(path: Path, *, max_bytes: int | None = None) -> tuple[bytes, str]:
    """Capture a regular file by descriptor, refusing symlinks and races."""
    import stat
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise GenerationError("ATTACHMENT_NOT_REGULAR")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(fd, 1024 * 1024):
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise GenerationError("ATTACHMENT_TOO_LARGE")
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or total != before.st_size:
        raise GenerationError("ATTACHMENT_CHANGED")
    data = b"".join(chunks)
    return data, hashlib.sha256(data).hexdigest()


def install_content_addressed_blob(root: Path, data: bytes, digest: str) -> Path:
    relative, _created = install_content_addressed_blob_owned(root, data, digest)
    return relative


def install_content_addressed_blob_owned(root: Path, data: bytes, digest: str) -> tuple[Path, bool]:
    relative = Path("blobs") / "sha256" / digest[:2] / digest
    destination = root / relative
    return relative, _install_blob(destination, data)


def _install_blob(destination: Path, data: bytes) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise GenerationError("ATTACHMENT_PATH_INVALID")
    if destination.exists():
        if destination.read_bytes() != data:
            raise GenerationError("ATTACHMENT_CONTENT_CONFLICT")
        return False
    try:
        with destination.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if destination.is_symlink() or destination.read_bytes() != data:
            raise GenerationError("ATTACHMENT_CONTENT_CONFLICT")
        return False
    _fsync_dir(destination.parent)
    return True


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: Any, *, durability: DurabilityAdapter | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            payload = canonical_json(value) + b"\n"
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("short atomic JSON write")
                offset += written
            if durability:
                durability.checkpoint(f"{path.name}-before-fsync")
            os.fsync(fd)
            if durability:
                durability.checkpoint(f"{path.name}-after-fsync")
        finally:
            os.close(fd)
        if durability:
            durability.checkpoint(f"{path.name}-before-rename")
        os.replace(temporary, path)
        if durability:
            durability.checkpoint(f"{path.name}-after-rename")
        (durability or DurabilityAdapter()).fsync_dir(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


@contextlib.contextmanager
def _lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def board_write_guard(board_dir: Path, *, owner: str = "native", token: str | None = None,
                      expected_generation: str | None = None) -> Iterator[None]:
    """The one shared guard for database and filesystem mutations."""
    del owner
    # Legacy boards retain their historical filesystem footprint.  Do not
    # create generation control files merely because a legacy mutation passed
    # through the shared seam.
    if not (
        (board_dir / "generation.writer-fence").exists()
        or pointer_path(board_dir).exists()
    ):
        yield
        return
    # Ordinary writers share the board custody.  A publication/GC fence takes
    # the same lock exclusively and therefore waits for every in-flight write
    # to finish before it can mutate the generation catalog.
    custody = contextlib.nullcontext() if str(board_dir.resolve()) in _HELD_FENCE_HANDLES else _lock(board_dir / "generation.writer-lock", exclusive=False)
    with custody:
        fence_path = board_dir / "generation.writer-fence"
        if fence_path.exists():
            validate_fence(board_dir, token or "")
        pointer = read_pointer(board_dir)
        if pointer is not None and expected_generation is not None:
            if pointer["generation_id"] != expected_generation:
                raise GenerationError("GENERATION_POINTER_CHANGED")
        yield


def acquire_fence(board_dir: Path, *, generation: str, owner: str) -> dict[str, Any]:
    path = board_dir / "generation.writer-fence"
    catalog_handle = (board_dir / "generation.catalog-lock").open("a+b")
    writer_handle = path.with_name("generation.writer-lock").open("a+b")
    import fcntl
    # Every custody path is catalog → writer → activation.  A fence is an
    # exclusive barrier, so it waits for an ordinary shared writer instead of
    # racing it or failing spuriously during a slow attachment write.
    fcntl.flock(catalog_handle.fileno(), fcntl.LOCK_EX)
    fcntl.flock(writer_handle.fileno(), fcntl.LOCK_EX)
    if path.exists():
        fcntl.flock(writer_handle.fileno(), fcntl.LOCK_UN)
        fcntl.flock(catalog_handle.fileno(), fcntl.LOCK_UN)
        writer_handle.close()
        catalog_handle.close()
        raise GenerationError("WRITER_FENCE_HELD")
    record = {
        "board": board_dir.name,
        "generation_id": generation,
        "epoch": time.time_ns(),
        "owner": owner,
        "owner_pid": os.getpid(),
        "owner_start_fingerprint": _start_fingerprint(os.getpid()),
        "token": secrets.token_urlsafe(24),
    }
    try:
        _atomic_json(path, record)
    except BaseException:
        fcntl.flock(writer_handle.fileno(), fcntl.LOCK_UN)
        fcntl.flock(catalog_handle.fileno(), fcntl.LOCK_UN)
        writer_handle.close()
        catalog_handle.close()
        raise
    _HELD_FENCE_HANDLES[str(board_dir.resolve())] = (catalog_handle, writer_handle)
    return record


def adopt_fence(
    board_dir: Path,
    *,
    token: str,
    owner: str,
    expected_epoch: int,
    expected_generation: str,
) -> dict[str, Any]:
    """Adopt a persisted fence after a crash without minting a second token."""
    key = str(board_dir.resolve())
    acquired_here = key not in _HELD_FENCE_HANDLES
    if acquired_here:
        catalog_handle = (board_dir / "generation.catalog-lock").open("a+b")
        handle = (board_dir / "generation.writer-lock").open("a+b")
        import fcntl
        fcntl.flock(catalog_handle.fileno(), fcntl.LOCK_EX)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _HELD_FENCE_HANDLES[key] = (catalog_handle, handle)
    try:
        record = validate_fence(board_dir, token)
        if int(record.get("epoch", -1)) != expected_epoch:
            raise GenerationError("WRITER_FENCE_EPOCH_MISMATCH")
        if str(record.get("generation_id", "")) != expected_generation:
            raise GenerationError("WRITER_FENCE_GENERATION_MISMATCH")
        try:
            owner_pid = int(record["owner_pid"])
            owner_fingerprint = str(record["owner_start_fingerprint"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GenerationError("WRITER_FENCE_IDENTITY_MISSING") from exc
        try:
            import psutil

            owner_live = psutil.pid_exists(owner_pid) and (
                _start_fingerprint(owner_pid) == owner_fingerprint
            )
        except (OSError, ValueError):
            owner_live = False
        if owner_live:
            raise GenerationError("WRITER_FENCE_OWNER_LIVE")
        record = {
            **record,
            "owner": owner,
            "owner_pid": os.getpid(),
            "owner_start_fingerprint": _start_fingerprint(os.getpid()),
            "epoch": time.time_ns(),
        }
        _atomic_json(board_dir / "generation.writer-fence", record)
        return record
    except BaseException:
        if acquired_here:
            import fcntl

            catalog_handle, handle = _HELD_FENCE_HANDLES.pop(key)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            fcntl.flock(catalog_handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            catalog_handle.close()
        raise


def release_fence(board_dir: Path, *, token: str) -> None:
    validate_fence(board_dir, token)
    (board_dir / "generation.writer-fence").unlink()
    _fsync_dir(board_dir)
    handles = _HELD_FENCE_HANDLES.pop(str(board_dir.resolve()), None)
    if handles is not None:
        import fcntl
        catalog_handle, writer_handle = handles
        fcntl.flock(writer_handle.fileno(), fcntl.LOCK_UN)
        fcntl.flock(catalog_handle.fileno(), fcntl.LOCK_UN)
        writer_handle.close()
        catalog_handle.close()


@contextlib.contextmanager
def fence_token_scope(token: str) -> Iterator[None]:
    marker = _ACTIVE_FENCE_TOKEN.set(token)
    try:
        yield
    finally:
        _ACTIVE_FENCE_TOKEN.reset(marker)


def current_fence_token() -> str | None:
    return _ACTIVE_FENCE_TOKEN.get()


def validate_fence(board_dir: Path, token: str) -> dict[str, Any]:
    try:
        record = json.loads((board_dir / "generation.writer-fence").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise GenerationError("WRITER_FENCE_REQUIRED") from exc
    if not isinstance(record, dict) or record.get("token") != token:
        raise GenerationError("WRITER_FENCE_TOKEN_INVALID")
    return record


def generation_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(6)


def pointer_path(board_dir: Path) -> Path:
    return board_dir / "generation.current"


def read_pointer(board_dir: Path) -> dict[str, Any] | None:
    path = pointer_path(board_dir)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise GenerationError("POINTER_INVALID") from exc
    if not isinstance(value, dict) or set(value) != POINTER_FIELDS or value.get("format_version") != 1:
        raise GenerationError("POINTER_INVALID")
    if any(not isinstance(value[key], str) or not value[key] for key in POINTER_FIELDS - {"format_version"}):
        raise GenerationError("POINTER_INVALID")
    return value


def write_pointer(board_dir: Path, value: dict[str, Any]) -> None:
    if set(value) != POINTER_FIELDS:
        raise GenerationError("POINTER_INVALID")
    _atomic_json(pointer_path(board_dir), value)


def row_manifest(conn: sqlite3.Connection) -> dict[str, Any]:
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    result: dict[str, Any] = {"tables": {}, "schema": []}
    for table in tables:
        quoted = '"' + table.replace('"', '""') + '"'
        result["tables"][table] = {"columns": [row[1] for row in conn.execute(f"PRAGMA table_info({quoted})")], "rows": [list(row) for row in conn.execute(f"SELECT * FROM {quoted} ORDER BY rowid")]}
    try:
        result["sequences"] = [list(row) for row in conn.execute("SELECT name, seq FROM sqlite_sequence ORDER BY name")]
    except sqlite3.OperationalError:
        result["sequences"] = []
    result["schema"] = [list(row) for row in conn.execute("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name")]
    return result


def manifest_digest(manifest: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(manifest))


def initialize_generation_db(db: Path, *, gid: str, nonce: str, control_root: Path) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS kanban_generation_meta (format_version INTEGER NOT NULL, generation_id TEXT PRIMARY KEY, generation_nonce TEXT NOT NULL, control_root_relpath TEXT NOT NULL)")
        conn.execute("DELETE FROM kanban_generation_meta")
        conn.execute("INSERT INTO kanban_generation_meta VALUES (1, ?, ?, ?)", (gid, nonce, os.path.relpath(control_root, db.parent)))


def _validate_blob_record(path: Path, record: dict[str, Any]) -> None:
    try:
        digest = str(record["blob_sha256"])
        size = int(record["size"])
        relative = Path(str(record["custody_relpath"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise GenerationError("BLOB_MANIFEST_INVALID") from exc
    expected = Path("blobs") / "sha256" / digest[:2] / digest
    root = path.resolve()
    resolved = (root / relative).resolve()
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or relative != expected
        or relative.is_absolute()
        or ".." in relative.parts
        or resolved != root / relative
        or resolved.is_symlink()
        or not resolved.is_file()
        or resolved.stat().st_size != size
        or sha256_file(resolved) != digest
    ):
        raise GenerationError("BLOB_MANIFEST_INVALID")


def _current_blob_manifest(
    path: Path, conn: sqlite3.Connection, *, exact_physical: bool
) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            "SELECT attachment_id, blob_sha256, size, custody_relpath "
            "FROM recovery_attachment_resolution ORDER BY attachment_id"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc):
            raise GenerationError("BLOB_MANIFEST_INVALID") from exc
        rows = []
    records = [
        {
            "attachment_id": int(row[0]),
            "blob_sha256": str(row[1]),
            "size": int(row[2]),
            "custody_relpath": str(row[3]),
        }
        for row in rows
    ]
    for record in records:
        _validate_blob_record(path, record)
    if exact_physical:
        blob_root = path / "blobs" / "sha256"
        physical: set[str] = set()
        if blob_root.exists():
            if blob_root.is_symlink() or not blob_root.is_dir():
                raise GenerationError("BLOB_MANIFEST_INVALID")
            for candidate in blob_root.rglob("*"):
                if candidate.is_symlink():
                    raise GenerationError("BLOB_MANIFEST_INVALID")
                if candidate.is_file():
                    physical.add(str(candidate.relative_to(path)))
        expected = {record["custody_relpath"] for record in records}
        if physical != expected:
            raise GenerationError("BLOB_MANIFEST_INVALID")
    return records


def build_ready(path: Path, *, gid: str, nonce: str) -> dict[str, Any]:
    with sqlite3.connect(path / "kanban.db") as conn:
        manifest = row_manifest(conn)
        blob_manifest = _current_blob_manifest(path, conn, exact_physical=True)
    ready = {"format_version": 1, "generation_id": gid, "generation_nonce": nonce, "database_sha256": sha256_file(path / "kanban.db"), "manifest": manifest, "manifest_digest": manifest_digest(manifest), "blob_manifest": blob_manifest}
    _atomic_json(path / "READY.json", ready)
    return ready


def validate_generation(path: Path, *, activated: bool = False) -> dict[str, Any]:
    _assert_not_poisoned(path)
    try:
        ready = json.loads((path / "READY.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise GenerationError("READY_INVALID") from exc
    required = {"format_version", "generation_id", "generation_nonce", "database_sha256", "manifest", "manifest_digest", "blob_manifest"}
    if not isinstance(ready, dict) or set(ready) != required or ready.get("format_version") != 1:
        raise GenerationError("READY_INVALID")
    db = path / "kanban.db"
    if not db.is_file() or (not activated and sha256_file(db) != ready["database_sha256"]):
        raise GenerationError("READY_INVALID")
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        try:
            meta = conn.execute("SELECT * FROM kanban_generation_meta").fetchone()
        except sqlite3.Error as exc:
            raise GenerationError("GENERATION_META_INVALID") from exc
        if meta is None or meta["generation_id"] != ready["generation_id"] or meta["generation_nonce"] != ready["generation_nonce"]:
            raise GenerationError("GENERATION_META_INVALID")
        if activated:
            try:
                integrity = [row[0] for row in conn.execute("PRAGMA integrity_check")]
                foreign_keys = list(conn.execute("PRAGMA foreign_key_check"))
            except sqlite3.Error as exc:
                raise GenerationError("GENERATION_META_INVALID") from exc
            if integrity != ["ok"] or foreign_keys:
                raise GenerationError("GENERATION_META_INVALID")
        try:
            manifest = row_manifest(conn)
            current_blobs = _current_blob_manifest(
                path, conn, exact_physical=not activated
            )
        except sqlite3.Error as exc:
            raise GenerationError("GENERATION_META_INVALID") from exc
        if not activated and (manifest_digest(manifest) != ready["manifest_digest"] or manifest != ready["manifest"]):
            raise GenerationError("READY_INVALID")
        if not isinstance(ready["blob_manifest"], list):
            raise GenerationError("READY_INVALID")
        for blob in ready["blob_manifest"]:
            if not isinstance(blob, dict):
                raise GenerationError("READY_INVALID")
            _validate_blob_record(path, blob)
        if not activated and current_blobs != ready["blob_manifest"]:
            raise GenerationError("READY_INVALID")
    if activated:
        try:
            marker = json.loads((path / "ACTIVATED.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise GenerationError("ACTIVATION_INVALID") from exc
        if (not isinstance(marker, dict) or set(marker) != ACTIVATED_FIELDS or
                marker.get("format_version") != 1 or
                not isinstance(marker.get("fence_epoch"), int) or
                marker.get("generation_id") != ready["generation_id"] or
                marker.get("generation_nonce") != ready["generation_nonce"] or
                marker.get("ready_sha256") != sha256_file(path / "READY.json")):
            raise GenerationError("ACTIVATION_INVALID")
    return ready


def active_manifest(path: Path) -> str:
    """Authenticate and return the current bytes of an activated generation."""
    validate_generation(path, activated=True)
    with sqlite3.connect(f"file:{path / 'kanban.db'}?mode=ro", uri=True) as conn:
        return manifest_digest(row_manifest(conn))


def _assert_not_poisoned(path: Path) -> None:
    if ((path / "ACTIVATION.poison.json").exists() or
            (path / "ACTIVATION.poison").exists()):
        raise GenerationError("ACTIVATION_POISONED")


def _mark_activation_poisoned(path: Path, payload: dict[str, Any]) -> None:
    """Publish poison through an independent fallback if JSON publication fails."""
    try:
        _atomic_json(path / "ACTIVATION.poison.json", payload)
        return
    except (OSError, TypeError, ValueError, GenerationError):
        sentinel = path / "ACTIVATION.poison"
        try:
            fd = os.open(sentinel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, canonical_json(payload) + b"\n")
                os.fsync(fd)
            finally:
                os.close(fd)
            _fsync_dir(path)
            return
        except FileExistsError:
            _fsync_dir(path)
            return
        except (OSError, TypeError, ValueError, GenerationError) as fallback_exc:
            raise GenerationError("ACTIVATION_POISON_DURABILITY_FAILED") from fallback_exc


def recover_publication(board_dir: Path) -> None:
    """Finish an interrupted publication while holding exclusive catalog custody."""
    held = str(board_dir.resolve()) in _HELD_FENCE_HANDLES
    custody = contextlib.nullcontext() if held else _lock(
        board_dir / "generation.catalog-lock", exclusive=True
    )
    with custody:
        _recover_publication_locked(board_dir)


def _recover_publication_locked(board_dir: Path) -> None:
    """Locked publication recovery; never call under shared catalog custody."""
    accepted_path = board_dir / "generation.accepted.json"
    if not accepted_path.exists():
        return
    accepted: Any = None
    try:
        accepted = json.loads(accepted_path.read_text(encoding="utf-8-sig"))
        if not isinstance(accepted, dict) or set(accepted) != ACCEPTED_FIELDS or not isinstance(accepted["fence_epoch"], int) or not isinstance(accepted["accepted_at"], (int, float)):
            raise ValueError("accepted record shape")
        candidate = board_dir / "generations" / accepted["generation_id"]
        # A first write legitimately changes the prepared database bytes after
        # publication.  The activation marker is the durable boundary that
        # tells recovery to validate the post-write generation as active rather
        # than re-authenticating its old prepared hash.
        ready = validate_generation(candidate, activated=(candidate / "ACTIVATED.json").is_file())
        if ready["generation_nonce"] != accepted["generation_nonce"] or sha256_file(candidate / "READY.json") != accepted["ready_sha256"]:
            raise ValueError("accepted generation mismatch")
        pointer = read_pointer(board_dir)
        expected = {key: accepted[key] for key in POINTER_FIELDS}
        if pointer is None:
            write_pointer(board_dir, expected)
        elif pointer != expected:
            raise ValueError("ambiguous publication")
        # Preserve the accepted fence epoch until first-write activation.  It
        # is the durable custody receipt that ACTIVATED.json authenticates.
    except (OSError, ValueError, KeyError, TypeError, GenerationError) as exc:
        if isinstance(exc, GenerationError) and str(exc) == "ACTIVATION_POISONED":
            raise
        _atomic_json(board_dir / "generation.publication.poison.json", {"reason": str(exc), "accepted": accepted})
        raise GenerationError("PUBLICATION_POISONED") from exc


def open_generation(board_dir: Path, *, require_activation: bool = False,
                    recover: bool = True) -> tuple[Path, dict[str, Any]]:
    if (board_dir / "generation.publication.poison.json").exists():
        raise GenerationError("PUBLICATION_POISONED")
    if recover:
        recover_publication(board_dir)
    pointer = read_pointer(board_dir)
    if pointer is None:
        raise GenerationError("BOARD_NOT_GENERATED")
    path = board_dir / "generations" / pointer["generation_id"]
    if not path.is_dir():
        raise GenerationError("POINTER_INVALID")
    ready = validate_generation(path, activated=(path / "ACTIVATED.json").is_file())
    if pointer["ready_sha256"] != sha256_file(path / "READY.json") or pointer["generation_nonce"] != ready["generation_nonce"]:
        raise GenerationError("POINTER_INVALID")
    if require_activation and not (path / "ACTIVATED.json").is_file():
        raise GenerationError("GENERATION_NOT_ACTIVATED")
    return path / "kanban.db", ready


def activate_first_write(path: Path, *, fence_epoch: int | None = None,
                         durability: DurabilityAdapter | None = None,
                         board_dir: Path | None = None) -> dict[str, Any]:
    ready = validate_generation(path)
    board_dir = board_dir or path.parents[1]
    with _lock(board_dir / "generation.activation-lock", exclusive=True):
        _assert_not_poisoned(path)
        accepted_path = board_dir / "generation.accepted.json"
        accepted = None
        if accepted_path.exists():
            try:
                accepted = json.loads(accepted_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as exc:
                raise GenerationError("PUBLICATION_POISONED") from exc
            expected_ready = sha256_file(path / "READY.json")
            if (not isinstance(accepted, dict) or set(accepted) != ACCEPTED_FIELDS or
                    accepted.get("format_version") != 1 or
                    not isinstance(accepted.get("fence_epoch"), int) or
                    accepted.get("generation_id") != ready["generation_id"] or
                    accepted.get("generation_nonce") != ready["generation_nonce"] or
                    accepted.get("ready_sha256") != expected_ready or
                    (fence_epoch is not None and fence_epoch != accepted["fence_epoch"])):
                raise GenerationError("ACTIVATION_FENCE_MISMATCH")
            fence_epoch = accepted["fence_epoch"]
        if not isinstance(fence_epoch, int):
            raise GenerationError("ACTIVATION_FENCE_REQUIRED")

        marker = {"format_version": 1, "generation_id": ready["generation_id"],
                  "generation_nonce": ready["generation_nonce"],
                  "ready_sha256": sha256_file(path / "READY.json"),
                  "fence_epoch": fence_epoch, "activated_at": time.time()}
        final = path / "ACTIVATED.json"
        if final.exists():
            try:
                _fsync_dir(path)
            except OSError as exc:
                try:
                    final.unlink()
                    _fsync_dir(path)
                except OSError as cleanup_exc:
                    _mark_activation_poisoned(path, {
                        "generation_id": ready["generation_id"],
                        "reason": str(cleanup_exc),
                        "ready_sha256": marker["ready_sha256"],
                    })
                    raise GenerationError("ACTIVATION_POISONED") from cleanup_exc
                raise GenerationError("ACTIVATION_RETRYABLE") from exc
            try:
                existing = json.loads(final.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as exc:
                _mark_activation_poisoned(path, {
                    "generation_id": ready["generation_id"],
                    "reason": str(exc),
                    "ready_sha256": marker["ready_sha256"],
                })
                raise GenerationError("ACTIVATION_POISONED") from exc
            if (not isinstance(existing, dict) or set(existing) != ACTIVATED_FIELDS or
                    any(existing.get(key) != marker[key] for key in
                        ("format_version", "generation_id", "generation_nonce",
                         "ready_sha256", "fence_epoch"))):
                _mark_activation_poisoned(path, {
                    "generation_id": ready["generation_id"],
                    "reason": "invalid surviving activation marker",
                    "ready_sha256": marker["ready_sha256"],
                })
                raise GenerationError("ACTIVATION_POISONED")
            if accepted_path.exists():
                accepted_path.unlink()
                _fsync_dir(board_dir)
            return existing
        try:
            _atomic_json(final, marker, durability=durability)
            if json.loads(final.read_text(encoding="utf-8-sig")) != marker:
                raise GenerationError("ACTIVATION_INVALID")
        except Exception as exc:
            # A reported in-process publication failure cannot be converted to
            # success from readback alone. Durably remove any marker and return
            # retryable; poison only when cleanup cannot be proven.
            try:
                if final.exists():
                    final.unlink()
                    _fsync_dir(path)
                if final.exists():
                    raise OSError("activation marker cleanup did not persist")
            except Exception as cleanup_exc:
                _mark_activation_poisoned(path, {
                    "generation_id": ready["generation_id"],
                    "reason": str(cleanup_exc),
                    "ready_sha256": marker["ready_sha256"],
                })
                raise GenerationError("ACTIVATION_POISONED") from cleanup_exc
            if isinstance(exc, OSError):
                raise
            raise GenerationError("ACTIVATION_RETRYABLE") from exc
        if accepted_path.exists():
            accepted_path.unlink()
            _fsync_dir(board_dir)
    return marker


def publish_prepared(path: Path, *, expected_manifest: str | None = None,
                     expected_target_manifest: str | None = None,
                     fence_token: str) -> dict[str, Any]:
    board_dir = path.parents[1]
    held = str(board_dir.resolve()) in _HELD_FENCE_HANDLES
    custody = contextlib.nullcontext() if held else _lock(board_dir / "generation.writer-lock", exclusive=True)
    catalog = contextlib.nullcontext() if held else _lock(board_dir / "generation.catalog-lock", exclusive=True)
    with catalog:
      with custody:
        fence = validate_fence(board_dir, fence_token)
        ready = validate_generation(path)
        if fence.get("generation_id") != ready["generation_id"]:
            raise GenerationError("WRITER_FENCE_GENERATION_MISMATCH")
        current = read_pointer(board_dir)
        if expected_target_manifest is not None:
            if current is None:
                if expected_target_manifest != "legacy":
                    raise GenerationError("TARGET_MANIFEST_MISMATCH")
            else:
                current_path = board_dir / "generations" / current["generation_id"]
                current_ready = validate_generation(current_path, activated=(current_path / "ACTIVATED.json").exists())
                current_manifest = active_manifest(current_path) if (current_path / "ACTIVATED.json").exists() else current_ready["manifest_digest"]
                if current_manifest != expected_target_manifest:
                    raise GenerationError("TARGET_MANIFEST_MISMATCH")
        elif expected_manifest != ready["manifest_digest"]:
            raise GenerationError("TARGET_MANIFEST_MISMATCH")
        pointer = {"format_version": 1, "generation_id": ready["generation_id"], "generation_nonce": ready["generation_nonce"], "ready_sha256": sha256_file(path / "READY.json")}
        with _lock(board_dir / "generation.activation-lock", exclusive=True):
            previous = pointer_path(board_dir)
            legacy = {"storage_mode": "legacy"}
            _atomic_json(board_dir / "generation.accepted.json", {
                "format_version": 1,
                "generation_id": ready["generation_id"],
                "generation_nonce": ready["generation_nonce"],
                "ready_sha256": pointer["ready_sha256"],
                "fence_epoch": fence["epoch"],
                "accepted_at": time.time(),
            })
            _atomic_json(board_dir / "generation.previous-pointer", json.loads(previous.read_text(encoding="utf-8-sig")) if previous.exists() else legacy)
            write_pointer(board_dir, pointer)
    return {"published": True, "pointer": pointer, "fence_epoch": fence["epoch"]}


def rollback_pre_activation(board_dir: Path, *, expected_active_manifest: str, fence_token: str) -> None:
    if (board_dir / "generation.publication.poison.json").exists():
        raise GenerationError("PUBLICATION_POISONED")
    fence = validate_fence(board_dir, fence_token)
    accepted_path = board_dir / "generation.accepted.json"
    pointer = read_pointer(board_dir)
    if accepted_path.is_file():
        try:
            accepted = json.loads(accepted_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise GenerationError("PUBLICATION_POISONED") from exc
        if accepted.get("generation_id") != fence.get("generation_id"):
            raise GenerationError("WRITER_FENCE_GENERATION_MISMATCH")
    elif pointer is None or pointer.get("generation_id") != fence.get("generation_id"):
        raise GenerationError("WRITER_FENCE_GENERATION_MISMATCH")
    recover_publication(board_dir)
    custody = contextlib.nullcontext() if str(board_dir.resolve()) in _HELD_FENCE_HANDLES else _lock(board_dir / "generation.writer-lock", exclusive=True)
    held = str(board_dir.resolve()) in _HELD_FENCE_HANDLES
    catalog = contextlib.nullcontext() if held else _lock(board_dir / "generation.catalog-lock", exclusive=True)
    with catalog:
      with custody:
        with _lock(board_dir / "generation.activation-lock", exclusive=True):
            _rollback_locked(board_dir, expected_active_manifest)


def _rollback_locked(board_dir: Path, expected_active_manifest: str) -> None:
    pointer = read_pointer(board_dir)
    if pointer is None:
        raise GenerationError("POINTER_INVALID")
    path = board_dir / "generations" / pointer["generation_id"]
    if (path / "ACTIVATED.json").exists():
        raise GenerationError("ROLLBACK_CLOSED")
    ready = validate_generation(path)
    if ready["manifest_digest"] != expected_active_manifest:
        raise GenerationError("ACTIVE_MANIFEST_MISMATCH")
    previous = board_dir / "generation.previous-pointer"
    if not previous.is_file():
        raise GenerationError("ROLLBACK_NOT_AVAILABLE")
    old_pointer = json.loads(previous.read_text(encoding="utf-8-sig"))
    if old_pointer == {"storage_mode": "legacy"}:
        pointer_path(board_dir).unlink(missing_ok=True)
        (board_dir / "generation.accepted.json").unlink(missing_ok=True)
        _fsync_dir(board_dir)
        return
    old_path = board_dir / "generations" / old_pointer["generation_id"]
    # ``expected_active_manifest`` authenticates the generation currently
    # published by this operation.  The retained pointer is necessarily the
    # prior generation and therefore has a different manifest after a real
    # republish; validate it independently before restoring it.
    validate_generation(old_path)
    write_pointer(board_dir, old_pointer)
    (board_dir / "generation.accepted.json").unlink(missing_ok=True)
    _fsync_dir(board_dir)


def _start_fingerprint(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return "unknown"


def acquire_reader_lease(board_dir: Path, *, consumer: str, operation_id: str,
                         consumer_pid: int | None = None,
                         require_activation: bool = True) -> dict[str, Any]:
    held = str(board_dir.resolve()) in _HELD_FENCE_HANDLES
    # Publication recovery mutates generation.current and therefore completes
    # under exclusive catalog custody before the shared selection transaction.
    recover_publication(board_dir)
    while True:
        catalog = contextlib.nullcontext() if held else _lock(
            board_dir / "generation.catalog-lock", exclusive=False
        )
        with catalog, _lock(
            board_dir / "generation.activation-lock", exclusive=False
        ):
            db, ready = open_generation(
                board_dir, require_activation=require_activation, recover=False
            )
            pid = os.getpid()
            selected_consumer_pid = (
                os.getpid() if consumer_pid is None else consumer_pid
            )
            lease_id = secrets.token_urlsafe(18)
            lease = {
                "lease_format_version": 1,
                "lease_id": lease_id,
                "board_slug": board_dir.name,
                "generation_id": ready["generation_id"],
                "generation_nonce": ready["generation_nonce"],
                "ready_sha256": sha256_file(db.parent / "READY.json"),
                "consumer": consumer,
                "lease_keeper_pid": pid,
                "lease_keeper_start_fingerprint": _start_fingerprint(pid),
                "consumer_pid": selected_consumer_pid,
                "consumer_start_fingerprint": _start_fingerprint(
                    selected_consumer_pid
                ),
                "consumer_kind": consumer,
                "created_at": time.time(),
            }
            leases = db.parent / "leases"
            leases.mkdir(exist_ok=True)
            _reclaim_stale_leases(leases)
            _atomic_json(leases / f"{lease_id}.json", lease)
            _fsync_dir(leases)

            # The durable lease precedes final selection validation. A
            # compliant publisher cannot move the pointer while these shared
            # locks are held, but revalidation also closes corruption and
            # non-cooperating-writer races without returning a stale path.
            try:
                current_db, current_ready = open_generation(
                    board_dir,
                    require_activation=require_activation,
                    recover=False,
                )
            except GenerationError as exc:
                release_reader_lease(db.parent, lease_id)
                raise GenerationError("LEASE_REVALIDATION_FAILED") from exc
            current_identity = (
                current_db,
                current_ready["generation_id"],
                current_ready["generation_nonce"],
                sha256_file(current_db.parent / "READY.json"),
            )
            selected_identity = (
                db,
                lease["generation_id"],
                lease["generation_nonce"],
                lease["ready_sha256"],
            )
            if current_identity != selected_identity:
                release_reader_lease(db.parent, lease_id)
                continue

            activated = db.parent / "ACTIVATED.json"
            return {
                "database_path": str(db),
                "storage_mode": "generation",
                "activated_sha256": (
                    sha256_file(activated) if activated.exists() else None
                ),
                "selected_at": datetime.now(UTC).isoformat().replace(
                    "+00:00", "Z"
                ),
                **{
                    key: lease[key]
                    for key in (
                        "generation_id", "generation_nonce", "ready_sha256",
                        "lease_id", "lease_keeper_pid",
                        "lease_keeper_start_fingerprint", "consumer_pid",
                        "consumer_start_fingerprint", "consumer_kind",
                    )
                },
            }


def _reclaim_stale_leases(leases: Path) -> None:
    for path in leases.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            import psutil
            pid = int(record["lease_keeper_pid"])
            fingerprint = str(record["lease_keeper_start_fingerprint"])
            consumer_pid = int(record["consumer_pid"])
            consumer_fp = str(record["consumer_start_fingerprint"])
            live = (psutil.pid_exists(pid) and _start_fingerprint(pid) == fingerprint) or (psutil.pid_exists(consumer_pid) and _start_fingerprint(consumer_pid) == consumer_fp)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise GenerationError("LEASE_INVALID") from exc
        reclaim_after = float(record.get(
            "reclaim_after",
            float(record["created_at"]) + LEASE_RECLAIM_GRACE_SECONDS,
        ))
        if not live and time.time() >= reclaim_after:
            path.unlink(missing_ok=True)


def release_reader_lease(generation: Path, lease_id: str) -> None:
    path = generation / "leases" / f"{lease_id}.json"
    if not path.is_file():
        raise GenerationError("LEASE_ID_MISMATCH")
    path.unlink()
    _fsync_dir(path.parent)


def discard_created_blobs(conn: sqlite3.Connection, paths: list[Path]) -> None:
    """Remove failed-transaction blobs while retaining any concurrent adoption.

    The transaction guard is released before an exception reaches its caller.
    Reacquire the same board writer custody, inspect the committed attachment
    rows, and unlink only blobs that have no durable reference.
    """
    if not paths:
        return
    from hermes_cli.kanban_db_connect import _main_db_file

    db_file = _main_db_file(conn)
    db_path = Path(db_file) if db_file else None
    board_dir = db_path.parents[2] if db_path and len(db_path.parents) >= 3 else None
    if board_dir is None or not (Path(db_file).parent / "READY.json").is_file():
        _discard_blob_paths(paths)
        return
    expected_generation = getattr(conn, "_kanban_generation_id", None)
    if expected_generation is None:
        try:
            expected_generation = _generation_id(db_path.parent)
        except GenerationError:
            expected_generation = None
    with board_write_guard(
        board_dir,
        owner="attachment-rollback",
        token=current_fence_token(),
        expected_generation=expected_generation,
    ):
        # The filesystem guard orders catalog effects, while IMMEDIATE orders
        # the metadata read against a producer's uncommitted attachment row.
        # Without this second barrier cleanup could observe the row before it
        # commits and unlink the blob that the producer is about to adopt.
        conn.execute("BEGIN IMMEDIATE")
        try:
            for path in paths:
                relative = str(path.relative_to(db_path.parent))
                referenced = conn.execute(
                    "SELECT 1 FROM task_attachments WHERE stored_path = ? LIMIT 1",
                    (relative,),
                ).fetchone()
                if referenced is None:
                    path.unlink(missing_ok=True)
                    _fsync_dir(path.parent)
            conn.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise


def _discard_blob_paths(paths: list[Path]) -> None:
    for path in paths:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
            _fsync_dir(path.parent)


def bind_reader_consumer(generation: Path, lease_id: str, *, pid: int, start_token: str) -> dict[str, Any]:
    """Bind the lease to the child incarnation before publishing reader-ready."""
    path = generation / "leases" / f"{lease_id}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise GenerationError("LEASE_INVALID") from exc
    if record.get("lease_id") != lease_id:
        raise GenerationError("LEASE_ID_MISMATCH")
    record = {
        **record,
        "consumer_pid": int(pid),
        "consumer_start_fingerprint": _start_fingerprint(int(pid)),
    }
    _atomic_json(path, record)
    return record


def resolve_attachment_blob(conn: sqlite3.Connection, attachment: Any, *, board: Any) -> Path:
    """Resolve one attachment through custody; never use a preserved old path."""
    if hasattr(attachment, "id"):
        attachment_id = int(attachment.id)
    elif isinstance(attachment, sqlite3.Row):
        attachment_id = int(attachment["id"])
    else:
        attachment_id = int(attachment)
    row = conn.execute("SELECT * FROM task_attachments WHERE id = ?", (attachment_id,)).fetchone()
    if row is None:
        raise GenerationError("ATTACHMENT_NOT_FOUND")
    generation = Path(board) if isinstance(board, (str, Path)) and Path(board).is_dir() and (Path(board) / "READY.json").exists() else None
    if generation is None:
        try:
            from hermes_cli import kanban_db_connect as _kbc
            db_file = _kbc._main_db_file(conn)
            if db_file and (Path(db_file).parent / "READY.json").is_file():
                generation = Path(db_file).parent
        except (ImportError, OSError, ValueError):
            generation = None
    if generation is None:
        try:
            from hermes_cli import kanban_db as _kb
            root = _kb.attachments_root(board=board).resolve()
        except (ImportError, OSError, ValueError) as exc:
            raise GenerationError("ATTACHMENT_PATH_INVALID") from exc
        raw_candidate = Path(str(row["stored_path"]))
        if raw_candidate.is_symlink():
            raise GenerationError("ATTACHMENT_PATH_INVALID")
        candidate = raw_candidate.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise GenerationError("ATTACHMENT_PATH_INVALID") from exc
        if not candidate.is_file():
            raise GenerationError("ATTACHMENT_MISSING")
        return candidate
    return attachment_path(conn, attachment_id, generation=generation)


def validate_generated_attachment_delete(conn: sqlite3.Connection, attachment_id: int) -> bool:
    """Validate generated custody and report whether GC owns the blob."""
    from hermes_cli import kanban_db_connect as _kbc
    db_file = _kbc._main_db_file(conn)
    if not db_file or not (Path(db_file).parent / "READY.json").is_file():
        return False
    resolve_attachment_blob(conn, attachment_id, board=None)
    return True


def attachment_delete_is_custodied(conn: sqlite3.Connection, attachment_id: int) -> bool:
    return validate_generated_attachment_delete(conn, attachment_id)


def attachment_path(conn: sqlite3.Connection, attachment_id: int, *, generation: Path) -> Path:
    row = conn.execute("SELECT * FROM task_attachments WHERE id = ?", (attachment_id,)).fetchone()
    if row is None:
        raise GenerationError("ATTACHMENT_NOT_FOUND")
    try:
        custody = conn.execute("SELECT generation_id, original_stored_path, source_row_sha256, blob_sha256, size, custody_relpath FROM recovery_attachment_resolution WHERE attachment_id = ?", (attachment_id,)).fetchone()
    except sqlite3.OperationalError as exc:
        raise GenerationError("ATTACHMENT_UNCUSTODIED") from exc
    original = row["stored_path"] if isinstance(row, sqlite3.Row) else row[3]
    row_digest = manifest_digest({key: row[key] for key in row.keys()}) if isinstance(row, sqlite3.Row) else None
    if custody is None or str(original) != str(custody[1]) or str(custody[0]) != _generation_id(generation) or row_digest != str(custody[2]):
        raise GenerationError("ATTACHMENT_ROW_MISMATCH")
    relative = Path(str(custody[5]))
    root = generation.resolve()
    resolved = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or resolved != root / relative or resolved.is_symlink():
        raise GenerationError("ATTACHMENT_PATH_INVALID")
    if not resolved.is_file() or resolved.stat().st_size != int(custody[4]) or sha256_file(resolved) != custody[3]:
        raise GenerationError("ATTACHMENT_CUSTODY_MISMATCH")
    return resolved


def _generation_id(path: Path) -> str:
    try:
        return json.loads((path / "READY.json").read_text(encoding="utf-8-sig"))["generation_id"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise GenerationError("READY_INVALID") from exc


def _mark_authenticated_import_journal(path: Path, marked: set[str]) -> None:
    journal_path = path / "import-journal.json"
    if not journal_path.exists():
        return
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8-sig"))
        if journal.get("state") not in {"incomplete", "prepared", "complete", "rolled-back"}:
            raise ValueError("journal state")
        ready = json.loads((path / "READY.json").read_text(encoding="utf-8-sig"))
        if journal.get("generation_id") != ready.get("generation_id") or not isinstance(journal.get("import_id"), str) or not journal["import_id"]:
            raise ValueError("journal identity")
        with sqlite3.connect(f"file:{path / 'kanban.db'}?mode=ro", uri=True) as conn:
            recovery = conn.execute(
                "SELECT 1 FROM recovery_imports WHERE import_id = ? LIMIT 1",
                (journal["import_id"],),
            ).fetchone()
            if recovery is None:
                raise ValueError("journal import metadata")
            blobs = conn.execute(
                "SELECT target_path, source_sha256, size FROM recovery_attachment_blobs WHERE import_id = ?",
                (journal["import_id"],),
            ).fetchall()
        root = path.resolve()
        for target_path, digest, size in blobs:
            relative = Path(str(target_path))
            resolved = (root / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or resolved != root / relative or resolved.is_symlink():
                raise ValueError("journal blob path")
            if not resolved.is_file() or resolved.stat().st_size != int(size) or sha256_file(resolved) != str(digest):
                raise ValueError("journal blob identity")
        if journal.get("state") in {"incomplete", "prepared"}:
            marked.add(path.name)
            for blob in journal.get("blobs", []):
                blob_path = Path(blob)
                if blob_path.is_absolute() or ".." in blob_path.parts:
                    raise ValueError("journal blob")
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        raise GenerationError(f"IMPORT_JOURNAL_INVALID:{path.name}") from exc


def _resolution_blob_digests(generation: Path, ready: dict[str, Any]) -> set[str]:
    """Return blob digests referenced by current and READY snapshot rows."""
    digests: set[str] = set()
    with sqlite3.connect(
        f"file:{generation / 'kanban.db'}?mode=ro", uri=True
    ) as conn:
        try:
            digests.update(
                str(row[0])
                for row in conn.execute(
                    "SELECT blob_sha256 FROM recovery_attachment_resolution"
                )
                if row[0]
            )
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise GenerationError("BLOB_GC_RESOLUTION_INVALID") from exc
    try:
        table = ready["manifest"]["tables"].get(
            "recovery_attachment_resolution"
        )
        if table:
            index = table["columns"].index("blob_sha256")
            digests.update(str(row[index]) for row in table["rows"] if row[index])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise GenerationError("BLOB_GC_READY_INVALID") from exc
    return digests


def _sweep_generation_blobs(
    generation: Path,
    *,
    mark_all: bool,
    ready: dict[str, Any],
    grace_seconds: int,
    dry_run: bool,
) -> tuple[list[str], list[str]]:
    blob_root = generation / "blobs" / "sha256"
    if not blob_root.is_dir():
        return [], []
    physical = sorted(path for path in blob_root.glob("*/*") if path.is_file())
    marked_digests = (
        {path.name for path in physical}
        if mark_all
        else _resolution_blob_digests(generation, ready)
    )
    marked = [str(path) for path in physical if path.name in marked_digests]
    cutoff = time.time() - grace_seconds
    sweep = [
        path
        for path in physical
        if path.name not in marked_digests and path.stat().st_mtime <= cutoff
    ]
    if not dry_run:
        trash = generation / "blobs" / ".trash"
        for path in sweep:
            trash.mkdir(parents=True, exist_ok=True)
            target = trash / f"{path.name}.{secrets.token_hex(8)}"
            os.replace(path, target)
            _fsync_dir(path.parent)
            _fsync_dir(trash)
            target.unlink()
            _fsync_dir(trash)
    return marked, [str(path) for path in sweep]


def gc_generations(board_dir: Path, *, dry_run: bool = False, grace_seconds: int = 3600) -> dict[str, Any]:  # health: allow CC -- custody mark/sweep must remain one catalog-locked crash boundary
    with _lock(board_dir / "generation.catalog-lock", exclusive=True), _lock(board_dir / "generation.writer-lock", exclusive=True), _lock(board_dir / "generation.activation-lock", exclusive=True):
        root = board_dir / "generations"
        candidates = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.exists() else []
        pointer = read_pointer(board_dir)
        active = pointer["generation_id"] if pointer else None
        marked = {active} if active else set()
        leased: set[str] = set()
        reclaimable_leases: list[Path] = []
        retained = board_dir / "generation.retained.json"
        if retained.exists() or retained.is_symlink():
            try:
                if retained.is_symlink() or not retained.is_file():
                    raise ValueError("retained manifest must be a regular file")
                saved = json.loads(retained.read_text(encoding="utf-8-sig"))
                if isinstance(saved, dict):
                    saved = saved.get("generation_ids", saved.get("retained", []))
                if not isinstance(saved, list) or not all(isinstance(item, str) for item in saved):
                    raise ValueError("retained manifest must be a list")
                marked.update(item for item in saved if item in candidates)
            except (OSError, ValueError, TypeError) as exc:
                raise GenerationError("RETAINED_INVALID") from exc
        previous = board_dir / "generation.previous-pointer"
        if previous.is_file():
            try:
                saved = json.loads(previous.read_text(encoding="utf-8-sig"))
                if saved.get("generation_id") in candidates:
                    marked.add(saved["generation_id"])
            except (OSError, ValueError, AttributeError) as exc:
                raise GenerationError("PREVIOUS_POINTER_INVALID") from exc
        for name in candidates:
            generation = root / name
            lease_files = list((generation / "leases").glob("*.json"))
            # Validate lease records before generation reachability so a
            # malformed lease is never downgraded into an ordinary
            # retained candidate or hidden by another malformed marker.
            for lease_path in lease_files:
                if not _lease_is_reclaimable(lease_path, grace_seconds):
                    marked.add(name)
                    leased.add(name)
                else:
                    reclaimable_leases.append(lease_path)
            ready = generation / "READY.json"
            if not ready.is_file():
                raise GenerationError("READY_INVALID")
            else:
                try:
                    validate_generation(generation, activated=(generation / "ACTIVATED.json").is_file())
                except GenerationError as exc:
                    raise GenerationError(f"READY_INVALID:{name}") from exc
            _mark_authenticated_import_journal(generation, marked)
            if time.time() - generation.stat().st_mtime < grace_seconds:
                marked.add(name)
        trash = board_dir / "generation.trash"
        if trash.exists():
            if not trash.is_dir():
                raise GenerationError("TRASH_INVALID")
            trash.mkdir(exist_ok=True)
            for orphan in list(trash.iterdir()):
                if not orphan.is_dir() or not (orphan / "READY.json").is_file():
                    raise GenerationError("TRASH_INVALID")
                try:
                    validate_generation(orphan, activated=(orphan / "ACTIVATED.json").is_file())
                except GenerationError as exc:
                    raise GenerationError("TRASH_INVALID") from exc
                if orphan.name not in candidates and not (root / orphan.name).exists():
                    os.replace(orphan, root / orphan.name)
            _fsync_dir(root)
        # Authenticate every blob root before any lease, generation, or blob
        # is removed.  Apply mode may then repeat the deterministic sweep while
        # the same catalog/writer/activation lock stack excludes mutation.
        for name in sorted(marked - {None}):
            generation = root / name
            if not generation.is_dir():
                continue
            ready = validate_generation(
                generation, activated=(generation / "ACTIVATED.json").is_file()
            )
            _sweep_generation_blobs(
                generation,
                mark_all=name in leased,
                ready=ready,
                grace_seconds=grace_seconds,
                dry_run=True,
            )
        # All custody markers have now been authenticated.  Only after the
        # complete mark phase succeeds may apply mode remove reclaimable lease
        # records; malformed state therefore causes no partial GC mutation.
        if not dry_run:
            for lease_path in reclaimable_leases:
                lease_path.unlink(missing_ok=True)
            if reclaimable_leases:
                _fsync_dir((reclaimable_leases[0]).parent)
        sweep = [name for name in candidates if name not in marked]
        if not dry_run:
            trash.mkdir(exist_ok=True)
            for name in sweep:
                os.replace(root / name, trash / name)
                _fsync_dir(root)
                _fsync_dir(trash)
                shutil.rmtree(trash / name)
            _fsync_dir(trash)
            _fsync_dir(root)
        blob_marked: list[str] = []
        blob_sweep: list[str] = []
        for name in sorted(marked - {None}):
            generation = root / name
            if not generation.is_dir():
                continue
            ready = validate_generation(
                generation, activated=(generation / "ACTIVATED.json").is_file()
            )
            kept, removed = _sweep_generation_blobs(
                generation,
                mark_all=name in leased,
                ready=ready,
                grace_seconds=grace_seconds,
                dry_run=dry_run,
            )
            blob_marked.extend(kept)
            blob_sweep.extend(removed)
        return {
            "active": active,
            "marked": sorted(marked - {None}),
            "sweep": sweep,
            "blob_marked": blob_marked,
            "blob_sweep": blob_sweep,
            "dry_run": dry_run,
        }


def _lease_is_reclaimable(path: Path, grace_seconds: int) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        if time.time() - float(record["created_at"]) < grace_seconds:
            return False
        return not (_identity_live(record["lease_keeper_pid"], record["lease_keeper_start_fingerprint"]) or _identity_live(record["consumer_pid"], record["consumer_start_fingerprint"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GenerationError("LEASE_INVALID") from exc


def _identity_live(pid: Any, fingerprint: Any) -> bool:
    try:
        return Path(f"/proc/{int(pid)}").exists() and _start_fingerprint(int(pid)) == str(fingerprint)
    except (TypeError, ValueError):
        return False
