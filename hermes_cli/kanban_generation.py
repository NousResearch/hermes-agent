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
from pathlib import Path
from typing import Any, Iterator


class GenerationError(ValueError):
    """A generation is malformed, inconsistent, or unsafe to consume."""


CHILD_TABLES = ("task_comments", "task_events", "task_runs", "task_attachments")
POINTER_FIELDS = {"format_version", "generation_id", "generation_nonce", "ready_sha256"}
ACCEPTED_FIELDS = {"format_version", "generation_id", "generation_nonce", "ready_sha256", "fence_epoch", "accepted_at"}
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
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, canonical_json(value) + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    if durability:
        durability.checkpoint(f"{path.name}-before-rename")
    os.replace(temporary, path)
    (durability or DurabilityAdapter()).fsync_dir(path.parent)


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
def board_write_guard(board_dir: Path, *, owner: str = "native", token: str | None = None) -> Iterator[None]:
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
    custody = contextlib.nullcontext() if str(board_dir.resolve()) in _HELD_FENCE_HANDLES else _lock(board_dir / "generation.writer-lock", exclusive=True)
    with custody:
        fence_path = board_dir / "generation.writer-fence"
        if fence_path.exists():
            validate_fence(board_dir, token or "")
        read_pointer(board_dir)
        yield


def acquire_fence(board_dir: Path, *, generation: str, owner: str) -> dict[str, Any]:
    path = board_dir / "generation.writer-fence"
    if path.exists():
        raise GenerationError("WRITER_FENCE_HELD")
    catalog_handle = (board_dir / "generation.catalog-lock").open("a+b")
    writer_handle = path.with_name("generation.writer-lock").open("a+b")
    import fcntl
    try:
        # Every custody path is catalog → writer → activation.  Retaining the
        # catalog handle here prevents publish/rollback/GC from taking the
        # locks in the opposite order while this fence is live.
        fcntl.flock(catalog_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(writer_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fcntl.flock(catalog_handle.fileno(), fcntl.LOCK_UN)
        catalog_handle.close()
        writer_handle.close()
        raise GenerationError("WRITER_FENCE_HELD") from exc
    record = {"board": board_dir.name, "generation_id": generation,
              "epoch": time.time_ns(), "owner": owner, "token": secrets.token_urlsafe(24)}
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


def adopt_fence(board_dir: Path, *, token: str, owner: str) -> dict[str, Any]:
    """Adopt a persisted fence after a crash without minting a second token."""
    key = str(board_dir.resolve())
    if key in _HELD_FENCE_HANDLES:
        custody = contextlib.nullcontext()
    else:
        catalog_handle = (board_dir / "generation.catalog-lock").open("a+b")
        handle = (board_dir / "generation.writer-lock").open("a+b")
        import fcntl
        fcntl.flock(catalog_handle.fileno(), fcntl.LOCK_EX)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _HELD_FENCE_HANDLES[key] = (catalog_handle, handle)
        custody = contextlib.nullcontext()
    with custody:
        record = validate_fence(board_dir, token)
        record = {**record, "owner": owner, "epoch": time.time_ns()}
        _atomic_json(board_dir / "generation.writer-fence", record)
        return record


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


def build_ready(path: Path, *, gid: str, nonce: str) -> dict[str, Any]:
    with sqlite3.connect(path / "kanban.db") as conn:
        manifest = row_manifest(conn)
    ready = {"format_version": 1, "generation_id": gid, "generation_nonce": nonce, "database_sha256": sha256_file(path / "kanban.db"), "manifest": manifest, "manifest_digest": manifest_digest(manifest)}
    _atomic_json(path / "READY.json", ready)
    return ready


def validate_generation(path: Path, *, activated: bool = False) -> dict[str, Any]:
    _assert_not_poisoned(path)
    try:
        ready = json.loads((path / "READY.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise GenerationError("READY_INVALID") from exc
    required = {"format_version", "generation_id", "generation_nonce", "database_sha256", "manifest", "manifest_digest"}
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
        manifest = row_manifest(conn)
        if not activated and (manifest_digest(manifest) != ready["manifest_digest"] or manifest != ready["manifest"]):
            raise GenerationError("READY_INVALID")
    if activated:
        try:
            marker = json.loads((path / "ACTIVATED.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise GenerationError("ACTIVATION_INVALID") from exc
        if marker.get("generation_id") != ready["generation_id"] or marker.get("generation_nonce") != ready["generation_nonce"] or marker.get("ready_sha256") != sha256_file(path / "READY.json"):
            raise GenerationError("ACTIVATION_INVALID")
    return ready


def active_manifest(path: Path) -> str:
    """Authenticate and return the current bytes of an activated generation."""
    validate_generation(path, activated=True)
    with sqlite3.connect(f"file:{path / 'kanban.db'}?mode=ro", uri=True) as conn:
        return manifest_digest(row_manifest(conn))


def _assert_not_poisoned(path: Path) -> None:
    if (path / "ACTIVATION.poison.json").exists():
        raise GenerationError("ACTIVATION_POISONED")


def recover_publication(board_dir: Path) -> None:
    """Finish or refuse an interrupted acceptance-before-pointer transition."""
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
        # Acceptance is a recovery journal, not a second active-state marker.
        # Once the pointer is durable and the candidate has either remained
        # prepared or crossed the first-write boundary, remove it so a later
        # reopen cannot mistake a mutated active DB for an unprepared one.
        accepted_path.unlink(missing_ok=True)
        _fsync_dir(board_dir)
    except (OSError, ValueError, KeyError, TypeError, GenerationError) as exc:
        _atomic_json(board_dir / "generation.publication.poison.json", {"reason": str(exc), "accepted": accepted})
        raise GenerationError("PUBLICATION_POISONED") from exc


def open_generation(board_dir: Path, *, require_activation: bool = False) -> tuple[Path, dict[str, Any]]:
    if (board_dir / "generation.publication.poison.json").exists():
        raise GenerationError("PUBLICATION_POISONED")
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


def activate_first_write(path: Path, *, fence_epoch: int,
                         durability: DurabilityAdapter | None = None,
                         board_dir: Path | None = None) -> dict[str, Any]:
    ready = validate_generation(path)
    marker = {"format_version": 1, "generation_id": ready["generation_id"], "generation_nonce": ready["generation_nonce"], "ready_sha256": sha256_file(path / "READY.json"), "fence_epoch": fence_epoch, "activated_at": time.time()}
    board_dir = board_dir or path.parents[1]
    with _lock(board_dir / "generation.activation-lock", exclusive=True):
        _assert_not_poisoned(path)
        final = path / "ACTIVATED.json"
        if final.exists():
            return json.loads(final.read_text(encoding="utf-8-sig"))
        try:
            _atomic_json(final, marker, durability=durability)
            if json.loads(final.read_text(encoding="utf-8-sig")) != marker:
                raise GenerationError("ACTIVATION_INVALID")
            accepted = path.parents[1] / "generation.accepted.json"
            if accepted.exists():
                try:
                    record = json.loads(accepted.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError) as exc:
                    raise GenerationError("PUBLICATION_POISONED") from exc
                if record.get("generation_id") != marker["generation_id"]:
                    raise GenerationError("PUBLICATION_POISONED")
                accepted.unlink()
                _fsync_dir(accepted.parent)
        except Exception as exc:
            with contextlib.suppress(OSError):
                final.unlink()
            with contextlib.suppress(OSError):
                _fsync_dir(path)
            _atomic_json(path / "ACTIVATION.poison.json", {
                "generation_id": ready["generation_id"],
                "reason": str(exc),
                "ready_sha256": marker["ready_sha256"],
            })
            if isinstance(exc, OSError):
                raise
            raise GenerationError("ACTIVATION_RETRYABLE") from exc
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
    recover_publication(board_dir)
    validate_fence(board_dir, fence_token)
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
    catalog = contextlib.nullcontext() if held else _lock(board_dir / "generation.catalog-lock", exclusive=False)
    with catalog, _lock(board_dir / "generation.activation-lock", exclusive=False):
        db, ready = open_generation(board_dir, require_activation=require_activation)
        pid = os.getpid()
        consumer_pid = os.getpid() if consumer_pid is None else consumer_pid
        lease_id = secrets.token_urlsafe(18)
        lease = {"lease_format_version": 1, "lease_id": lease_id, "board_slug": board_dir.name, "generation_id": ready["generation_id"], "generation_nonce": ready["generation_nonce"], "ready_sha256": sha256_file(db.parent / "READY.json"), "consumer": consumer, "lease_keeper_pid": pid, "lease_keeper_start_fingerprint": _start_fingerprint(pid), "consumer_pid": consumer_pid, "consumer_start_fingerprint": _start_fingerprint(consumer_pid), "consumer_kind": consumer, "operation_id": operation_id, "created_at": time.time(), "reclaim_after": time.time() + LEASE_RECLAIM_GRACE_SECONDS}
        leases = db.parent / "leases"
        leases.mkdir(exist_ok=True)
        _reclaim_stale_leases(leases)
        _atomic_json(leases / f"{lease_id}.json", lease)
        _fsync_dir(leases)
        activated = db.parent / "ACTIVATED.json"
        return {"database_path": str(db), "storage_mode": "generation",
                "activated_sha256": sha256_file(activated) if activated.exists() else None,
                **{key: lease[key] for key in ("generation_id", "generation_nonce", "ready_sha256", "lease_id", "lease_keeper_pid", "lease_keeper_start_fingerprint", "consumer_pid", "consumer_start_fingerprint")}}


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
        if not live and ("reclaim_after" not in record or time.time() >= float(record["reclaim_after"])):
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
    with board_write_guard(
        board_dir,
        owner="attachment-rollback",
        token=current_fence_token(),
    ):
        for path in paths:
            relative = str(path.relative_to(db_path.parent))
            referenced = conn.execute(
                "SELECT 1 FROM task_attachments WHERE stored_path = ? LIMIT 1",
                (relative,),
            ).fetchone()
            if referenced is None:
                path.unlink(missing_ok=True)
                _fsync_dir(path.parent)


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
    record = {**record, "consumer_pid": int(pid),
              "consumer_start_fingerprint": _start_fingerprint(int(pid)),
              "start_token": start_token, "bound_at": time.time()}
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
    try:
        return validate_generated_attachment_delete(conn, attachment_id)
    except (ImportError, ValueError, OSError, sqlite3.Error):
        return False


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


def gc_generations(board_dir: Path, *, dry_run: bool = False, grace_seconds: int = 3600) -> dict[str, Any]:  # health: allow CC -- custody mark/sweep must remain one catalog-locked crash boundary
    with _lock(board_dir / "generation.catalog-lock", exclusive=True), _lock(board_dir / "generation.writer-lock", exclusive=True), _lock(board_dir / "generation.activation-lock", exclusive=True):
        root = board_dir / "generations"
        candidates = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.exists() else []
        pointer = read_pointer(board_dir)
        active = pointer["generation_id"] if pointer else None
        marked = {active} if active else set()
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
            if (generation / "import-journal.json").exists():
                try:
                    journal = json.loads((generation / "import-journal.json").read_text(encoding="utf-8-sig"))
                    if journal.get("state") not in {"prepared", "complete", "rolled-back"}:
                        raise ValueError("journal state")
                except (OSError, ValueError, AttributeError) as exc:
                    raise GenerationError(f"IMPORT_JOURNAL_INVALID:{name}") from exc
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
        return {"active": active, "marked": sorted(marked - {None}), "sweep": sweep, "dry_run": dry_run}


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
