"""Native localization worker for UTF-8 text and Markdown files."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home

MAX_RESULT_BYTES = 64_000
LEASE_TTL_SECONDS = 300
LOCAL_SOURCE_TOKEN_BUDGET = 2048
TERMINAL_STATES = {"ABORTED", "FAILED", "BLOCKED", "NEEDS_REVIEW", "COMPLETED"}
PLACEHOLDER_RE = re.compile(r"\{[^{}\r\n]+\}|%\([^)]+\)[a-zA-Z]|%[a-zA-Z]|\$\{[^{}\r\n]+\}")
NUMBER_RE = re.compile(r"(?<![0-9.,])[+-]?(?:\d+(?:[.,]\d+)*|[.,]\d+)(?![0-9.,])")
_SOURCE_ROOTS: tuple[Path, ...] = ()


def _source_roots() -> tuple[Path, ...]:
    return _SOURCE_ROOTS or ((get_hermes_home() / "localization-input").resolve(),)


def _data_dir() -> Path:
    path = get_hermes_home() / "plugins" / "localization-worker"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_data_dir() / "localization-worker.sqlite3", timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if str(mode).lower() != "wal":
        conn.close()
        raise RuntimeError("WAL_REQUIRED")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS jobs(
          id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE, source_path TEXT NOT NULL,
          output_path TEXT NOT NULL, target_locale TEXT NOT NULL, state TEXT NOT NULL,
          source_hash TEXT, bom INTEGER, newline TEXT, final_newline INTEGER,
          output_hash TEXT, verification_receipt TEXT, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS segments(
          job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, segment_id TEXT NOT NULL,
          ordinal INTEGER NOT NULL, source_text TEXT NOT NULL, source_hash TEXT NOT NULL,
          translation TEXT, PRIMARY KEY(job_id, segment_id));
        CREATE TABLE IF NOT EXISTS chunks(
          job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, id TEXT NOT NULL,
          state TEXT NOT NULL, fencing_token TEXT, worker_id TEXT, lease_expires_at REAL,
          submission_hash TEXT,
          PRIMARY KEY(job_id, id));
        CREATE TABLE IF NOT EXISTS chunk_segments(
          job_id TEXT NOT NULL, chunk_id TEXT NOT NULL, segment_id TEXT NOT NULL,
          PRIMARY KEY(job_id, chunk_id, segment_id),
          FOREIGN KEY(job_id, chunk_id) REFERENCES chunks(job_id, id) ON DELETE CASCADE,
          FOREIGN KEY(job_id, segment_id) REFERENCES segments(job_id, segment_id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS audit_events(
          seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id),
          event TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL);
        """
    )
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > 1:
        conn.close()
        raise RuntimeError("UNSUPPORTED_DATABASE_VERSION")
    job_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    chunk_columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
    if "output_hash" not in job_columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN output_hash TEXT")
    if "lease_expires_at" not in chunk_columns:
        conn.execute("ALTER TABLE chunks ADD COLUMN lease_expires_at REAL")
    conn.execute("PRAGMA user_version=1")
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _epoch() -> float:
    return time.time()


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text.encode("utf-8")) + 3) // 4 + 8)


def _dump(value: dict[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(raw.encode("utf-8")) > MAX_RESULT_BYTES:
        return json.dumps({"ok": False, "error": {"code": "RESULT_TOO_LARGE"}})
    return raw


def _error(code: str, message: str = "") -> str:
    return _dump({"ok": False, "error": {"code": code, "message": message or code}})


def _args(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _event(conn: sqlite3.Connection, job_id: str, event: str, detail: dict[str, Any] | None = None) -> None:
    conn.execute(
        "INSERT INTO audit_events(job_id,event,detail,created_at) VALUES(?,?,?,?)",
        (job_id, event, json.dumps(detail or {}, sort_keys=True), _now()),
    )


def _job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise KeyError("JOB_NOT_FOUND")
    return row


def _require(row: sqlite3.Row, expected: str) -> None:
    if row["state"] != expected:
        raise ValueError(f"INVALID_TRANSITION:{row['state']}->{expected}")


def _source_location(path: str) -> tuple[Path, tuple[str, ...]]:
    candidate = Path(path).expanduser()
    roots = _source_roots()
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    normalized = Path(os.path.abspath(candidate))
    for root in roots:
        root_abs = Path(os.path.abspath(root))
        try:
            relative = normalized.relative_to(root_abs)
        except ValueError:
            continue
        if relative.parts and all(part not in {"", ".", ".."} for part in relative.parts):
            return root_abs, relative.parts
    raise ValueError("SOURCE_OUTSIDE_ALLOWED_ROOTS")


def _read_source_no_follow(path: str) -> tuple[Path, bytes]:
    root, parts = _source_location(path)
    directory_fd: int | None = None
    file_fd: int | None = None
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        with _open_absolute_dir_no_follow(root) as trusted_root_fd:
            directory_fd = os.dup(trusted_root_fd)
        for component in parts[:-1]:
            next_fd = os.open(component, flags | os.O_DIRECTORY, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        final_mode = os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False).st_mode
        if stat.S_ISLNK(final_mode):
            raise ValueError("SOURCE_OUTSIDE_ALLOWED_ROOTS")
        if not stat.S_ISREG(final_mode):
            raise ValueError("SOURCE_NOT_FILE")
        file_fd = os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd=directory_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ValueError("SOURCE_NOT_FILE")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return root.joinpath(*parts), b"".join(chunks)
    except ValueError:
        raise
    except OSError as exc:
        if getattr(exc, "errno", None) in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("SOURCE_OUTSIDE_ALLOWED_ROOTS") from None
        if getattr(exc, "errno", None) == errno.EOPNOTSUPP:
            raise ValueError("SOURCE_NOT_FILE") from None
        if isinstance(exc, FileNotFoundError):
            raise ValueError("SOURCE_PATH_INVALID") from None
        raise ValueError("SOURCE_READ_FAILED") from None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _mark_source_failed(job_id: str, event: str) -> None:
    with _connect() as failure_conn:
        failure_conn.execute("UPDATE jobs SET state='FAILED',verification_receipt=NULL WHERE id=?", (job_id,))
        _event(failure_conn, job_id, event)


def _source_bytes_or_fail(conn: sqlite3.Connection, row: sqlite3.Row) -> bytes:
    try:
        _, raw = _read_source_no_follow(row["source_path"])
    except ValueError as exc:
        conn.rollback()
        _mark_source_failed(row["id"], str(exc))
        raise
    if not row["source_hash"] or hashlib.sha256(raw).hexdigest() != row["source_hash"]:
        conn.rollback()
        _mark_source_failed(row["id"], "SOURCE_CHANGED")
        raise ValueError("SOURCE_CHANGED")
    return raw


def _output_components(row: sqlite3.Row) -> tuple[Path, str, str]:
    data_root = _data_dir()
    output = Path(row["output_path"])
    expected_parent = data_root / "jobs" / row["id"]
    if output.parent != expected_parent or output.name in {"", ".", ".."}:
        raise ValueError("OUTPUT_PATH_UNSAFE")
    return data_root, row["id"], output.name


def _mark_output_review(conn: sqlite3.Connection, row: sqlite3.Row, event: str) -> None:
    conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',verification_receipt=NULL WHERE id=?", (row["id"],))
    _event(conn, row["id"], event)
    conn.commit()


@contextmanager
def _open_absolute_dir_no_follow(path: Path):
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError("OUTPUT_PATH_UNSAFE")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_fd = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        yield current_fd
    finally:
        os.close(current_fd)


@contextmanager
def _output_dir_fd(conn: sqlite3.Connection, row: sqlite3.Row, *, create: bool):
    data_root, job_id, output_name = _output_components(row)
    data_fd: int | None = None
    jobs_fd: int | None = None
    job_fd: int | None = None
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        with _open_absolute_dir_no_follow(data_root) as trusted_data_fd:
            data_fd = os.dup(trusted_data_fd)
        if create:
            try:
                os.mkdir("jobs", 0o700, dir_fd=data_fd)
            except FileExistsError:
                pass
        jobs_fd = os.open("jobs", flags, dir_fd=data_fd)
        if create:
            try:
                os.mkdir(job_id, 0o700, dir_fd=jobs_fd)
            except FileExistsError:
                pass
        job_fd = os.open(job_id, flags, dir_fd=jobs_fd)
    except (OSError, ValueError, AttributeError, TypeError):
        for fd in (job_fd, jobs_fd, data_fd):
            if fd is not None:
                os.close(fd)
        _mark_output_review(conn, row, "OUTPUT_PATH_UNSAFE")
        raise ValueError("OUTPUT_PATH_UNSAFE") from None
    try:
        yield output_name, job_fd
    finally:
        for fd in (job_fd, jobs_fd, data_fd):
            if fd is not None:
                os.close(fd)


def _write_output_atomic(conn: sqlite3.Connection, row: sqlite3.Row, raw: bytes) -> None:
    temporary: str | None = None
    try:
        with _output_dir_fd(conn, row, create=True) as (output_name, dir_fd):
            temporary = f".{output_name}.{secrets.token_hex(8)}.tmp"
            file_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
            try:
                view = memoryview(raw)
                while view:
                    written = os.write(file_fd, view)
                    if written <= 0:
                        raise OSError("short write")
                    view = view[written:]
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            os.rename(temporary, output_name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            temporary = None
            os.fsync(dir_fd)
    except ValueError:
        raise
    except (OSError, TypeError):
        _mark_output_review(conn, row, "OUTPUT_WRITE_FAILED")
        raise ValueError("OUTPUT_WRITE_FAILED") from None
    finally:
        if temporary is not None:
            try:
                with _output_dir_fd(conn, row, create=False) as (_, dir_fd):
                    os.unlink(temporary, dir_fd=dir_fd)
            except (OSError, ValueError):
                pass


def _read_output_no_follow(conn: sqlite3.Connection, row: sqlite3.Row) -> bytes:
    try:
        with _output_dir_fd(conn, row, create=False) as (output_name, dir_fd):
            if not stat.S_ISREG(os.stat(output_name, dir_fd=dir_fd, follow_symlinks=False).st_mode):
                raise OSError("output is not a regular file")
            file_fd = os.open(output_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
            try:
                if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                    raise OSError("output is not a regular file")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(file_fd, 64 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                os.close(file_fd)
    except FileNotFoundError:
        raise ValueError("OUTPUT_MISSING") from None
    except OSError:
        _mark_output_review(conn, row, "OUTPUT_PATH_UNSAFE")
        raise ValueError("OUTPUT_PATH_UNSAFE") from None


def _handler(fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[..., str]:
    def wrapped(arguments: Any = None, **kwargs: Any) -> str:
        try:
            return _dump(fn({**_args(arguments), **kwargs}))
        except KeyError as exc:
            return _error(str(exc.args[0]))
        except ValueError as exc:
            text = str(exc)
            code = text.split(":", 1)[0]
            return _error(code, text)
        except RuntimeError as exc:
            return _error(str(exc))
        except sqlite3.Error:
            return _error("DATABASE_ERROR")
        except (TypeError, AttributeError, IndexError):
            return _error("INVALID_ARGUMENTS")
        except (OSError, UnicodeError) as exc:
            return _error("IO_ERROR", str(exc))
    return wrapped


@_handler
def create_job(a: dict[str, Any]) -> dict[str, Any]:
    if "output_path" in a:
        raise ValueError("CALLER_CONTROLLED_OUTPUT_PATH")
    source, source_raw = _read_source_no_follow(a["source_path"])
    if source.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("UNSUPPORTED_FORMAT")
    target_locale = a["target_locale"]
    if not isinstance(target_locale, str) or not re.fullmatch(r"[A-Za-z]{2,8}(?:[-_][A-Za-z0-9]{2,8})*", target_locale):
        raise ValueError("INVALID_TARGET_LOCALE")
    source_hash = hashlib.sha256(source_raw).hexdigest()
    key_material = json.dumps([str(source), source_hash, target_locale], separators=(",", ":"))
    key = hashlib.sha256(key_material.encode()).hexdigest()
    job_id = key[:24]
    job_dir = _data_dir() / "jobs" / job_id
    safe_locale = target_locale.replace("_", "-")
    output = job_dir / f"{source.stem}.{safe_locale}{source.suffix.lower()}"
    with _connect() as conn:
        existing = conn.execute("SELECT id,state,output_path FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
        if existing:
            return {"ok": True, "job_id": existing["id"], "state": existing["state"], "idempotent": True, "data_dir": str(_data_dir()), "output_path": existing["output_path"]}
        conn.execute(
            "INSERT INTO jobs(id,idempotency_key,source_path,output_path,target_locale,state,source_hash,created_at) VALUES(?,?,?,?,?,'CREATED',?,?)",
            (job_id, key, str(source), str(output), target_locale, source_hash, _now()),
        )
        _event(conn, job_id, "JOB_CREATED")
    return {"ok": True, "job_id": job_id, "state": "CREATED", "idempotent": False, "data_dir": str(_data_dir()), "output_path": str(output)}


@_handler
def inspect_job(a: dict[str, Any]) -> dict[str, Any]:
    with _connect() as conn:
        row = _job(conn, a["job_id"]); _require(row, "CREATED")
        if Path(row["source_path"]).suffix.lower() not in {".txt", ".md"}: raise ValueError("UNSUPPORTED_FORMAT")
        raw = _source_bytes_or_fail(conn, row); bom = raw.startswith(b"\xef\xbb\xbf")
        try:
            text = raw[3:].decode("utf-8") if bom else raw.decode("utf-8")
        except UnicodeDecodeError:
            conn.rollback()
            _mark_source_failed(row["id"], "SOURCE_NOT_UTF8")
            raise ValueError("SOURCE_NOT_UTF8") from None
        unsupported_separators = {"\u0085", "\u2028", "\u2029", "\u000b", "\u000c"}
        if any(separator in text for separator in unsupported_separators):
            conn.rollback()
            _mark_source_failed(row["id"], "UNSUPPORTED_NEWLINE_STYLE")
            raise ValueError("UNSUPPORTED_NEWLINE_STYLE")
        crlf_count = text.count("\r\n")
        remainder = text.replace("\r\n", "")
        if "\r" in remainder or (crlf_count and "\n" in remainder):
            conn.execute("UPDATE jobs SET state='FAILED' WHERE id=?", (row["id"],))
            _event(conn, row["id"], "UNSUPPORTED_NEWLINE_STYLE")
            conn.commit()
            raise ValueError("UNSUPPORTED_NEWLINE_STYLE")
        newline = "\r\n" if crlf_count else "\n"
        final = text.endswith(("\n", "\r"))
        digest = hashlib.sha256(raw).hexdigest()
        conn.execute("UPDATE jobs SET state='INSPECTED',bom=?,newline=?,final_newline=? WHERE id=?", (bom, newline, final, row["id"]))
        _event(conn, row["id"], "JOB_INSPECTED", {"source_hash": digest})
    return {"ok": True, "job_id": row["id"], "state": "INSPECTED", "source_hash": digest}


@_handler
def extract_segments(a: dict[str, Any]) -> dict[str, Any]:
    with _connect() as conn:
        row = _job(conn, a["job_id"]); _require(row, "INSPECTED")
        raw = _source_bytes_or_fail(conn, row)
        text = raw[3:].decode() if row["bom"] else raw.decode()
        lines = text.splitlines()
        if not lines:
            conn.execute("UPDATE jobs SET state='FAILED' WHERE id=?", (row["id"],))
            _event(conn, row["id"], "EMPTY_DOCUMENT")
            conn.commit()
            raise ValueError("EMPTY_DOCUMENT")
        segments = []
        for ordinal, text_line in enumerate(lines):
            source_hash = hashlib.sha256(text_line.encode()).hexdigest()
            segment_id = hashlib.sha256(f"{row['source_hash']}:{ordinal}:{source_hash}".encode()).hexdigest()[:24]
            conn.execute("INSERT INTO segments VALUES(?,?,?,?,?,NULL)", (row["id"], segment_id, ordinal, text_line, source_hash))
            segments.append({"segment_id": segment_id, "source_text": text_line, "source_hash": source_hash})
        conn.execute("UPDATE jobs SET state='EXTRACTED' WHERE id=?", (row["id"],)); _event(conn, row["id"], "SEGMENTS_EXTRACTED", {"count": len(segments)})
    return {"ok": True, "job_id": row["id"], "state": "EXTRACTED", "segment_count": len(segments)}


@_handler
def create_chunks(a: dict[str, Any]) -> dict[str, Any]:
    with _connect() as conn:
        row = _job(conn, a["job_id"]); _require(row, "EXTRACTED")
        segments = list(conn.execute("SELECT segment_id,source_text FROM segments WHERE job_id=? ORDER BY ordinal", (row["id"],)))
        groups: list[tuple[list[str], int]] = []
        current: list[str] = []
        current_tokens = 0
        for segment in segments:
            estimated = _estimate_tokens(segment["source_text"])
            if estimated > LOCAL_SOURCE_TOKEN_BUDGET:
                conn.rollback()
                _mark_source_failed(row["id"], "OVERSIZED_SEGMENT")
                raise ValueError("OVERSIZED_SEGMENT")
            if current and current_tokens + estimated > LOCAL_SOURCE_TOKEN_BUDGET:
                groups.append((current, current_tokens))
                current, current_tokens = [], 0
            current.append(segment["segment_id"])
            current_tokens += estimated
        if current:
            groups.append((current, current_tokens))
        chunks = []
        for index, (segment_ids, estimated_tokens) in enumerate(groups):
            chunk_id = hashlib.sha256(f"{row['id']}:{index}".encode()).hexdigest()[:24]
            conn.execute("INSERT INTO chunks VALUES(?,?,'READY',NULL,NULL,NULL,NULL)", (row["id"], chunk_id))
            conn.executemany("INSERT INTO chunk_segments VALUES(?,?,?)", [(row["id"], chunk_id, segment_id) for segment_id in segment_ids])
            chunks.append({"chunk_id": chunk_id, "estimated_tokens": estimated_tokens, "segment_count": len(segment_ids)})
        conn.execute("UPDATE jobs SET state='CHUNKED' WHERE id=?", (row["id"],)); _event(conn, row["id"], "CHUNKS_CREATED", {"count": len(chunks)})
    return {
        "ok": True,
        "job_id": row["id"],
        "state": "CHUNKED",
        "chunk_count": len(chunks),
        "max_estimated_tokens": max((chunk["estimated_tokens"] for chunk in chunks), default=0),
    }


@_handler
def claim_chunk(a: dict[str, Any]) -> dict[str, Any]:
    worker_id = a.get("worker_id")
    if not isinstance(worker_id, str) or not worker_id.strip() or len(worker_id.encode("utf-8")) > 128:
        raise ValueError("INVALID_WORKER_ID")
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _job(conn, a["job_id"])
        if row["state"] not in {"CHUNKED", "PROCESSING"}: raise ValueError("INVALID_TRANSITION")
        now = _epoch()
        chunk = conn.execute(
            "SELECT * FROM chunks WHERE job_id=? AND (state='READY' OR (state='LEASED' AND lease_expires_at<=?)) ORDER BY id LIMIT 1",
            (row["id"], now),
        ).fetchone()
        if not chunk: raise ValueError("NO_CHUNK_AVAILABLE")
        token = secrets.token_hex(16)
        updated = conn.execute(
            "UPDATE chunks SET state='LEASED',fencing_token=?,worker_id=?,lease_expires_at=? "
            "WHERE job_id=? AND id=? AND (state='READY' OR (state='LEASED' AND lease_expires_at<=?))",
            (token, worker_id, now + LEASE_TTL_SECONDS, row["id"], chunk["id"], now),
        ).rowcount
        if updated != 1: raise ValueError("NO_CHUNK_AVAILABLE")
        conn.execute("UPDATE jobs SET state='PROCESSING' WHERE id=?", (row["id"],)); _event(conn, row["id"], "CHUNK_CLAIMED", {"chunk_id": chunk["id"], "worker_id": worker_id})
        segments = [dict(x) for x in conn.execute("SELECT s.segment_id,s.source_text,s.source_hash FROM segments s JOIN chunk_segments c USING(job_id,segment_id) WHERE c.job_id=? AND c.chunk_id=? ORDER BY s.ordinal", (row["id"], chunk["id"]))]
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"ok": True, "job_id": row["id"], "state": "PROCESSING", "chunk_id": chunk["id"], "fencing_token": token, "lease_expires_at": now + LEASE_TTL_SECONDS, "segments": segments}


@_handler
def submit_chunk(a: dict[str, Any]) -> dict[str, Any]:
    translations = a.get("translations") or []
    canonical = json.dumps(translations, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(canonical.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ValueError("SUBMISSION_TOO_LARGE")
    submission_hash = hashlib.sha256(canonical.encode()).hexdigest()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _job(conn, a["job_id"])
        if row["state"] not in {"PROCESSING", "VALIDATING"}:
            raise ValueError("INVALID_TRANSITION")
        chunk = conn.execute("SELECT * FROM chunks WHERE job_id=? AND id=?", (row["id"], a["chunk_id"])).fetchone()
        if not chunk: raise ValueError("CHUNK_NOT_FOUND")
        if chunk["fencing_token"] != a.get("fencing_token"):
            raise ValueError("STALE_LEASE")
        if chunk["state"] in {"SUBMITTED", "VALIDATED"} and chunk["submission_hash"] == submission_hash:
            conn.commit()
            return {"ok": True, "accepted": True, "idempotent": True, "state": row["state"]}
        if row["state"] != "PROCESSING":
            raise ValueError("INVALID_TRANSITION")
        if chunk["state"] != "LEASED" or chunk["lease_expires_at"] <= _epoch():
            raise ValueError("STALE_LEASE")
        expected = {x[0] for x in conn.execute("SELECT segment_id FROM chunk_segments WHERE job_id=? AND chunk_id=?", (row["id"], chunk["id"]))}
        provided = [x.get("segment_id") for x in translations]
        if len(provided) != len(set(provided)) or set(provided) != expected: raise ValueError("SEGMENT_ID_SET_MISMATCH")
        for item in translations:
            source = conn.execute("SELECT source_text FROM segments WHERE job_id=? AND segment_id=?", (row["id"], item["segment_id"])).fetchone()[0]
            target = item["text"]
            if not isinstance(target, str): raise ValueError("INVALID_TRANSLATION")
            if "\n" in target or "\r" in target: raise ValueError("TRANSLATION_CONTAINS_NEWLINE")
            if source == "":
                if target != "": raise ValueError("BLANK_LINE_CHANGED")
            elif not target.strip():
                raise ValueError("EMPTY_TRANSLATION")
            if sorted(PLACEHOLDER_RE.findall(source)) != sorted(PLACEHOLDER_RE.findall(target)): raise ValueError("PLACEHOLDER_MISMATCH")
            if NUMBER_RE.findall(source) != NUMBER_RE.findall(target): raise ValueError("NUMBER_MISMATCH")
            conn.execute("UPDATE segments SET translation=? WHERE job_id=? AND segment_id=?", (target, row["id"], item["segment_id"]))
        conn.execute("UPDATE chunks SET state='SUBMITTED',submission_hash=? WHERE job_id=? AND id=?", (submission_hash, row["id"], chunk["id"])); _event(conn, row["id"], "CHUNK_SUBMITTED", {"chunk_id": chunk["id"]})
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"ok": True, "accepted": True, "idempotent": False, "state": "PROCESSING"}


@_handler
def validate_chunk(a: dict[str, Any]) -> dict[str, Any]:
    with _connect() as conn:
        row = _job(conn, a["job_id"]); _require(row, "PROCESSING")
        chunk = conn.execute("SELECT state FROM chunks WHERE job_id=? AND id=?", (row["id"], a["chunk_id"])).fetchone()
        if not chunk or chunk["state"] != "SUBMITTED": raise ValueError("CHUNK_NOT_SUBMITTED")
        conn.execute("UPDATE chunks SET state='VALIDATED' WHERE job_id=? AND id=?", (row["id"], a["chunk_id"]))
        remaining = conn.execute("SELECT count(*) FROM chunks WHERE job_id=? AND state!='VALIDATED'", (row["id"],)).fetchone()[0]
        next_state = "VALIDATING" if remaining == 0 else "PROCESSING"
        conn.execute("UPDATE jobs SET state=? WHERE id=?", (next_state, row["id"]))
        _event(conn, row["id"], "CHUNK_VALIDATED", {"chunk_id": a["chunk_id"], "remaining": remaining})
    return {"ok": True, "job_id": row["id"], "state": next_state}


@_handler
def assemble_output(a: dict[str, Any]) -> dict[str, Any]:
    with _connect() as conn:
        row = _job(conn, a["job_id"]); _require(row, "VALIDATING")
        _source_bytes_or_fail(conn, row)
        if conn.execute("SELECT count(*) FROM chunks WHERE job_id=? AND state!='VALIDATED'", (row["id"],)).fetchone()[0]: raise ValueError("UNVALIDATED_CHUNKS")
        translations = [x[0] for x in conn.execute("SELECT translation FROM segments WHERE job_id=? ORDER BY ordinal", (row["id"],))]
        if any(x is None for x in translations): raise ValueError("MISSING_TRANSLATION")
        text = row["newline"].join(translations) + (row["newline"] if row["final_newline"] else "")
        raw = (("\ufeff" if row["bom"] else "") + text).encode("utf-8")
        _write_output_atomic(conn, row, raw)
        conn.execute("UPDATE jobs SET state='ASSEMBLING' WHERE id=?", (row["id"],)); _event(conn, row["id"], "OUTPUT_ASSEMBLED", {"output_hash": hashlib.sha256(raw).hexdigest()})
    return {"ok": True, "job_id": row["id"], "state": "ASSEMBLING", "output_path": row["output_path"]}


@_handler
def verify_output(a: dict[str, Any]) -> dict[str, Any]:
    with _connect() as conn:
        row = _job(conn, a["job_id"]); _require(row, "ASSEMBLING")
        _source_bytes_or_fail(conn, row)
        try:
            raw = _read_output_no_follow(conn, row)
        except ValueError as exc:
            code = str(exc)
            event = "OUTPUT_MISSING_BEFORE_VERIFICATION" if code == "OUTPUT_MISSING" else "OUTPUT_PATH_UNSAFE"
            conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',verification_receipt=NULL WHERE id=?", (row["id"],))
            _event(conn, row["id"], event)
            conn.commit()
            raise
        conn.execute("UPDATE jobs SET state='VERIFYING' WHERE id=?", (row["id"],)); _event(conn, row["id"], "OUTPUT_VERIFYING")
        bom = raw.startswith(b"\xef\xbb\xbf")
        try:
            text = (raw[3:] if bom else raw).decode("utf-8")
        except UnicodeDecodeError:
            conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',verification_receipt=NULL WHERE id=?", (row["id"],))
            _event(conn, row["id"], "OUTPUT_NOT_UTF8")
            conn.commit()
            raise ValueError("OUTPUT_NOT_UTF8") from None
        payload = text[:-len(row["newline"])] if row["final_newline"] and text.endswith(row["newline"]) else text
        reparsed = payload.split(row["newline"])
        expected = [x[0] for x in conn.execute("SELECT translation FROM segments WHERE job_id=? ORDER BY ordinal", (row["id"],))]
        without_expected_newlines = text.replace(row["newline"], "")
        newline_ok = "\r" not in without_expected_newlines and "\n" not in without_expected_newlines
        if reparsed != expected or not newline_ok or bom != bool(row["bom"]) or text.endswith(("\n", "\r")) != bool(row["final_newline"]):
            conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',verification_receipt=NULL WHERE id=?", (row["id"],)); _event(conn, row["id"], "OUTPUT_VERIFICATION_FAILED")
            conn.commit()
            raise ValueError("OUTPUT_REPARSE_MISMATCH")
        output_hash = hashlib.sha256(raw).hexdigest()
        receipt = hashlib.sha256((row["id"] + output_hash + _now()).encode()).hexdigest()
        conn.execute("UPDATE jobs SET state='COMPLETED',output_hash=?,verification_receipt=? WHERE id=?", (output_hash, receipt, row["id"])); _event(conn, row["id"], "OUTPUT_VERIFIED", {"receipt": receipt, "output_hash": output_hash})
    return {"ok": True, "job_id": row["id"], "state": "COMPLETED", "verification_receipt": receipt}


@_handler
def get_job_status(a: dict[str, Any]) -> dict[str, Any]:
    with _connect() as conn:
        row = _job(conn, a["job_id"])
        if row["state"] == "COMPLETED":
            try:
                raw = _read_output_no_follow(conn, row)
                actual_hash = hashlib.sha256(raw).hexdigest()
            except ValueError as exc:
                if str(exc) == "OUTPUT_MISSING":
                    actual_hash = None
                else:
                    conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',verification_receipt=NULL WHERE id=?", (row["id"],))
                    _event(conn, row["id"], "OUTPUT_PATH_UNSAFE")
                    conn.commit()
                    raise
            if not row["output_hash"] or actual_hash != row["output_hash"]:
                conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',verification_receipt=NULL WHERE id=?", (row["id"],))
                _event(conn, row["id"], "OUTPUT_CHANGED_AFTER_VERIFICATION", {"expected_hash": row["output_hash"], "actual_hash": actual_hash})
                conn.commit()
                raise ValueError("OUTPUT_CHANGED_AFTER_VERIFICATION")
        if row["state"] == "NEEDS_REVIEW" and conn.execute(
            "SELECT 1 FROM audit_events WHERE job_id=? AND event='OUTPUT_CHANGED_AFTER_VERIFICATION' LIMIT 1",
            (row["id"],),
        ).fetchone():
            raise ValueError("OUTPUT_CHANGED_AFTER_VERIFICATION")
        count = conn.execute("SELECT count(*) FROM audit_events WHERE job_id=?", (row["id"],)).fetchone()[0]
    return {"ok": True, "job_id": row["id"], "state": row["state"], "verification_receipt": row["verification_receipt"], "audit_event_count": count}


@_handler
def abort_job(a: dict[str, Any]) -> dict[str, Any]:
    reason = a.get("reason", "")
    if not isinstance(reason, str) or len(reason.encode("utf-8")) > 1024:
        raise ValueError("ABORT_REASON_TOO_LARGE")
    with _connect() as conn:
        row = _job(conn, a["job_id"])
        if row["state"] in TERMINAL_STATES: raise ValueError("INVALID_TRANSITION")
        conn.execute("UPDATE jobs SET state='ABORTED' WHERE id=?", (row["id"],)); _event(conn, row["id"], "JOB_ABORTED", {"reason": reason})
    return {"ok": True, "job_id": row["id"], "state": "ABORTED"}


_HANDLERS = {name: globals()[name] for name in (
    "create_job", "inspect_job", "extract_segments", "create_chunks", "claim_chunk",
    "submit_chunk", "validate_chunk", "assemble_output", "verify_output", "get_job_status", "abort_job",
)}


def _schema(name: str) -> dict[str, Any]:
    specs: dict[str, dict[str, Any]] = {
        "create_job": {"source_path": {"type": "string", "minLength": 1}, "target_locale": {"type": "string", "minLength": 2}},
        "inspect_job": {"job_id": {"type": "string", "minLength": 1}},
        "extract_segments": {"job_id": {"type": "string", "minLength": 1}},
        "create_chunks": {"job_id": {"type": "string", "minLength": 1}},
        "claim_chunk": {"job_id": {"type": "string", "minLength": 1}, "worker_id": {"type": "string", "minLength": 1, "maxLength": 128}},
        "submit_chunk": {
            "job_id": {"type": "string", "minLength": 1},
            "chunk_id": {"type": "string", "minLength": 1},
            "fencing_token": {"type": "string", "minLength": 1},
            "translations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2048,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["segment_id", "text"],
                    "properties": {
                        "segment_id": {"type": "string", "minLength": 1},
                        "text": {"type": "string", "maxLength": 64000},
                    },
                },
            },
        },
        "validate_chunk": {"job_id": {"type": "string", "minLength": 1}, "chunk_id": {"type": "string", "minLength": 1}},
        "assemble_output": {"job_id": {"type": "string", "minLength": 1}},
        "verify_output": {"job_id": {"type": "string", "minLength": 1}},
        "get_job_status": {"job_id": {"type": "string", "minLength": 1}},
        "abort_job": {"job_id": {"type": "string", "minLength": 1}, "reason": {"type": "string", "maxLength": 1024}},
    }
    properties = specs[name]
    required = list(properties) if name != "abort_job" else ["job_id"]
    return {
        "name": f"localization_{name}",
        "description": f"Localization worker: {name.replace('_', ' ')}.",
        "parameters": {"type": "object", "additionalProperties": False, "required": required, "properties": properties},
    }


def _require_security_primitives() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    required_dir_fd = (os.open, os.mkdir, os.unlink, os.rename, os.stat)
    if (
        os.name != "posix"
        or any(not hasattr(os, flag) for flag in required_flags)
        or any(function not in os.supports_dir_fd for function in required_dir_fd)
        or os.stat not in os.supports_follow_symlinks
    ):
        raise RuntimeError("UNSUPPORTED_PLATFORM_SECURITY_PRIMITIVES")


def register(ctx: Any) -> None:
    global _SOURCE_ROOTS
    _require_security_primitives()
    configured = ctx.get_config("source_roots", [str(get_hermes_home() / "localization-input")])
    if not isinstance(configured, list) or not configured or not all(isinstance(item, str) and item.strip() for item in configured):
        raise ValueError("INVALID_SOURCE_ROOTS")
    _SOURCE_ROOTS = tuple(Path(item).expanduser().resolve() for item in configured)
    for name, handler in _HANDLERS.items():
        ctx.register_tool(name=f"localization_{name}", toolset="localization_worker", schema=_schema(name), handler=handler, emoji="🌐")
