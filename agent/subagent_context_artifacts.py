from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from hermes_constants import get_hermes_home

_SCHEMA_VERSION = 1
_KIND = "subagent_context_payload"
_SAFE_PATH_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _now() -> float:
    return time.time()


def _safe_child_id(child_session_id: str) -> str:
    safe = _SAFE_PATH_RE.sub("_", str(child_session_id or "").strip())
    safe = safe.strip("._")
    return safe or "unknown-child"


def _session_home(session_db: Any = None) -> Path:
    db_path = getattr(session_db, "db_path", None)
    if db_path:
        try:
            return Path(db_path).expanduser().resolve().parent
        except Exception:
            pass
    return get_hermes_home()


def _artifact_path(child_session_id: str, session_db: Any = None) -> Path:
    return _session_home(session_db) / "artifacts" / "subagent-context" / _safe_child_id(child_session_id) / "latest.json"


def _private_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass


def _private_file(path: Path) -> None:
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def _json_safe(value: Any, warnings: list[str], path: str = "$") -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        warnings.append(f"non-serializable bytes at {path}")
        return f"[non_serializable:bytes len={len(value)}]"
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v, warnings, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                safe_key = key
            else:
                safe_key = str(key)
                warnings.append(f"non-string dict key at {path}: {key!r}")
            out[safe_key] = _json_safe(item, warnings, f"{path}.{safe_key}")
        return out
    try:
        copied = copy.deepcopy(value)
        json.dumps(copied)
        return copied
    except Exception:
        warnings.append(f"non-serializable value at {path}: {type(value).__name__}")
        return f"[non_serializable:{type(value).__name__}]"


def json_safe_copy(value: Any) -> Tuple[Any, list[str]]:
    warnings: list[str] = []
    safe = _json_safe(value, warnings)
    return safe, warnings


def _db(session_db: Any = None) -> tuple[Any, bool]:
    if session_db is not None:
        return session_db, False
    from hermes_state import SessionDB

    return SessionDB(), True


def _row_to_dict(row: Any) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    return dict(row)


def _ensure_schema(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS subagent_context_artifacts (
            child_session_id TEXT PRIMARY KEY,
            parent_session_id TEXT,
            subagent_id TEXT,
            latest_artifact_path TEXT NOT NULL,
            capture_sequence INTEGER NOT NULL DEFAULT 0,
            latest_api_call_count INTEGER,
            latest_retry_count INTEGER,
            status TEXT NOT NULL DEFAULT 'capturing',
            role TEXT,
            profile TEXT,
            model TEXT,
            provider TEXT,
            api_mode TEXT,
            base_url TEXT,
            toolsets_json TEXT,
            artifact_sha256 TEXT,
            artifact_size_bytes INTEGER,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            captured_at REAL,
            finalized_at REAL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_subagent_context_parent ON subagent_context_artifacts(parent_session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_subagent_context_subagent ON subagent_context_artifacts(subagent_id)"
    )


def _fetch_pointer(conn: Any, child_session_id: str) -> Optional[dict[str, Any]]:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM subagent_context_artifacts WHERE child_session_id = ?",
        (child_session_id,),
    ).fetchone()
    pointer = _row_to_dict(row)
    if pointer and pointer.get("toolsets_json"):
        try:
            pointer["toolsets"] = json.loads(pointer["toolsets_json"])
        except Exception:
            pointer["toolsets"] = []
    elif pointer:
        pointer["toolsets"] = []
    return pointer


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> tuple[str, int]:
    _private_mkdir(path.parent.parent)
    _private_mkdir(path.parent)
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    _private_file(tmp)
    os.replace(tmp, path)
    _private_file(path)
    return hashlib.sha256(data).hexdigest(), len(data)


def _initial_artifact(pointer: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": _KIND,
        "raw_unredacted_by_viewer": True,
        "child_session_id": pointer.get("child_session_id"),
        "parent_session_id": pointer.get("parent_session_id"),
        "subagent_id": pointer.get("subagent_id"),
        "capture_sequence": pointer.get("capture_sequence", 0),
        "api_call_count": None,
        "retry_count": None,
        "captured_at": None,
        "finalized_at": pointer.get("finalized_at"),
        "status": pointer.get("status", "capturing"),
        "role": pointer.get("role"),
        "profile": pointer.get("profile"),
        "model": pointer.get("model"),
        "provider": pointer.get("provider"),
        "api_mode": pointer.get("api_mode"),
        "base_url": pointer.get("base_url"),
        "toolsets": pointer.get("toolsets", []),
        "canonical_messages": [],
        "provider_request": {},
        "provider_request_keys": [],
        "serialization_warnings": [],
    }


def create_subagent_context_artifact_pointer(
    *,
    child_session_id: str,
    parent_session_id: Optional[str] = None,
    subagent_id: Optional[str] = None,
    role: Optional[str] = None,
    profile: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_mode: Optional[str] = None,
    base_url: Optional[str] = None,
    toolsets: Optional[Iterable[str]] = None,
    session_db: Any = None,
) -> dict[str, Any]:
    db, close = _db(session_db)
    try:
        now = _now()
        child_session_id = str(child_session_id or "")
        path = _artifact_path(child_session_id, db)
        toolsets_list = list(toolsets or [])
        toolsets_json = json.dumps(toolsets_list, ensure_ascii=False)

        def write(conn: Any) -> dict[str, Any]:
            _ensure_schema(conn)
            existing = _fetch_pointer(conn, child_session_id)
            if existing:
                return existing
            conn.execute(
                """
                INSERT INTO subagent_context_artifacts (
                    child_session_id, parent_session_id, subagent_id,
                    latest_artifact_path, capture_sequence, status,
                    role, profile, model, provider, api_mode, base_url,
                    toolsets_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 0, 'capturing', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    child_session_id,
                    parent_session_id,
                    subagent_id,
                    str(path),
                    role,
                    profile,
                    model,
                    provider,
                    api_mode,
                    base_url,
                    toolsets_json,
                    now,
                    now,
                ),
            )
            return _fetch_pointer(conn, child_session_id) or {}

        pointer = db._execute_write(write)
        sha, size = _write_json_atomic(path, _initial_artifact(pointer))

        def update_meta(conn: Any) -> dict[str, Any]:
            _ensure_schema(conn)
            conn.execute(
                """
                UPDATE subagent_context_artifacts
                   SET artifact_sha256 = ?, artifact_size_bytes = ?, updated_at = ?
                 WHERE child_session_id = ?
                """,
                (sha, size, _now(), child_session_id),
            )
            return _fetch_pointer(conn, child_session_id) or {}

        return db._execute_write(update_meta)
    finally:
        if close:
            db.close()


def update_subagent_context_artifact_capture(
    child_session_id: str,
    payload: dict[str, Any],
    *,
    session_db: Any = None,
) -> dict[str, Any]:
    db, close = _db(session_db)
    try:
        child_session_id = str(child_session_id or "")
        pointer = _fetch_pointer(db._conn, child_session_id)
        if not pointer:
            return {"ok": False, "error": {"code": "pointer_missing", "message": "subagent context pointer not found"}}
        if pointer.get("status") == "finalized":
            return {"ok": False, "ignored": True, "status": "finalized"}
        safe_payload, warnings = json_safe_copy(payload)
        if not isinstance(safe_payload, dict):
            safe_payload = {"payload": safe_payload}
        sequence = int(pointer.get("capture_sequence") or 0) + 1
        existing_warnings = list(safe_payload.get("serialization_warnings") or [])
        safe_payload.update(
            {
                "schema_version": safe_payload.get("schema_version", _SCHEMA_VERSION),
                "kind": safe_payload.get("kind", _KIND),
                "raw_unredacted_by_viewer": True,
                "child_session_id": child_session_id,
                "parent_session_id": safe_payload.get("parent_session_id", pointer.get("parent_session_id")),
                "subagent_id": safe_payload.get("subagent_id", pointer.get("subagent_id")),
                "capture_sequence": sequence,
                "status": "capturing",
                "finalized_at": None,
                "role": safe_payload.get("role", pointer.get("role")),
                "profile": safe_payload.get("profile", pointer.get("profile")),
                "model": safe_payload.get("model", pointer.get("model")),
                "provider": safe_payload.get("provider", pointer.get("provider")),
                "api_mode": safe_payload.get("api_mode", pointer.get("api_mode")),
                "base_url": safe_payload.get("base_url", pointer.get("base_url")),
                "toolsets": safe_payload.get("toolsets", pointer.get("toolsets", [])),
                "serialization_warnings": existing_warnings + warnings,
            }
        )
        provider_request = safe_payload.get("provider_request")
        safe_payload["provider_request_keys"] = sorted(provider_request.keys()) if isinstance(provider_request, dict) else []
        path = Path(pointer["latest_artifact_path"])
        sha, size = _write_json_atomic(path, safe_payload)
        now = _now()

        def update(conn: Any) -> dict[str, Any]:
            _ensure_schema(conn)
            conn.execute(
                """
                UPDATE subagent_context_artifacts
                   SET capture_sequence = ?, latest_api_call_count = ?, latest_retry_count = ?,
                       status = 'capturing', artifact_sha256 = ?, artifact_size_bytes = ?,
                       updated_at = ?, captured_at = ?
                 WHERE child_session_id = ?
                """,
                (
                    sequence,
                    safe_payload.get("api_call_count"),
                    safe_payload.get("retry_count"),
                    sha,
                    size,
                    now,
                    safe_payload.get("captured_at") or now,
                    child_session_id,
                ),
            )
            pointer = _fetch_pointer(conn, child_session_id) or {}
            pointer["ok"] = True
            return pointer

        return db._execute_write(update)
    finally:
        if close:
            db.close()


def finalize_subagent_context_artifact(child_session_id: str, *, session_db: Any = None) -> dict[str, Any]:
    db, close = _db(session_db)
    try:
        child_session_id = str(child_session_id or "")
        pointer = _fetch_pointer(db._conn, child_session_id)
        if not pointer:
            return {"ok": False, "error": {"code": "pointer_missing", "message": "subagent context pointer not found"}}
        path = Path(pointer["latest_artifact_path"])
        finalized_at = _now()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                artifact = json.load(fh)
        except Exception:
            artifact = _initial_artifact(pointer)
        artifact["status"] = "finalized"
        artifact["finalized_at"] = finalized_at
        sha, size = _write_json_atomic(path, artifact)

        def update(conn: Any) -> dict[str, Any]:
            _ensure_schema(conn)
            conn.execute(
                """
                UPDATE subagent_context_artifacts
                   SET status = 'finalized', finalized_at = ?, updated_at = ?,
                       artifact_sha256 = ?, artifact_size_bytes = ?
                 WHERE child_session_id = ?
                """,
                (finalized_at, finalized_at, sha, size, child_session_id),
            )
            out = _fetch_pointer(conn, child_session_id) or {}
            out["ok"] = True
            return out

        return db._execute_write(update)
    finally:
        if close:
            db.close()


def get_subagent_context_artifact(child_session_id: str, *, session_db: Any = None) -> dict[str, Any]:
    db, close = _db(session_db)
    try:
        pointer = _fetch_pointer(db._conn, str(child_session_id or ""))
        if not pointer:
            return {"ok": False, "error": {"code": "pointer_missing", "message": "subagent context pointer not found"}}
        path = Path(pointer["latest_artifact_path"])
        if not path.exists():
            return {"ok": False, "error": {"code": "artifact_missing", "message": "artifact file is missing"}, "pointer": pointer}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                artifact = json.load(fh)
        except Exception as exc:
            return {"ok": False, "error": {"code": "artifact_corrupt", "message": str(exc)}, "pointer": pointer}
        return {"ok": True, "pointer": pointer, "artifact": artifact}
    finally:
        if close:
            db.close()


def delete_subagent_context_artifacts_for_sessions(
    session_ids: Iterable[str],
    *,
    session_db: Any = None,
) -> dict[str, Any]:
    ids = [str(sid) for sid in session_ids if sid]
    if not ids:
        return {"deleted": 0, "files_deleted": 0}
    db, close = _db(session_db)
    try:
        placeholders = ",".join("?" for _ in ids)
        _ensure_schema(db._conn)
        rows = db._conn.execute(
            f"""
            SELECT child_session_id, latest_artifact_path
              FROM subagent_context_artifacts
             WHERE child_session_id IN ({placeholders}) OR parent_session_id IN ({placeholders})
            """,
            ids + ids,
        ).fetchall()
        paths = [Path(row["latest_artifact_path"]) for row in rows]
        child_ids = [row["child_session_id"] for row in rows]

        def delete_rows(conn: Any) -> int:
            _ensure_schema(conn)
            if not child_ids:
                return 0
            ph = ",".join("?" for _ in child_ids)
            conn.execute(f"DELETE FROM subagent_context_artifacts WHERE child_session_id IN ({ph})", child_ids)
            return len(child_ids)

        deleted = db._execute_write(delete_rows)
        files_deleted = 0
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
                    files_deleted += 1
                # Clean empty child directory only.
                try:
                    path.parent.rmdir()
                except OSError:
                    pass
            except Exception:
                pass
        return {"deleted": deleted, "files_deleted": files_deleted}
    finally:
        if close:
            db.close()


def build_subagent_context_payload(
    *,
    ref: dict[str, Any],
    canonical_messages: Any,
    provider_request: Any,
    api_call_count: Optional[int] = None,
    retry_count: Optional[int] = None,
) -> dict[str, Any]:
    messages_safe, msg_warnings = json_safe_copy(canonical_messages)
    request_safe, req_warnings = json_safe_copy(provider_request)
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": _KIND,
        "raw_unredacted_by_viewer": True,
        "child_session_id": ref.get("child_session_id"),
        "parent_session_id": ref.get("parent_session_id"),
        "subagent_id": ref.get("subagent_id"),
        "api_call_count": api_call_count,
        "retry_count": retry_count,
        "captured_at": _now(),
        "finalized_at": None,
        "status": "capturing",
        "role": ref.get("role"),
        "profile": ref.get("profile"),
        "model": ref.get("model"),
        "provider": ref.get("provider"),
        "api_mode": ref.get("api_mode"),
        "base_url": ref.get("base_url"),
        "toolsets": ref.get("toolsets") or [],
        "canonical_messages": messages_safe,
        "provider_request": request_safe,
        "provider_request_keys": sorted(request_safe.keys()) if isinstance(request_safe, dict) else [],
        "serialization_warnings": msg_warnings + req_warnings,
    }
