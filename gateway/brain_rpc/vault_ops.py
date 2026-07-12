"""Vault list / stat / read handlers (contract §4.3–4.5).

Never log file bodies. Metadata-only for list/stat; size-capped single-file read.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.brain_rpc.auth import AuthContext, path_allowed
from gateway.brain_rpc.config import (
    DEFAULT_LIST_LIMIT,
    DEFAULT_READ_MAX_BYTES,
    HARD_LIST_LIMIT,
    HARD_READ_MAX_BYTES,
    BrainRpcHostConfig,
)
from gateway.brain_rpc.errors import (
    FORBIDDEN,
    INVALID_ARGUMENT,
    NOT_FOUND,
    PAYLOAD_TOO_LARGE,
    UNAVAILABLE,
    BrainRpcError,
)
from gateway.brain_rpc.paths import normalize_vault_path, resolve_under_vault

logger = logging.getLogger(__name__)


def _mtime_iso(path: Path) -> Optional[str]:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    except OSError:
        return None


def _guess_mime(path: Path, kind: str) -> Optional[str]:
    if kind != "file":
        return None
    mime, _ = mimetypes.guess_type(str(path))
    return mime


def _entry_for(path: Path, vault_path: str) -> Dict[str, Any]:
    if path.is_dir():
        kind = "directory"
        size = None
    elif path.is_file():
        kind = "file"
        try:
            size = path.stat().st_size
        except OSError:
            size = None
    else:
        # symlink to nowhere / special — treat as not a normal entry
        kind = "file"
        size = None
    return {
        "name": path.name if vault_path != "/" else path.name,
        "path": vault_path if vault_path != "/" or path.name else "/",
        "kind": kind,
        "mime_type": _guess_mime(path, kind),
        "size_bytes": size,
        "mtime": _mtime_iso(path),
    }


def _require_path_acl(vault_path: str, auth: AuthContext) -> None:
    if not path_allowed(vault_path, auth):
        raise BrainRpcError(
            FORBIDDEN,
            "path not allowed for profile",
            details={"path": vault_path},
        )


def _ensure_vault_root(host: BrainRpcHostConfig) -> Path:
    root = host.vault_root
    if not root.exists() or not root.is_dir():
        raise BrainRpcError(
            UNAVAILABLE,
            "vault root not readable",
            details={},
        )
    try:
        # Probe readability without listing contents into logs.
        next(root.iterdir(), None)
    except OSError as exc:
        raise BrainRpcError(UNAVAILABLE, "vault root not readable") from exc
    return root


def vault_list(
    params: Dict[str, Any],
    auth: AuthContext,
    host: BrainRpcHostConfig,
) -> Dict[str, Any]:
    root = _ensure_vault_root(host)
    raw_path = params.get("path", "/")
    norm = normalize_vault_path(raw_path if raw_path is not None else "/")
    _require_path_acl(norm, auth)
    fs_path, norm = resolve_under_vault(root, norm)

    if not fs_path.exists():
        raise BrainRpcError(NOT_FOUND, "path not found", details={"path": norm})
    if not fs_path.is_dir():
        raise BrainRpcError(
            INVALID_ARGUMENT,
            "path is not a directory",
            details={"path": norm},
        )

    limit = params.get("limit", DEFAULT_LIST_LIMIT)
    try:
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise BrainRpcError(INVALID_ARGUMENT, "invalid limit") from exc
    if limit < 1:
        raise BrainRpcError(INVALID_ARGUMENT, "limit must be >= 1")
    limit = min(limit, HARD_LIST_LIMIT)

    cursor = params.get("cursor")
    offset = 0
    if cursor is not None and cursor != "":
        try:
            offset = int(cursor)
            if offset < 0:
                raise ValueError("negative")
        except (TypeError, ValueError) as exc:
            raise BrainRpcError(INVALID_ARGUMENT, "invalid cursor") from exc

    try:
        names = sorted(p.name for p in fs_path.iterdir())
    except OSError as exc:
        raise BrainRpcError(UNAVAILABLE, "cannot list directory") from exc

    # Filter entries the subject cannot see (prefix ACL on children).
    entries: List[Dict[str, Any]] = []
    for name in names:
        child_vault = f"{norm.rstrip('/')}/{name}" if norm != "/" else f"/{name}"
        if not path_allowed(child_vault, auth):
            continue
        child_fs = fs_path / name
        # For the root listing entry name, use the file name.
        ent = _entry_for(child_fs, child_vault)
        ent["name"] = name
        entries.append(ent)

    page = entries[offset : offset + limit]
    next_offset = offset + len(page)
    truncated = next_offset < len(entries)
    next_cursor = str(next_offset) if truncated else None

    return {
        "path": norm,
        "entries": page,
        "next_cursor": next_cursor,
        "truncated": truncated,
    }


def vault_stat(
    params: Dict[str, Any],
    auth: AuthContext,
    host: BrainRpcHostConfig,
) -> Dict[str, Any]:
    root = _ensure_vault_root(host)
    norm = normalize_vault_path(params.get("path"))
    _require_path_acl(norm, auth)
    fs_path, norm = resolve_under_vault(root, norm)
    if not fs_path.exists():
        raise BrainRpcError(NOT_FOUND, "path not found", details={"path": norm})
    ent = _entry_for(fs_path, norm)
    # Name for root
    if norm == "/":
        ent["name"] = ""
        ent["path"] = "/"
        ent["kind"] = "directory"
        ent["mime_type"] = None
        ent["size_bytes"] = None
    else:
        ent["name"] = Path(norm).name
    return ent


def vault_read(
    params: Dict[str, Any],
    auth: AuthContext,
    host: BrainRpcHostConfig,
) -> Dict[str, Any]:
    root = _ensure_vault_root(host)
    norm = normalize_vault_path(params.get("path"))
    _require_path_acl(norm, auth)
    fs_path, norm = resolve_under_vault(root, norm)

    if not fs_path.exists():
        raise BrainRpcError(NOT_FOUND, "path not found", details={"path": norm})
    if fs_path.is_dir():
        raise BrainRpcError(
            INVALID_ARGUMENT,
            "path is a directory",
            details={"path": norm},
        )
    if not fs_path.is_file():
        raise BrainRpcError(NOT_FOUND, "path not found", details={"path": norm})

    caller_max = params.get("max_bytes", DEFAULT_READ_MAX_BYTES)
    try:
        caller_max = int(caller_max)
    except (TypeError, ValueError) as exc:
        raise BrainRpcError(INVALID_ARGUMENT, "invalid max_bytes") from exc
    if caller_max < 1:
        raise BrainRpcError(INVALID_ARGUMENT, "max_bytes must be >= 1")

    host_cap = min(host.read_max_bytes, host.hard_read_max_bytes, HARD_READ_MAX_BYTES)
    max_bytes = min(caller_max, host_cap)

    try:
        size = fs_path.stat().st_size
    except OSError as exc:
        raise BrainRpcError(UNAVAILABLE, "cannot stat file") from exc

    if size > max_bytes:
        raise BrainRpcError(
            PAYLOAD_TOO_LARGE,
            "file exceeds max_bytes",
            details={"path": norm, "size_bytes": size, "max_bytes": max_bytes},
        )

    try:
        # Read size+1 to detect race growth past cap without logging content.
        data = fs_path.read_bytes()
    except OSError as exc:
        raise BrainRpcError(UNAVAILABLE, "cannot read file") from exc

    if len(data) > max_bytes:
        raise BrainRpcError(
            PAYLOAD_TOO_LARGE,
            "file exceeds max_bytes",
            details={"path": norm, "size_bytes": len(data), "max_bytes": max_bytes},
        )

    mime = _guess_mime(fs_path, "file") or "application/octet-stream"
    encoding, content = _encode_content(data, mime)

    # Intentionally do not log `content` / `data`.
    logger.info(
        "brain_rpc vault.read path=%s size=%d encoding=%s",
        norm,
        len(data),
        encoding,
    )

    return {
        "path": norm,
        "kind": "file",
        "mime_type": mime,
        "encoding": encoding,
        "content": content,
        "size_bytes": len(data),
    }


def _encode_content(data: bytes, mime: str) -> tuple[str, str]:
    text_like = (
        mime.startswith("text/")
        or mime in {"application/json", "application/xml", "application/javascript"}
        or mime.endswith("+json")
        or mime.endswith("+xml")
        or mime == "application/x-yaml"
        or mime == "application/yaml"
    )
    if text_like or _looks_like_text(data):
        try:
            return "utf-8", data.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return "base64", base64.b64encode(data).decode("ascii")


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return True
    sample = data[: 8 * 1024]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False
