from __future__ import annotations

"""LCM SQLite storage security helpers.

This module keeps the concrete data-at-rest controls out of the query-heavy
store/DAG modules: private file modes, optional per-row AEAD, retention status,
and plaintext-path policy checks.
"""

import base64
import os
import time
from pathlib import Path
from typing import Any

try:  # optional dependency: encryption is opt-in and fails loud when missing
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:  # pragma: no cover - exercised by monkeypatch tests
    AESGCM = None  # type: ignore[assignment]

AEAD_PREFIX = "lcm-aead-v1:"
UNSYNCED_PATH_MARKER = ".lcm-unsynced-ok"
_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_DEFAULT_KEY_FILENAME = "lcm-row.key"
_SYNCED_PATH_MARKERS = (
    "Mobile Documents",
    "iCloud Drive",
    "Dropbox",
    "Google Drive",
    "OneDrive",
    "Syncthing",
)


def aead_available() -> bool:
    return AESGCM is not None


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        pass


def sqlite_sidecar_paths(db_path: str | Path) -> list[Path]:
    path = Path(db_path)
    return [path, Path(str(path) + "-wal"), Path(str(path) + "-shm")]


def ensure_lcm_file_permissions(db_path: str | Path) -> None:
    """Enforce private modes for the LCM DB parent and SQLite sidecars."""
    path = Path(db_path)
    path.parent.mkdir(mode=_PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    _chmod_best_effort(path.parent, _PRIVATE_DIR_MODE)
    for candidate in sqlite_sidecar_paths(path):
        if candidate.exists():
            _chmod_best_effort(candidate, _PRIVATE_FILE_MODE)


def _default_key_path(hermes_home: str | Path, db_path: str | Path) -> Path:
    if hermes_home:
        return Path(hermes_home).expanduser() / _DEFAULT_KEY_FILENAME
    return Path(db_path).expanduser().parent / _DEFAULT_KEY_FILENAME


def _load_or_create_key(path: Path) -> bytes:
    path.parent.mkdir(mode=_PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    _chmod_best_effort(path.parent, _PRIVATE_DIR_MODE)
    if path.exists():
        key = path.read_bytes()
        if len(key) != 32:
            raise RuntimeError(f"LCM encryption key at {path} must be exactly 32 bytes")
        _chmod_best_effort(path, _PRIVATE_FILE_MODE)
        return key
    key = os.urandom(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(path), flags, _PRIVATE_FILE_MODE)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    _chmod_best_effort(path, _PRIVATE_FILE_MODE)
    return key


class RowCipher:
    """Small wrapper around AES-GCM for text columns."""

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key
        self.enabled = key is not None
        self._aead = AESGCM(key) if key is not None and AESGCM is not None else None

    @classmethod
    def disabled(cls) -> "RowCipher":
        return cls(None)

    @classmethod
    def from_config(cls, config: Any, *, hermes_home: str = "", db_path: str | Path = "") -> "RowCipher":
        if not bool(getattr(config, "encryption_enabled", False)):
            return cls.disabled()
        if AESGCM is None:
            raise RuntimeError(
                "cryptography AESGCM is required for LCM encryption; "
                "install cryptography or disable LCM encryption"
            )
        configured_path = str(getattr(config, "encryption_key_path", "") or "").strip()
        key_path = Path(configured_path).expanduser() if configured_path else _default_key_path(hermes_home, db_path)
        return cls(_load_or_create_key(key_path))

    def encrypt_text(self, value: str | None, *, field: str) -> str | None:
        if value is None or not self.enabled:
            return value
        if value.startswith(AEAD_PREFIX):
            return value
        if self._aead is None:  # pragma: no cover - defensive; constructor guards this
            raise RuntimeError("LCM encryption is enabled but AESGCM is unavailable")
        nonce = os.urandom(12)
        aad = f"lcm:{field}:v1".encode("utf-8")
        ciphertext = self._aead.encrypt(nonce, value.encode("utf-8"), aad)
        return AEAD_PREFIX + base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")

    def decrypt_text(self, value: str | None, *, field: str) -> str | None:
        if value is None or not isinstance(value, str) or not value.startswith(AEAD_PREFIX):
            return value
        if not self.enabled or self._aead is None:
            raise RuntimeError("LCM encrypted row cannot be read without the profile encryption key")
        payload = base64.urlsafe_b64decode(value[len(AEAD_PREFIX):].encode("ascii"))
        nonce, ciphertext = payload[:12], payload[12:]
        aad = f"lcm:{field}:v1".encode("utf-8")
        return self._aead.decrypt(nonce, ciphertext, aad).decode("utf-8")


def _safe_min_timestamp(conn: Any, query: str) -> float | None:
    try:
        row = conn.execute(query).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def _safe_count(conn: Any, table: str) -> int:
    try:
        row = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
    except Exception:
        return 0
    return int(row[0] or 0) if row else 0


def retention_status(
    conn: Any,
    db_path: str | Path,
    *,
    ttl_days: int,
    max_bytes: int,
    now: float | None = None,
) -> dict[str, Any]:
    current = time.time() if now is None else float(now)
    paths = sqlite_sidecar_paths(db_path)
    sizes = {str(path): int(path.stat().st_size) for path in paths if path.exists()}
    total_size = sum(sizes.values())
    oldest_message_at = _safe_min_timestamp(conn, "SELECT MIN(timestamp) FROM messages")
    oldest_node_at = _safe_min_timestamp(
        conn,
        "SELECT MIN(COALESCE(earliest_at, created_at)) FROM summary_nodes",
    )
    candidates = [ts for ts in (oldest_message_at, oldest_node_at) if ts is not None]
    oldest_row_at = min(candidates) if candidates else None
    oldest_age_days = None
    if oldest_row_at is not None:
        oldest_age_days = max(0.0, (current - oldest_row_at) / 86400.0)
    return {
        "database_path": str(Path(db_path)),
        "sidecar_sizes": sizes,
        "total_size_bytes": total_size,
        "ttl_days": int(ttl_days),
        "max_bytes": int(max_bytes),
        "oldest_row_at": oldest_row_at,
        "oldest_row_age_days": oldest_age_days,
        "oldest_message_at": oldest_message_at,
        "oldest_summary_at": oldest_node_at,
        "message_rows": _safe_count(conn, "messages"),
        "summary_rows": _safe_count(conn, "summary_nodes"),
        "ttl_exceeded": bool(oldest_age_days is not None and oldest_age_days > float(ttl_days)),
        "max_bytes_exceeded": bool(max_bytes > 0 and total_size > int(max_bytes)),
    }


def _has_unsynced_marker(path: Path) -> str:
    current = path.parent if path.suffix else path
    for candidate in [current, *current.parents]:
        marker = candidate / UNSYNCED_PATH_MARKER
        if marker.exists():
            return str(marker)
    return ""


def plaintext_path_policy(db_path: str | Path) -> dict[str, Any]:
    """Return whether an unencrypted LCM DB path is explicitly local-only.

    Plaintext is permitted only when an operator places UNSYNCED_PATH_MARKER in
    the DB directory (or an ancestor) and the path is not under common synced or
    backed-up user folders. This is a conservative checker for Aegis-style local
    break-glass deployments; encrypted mode does not need it.
    """
    path = Path(db_path).expanduser()
    path_text = str(path)
    path_parts = {part.lower() for part in path.parts}
    synced_marker = next(
        (marker for marker in _SYNCED_PATH_MARKERS if marker.lower() in path_text.lower() or marker.lower() in path_parts),
        "",
    )
    marker_path = _has_unsynced_marker(path)
    allowed = bool(marker_path) and not bool(synced_marker) and path.is_absolute()
    return {
        "path": str(path),
        "allowed": allowed,
        "explicit_unsynced_marker": marker_path,
        "required_marker": UNSYNCED_PATH_MARKER,
        "synced_path_detected": bool(synced_marker),
        "synced_path_marker": synced_marker,
        "absolute_path": path.is_absolute(),
    }
