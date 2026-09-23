"""Managed Hermes API keys (Dashboard ``/api/api-server/keys``).

A small, atomic, versioned JSON store that lives next to ``auth.json`` at
``~/.hermes/api_keys.json`` and is the single source of truth for the **named,
revocable** Hermes API keys that authenticate the OpenAI-compatible API in
addition to the legacy single ``API_SERVER_KEY`` env var.

Design contract:

* Plaintext secrets are NEVER persisted. The on-disk row stores a per-key
  random salt and ``sha256(salt || token)``; the plaintext is only ever
  returned once by :func:`create_api_key`.
* The on-disk format is a tiny versioned JSON dict (see :data:`_SCHEMA_VERSION`)
  — no external dependency on ``state.db`` or the OAuth ``auth.json`` shape.
* Writes are atomic (tempfile + fsync + ``utils.atomic_replace``) and the
  in-process ``_STORE_LOCK`` serializes concurrent writers so two requests
  cannot interleave a read-modify-write.
* The file is created with mode 0600 on POSIX; inside a Docker container or
  under a managed-scope activation we leave permissions alone (matches
  ``_secure_file`` in ``hermes_cli.config``).

This module is intentionally dependency-light (stdlib + ``utils.atomic_replace``)
so it can be imported from both the dashboard FastAPI routes and the gateway
``api_server`` adapter without cycles.
"""
from __future__ import annotations

import base64
import contextlib
import datetime
import hashlib
import hmac
import json
import logging
import os
import secrets
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Cross-device, busy-file fallback lives in utils.atomic_replace; reuse the
# canonical implementation so atomic-write semantics stay consistent with
# .env writes in hermes_cli.config.
from utils import atomic_replace

logger = logging.getLogger(__name__)


# -- on-disk schema ---------------------------------------------------------

SCHEMA_VERSION = 1
KEY_PREFIX = "hm_live_"
"""Recognizable plaintext prefix; identifies a token as a managed Hermes key."""
KEY_RANDOM_BYTES = 32
"""256 bits of entropy — same strength as a `secrets.token_urlsafe(32)` value."""
_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


# File location. The Hermes home is resolved lazily so this module is
# importable before ``HERMES_HOME`` is materialised in the process.
def _store_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "api_keys.json"


# Process-wide writer lock. ``fcntl.flock`` would also work, but the dashboard
# may issue create/revoke from multiple worker threads; a single in-process
# lock keeps every read-modify-write serialized cheaply.
_STORE_LOCK = threading.Lock()


# -- helpers -----------------------------------------------------------------

def _new_id() -> str:
    """Short, opaque key identifier — visible in audit rows, never secret."""
    return "key_" + "".join(secrets.choice(_ID_ALPHABET) for _ in range(12))


def _new_salt() -> str:
    """Per-key salt (16 random bytes → 22 chars base64url, no padding)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(16)).rstrip(b"=").decode("ascii")


def _hash_secret(plaintext: str, salt: str) -> str:
    """SHA-256 over ``salt || plaintext``, hex-encoded.

    SHA-256 (not argon2) is deliberate: the lookup runs on every API request
    on the hot path, and the salt is per-key random — the search space is too
    large to brute-force regardless of hash function. The salt+SHA-256 design
    also keeps this module dependency-free; we can swap in argon2 later
    without an on-disk migration (just rerun create-and-replace for each
    key) without changing this file's public API.
    """
    return hashlib.sha256((salt + plaintext).encode("utf-8")).hexdigest()


def _constant_time_eq(a: str, b: str) -> bool:
    """String compare that doesn't raise on non-ASCII bytes (mirrors the
    guard at ``gateway/platforms/api_server.py:_check_auth``)."""
    try:
        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except (UnicodeEncodeError, TypeError):
        return False


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        # ``fromisoformat`` accepts the trailing ``+00:00`` we emit.
        return datetime.datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return None


# -- low-level load / save ---------------------------------------------------

def _read_store(path: Path) -> Dict[str, Any]:
    """Read and validate the on-disk JSON. Returns an empty store on first run."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"version": SCHEMA_VERSION, "keys": []}
    except OSError as exc:
        logger.warning("api_keys: could not read %s: %s", path, exc)
        return {"version": SCHEMA_VERSION, "keys": []}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("api_keys: %s is not valid JSON (%s); treating as empty", path, exc)
        return {"version": SCHEMA_VERSION, "keys": []}
    if not isinstance(data, dict):
        return {"version": SCHEMA_VERSION, "keys": []}
    if data.get("version") != SCHEMA_VERSION:
        # Future migrations land here. Today: silent reset rather than crash.
        logger.info("api_keys: schema version %s != %s; resetting", data.get("version"), SCHEMA_VERSION)
    keys = data.get("keys")
    if not isinstance(keys, list):
        keys = []
    return {"version": SCHEMA_VERSION, "keys": keys}


def _write_store(path: Path, data: Dict[str, Any]) -> None:
    """Atomic write with mode tightening. Never raises on permission tightening
    failure (matches ``_secure_file`` behaviour in ``hermes_cli.config``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"

    # Preserve mode on overwrite so a manually-chmod'd file stays as the
    # operator set it (mirrors ``_write_env_lines``).
    original_mode: Optional[int] = None
    try:
        original_mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        pass

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".api_keys_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    # Tighten permissions on new / pre-existing files.
    if original_mode is not None:
        with contextlib.suppress(OSError):
            os.chmod(path, original_mode)
    else:
        # Best-effort 0600 — silently skipped in containers where chmod is
        # mocked, matching the rest of the codebase.
        try:
            if hasattr(os, "geteuid") and os.geteuid() != 0:
                os.chmod(path, 0o600)
        except (OSError, NotImplementedError):
            pass


# -- public API --------------------------------------------------------------

def list_keys(*, include_revoked: bool = False) -> List[Dict[str, Any]]:
    """Return metadata rows. NEVER returns plaintext secrets or hashes —
    the dashboard uses this to render the key list.

    ``include_revoked=False`` (default) hides revoked rows from the active
    list. Pass ``True`` for an audit view.
    """
    path = _store_path()
    with _STORE_LOCK:
        data = _read_store(path)
    rows: List[Dict[str, Any]] = []
    for entry in data.get("keys", []):
        if not isinstance(entry, dict):
            continue
        revoked = bool(entry.get("revoked_at"))
        if revoked and not include_revoked:
            continue
        rows.append(_public_row(entry))
    return rows


def has_active_keys() -> bool:
    """Cheap probe used by ``/v1/capabilities`` and the auth gate. Avoids
    parsing timestamps when only existence matters."""
    path = _store_path()
    try:
        data = _read_store(path)
    except Exception:  # pragma: no cover — defensive
        return False
    return any(
        isinstance(e, dict) and not e.get("revoked_at")
        for e in data.get("keys", [])
    )


def get_key_by_id(key_id: str) -> Optional[Dict[str, Any]]:
    """Public lookup (no secret material). Used by the admin DELETE handler."""
    path = _store_path()
    with _STORE_LOCK:
        data = _read_store(path)
    for entry in data.get("keys", []):
        if isinstance(entry, dict) and entry.get("id") == key_id:
            return _public_row(entry)
    return None


def create_api_key(*, name: str, description: str = "") -> Dict[str, Any]:
    """Create a managed key, return its full row including the plaintext
    secret. The plaintext is shown to the user ONCE — never persisted, never
    returned by ``list_keys`` or ``get_key_by_id``.

    Caller is responsible for surfacing the plaintext to the user immediately
    (the FastAPI handler does so via ``detail.plaintext``).
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("name is required")
    if len(name) > 128:
        raise ValueError("name must be 128 characters or fewer")

    plaintext = KEY_PREFIX + secrets.token_urlsafe(KEY_RANDOM_BYTES)
    salt = _new_salt()
    key_id = _new_id()
    now = _now_iso()
    prefix = plaintext[: len(KEY_PREFIX) + 6]  # e.g. ``hm_live_AbCdEf``

    row = {
        "id": key_id,
        "name": name,
        "description": (description or "").strip(),
        "prefix": prefix,
        "secret_hash": _hash_secret(plaintext, salt),
        "salt": salt,
        "created_at": now,
        "last_used_at": None,
        "revoked_at": None,
    }

    path = _store_path()
    with _STORE_LOCK:
        data = _read_store(path)
        data.setdefault("keys", []).append(row)
        _write_store(path, data)

    # Return shape: full metadata + plaintext (single-use). Caller MUST NOT
    # log or persist ``plaintext``.
    public = _public_row(row)
    public["plaintext"] = plaintext
    return public


def revoke_api_key(key_id: str) -> bool:
    """Idempotent revoke. Returns True if the key existed (and is now revoked
    or was already revoked). Returns False for unknown ids."""
    path = _store_path()
    with _STORE_LOCK:
        data = _read_store(path)
        keys = data.get("keys", [])
        found = False
        for entry in keys:
            if isinstance(entry, dict) and entry.get("id") == key_id:
                found = True
                if not entry.get("revoked_at"):
                    entry["revoked_at"] = _now_iso()
                break
        if found:
            _write_store(path, data)
    return found


def verify_api_key(token: str) -> Optional[Dict[str, Any]]:
    """Tier-2 auth check used by the gateway's ``_check_auth``.

    Returns a minimal identity dict ``{"id": ..., "name": ...}`` on success,
    or ``None`` if no active key matches. NEVER returns the secret_hash or
    salt; the caller only needs an opaque identity for the request log and
    the idempotency principal.

    Side-effect: on success, refreshes ``last_used_at`` (best-effort,
    debounced so we don't write the file on every chat completion).
    """
    if not token or not token.startswith(KEY_PREFIX):
        return None

    path = _store_path()
    try:
        data = _read_store(path)
    except Exception:
        return None

    keys = data.get("keys") or []
    match: Optional[Dict[str, Any]] = None
    for entry in keys:
        if not isinstance(entry, dict):
            continue
        if entry.get("revoked_at"):
            continue
        salt = entry.get("salt") or ""
        secret_hash = entry.get("secret_hash") or ""
        if not salt or not secret_hash:
            continue
        if _constant_time_eq(_hash_secret(token, salt), secret_hash):
            match = entry
            break

    if match is None:
        return None

    # Debounce last_used_at writes — once every 60s per key.
    last_ts = _parse_iso(match.get("last_used_at"))
    now = time.time()
    if last_ts is None or (now - last_ts) >= 60.0:
        match["last_used_at"] = _now_iso()
        with _STORE_LOCK:
            # Re-read inside the lock so concurrent revoke/create is honored.
            current = _read_store(path)
            current_keys = current.get("keys") or []
            for entry in current_keys:
                if isinstance(entry, dict) and entry.get("id") == match.get("id"):
                    if not entry.get("revoked_at"):
                        entry["last_used_at"] = match["last_used_at"]
                    break
            _write_store(path, current)

    return {"id": match.get("id", ""), "name": match.get("name", "")}


# -- private helpers ---------------------------------------------------------

def _public_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Strip secret material from an internal row. The output is safe to send
    over the dashboard API and to log at INFO level (no plaintext, no hash)."""
    return {
        "id": entry.get("id", ""),
        "name": entry.get("name", ""),
        "description": entry.get("description", ""),
        "prefix": entry.get("prefix", ""),
        "created_at": entry.get("created_at"),
        "last_used_at": entry.get("last_used_at"),
        "revoked_at": entry.get("revoked_at"),
        "active": not bool(entry.get("revoked_at")),
    }