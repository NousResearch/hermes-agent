"""Durable credential store for enrolled WebAuthn passkeys (public-key material only).

A JSON document at ``dashboard.webauthn.credential_store`` (default
``~/.hermes/dashboard_webauthn.json``, 0600, atomic temp+rename writes,
``version: 1``). Holds PUBLIC keys: leaking the *contents* is harmless, but
tampering must not silently change authentication, so a load failure fails
CLOSED (the provider declines to register rather than trusting garbage) —
integrity, not secrecy. Precedents: ``~/.hermes/auth.json``,
``sessions/sessions.json``; deliberately NOT config.yaml (human-managed) or
state.db (a session-history DB, not auth-shaped).

One-shot enrolment tokens from ``hermes dashboard webauthn enrol`` (CLI) ride
the same file under ``pending_enrolments`` so a headless bootstrap can mint a
URL the running dashboard process honours without any shared memory.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

FILE_VERSION = 1
# CLI-issued one-shot enrolment tokens: TTL + shape (single use, consumed on
# the register/verify completion the design doc §2.2 describes).
ENROLMENT_TOKEN_TTL_SECONDS = 600
_ENROLMENT_TOKEN_BYTES = 32
_STORE_UMASK_PLACEHOLDER = 0o777

_lock = threading.Lock()


class StoreError(Exception):
    """Credential store unusable (corrupt / tampered / unknown version / wrong
    rp_id). Registration must fail closed — never half-trust the file."""


def resolve_store_path(configured: str) -> Path:
    """Absolute ``configured`` wins; relative or default anchors at
    :func:`hermes_constants.get_hermes_home` (HX001) — never a hardcoded home."""
    from hermes_constants import get_hermes_home

    if configured:
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path
        return get_hermes_home() / path
    return get_hermes_home() / "dashboard_webauthn.json"


def load(path: Path) -> dict:
    """Read a store document; a missing file yields a fresh ``version: 1`` doc
    (no credentials yet). Any unreadable/unparsable/wrong-version corpus raises
    ``StoreError`` — fail closed."""
    if not path.exists():
        return {"version": FILE_VERSION, "rp_id": None, "credentials": [],
                "pending_enrolments": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise StoreError(f"cannot read credential store {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise StoreError(f"credential store {path} is not a JSON object")
    if doc.get("version") != FILE_VERSION:
        raise StoreError(
            f"credential store {path} has unsupported version {doc.get('version')!r} "
            f"(this build reads version {FILE_VERSION}); refusing to half-trust it")
    if not isinstance(doc.get("credentials"), list):
        raise StoreError(f"credential store {path} has no credentials list")
    return doc


def save(path: Path, doc: Mapping[str, Any]) -> None:
    """Atomically persist ``doc`` (temp+rename in the same directory), 0600,
    owners-only directory creation. Explicit ``encoding="utf-8"`` everywhere
    (PLW1514); ``os.replace`` is atomic on POSIX and same-volume Windows."""
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".webauthn-store-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(dict(doc), fh, indent=1, sort_keys=True)
                fh.write("\n")
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    try:  # best-effort tighten the directory too (0600 file is the contract)
        os.chmod(path.parent, 0o700)
    except OSError:
        pass


def add_pending_enrolment(doc: dict, *, now: Optional[int] = None) -> str:
    """Append a fresh single-use enrolment token to ``doc`` (mutates + returns it)."""
    token = secrets.token_urlsafe(_ENROLMENT_TOKEN_BYTES)
    now = int(time.time()) if now is None else now
    pending = doc.setdefault("pending_enrolments", {})
    pending[token] = {
        "created_at": now,
        "expires_at": now + ENROLMENT_TOKEN_TTL_SECONDS,
    }
    return token


def check_pending_enrolment(
    doc: Mapping[str, Any], token: str, *, now: Optional[int] = None) -> bool:
    """Peek whether ``token`` names an unconsumed, unexpired enrolment entry
    (constant-time compare first so a wrong token takes comparable time)."""
    pending = doc.get("pending_enrolments")
    if not isinstance(pending, dict) or not token:
        return False
    now = int(time.time()) if now is None else now
    import hmac as _hmac

    for candidate in pending:
        if _hmac.compare_digest(str(candidate), token):
            entry = pending[candidate]
            expired = isinstance(entry, dict) and int(entry.get("expires_at", 0)) < now
            return not expired
    return False


def consume_pending_enrolment(doc: dict, token: str) -> bool:
    """Drop ``token`` from the pending set (single use). True when consumed."""
    pending = doc.get("pending_enrolments")
    if not isinstance(pending, dict) or not token:
        return False
    import hmac as _hmac

    for candidate in list(pending):
        if _hmac.compare_digest(str(candidate), token):
            pending.pop(candidate, None)
            return True
    return False


def prune_pending_enrolments(doc: dict, *, now: Optional[int] = None) -> None:
    """Drop expired pending enrolments whenever the store is written."""
    now = int(time.time()) if now is None else now
    pending = doc.get("pending_enrolments")
    if isinstance(pending, dict):
        for candidate in list(pending):
            entry = pending[candidate]
            if not isinstance(entry, dict) or int(entry.get("expires_at", 0)) < now:
                pending.pop(candidate, None)