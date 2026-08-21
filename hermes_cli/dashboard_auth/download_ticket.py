"""Short-lived signed download tickets for external viewers (WPS Office etc.).

The gated dashboard authenticates with session cookies. An external viewer
like WPS Office opens a plain GET URL and can attach neither cookies nor
headers, so the cookie gate would 401 every direct link. These tickets let a
first-party caller (agent, App, or the /api/files/download-ticket endpoint)
mint a time-limited, path-scoped signed URL for ``/api/files/download`` that
bypasses the cookie gate for exactly one file for a few minutes.

Security model:

* HMAC-SHA256 over ``f"{path}|{expiry}|{mtime_ns}|{size}"`` with an
  in-process random secret, so the signature cannot be forged without
  process access and dies with the dashboard process. A dashboard restart
  rotates the secret and therefore invalidates every outstanding ticket —
  a viewer mid-download gets a 401 and the caller must mint a fresh URL.
  This restart-invalidation is deliberate: the secret is never persisted
  to disk.
* Tickets are short-lived (default 300s) and single-file: even a leaked URL
  is bounded in both time and scope.
* A ticket is bound to the file's identity (``st_mtime_ns`` + ``st_size``
  at mint time), so a file replaced at the same path between mint and
  download does not ride the old ticket; a deleted file invalidates it too.
* Bypassing the gate only skips *authentication* — every file-level guard in
  ``download_managed_file`` (managed-root resolution, sensitive-path
  denylist, size cap) still runs unchanged.

Mobile/gateway host is unchanged; this is a dashboard-side capability.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
import urllib.parse

from fastapi import Request

_TICKET_TTL_SECONDS = 300  # 5 minutes
_SECRET = secrets.token_bytes(32)


def _sign(payload: str) -> str:
    return (
        base64.urlsafe_b64encode(
            hmac.new(_SECRET, payload.encode(), hashlib.sha256).digest()
        )
        .decode()
        .rstrip("=")
    )


def _file_identity(path: str) -> tuple[int, int] | None:
    """Return ``(st_mtime_ns, st_size)`` for ``path``, or None if unstat-able.

    The identity is what a ticket is bound to: a file replaced at the same
    path (new mtime/size) or deleted no longer satisfies the ticket.
    ``~`` is expanded so relative-to-home paths behave like the managed-file
    resolver's.
    """
    try:
        st = os.stat(os.path.expanduser(path))
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def build_download_url(
    base_url: str,
    path: str,
    ttl_seconds: int = _TICKET_TTL_SECONDS,
) -> str:
    """Mint a signed download URL for ``path`` valid for ``ttl_seconds``.

    ``path`` is the absolute host path (or ``~``-expanded); the signed
    payload uses the raw path so verification is exact. The ticket is bound
    to the file's current identity (mtime + size), so a file swapped at the
    same path after minting no longer satisfies the ticket. Raises
    :class:`FileNotFoundError` when ``path`` cannot be stat'ed (e.g. deleted
    between the caller's existence check and the mint).
    """
    exp = int(time.time()) + ttl_seconds
    identity = _file_identity(path)
    if identity is None:
        raise FileNotFoundError(
            f"cannot mint download ticket for missing/unreadable file: {path}"
        )
    mtime_ns, size = identity
    sig = _sign(f"{path}|{exp}|{mtime_ns}|{size}")
    enc_path = urllib.parse.quote(path)
    return f"{base_url.rstrip('/')}/api/files/download?path={enc_path}&exp={exp}&sig={sig}"


def verify_download_ticket(request: Request) -> bool:
    """True iff the request carries a valid, unexpired ticket for its path.

    Only acts on ``/api/files/download``; every other path returns False so
    this helper can never widen the auth surface of a different route.

    The signed payload binds path, expiry, and the file's mtime/size at
    mint time, so verification also stats the file: a ticket whose file was
    deleted (or replaced at the same path) since minting is rejected.
    """
    if request.url.path != "/api/files/download":
        return False
    path = request.query_params.get("path", "")
    exp_raw = request.query_params.get("exp", "")
    sig = request.query_params.get("sig", "")
    if not path or not exp_raw or not sig:
        return False
    try:
        exp = int(exp_raw)
    except ValueError:
        return False
    if exp < time.time():
        return False
    identity = _file_identity(path)
    if identity is None:
        return False
    mtime_ns, size = identity
    expected = _sign(f"{path}|{exp}|{mtime_ns}|{size}")
    return hmac.compare_digest(sig, expected)
