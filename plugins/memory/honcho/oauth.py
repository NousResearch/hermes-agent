"""OAuth credential storage and refresh for the Honcho memory provider.

The access token is stored as the host's ``apiKey``; the refresh token is exchanged before expiry.
Refresh tokens rotate with single-use reuse detection, so every refresh persists atomically and is
serialized (in-process lock + cross-process file lock). A failed exchange never raises into the agent:
transient failures retry once immediately; a permanent error (invalid_grant) marks the grant dead so
callers surface a re-login prompt. A server-side 401 on a locally-valid token uses ``force_refresh_token``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ACCESS_TOKEN_PREFIX = "hch-at-"
REFRESH_TOKEN_PREFIX = "hch-rt-"

_REFRESH_SKEW_SECONDS = 120  # refresh this early so an in-flight request never races expiry
_REFRESH_TIMEOUT_SECONDS = 15.0  # short: sits on the path to a memory call
_REFRESH_RETRY_DELAY_SECONDS = 2.0  # replayed refresh tokens are honored only briefly after rotation
# One exchange cycle (attempt + pause + retry) runs under the global refresh locks: cap it below 2 timeouts.
_REFRESH_TOTAL_BUDGET_SECONDS = 20.0
# After a transient failure, fail open without re-exchanging for this long so N waiting threads
# don't serialize N exchange cycles against a failing endpoint.
_REFRESH_FAILURE_COOLDOWN_SECONDS = 30.0
# OAuth error codes a retry can never fix — the grant itself is dead.
_PERMANENT_OAUTH_ERRORS = frozenset({"invalid_grant", "invalid_client", "unauthorized_client"})
# Derived from the canonical prefixes so a prefix change can't silently break redaction.
_TOKEN_VALUE_RE = re.compile(rf"({re.escape(ACCESS_TOKEN_PREFIX)}|{re.escape(REFRESH_TOKEN_PREFIX)})[A-Za-z0-9._~+/=-]+")

def redact_tokens(text: str) -> str:
    """Replace any embedded token values with their prefix plus a placeholder."""
    return _TOKEN_VALUE_RE.sub(lambda m: f"{m.group(1)}[redacted]", text)

class OAuthRefreshError(Exception):
    """Token endpoint rejected the refresh. ``permanent`` means re-login is required."""

    def __init__(self, message: str, *, error: str = "", permanent: bool = False):
        super().__init__(message)
        self.error, self.permanent = error, permanent

# Serializes refresh across threads; state is re-checked under it so racers don't replay a rotated token.
_refresh_lock = threading.Lock()

def _os_lock(fh, lock: bool) -> None:
    if os.name == "nt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK if lock else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX if lock else fcntl.LOCK_UN)

@contextmanager
def _config_refresh_lock(path: Path):
    """Machine-wide advisory lock (``<config>.lock``) around read-refresh-persist: a sibling process sharing
    this honcho.json must not replay the single-use refresh token. Best-effort — degrades to in-process only."""
    fh = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(f"{path}.lock", "a+b")
        _os_lock(fh, True)
    except Exception:
        logger.debug("Honcho OAuth cross-process lock unavailable; in-process only", exc_info=True)
        if fh is not None:
            fh.close()
        fh = None
    try:
        yield
    finally:
        if fh is not None:
            with suppress(Exception):
                _os_lock(fh, False)
            fh.close()


# Per-grant state keyed by (config path, host); single-key dict ops are atomic under the GIL.
# (expires_at, access): lets the hot path skip the honcho.json read while the token is well clear of
# expiry. A stale entry can't break auth; it only defers noticing out-of-band rotation.
_expiry_cache: dict[tuple[str, str], tuple[float, str]] = {}
# sha256 of the permanently rejected refresh token; a re-login rotates the token, so it self-clears.
_dead_grants: dict[tuple[str, str], str] = {}
# monotonic time of the last transient exchange failure (drives the fail-open cooldown).
_refresh_failure_at: dict[tuple[str, str], float] = {}
# (config mtime_ns, verdict): reauth_required only changes when the file is rewritten.
_reauth_check_cache: dict[tuple[str, str], tuple[int, bool]] = {}

def _in_failure_cooldown(key: tuple[str, str]) -> bool:
    failed_at = _refresh_failure_at.get(key)
    return failed_at is not None and (time.monotonic() - failed_at) < _REFRESH_FAILURE_COOLDOWN_SECONDS

def _grant_is_dead(key: tuple[str, str], cred: OAuthCredential) -> bool:
    return _dead_grants.get(key) == hashlib.sha256(cred.refresh_token.encode("utf-8")).hexdigest()

def _mark_grant_dead(key: tuple[str, str], cred: OAuthCredential) -> None:
    _dead_grants[key] = hashlib.sha256(cred.refresh_token.encode("utf-8")).hexdigest()
    _reauth_check_cache.pop(key, None)  # verdict changed without a config rewrite

def _read_config(path: Path) -> dict[str, Any]:
    """Tolerant reader for the read paths: an unreadable store reads as no credential. Writers use
    ``_read_config_strict``."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}

def _read_config_strict(path: Path) -> dict[str, Any]:
    """Reader for the write paths: a missing file is ``{}``, an unreadable one raises. A write seeded
    from ``{}`` would replace every other host's credentials."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Honcho config at %s exists but could not be read (%s); refusing to overwrite it.", path, exc)
        raise

def _load_cred(path: Path, host: str, raw: dict[str, Any] | None = None) -> OAuthCredential | None:
    """Credential from ``host``'s block in ``raw`` (or the file at ``path``)."""
    source = raw if raw is not None else _read_config(path)
    return OAuthCredential.from_host_block((source.get("hosts") or {}).get(host) or {})

def reauth_required(path: Path, host: str) -> bool:
    """True when ``host``'s OAuth grant is dead and only a new login fixes it."""
    key = (str(path), host)
    if key not in _dead_grants:
        return False
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = -1
    cached = _reauth_check_cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    cred = _load_cred(path, host)
    result = cred is not None and _grant_is_dead(key, cred)
    _reauth_check_cache[key] = (mtime, result)
    return result

def any_dead_grants() -> bool:
    """Cheap predicate letting hot-path callers skip config resolution when healthy."""
    return bool(_dead_grants)

def _num(value, cast):
    """``cast(value)``, or ``cast(0)`` when the stored value is malformed."""
    try:
        return cast(value)
    except (TypeError, ValueError):
        return cast(0)

def is_oauth_access_token(value: str | None) -> bool:
    """True when ``value`` is an OAuth access token (vs a static API key)."""
    return bool(value) and value.startswith(ACCESS_TOKEN_PREFIX)

@dataclass
class OAuthCredential:
    """An OAuth grant as stored in a honcho.json host block: ``access_token`` is the host's
    ``apiKey``, the rest lives in its ``oauth`` sub-block. ``expires_at`` is absolute epoch seconds."""

    access_token: str
    refresh_token: str
    expires_at: float
    client_id: str
    token_endpoint: str
    scope: str = "write"
    token_type: str = "Bearer"
    consent_peer_name: str | None = None  # transient: set only on a fresh grant, never persisted

    @classmethod
    def from_host_block(cls, block: dict[str, Any]) -> "OAuthCredential | None":
        """Build a credential from a honcho.json host block, or None if incomplete."""
        oauth, access = block.get("oauth"), block.get("apiKey")
        if not isinstance(oauth, dict) or not is_oauth_access_token(access):
            return None
        refresh, endpoint, client_id = oauth.get("refreshToken"), oauth.get("tokenEndpoint"), oauth.get("clientId")
        if not (refresh and endpoint and client_id):
            return None
        return cls(access, str(refresh), _num(oauth.get("expiresAt", 0), float), str(client_id), str(endpoint),
                   scope=str(oauth.get("scope", "write")), token_type=str(oauth.get("tokenType", "Bearer")))

    @classmethod
    def from_token_response(
        cls, body: dict[str, Any], *, now: float, client_id: str, token_endpoint: str,
        scope: str = "write", token_type: str = "Bearer", what: str = "grant",
    ) -> "OAuthCredential":
        """Build a credential from an OAuth token response; ``expires_in`` is relative to ``now``."""
        access, refresh = body.get("access_token"), body.get("refresh_token")
        if not is_oauth_access_token(access) or not refresh:
            raise ValueError(f"{what} missing access_token/refresh_token")
        return cls(access, str(refresh), now + _num(body.get("expires_in", 0), int), client_id, token_endpoint,
                   scope=str(body.get("scope", scope)), token_type=str(body.get("token_type", token_type)))

    def oauth_block(self) -> dict[str, Any]:
        """The ``oauth`` sub-block to persist (the access token lives in apiKey)."""
        return {
            "refreshToken": self.refresh_token, "expiresAt": int(self.expires_at), "clientId": self.client_id,
            "tokenEndpoint": self.token_endpoint, "scope": self.scope, "tokenType": self.token_type,
        }

    def is_expired(self, *, now: float, skew: float = _REFRESH_SKEW_SECONDS) -> bool:
        """True when the access token is within ``skew`` seconds of expiry."""
        return now >= (self.expires_at - skew)


# HTTP indirection: tests monkeypatch these module attributes; callers look them up at call time.
def _http_json(method: str, url: str, *, data=None, timeout: float, strict: bool = True) -> tuple[int, Any]:
    """Return ``(status, parsed JSON body)``. ``strict`` raises on non-2xx / non-JSON; otherwise a 4xx
    passes through (RFC 8628 polling reads the OAuth error off a 400) and a non-object body parses to ``{}``."""
    import httpx

    resp = httpx.request(method, url, data=data, timeout=timeout)
    if strict:
        resp.raise_for_status()
        return resp.status_code, resp.json()
    body = {}
    with suppress(ValueError):
        body = resp.json()
    return resp.status_code, body if isinstance(body, dict) else {}

def _http_post_form_status(url: str, data: dict[str, str], timeout: float) -> tuple[int, dict[str, Any]]:
    """POST form-encoded ``data``; return ``(status, body)`` without raising on 4xx."""
    return _http_json("POST", url, data=data, timeout=timeout, strict=False)

def _exchange_refresh_token(
    cred: OAuthCredential, *, now: float, timeout: float = _REFRESH_TIMEOUT_SECONDS
) -> OAuthCredential:
    """Run the refresh_token grant and return the rotated credential. Raises ``OAuthRefreshError`` (with
    the endpoint's error body) on an error response, transport errors as-is; callers fail open."""
    form = {"grant_type": "refresh_token", "client_id": cred.client_id, "refresh_token": cred.refresh_token}
    status, body = _http_post_form_status(cred.token_endpoint, form, timeout)
    if status >= 400:
        error, description = str(body.get("error") or ""), str(body.get("error_description") or "")
        detail = " — ".join(p for p in (error, description) if p) or "no error body"
        message = redact_tokens(f"token endpoint returned HTTP {status}: {detail}")
        raise OAuthRefreshError(message, error=error, permanent=error in _PERMANENT_OAUTH_ERRORS)
    return OAuthCredential.from_token_response(
        body, now=now, client_id=cred.client_id, token_endpoint=cred.token_endpoint,
        scope=cred.scope, token_type=cred.token_type, what="refresh response",
    )

def _exchange_with_retry(cred: OAuthCredential, *, now: float) -> OAuthCredential:
    """Exchange the refresh token, retrying once on transient failure. The retry cannot wait (replay grace
    window is short) and the cycle is capped by ``_REFRESH_TOTAL_BUDGET_SECONDS`` (runs under the locks)."""
    deadline = time.monotonic() + _REFRESH_TOTAL_BUDGET_SECONDS
    try:
        return _exchange_refresh_token(cred, now=now)
    except Exception as exc:
        if isinstance(exc, OAuthRefreshError) and exc.permanent:
            raise
        first = exc
    remaining = deadline - time.monotonic() - _REFRESH_RETRY_DELAY_SECONDS
    if remaining <= 0:
        raise first
    logger.warning("Honcho OAuth token exchange failed, retrying once: %s", redact_tokens(str(first)))
    time.sleep(_REFRESH_RETRY_DELAY_SECONDS)
    return _exchange_refresh_token(cred, now=now, timeout=min(remaining, _REFRESH_TIMEOUT_SECONDS))

def _rotate_and_persist(
    path: Path, host: str, key: tuple[str, str], cred: OAuthCredential, *, now: float, op_label: str = "refresh"
) -> OAuthCredential | None:
    """Exchange ``cred`` and persist the rotation; ``None`` on failure (logged). A permanent OAuth error
    marks the grant dead so later calls skip the endpoint until a new login rotates the refresh token."""
    try:
        # Read before spending the single-use refresh token: an exchange that cannot be persisted loses the grant.
        raw = _read_config_strict(path)
    except (OSError, ValueError):
        _refresh_failure_at[key] = time.monotonic()
        return None
    try:
        rotated = _exchange_with_retry(cred, now=now)
    except Exception as exc:
        if isinstance(exc, OAuthRefreshError) and exc.permanent:
            # The file lock is best-effort: a sibling may have rotated first, making our
            # refresh token a replay. If disk moved on, adopt that grant instead of killing it.
            disk = _load_cred(path, host)
            if disk is not None and disk.refresh_token != cred.refresh_token:
                _dead_grants.pop(key, None)
                _expiry_cache[key] = (disk.expires_at, disk.access_token)
                logger.info("Honcho OAuth %s for host %s lost a rotation race; "
                            "adopted the newer on-disk grant", op_label, host)
                return disk
            _mark_grant_dead(key, cred)
            logger.error("Honcho OAuth grant for host %s is no longer valid (%s); "
                         "run 'hermes honcho setup' to re-authenticate", host, exc)
            return None
        _refresh_failure_at[key] = time.monotonic()
        logger.warning("Honcho OAuth %s failed for host %s: %s", op_label, host, redact_tokens(str(exc)))
        return None
    _persist_credential(path, host, rotated, raw=raw)
    return rotated

def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` in place (overlay wins on scalars/lists)."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            value = _deep_merge(base[key], value)
        base[key] = value
    return base

def _persist_credential(path: Path, host: str, cred: OAuthCredential, raw: dict[str, Any] | None = None) -> None:
    """Write ``cred`` into ``host``'s block of ``raw`` (default: a strict read of the file) and mark
    the grant live."""
    from utils import atomic_json_write

    raw = _read_config_strict(path) if raw is None else raw
    block = raw.setdefault("hosts", {}).setdefault(host, {})
    block["apiKey"], block["oauth"] = cred.access_token, cred.oauth_block()
    atomic_json_write(path, raw, mode=0o600)
    key = (str(path), host)
    _expiry_cache[key] = (cred.expires_at, cred.access_token)
    _dead_grants.pop(key, None)
    _refresh_failure_at.pop(key, None)

def ensure_fresh_token(
    path: Path, host: str, raw: dict[str, Any] | None = None, *, now: float | None = None
) -> tuple[str | None, bool]:
    """Return ``(access_token, refreshed)`` for ``host``, refreshing if near expiry; ``(None, False)`` when
    the host has no OAuth credential. Refresh failures are swallowed: the current (possibly stale) token comes
    back with ``refreshed=False``; a permanently rejected grant is marked dead and 401 recovery escalates it."""
    now = time.time() if now is None else now
    key = (str(path), host)
    # Hot path: trust the cached expiry while well clear of the skew window (no disk read); bypassed
    # when an explicit ``raw`` is supplied.
    if raw is None:
        cached = _expiry_cache.get(key)
        if cached is not None and now < cached[0] - _REFRESH_SKEW_SECONDS:
            return cached[1], False
    cred = _load_cred(path, host, raw)
    if cred is None:
        _expiry_cache.pop(key, None)
        return None, False
    _expiry_cache[key] = (cred.expires_at, cred.access_token)
    if not cred.is_expired(now=now) or _in_failure_cooldown(key):
        return cred.access_token, False

    with _refresh_lock, _config_refresh_lock(path):
        # Re-read under both locks: another thread/process may have just rotated — adopt theirs.
        current = _load_cred(path, host) or cred
        if not current.is_expired(now=now):
            return current.access_token, current.access_token != cred.access_token
        # The lock holder we waited on may have just failed; fail open too.
        if _grant_is_dead(key, current) or _in_failure_cooldown(key):
            return current.access_token, False
        rotated = _rotate_and_persist(path, host, key, current, now=now)
        if rotated is not None:
            logger.info("Honcho OAuth token refreshed for host %s", host)
        return (rotated.access_token, True) if rotated is not None else (current.access_token, False)

def force_refresh_token(
    path: Path, host: str, *, failed_access_token: str | None = None
) -> str | None:
    """Rotate ``host``'s token now, ignoring local expiry (recovers a 401 on a
    token the local clock still thinks is valid). ``failed_access_token`` is the
    bearer the server rejected: when disk already holds a different one, a sibling
    rotated the single-use refresh token first, so adopt its grant instead of
    re-exchanging (a replay can revoke the whole grant)."""
    now = time.time()
    key = (str(path), host)
    with _refresh_lock, _config_refresh_lock(path):
        cred = _load_cred(path, host)
        if cred is None:
            _expiry_cache.pop(key, None)
            return None
        # Dead grant, or an exchange just failed transiently: callers fail open.
        if _grant_is_dead(key, cred) or _in_failure_cooldown(key):
            return None
        # Disk moving off the rejected bearer is the authoritative signal: the expiry cache is
        # empty in sibling processes and already equals disk after this process's first waiter.
        if failed_access_token and cred.access_token != failed_access_token and not cred.is_expired(now=now):
            _expiry_cache[key] = (cred.expires_at, cred.access_token)
            return cred.access_token
        cached = _expiry_cache.get(key)
        # Another thread or process already rotated: adopt the newer on-disk token.
        if cached is not None and cred.access_token != cached[1] and not cred.is_expired(now=now):
            _expiry_cache[key] = (cred.expires_at, cred.access_token)
            return cred.access_token
        rotated = _rotate_and_persist(path, host, key, cred, now=now, op_label="forced refresh")
        if rotated is not None:
            logger.info("Honcho OAuth token force-refreshed for host %s after an auth failure", host)
        return rotated.access_token if rotated is not None else None

def install_grant(
    path: Path, host: str, grant: dict[str, Any], *,
    client_id: str, token_endpoint: str, apply_config: bool = True, now: float | None = None,
) -> OAuthCredential:
    """Apply a fresh OAuth grant (an OAuthTokenResponse dict) to ``path`` for ``host``: deep-merge the
    grant's ``config`` into the file root (preserving other hosts and root keys), then write the host's
    ``apiKey`` and ``oauth`` block. ``apply_config=False`` stores tokens only. Runs under the same locks
    as a refresh so a login cannot interleave with a sibling's read-modify-write of the file."""
    now = time.time() if now is None else now
    cred = OAuthCredential.from_token_response(grant, now=now, client_id=client_id, token_endpoint=token_endpoint)
    with _refresh_lock, _config_refresh_lock(path):
        # Strict read: a login must not seed its root merge from a store it could not read.
        raw = _read_config_strict(path)
        granted_config = grant.get("config")
        if isinstance(granted_config, dict):
            cred.consent_peer_name = granted_config.get("peerName")
            if apply_config:
                _deep_merge(raw, granted_config)
        _persist_credential(path, host, cred, raw)
    return cred

def apply_token_to_client(client: Any, token: str) -> bool:
    """Rotate the live Honcho client's Bearer in place. The SDK builds its auth header per request from
    ``_http.api_key``, so mutating it rotates every holder of the singleton. False on an SDK shape change."""
    http = getattr(client, "_http", None)
    if http is None or not hasattr(http, "api_key"):
        return False
    http.api_key = token
    return True


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Callable  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
