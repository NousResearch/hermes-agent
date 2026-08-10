"""OAuth credential storage and refresh for the Honcho memory provider.

An access token authenticates exactly like a scoped API key, so it is stored
as the host's ``apiKey``; this module exchanges the refresh token before
expiry to keep it live.

Refresh tokens rotate with single-use reuse detection: a replayed stale token
revokes the whole grant. So every refresh must persist the rotated token
atomically and be serialized. A failed exchange never raises into the agent:
transient failures retry once immediately (the server re-rotates a replayed
refresh token only within a short grace window, so waiting for the next
memory call is too late), and a permanent OAuth error such as invalid_grant
marks the grant dead so nothing keeps hitting the token endpoint — callers
surface a re-login prompt instead. A server-side 401 on a locally-valid
token is recovered via ``force_refresh_token``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

ACCESS_TOKEN_PREFIX = "hch-at-"
REFRESH_TOKEN_PREFIX = "hch-rt-"

# Refresh this many seconds before the access token actually expires, so an
# in-flight request never races the expiry boundary.
_REFRESH_SKEW_SECONDS = 120

# Default HTTP timeout for the token exchange. Kept short — the refresh happens
# on the path to a memory call, and a stalled auth server must not hang it.
_REFRESH_TIMEOUT_SECONDS = 15.0

# Retry pause, kept short: the server honors a replayed refresh token only briefly after rotating it.
_REFRESH_RETRY_DELAY_SECONDS = 2.0

# Total wall-clock budget for one exchange cycle (first attempt + pause + retry).
# The exchange runs while holding the global refresh locks on the path to a
# memory call, so a stalled token endpoint must not hold them for two full
# HTTP timeouts back to back.
_REFRESH_TOTAL_BUDGET_SECONDS = 20.0

# After a transient exchange failure, fail open without re-exchanging for this
# long. Prevents N waiting threads (or turns) from serializing N full exchange
# cycles against an endpoint that just failed.
_REFRESH_FAILURE_COOLDOWN_SECONDS = 30.0

# OAuth error codes that a retry can never fix — the grant itself is dead.
_PERMANENT_OAUTH_ERRORS = frozenset({"invalid_grant", "invalid_client", "unauthorized_client"})

# Token values are secret even though their prefixes are not; redact before logging.
# Derived from the canonical prefixes above so a prefix change can't silently
# break redaction.
_TOKEN_VALUE_RE = re.compile(
    rf"({re.escape(ACCESS_TOKEN_PREFIX)}|{re.escape(REFRESH_TOKEN_PREFIX)})[A-Za-z0-9._~+/=-]+"
)


def redact_tokens(text: str) -> str:
    """Replace any embedded token values with their prefix plus a placeholder."""
    return _TOKEN_VALUE_RE.sub(lambda m: f"{m.group(1)}[redacted]", text)


# Backward-compat alias for oauth-internal call sites and older importers.
_redact_tokens = redact_tokens


class OAuthRefreshError(Exception):
    """Token endpoint rejected the refresh. ``permanent`` means re-login is required."""

    def __init__(self, message: str, *, error: str = "", permanent: bool = False):
        super().__init__(message)
        self.error = error
        self.permanent = permanent

# Serializes refresh across threads sharing one process's config. Re-checked
# under the lock (double-checked) so racing callers don't replay a rotated
# refresh token and trip reuse detection.
_refresh_lock = threading.Lock()


@contextmanager
def _config_refresh_lock(path: Path):
    """Machine-wide advisory lock around read-refresh-persist.

    The in-process ``_refresh_lock`` can't stop a second process (a sibling
    Hermes profile or the desktop app sharing this honcho.json) from replaying
    the single-use refresh token and tripping reuse-detection — which revokes
    the whole grant. An OS file lock on ``<config>.lock`` serializes rotation
    across processes; best-effort, so a platform without flock degrades to
    in-process serialization only.
    """
    lock_path = Path(f"{path}.lock")
    fh = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+b")
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    except Exception:
        logger.debug("Honcho OAuth cross-process lock unavailable; in-process only", exc_info=True)
        if fh is not None:
            fh.close()
            fh = None
    try:
        yield
    finally:
        if fh is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            fh.close()

# In-memory expiry cache keyed by (config path, host) → (expires_at, access).
# Lets the hot path (every memory access calls this) skip the honcho.json read
# while the token is comfortably live; disk is only touched near expiry, on a
# cache miss, or when an explicit ``raw`` is supplied. Single-key dict ops are
# atomic under the GIL, so no separate lock is needed. An access token stays
# valid until its own expiry regardless of out-of-band rotation, so a stale
# cache entry can't break auth — it just defers picking up external changes
# until the token nears expiry and disk is read again.
_expiry_cache: dict[tuple[str, str], tuple[float, str]] = {}

# Permanently rejected grants: (config path, host) → sha256 of the dead refresh token; a re-login rotates the token, so the digest check self-clears.
_dead_grants: dict[tuple[str, str], str] = {}

# Last transient exchange failure per grant: key → monotonic timestamp. While
# inside the cooldown window callers fail open to the stale token without
# re-exchanging, so waiting threads don't serialize repeated full exchange
# cycles against an endpoint that just failed.
_refresh_failure_at: dict[tuple[str, str], float] = {}


def _in_failure_cooldown(key: tuple[str, str]) -> bool:
    failed_at = _refresh_failure_at.get(key)
    return (
        failed_at is not None
        and (time.monotonic() - failed_at) < _REFRESH_FAILURE_COOLDOWN_SECONDS
    )


# Memoized reauth_required verdict per grant: key → (config mtime_ns, result).
# The verdict only changes when the config file is rewritten (re-login), so an
# unchanged mtime short-circuits the read+parse on the dead-grant hot path.
_reauth_check_cache: dict[tuple[str, str], tuple[int, bool]] = {}


def _refresh_token_digest(cred: OAuthCredential) -> str:
    return hashlib.sha256(cred.refresh_token.encode("utf-8")).hexdigest()


def _grant_is_dead(key: tuple[str, str], cred: OAuthCredential) -> bool:
    return _dead_grants.get(key) == _refresh_token_digest(cred)


def _mark_grant_dead(key: tuple[str, str], cred: OAuthCredential) -> None:
    _dead_grants[key] = _refresh_token_digest(cred)
    # The verdict changed without a config rewrite; drop any memoized answer.
    _reauth_check_cache.pop(key, None)


def reauth_required(path: Path, host: str) -> bool:
    """True when ``host``'s OAuth grant is dead and only a new login fixes it."""
    key = (str(path), host)
    if key not in _dead_grants:
        return False
    # A re-login rewrites the config file, so gate the read+parse on mtime:
    # while the file is unchanged the answer cannot change.
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = -1
    cached = _reauth_check_cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    block = (_read_config(path).get("hosts") or {}).get(host) or {}
    cred = OAuthCredential.from_host_block(block)
    result = cred is not None and _grant_is_dead(key, cred)
    _reauth_check_cache[key] = (mtime, result)
    return result


def any_dead_grants() -> bool:
    """Cheap predicate: has any grant in this process been marked dead?

    Lets hot-path callers skip config-path resolution entirely in the
    overwhelmingly common healthy state.
    """
    return bool(_dead_grants)


def is_oauth_access_token(value: str | None) -> bool:
    """True when ``value`` is an OAuth access token (vs a static API key)."""
    return bool(value) and value.startswith(ACCESS_TOKEN_PREFIX)


@dataclass
class OAuthCredential:
    """An OAuth grant as stored in a honcho.json host block.

    ``access_token`` mirrors the host's ``apiKey``; the remaining fields live in
    the host's ``oauth`` sub-block. ``expires_at`` is absolute epoch seconds.
    """

    access_token: str
    refresh_token: str
    expires_at: float
    client_id: str
    token_endpoint: str
    scope: str = "write"
    token_type: str = "Bearer"
    # Transient consent peer name — set only on a fresh grant, never persisted.
    consent_peer_name: str | None = None

    @classmethod
    def from_host_block(cls, block: dict[str, Any]) -> "OAuthCredential | None":
        """Build a credential from a honcho.json host block, or None if incomplete."""
        oauth = block.get("oauth")
        access = block.get("apiKey")
        if not isinstance(oauth, dict) or not is_oauth_access_token(access):
            return None
        refresh = oauth.get("refreshToken")
        endpoint = oauth.get("tokenEndpoint")
        client_id = oauth.get("clientId")
        if not (refresh and endpoint and client_id):
            return None
        try:
            expires_at = float(oauth.get("expiresAt", 0))
        except (TypeError, ValueError):
            expires_at = 0.0
        return cls(
            access_token=access,
            refresh_token=str(refresh),
            expires_at=expires_at,
            client_id=str(client_id),
            token_endpoint=str(endpoint),
            scope=str(oauth.get("scope", "write")),
            token_type=str(oauth.get("tokenType", "Bearer")),
        )

    def oauth_block(self) -> dict[str, Any]:
        """The ``oauth`` sub-block to persist (the access token lives in apiKey)."""
        return {
            "refreshToken": self.refresh_token,
            "expiresAt": int(self.expires_at),
            "clientId": self.client_id,
            "tokenEndpoint": self.token_endpoint,
            "scope": self.scope,
            "tokenType": self.token_type,
        }

    def is_expired(self, *, now: float, skew: float = _REFRESH_SKEW_SECONDS) -> bool:
        """True when the access token is within ``skew`` seconds of expiry."""
        return now >= (self.expires_at - skew)


# Indirection so tests can drive the exchange without a live server.
def _http_post_form(url: str, data: dict[str, str], timeout: float) -> dict[str, Any]:
    """POST form-encoded ``data`` to ``url`` and return the parsed JSON body."""
    import httpx

    resp = httpx.post(url, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _http_post_form_status(
    url: str, data: dict[str, str], timeout: float
) -> tuple[int, dict[str, Any]]:
    """POST form-encoded ``data``; return ``(status, parsed JSON body)``.

    Unlike ``_http_post_form``, 4xx does not raise — RFC 8628 polling reads the
    OAuth error body off a 400. A non-JSON body parses to ``{}``.
    """
    import httpx

    resp = httpx.post(url, data=data, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    return resp.status_code, body


def _http_get_json(url: str, timeout: float) -> dict[str, Any]:
    """GET ``url`` and return the parsed JSON body. Raises on non-2xx/non-JSON."""
    import httpx

    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    return body if isinstance(body, dict) else {}


def _exchange_refresh_token(
    cred: OAuthCredential, *, now: float, timeout: float = _REFRESH_TIMEOUT_SECONDS
) -> OAuthCredential:
    """Run the refresh_token grant and return the rotated credential.

    Raises ``OAuthRefreshError`` (with the endpoint's error body) on an error
    response, transport errors as-is; callers fail open.
    """
    status, body = _http_post_form_status(
        cred.token_endpoint,
        {
            "grant_type": "refresh_token",
            "client_id": cred.client_id,
            "refresh_token": cred.refresh_token,
        },
        timeout,
    )
    if status >= 400:
        error = str(body.get("error") or "")
        description = str(body.get("error_description") or "")
        detail = " — ".join(p for p in (error, description) if p) or "no error body"
        raise OAuthRefreshError(
            _redact_tokens(f"token endpoint returned HTTP {status}: {detail}"),
            error=error,
            permanent=error in _PERMANENT_OAUTH_ERRORS,
        )
    access = body.get("access_token")
    refresh = body.get("refresh_token")
    if not is_oauth_access_token(access) or not refresh:
        raise ValueError("refresh response missing access_token/refresh_token")
    try:
        expires_in = int(body.get("expires_in", 0))
    except (TypeError, ValueError):
        expires_in = 0
    return OAuthCredential(
        access_token=access,
        refresh_token=str(refresh),
        expires_at=now + expires_in,
        client_id=cred.client_id,
        token_endpoint=cred.token_endpoint,
        scope=str(body.get("scope", cred.scope)),
        token_type=str(body.get("token_type", cred.token_type)),
    )


def _exchange_with_retry(cred: OAuthCredential, *, now: float) -> OAuthCredential:
    """Exchange the refresh token, retrying once on transient failure.

    The server accepts a replayed token only briefly after rotating it, so the
    retry cannot wait — and the whole cycle is capped by
    ``_REFRESH_TOTAL_BUDGET_SECONDS`` because it runs under the global refresh
    locks: a fast first failure gets a full-timeout retry, a slow (timed-out)
    first attempt gets only the remaining budget.
    """
    deadline = time.monotonic() + _REFRESH_TOTAL_BUDGET_SECONDS
    try:
        return _exchange_refresh_token(cred, now=now)
    except OAuthRefreshError as exc:
        if exc.permanent:
            raise
        first: Exception = exc
    except Exception as exc:
        first = exc
    remaining = deadline - time.monotonic() - _REFRESH_RETRY_DELAY_SECONDS
    if remaining <= 0:
        raise first
    logger.warning(
        "Honcho OAuth token exchange failed, retrying once: %s",
        _redact_tokens(str(first)),
    )
    time.sleep(_REFRESH_RETRY_DELAY_SECONDS)
    return _exchange_refresh_token(
        cred, now=now, timeout=min(remaining, _REFRESH_TIMEOUT_SECONDS)
    )


def _rotate_and_persist(
    path: Path,
    host: str,
    key: tuple[str, str],
    cred: OAuthCredential,
    *,
    now: float,
    op_label: str = "refresh",
) -> OAuthCredential | None:
    """Exchange ``cred`` and persist the rotation; ``None`` on failure (logged).

    A permanent OAuth error marks the grant dead so later calls skip the
    endpoint until a new login rotates the refresh token.
    """
    try:
        rotated = _exchange_with_retry(cred, now=now)
    except OAuthRefreshError as exc:
        if exc.permanent:
            _mark_grant_dead(key, cred)
            logger.error(
                "Honcho OAuth grant for host %s is no longer valid (%s); "
                "run 'hermes honcho setup' to re-authenticate", host, exc,
            )
        else:
            _refresh_failure_at[key] = time.monotonic()
            logger.warning("Honcho OAuth %s failed for host %s: %s", op_label, host, exc)
        return None
    except Exception as exc:
        _refresh_failure_at[key] = time.monotonic()
        logger.warning(
            "Honcho OAuth %s failed for host %s: %s",
            op_label, host, _redact_tokens(str(exc)),
        )
        return None
    _persist_credential(path, host, rotated)
    return rotated


def _read_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_write_config(path: Path, raw: dict[str, Any]) -> None:
    """Write ``raw`` to ``path`` atomically, preserving 0600 on the new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    text = json.dumps(raw, indent=2) + "\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (overlay wins on scalars/lists)."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _persist_credential(path: Path, host: str, cred: OAuthCredential) -> None:
    """Persist ``cred`` into ``host``'s block (apiKey + oauth), leaving all else intact."""
    raw = _read_config(path)
    hosts = raw.setdefault("hosts", {})
    block = hosts.setdefault(host, {})
    block["apiKey"] = cred.access_token
    block["oauth"] = cred.oauth_block()
    _atomic_write_config(path, raw)
    _expiry_cache[(str(path), host)] = (cred.expires_at, cred.access_token)
    _dead_grants.pop((str(path), host), None)
    _refresh_failure_at.pop((str(path), host), None)


def ensure_fresh_token(
    path: Path,
    host: str,
    raw: dict[str, Any] | None = None,
    *,
    now: float | None = None,
) -> tuple[str | None, bool]:
    """Return ``(access_token, refreshed)`` for ``host``, refreshing if near expiry.

    Returns ``(None, False)`` when the host has no OAuth credential (e.g. a plain
    API key) so callers leave the existing token untouched. Refresh failures are
    swallowed: the current (possibly stale) token is returned with
    ``refreshed=False``, transient failures retry once immediately, and a
    permanently rejected grant is marked dead so later calls skip the endpoint.
    The 401 recovery in session.py escalates dead grants to the user.
    """
    now = time.time() if now is None else now
    key = (str(path), host)

    # Hot path: trust the cached expiry while the token is well clear of the
    # skew window — no disk read. Bypassed when an explicit ``raw`` is supplied.
    if raw is None:
        cached = _expiry_cache.get(key)
        if cached is not None and now < cached[0] - _REFRESH_SKEW_SECONDS:
            return cached[1], False

    source = raw if raw is not None else _read_config(path)
    block = (source.get("hosts") or {}).get(host) or {}
    cred = OAuthCredential.from_host_block(block)
    if cred is None:
        _expiry_cache.pop(key, None)
        return None, False

    _expiry_cache[key] = (cred.expires_at, cred.access_token)
    if not cred.is_expired(now=now):
        return cred.access_token, False
    if _in_failure_cooldown(key):
        # An exchange just failed transiently; don't pile on the endpoint.
        return cred.access_token, False

    with _refresh_lock, _config_refresh_lock(path):
        # Re-read under both locks: another thread or process may have just
        # rotated the token — adopt theirs instead of replaying the old one.
        fresh_block = (_read_config(path).get("hosts") or {}).get(host) or {}
        current = OAuthCredential.from_host_block(fresh_block) or cred
        if not current.is_expired(now=now):
            return current.access_token, current.access_token != cred.access_token
        if _grant_is_dead(key, current):
            return current.access_token, False
        if _in_failure_cooldown(key):
            # The lock holder we waited on just failed; fail open too.
            return current.access_token, False
        rotated = _rotate_and_persist(path, host, key, current, now=now)
        if rotated is None:
            return current.access_token, False
        logger.info("Honcho OAuth token refreshed for host %s", host)
        return rotated.access_token, True


def force_refresh_token(path: Path, host: str) -> str | None:
    """Rotate ``host``'s access token now, ignoring local expiry.

    Recovers a 401 on a token the local clock still thinks is valid.
    """
    now = time.time()
    key = (str(path), host)
    with _refresh_lock, _config_refresh_lock(path):
        block = (_read_config(path).get("hosts") or {}).get(host) or {}
        cred = OAuthCredential.from_host_block(block)
        if cred is None:
            _expiry_cache.pop(key, None)
            return None
        if _grant_is_dead(key, cred):
            return None
        if _in_failure_cooldown(key):
            # An exchange just failed transiently; don't force another full
            # cycle — callers fail open and retry after the cooldown.
            return None
        cached = _expiry_cache.get(key)
        # Another thread or process already rotated: adopt the newer on-disk token.
        if cached is not None and cred.access_token != cached[1] and not cred.is_expired(now=now):
            _expiry_cache[key] = (cred.expires_at, cred.access_token)
            return cred.access_token
        rotated = _rotate_and_persist(path, host, key, cred, now=now, op_label="forced refresh")
        if rotated is None:
            return None
        logger.info("Honcho OAuth token force-refreshed for host %s after an auth failure", host)
        return rotated.access_token


def install_grant(
    path: Path,
    host: str,
    grant: dict[str, Any],
    *,
    client_id: str,
    token_endpoint: str,
    apply_config: bool = True,
    now: float | None = None,
) -> OAuthCredential:
    """Apply a fresh OAuth grant to ``path`` for ``host``.

    Deep-merges the grant's ``config`` (the manifest default_config) into the
    file root — preserving other hosts and root keys — then writes the host's
    ``apiKey`` and ``oauth`` block. ``grant`` is an OAuthTokenResponse dict
    (access_token, refresh_token, expires_in, scope, config).
    ``apply_config=False`` skips the config merge and stores tokens only.
    """
    now = time.time() if now is None else now
    access = grant.get("access_token")
    refresh = grant.get("refresh_token")
    if not is_oauth_access_token(access) or not refresh:
        raise ValueError("grant missing access_token/refresh_token")
    try:
        expires_in = int(grant.get("expires_in", 0))
    except (TypeError, ValueError):
        expires_in = 0

    cred = OAuthCredential(
        access_token=access,
        refresh_token=str(refresh),
        expires_at=now + expires_in,
        client_id=client_id,
        token_endpoint=token_endpoint,
        scope=str(grant.get("scope", "write")),
        token_type=str(grant.get("token_type", "Bearer")),
    )

    raw = _read_config(path)
    granted_config = grant.get("config")
    if isinstance(granted_config, dict):
        cred.consent_peer_name = granted_config.get("peerName")
        if apply_config:
            _deep_merge(raw, granted_config)
    _expiry_cache[(str(path), host)] = (cred.expires_at, cred.access_token)
    _dead_grants.pop((str(path), host), None)
    _refresh_failure_at.pop((str(path), host), None)
    hosts = raw.setdefault("hosts", {})
    block = hosts.setdefault(host, {})
    block["apiKey"] = cred.access_token
    block["oauth"] = cred.oauth_block()
    _atomic_write_config(path, raw)
    return cred


def apply_token_to_client(client: Any, token: str) -> bool:
    """Rotate the live Honcho client's Bearer in place. Returns success.

    The SDK builds its auth header per request from the HTTP client's
    ``api_key``, so mutating it rotates every holder of the singleton without a
    rebuild. Guarded: an SDK shape change degrades to False and the caller can
    fall back to resetting the client.
    """
    http = getattr(client, "_http", None)
    if http is None or not hasattr(http, "api_key"):
        return False
    http.api_key = token
    return True
