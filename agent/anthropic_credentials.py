"""Anthropic credential sources, OAuth flows, and token resolution.

Extracted from ``agent/anthropic_adapter.py``: the adapter is a message/HTTP
translation layer, while everything below owns *where an Anthropic credential
comes from* and *how a rotated one is committed*. Keeping the two apart means
the refresh transaction has a single home instead of being interleaved with
request building.

Sources, in the order ``resolve_anthropic_token()`` consults them:

1. ``ANTHROPIC_TOKEN`` / ``CLAUDE_CODE_OAUTH_TOKEN`` (explicit OAuth env)
2. ``ANTHROPIC_API_KEY`` (explicit API key)
3. ``~/.hermes/.anthropic_oauth.json`` (Hermes PKCE login)
4. ``~/.claude/.credentials.json`` / macOS Keychain (Claude Code)
5. the credential pool in ``auth.json``

Sources 3 and 4 are *singletons*: ``credential_pool._seed_from_singletons()``
re-reads them on every ``load_pool()`` and writes what it finds over the pool
row, which is why a failed write here is a failed refresh (see
``CredentialPersistError``) rather than a best-effort cache miss.

``agent.anthropic_adapter`` re-exports every public name below, so existing
``from agent.anthropic_adapter import resolve_anthropic_token`` imports keep
working.
"""

import json
import logging
import os
import platform
import secrets
import stat
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from agent.secret_scope import get_secret as _get_secret

logger = logging.getLogger(__name__)


def _getenv(name: str, default: str = "") -> str:
    """Profile-scoped replacement for os.getenv on credential reads.

    Routes through the secret scope (Workstream A): identical to os.getenv
    when multiplexing is off, scope-aware (and fail-closed on an unscoped
    read) when on. Mirrors the same wrapper in hermes_cli/runtime_provider.py.
    """
    val = _get_secret(name, default)
    return val if val is not None else default


def _is_oauth_token(key: str) -> bool:
    """Check if the key is an Anthropic OAuth/setup token.

    Positively identifies Anthropic OAuth tokens by their key format:
    - ``sk-ant-`` prefix (but NOT ``sk-ant-api``) → setup tokens, managed keys
    - ``eyJ`` prefix → JWTs from the Anthropic OAuth flow
    - ``cc-`` prefix → Claude Code OAuth access tokens (from CLAUDE_CODE_OAUTH_TOKEN)

    Non-Anthropic keys (MiniMax, Alibaba, etc.) don't match any pattern
    and correctly return False.
    """
    if not key:
        return False
    # Regular Anthropic Console API keys — x-api-key auth, never OAuth
    if key.startswith("sk-ant-api"):
        return False
    # Anthropic-issued tokens (setup-tokens sk-ant-oat-*, managed keys)
    if key.startswith("sk-ant-"):
        return True
    # JWTs from Anthropic OAuth flow
    if key.startswith("eyJ"):
        return True
    # Claude Code OAuth access tokens (opaque, from CLAUDE_CODE_OAUTH_TOKEN)
    if key.startswith("cc-"):
        return True
    return False



class CredentialPersistError(RuntimeError):
    """A rotated single-use credential could not be durably committed.

    Anthropic OAuth refresh tokens are single-use: a successful refresh POST
    consumes the old refresh token server-side and returns a replacement. The
    replacement exists only in memory until it reaches its authoritative
    on-disk store (``~/.claude/.credentials.json`` for ``claude_code``,
    ``~/.hermes/.anthropic_oauth.json`` for ``hermes_pkce``).

    If that write fails and the caller reports success anyway, the on-disk
    (already consumed) pair survives and is re-seeded on the next
    ``load_pool()``, so the following refresh replays a spent token and fails
    with ``invalid_grant`` / ``refresh_token_reused``. Callers must therefore
    treat this as a failed refresh, not a successful one, and fail closed.
    """

    def __init__(self, path: Any, cause: BaseException) -> None:
        super().__init__(
            f"failed to durably persist rotated Anthropic credentials to {path}: {cause}"
        )
        self.path = path


# Fingerprints of Anthropic secrets whose refresh POST succeeded (so the
# server-side pair was rotated and the old refresh token is spent) but whose
# replacement never reached its authoritative store.  The pre-rotation pair
# survives on disk and is re-seeded on the next ``load_pool()``, so without an
# explicit verdict the resolver happily hands that already-consumed credential
# back from a later source and the caller reads a silent success.
#
# Kept as non-reversible digests and bounded: a spent secret is spent forever,
# so entries never need clearing (a re-auth mints new tokens with new
# fingerprints).
#
# The registry has TWO scopes, because the credential it protects does:
#   * process-local (this OrderedDict) — fast path, always recorded;
#   * durable sidecar file next to the shared credential source — the
#     authority boundary of ``claude_code``/``hermes_pkce`` is the shared
#     singleton file, which other Hermes processes/profiles read with fresh
#     interpreters.  A process-local verdict only stops the process that
#     lost the commit from lying to itself; the sidecar stops every OTHER
#     process from leasing the stale pair or re-POSTing the spent refresh
#     token.  The sidecar stores only one-way fingerprints (never secrets)
#     and is written under the same path-keyed cross-process lock that
#     serializes refreshes of that source.
_SPENT_ROTATION_LOCK = threading.Lock()
_SPENT_ROTATION_FINGERPRINTS: "OrderedDict[str, None]" = OrderedDict()
_SPENT_ROTATION_MAX_TRACKED = 64
_SPENT_ROTATION_SIDECAR_VERSION = 1


def _spent_rotation_sidecar_path(source_path: Path) -> Path:
    """Sidecar registry path for a shared credential source file."""
    return source_path.with_name(source_path.name + ".hermes-spent-rotations.json")


def spent_rotation_source_path(source: Any) -> Optional[Path]:
    """Map a pool-entry source to the shared singleton file it borrows from.

    Only singleton-backed sources have a cross-process authority boundary;
    profile-owned rows are already protected by the process-local registry
    plus the pool quarantine.
    """
    if source == "claude_code":
        return claude_code_credentials_path()
    if source == "hermes_pkce":
        return _get_hermes_oauth_file()
    return None


def _read_spent_rotation_sidecar(source_path: Optional[Path]) -> set:
    if source_path is None:
        return set()
    try:
        raw = json.loads(
            _spent_rotation_sidecar_path(source_path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return set()
    fingerprints = raw.get("fingerprints") if isinstance(raw, dict) else None
    if not isinstance(fingerprints, list):
        return set()
    return {fp for fp in fingerprints if isinstance(fp, str) and fp}


def _append_spent_rotation_sidecar(source_path: Path, fingerprints: list) -> None:
    """Merge fingerprints into the sidecar registry (atomic replace).

    Callers on the refresh path already hold the path-keyed cross-process
    lock for ``source_path``, so concurrent merge-writes are serialized.
    Fail-soft: a sidecar write failure must never mask the fail-closed
    verdict already recorded in the process-local registry.
    """
    sidecar = _spent_rotation_sidecar_path(source_path)
    try:
        merged = _read_spent_rotation_sidecar(source_path)
        merged.update(fingerprints)
        bounded = sorted(merged)[-_SPENT_ROTATION_MAX_TRACKED * 4 :]
        payload = json.dumps(
            {
                "version": _SPENT_ROTATION_SIDECAR_VERSION,
                "comment": (
                    "Non-secret one-way fingerprints of Anthropic OAuth "
                    "credentials whose rotation was consumed server-side but "
                    "never durably committed. Written by Hermes so sibling "
                    "processes sharing this credential source fail closed "
                    "instead of replaying a spent single-use refresh token."
                ),
                "fingerprints": bounded,
            },
            indent=2,
        )
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        tmp = sidecar.with_name(sidecar.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, sidecar)
    except Exception:
        logger.debug(
            "Failed to persist spent-rotation fingerprints to %s", sidecar,
            exc_info=True,
        )


def mark_rotation_consumed_uncommitted(
    *secrets: Any, source_path: Optional[Path] = None
) -> None:
    """Record secrets consumed by a refresh whose replacement never committed.

    Called from every commit-failure path (the direct resolver here and
    ``CredentialPool._fail_closed_unpersisted_rotation``).  Recording the
    *pre-rotation* pair is what lets later resolution steps recognise the stale
    copy they read back off disk as unusable rather than as a working token.

    When ``source_path`` names the shared singleton file the credential was
    borrowed from, the verdict is additionally persisted to that source's
    sidecar registry so other processes/profiles sharing the file adopt it too.
    """
    from agent.credential_persistence import fingerprint_secret_value

    recorded: list = []
    with _SPENT_ROTATION_LOCK:
        for secret in secrets:
            value = str(secret or "").strip()
            if not value:
                continue
            fingerprint = fingerprint_secret_value(value)
            if not fingerprint:
                continue
            recorded.append(fingerprint)
            _SPENT_ROTATION_FINGERPRINTS.pop(fingerprint, None)
            _SPENT_ROTATION_FINGERPRINTS[fingerprint] = None
            while len(_SPENT_ROTATION_FINGERPRINTS) > _SPENT_ROTATION_MAX_TRACKED:
                _SPENT_ROTATION_FINGERPRINTS.popitem(last=False)
    if recorded and source_path is not None:
        _append_spent_rotation_sidecar(source_path, recorded)


def is_rotation_consumed_uncommitted(
    secret: Any, *, source_path: Optional[Path] = None
) -> bool:
    """True when *secret* belongs to a rotation that was spent but not committed.

    Checks the process-local registry first, then (when ``source_path`` is
    given) the durable sidecar registry of the shared credential source, so a
    fresh interpreter in another process still sees the terminal verdict.
    """
    from agent.credential_persistence import fingerprint_secret_value

    value = str(secret or "").strip()
    if not value:
        return False
    fingerprint = fingerprint_secret_value(value)
    if not fingerprint:
        return False
    with _SPENT_ROTATION_LOCK:
        if fingerprint in _SPENT_ROTATION_FINGERPRINTS:
            return True
    return fingerprint in _read_spent_rotation_sidecar(source_path)


def _read_claude_code_credentials_from_keychain() -> Optional[Dict[str, Any]]:
    """Read Claude Code OAuth credentials from the macOS Keychain.

    Claude Code >=2.1.114 stores credentials in the macOS Keychain under the
    service name "Claude Code-credentials" rather than (or in addition to)
    the JSON file at ~/.claude/.credentials.json.

    The password field contains a JSON string with the same claudeAiOauth
    structure as the JSON file.

    Returns dict with {accessToken, refreshToken?, expiresAt?} or None.
    """
    if platform.system() != "Darwin":
        return None

    try:
        # Read the "Claude Code-credentials" generic password entry
        result = subprocess.run(
            ["security", "find-generic-password",
             "-s", "Claude Code-credentials",
             "-w"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("Keychain: security command not available or timed out")
        return None

    if result.returncode != 0:
        logger.debug("Keychain: no entry found for 'Claude Code-credentials'")
        return None

    raw = result.stdout.strip()
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Keychain: credentials payload is not valid JSON")
        return None

    oauth_data = data.get("claudeAiOauth")
    if oauth_data and isinstance(oauth_data, dict):
        access_token = oauth_data.get("accessToken", "")
        if access_token:
            return {
                "accessToken": access_token,
                "refreshToken": oauth_data.get("refreshToken", ""),
                "expiresAt": oauth_data.get("expiresAt", 0),
                "source": "macos_keychain",
            }

    return None


def claude_code_credentials_path() -> Path:
    """Location Claude Code CLI writes its shared OAuth credentials file.

    This file is not profile-owned: every Hermes profile's credential pool
    reads and writes the *same* path, so cross-profile refresh races on a
    ``claude_code`` pool entry must be serialized against this exact path
    (see ``CredentialPool._claude_code_credentials_lock`` in
    ``agent/credential_pool.py``).
    """
    return Path.home() / ".claude" / ".credentials.json"


def _read_claude_code_credentials_from_file() -> Optional[Dict[str, Any]]:
    """Read Claude Code OAuth credentials from ~/.claude/.credentials.json.

    Returns dict with {accessToken, refreshToken?, expiresAt?, source} or None.
    """
    cred_path = claude_code_credentials_path()
    if not cred_path.exists():
        return None
    try:
        data = json.loads(cred_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, IOError) as e:
        logger.debug("Failed to read ~/.claude/.credentials.json: %s", e)
        return None

    oauth_data = data.get("claudeAiOauth")
    if not (oauth_data and isinstance(oauth_data, dict)):
        return None
    access_token = oauth_data.get("accessToken", "")
    if not access_token:
        return None
    return {
        "accessToken": access_token,
        "refreshToken": oauth_data.get("refreshToken", ""),
        "expiresAt": oauth_data.get("expiresAt", 0),
        "source": "claude_code_credentials_file",
    }


def read_claude_code_credentials() -> Optional[Dict[str, Any]]:
    """Read refreshable Claude Code OAuth credentials.

    Reads from two possible sources and reconciles them:
      1. macOS Keychain (Darwin only) — "Claude Code-credentials" entry
      2. ~/.claude/.credentials.json file

    Selection rules when both are present:
      - If exactly one is non-expired, prefer that one. (Handles the case
        where Claude Code refreshes one source but not the other — observed
        in the wild on Claude Code 2.1.x.)
      - Otherwise, prefer the source with the later ``expiresAt`` so that
        any subsequent refresh uses the most recent ``refreshToken``.

    This intentionally excludes ~/.claude.json primaryApiKey. Opencode's
    subscription flow is OAuth/setup-token based with refreshable credentials,
    and native direct Anthropic provider usage should follow that path rather
    than auto-detecting Claude's first-party managed key.

    Returns dict with {accessToken, refreshToken?, expiresAt?, source} or None.
    """
    kc_creds = _read_claude_code_credentials_from_keychain()
    file_creds = _read_claude_code_credentials_from_file()

    if kc_creds and file_creds:
        kc_valid = is_claude_code_token_valid(kc_creds)
        file_valid = is_claude_code_token_valid(file_creds)
        if kc_valid and not file_valid:
            return kc_creds
        if file_valid and not kc_valid:
            return file_creds
        # Both valid or both expired: prefer the later expiresAt so the
        # downstream refresh path uses the freshest refresh_token.
        kc_exp = kc_creds.get("expiresAt", 0) or 0
        file_exp = file_creds.get("expiresAt", 0) or 0
        return kc_creds if kc_exp >= file_exp else file_creds

    return kc_creds or file_creds


def is_claude_code_token_valid(creds: Dict[str, Any]) -> bool:
    """Check if Claude Code credentials have a non-expired access token."""
    import time

    expires_at = creds.get("expiresAt", 0)
    if not expires_at:
        # No expiry set (managed keys) — valid if token is present
        return bool(creds.get("accessToken"))

    # expiresAt is in milliseconds since epoch
    now_ms = int(time.time() * 1000)
    # Allow 60 seconds of buffer
    return now_ms < (expires_at - 60_000)


def refresh_anthropic_oauth_pure(refresh_token: str, *, use_json: bool = False) -> Dict[str, Any]:
    """Refresh an Anthropic OAuth token without mutating local credential files."""
    import time
    import urllib.parse
    import urllib.request

    if not refresh_token:
        raise ValueError("refresh_token is required")

    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    if use_json:
        data = json.dumps({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }).encode()
        content_type = "application/json"
    else:
        data = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }).encode()
        content_type = "application/x-www-form-urlencoded"

    token_endpoints = [
        "https://platform.claude.com/v1/oauth/token",
        "https://console.anthropic.com/v1/oauth/token",
    ]
    last_error = None
    for endpoint in token_endpoints:
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": content_type,
                "User-Agent": _OAUTH_TOKEN_USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
        except Exception as exc:
            last_error = exc
            logger.debug("Anthropic token refresh failed at %s: %s", endpoint, exc)
            continue

        access_token = result.get("access_token", "")
        if not access_token:
            raise ValueError("Anthropic refresh response was missing access_token")
        next_refresh = result.get("refresh_token", refresh_token)
        expires_in = result.get("expires_in", 3600)
        return {
            "access_token": access_token,
            "refresh_token": next_refresh,
            "expires_at_ms": int(time.time() * 1000) + (expires_in * 1000),
        }

    if last_error is not None:
        raise last_error
    raise ValueError("Anthropic token refresh failed")


def _refresh_oauth_token(creds: Dict[str, Any]) -> Optional[str]:
    """Attempt to refresh an expired Claude Code OAuth token.

    Claude Code's OAuth refresh tokens are single-use: a successful refresh
    rotates the pair and invalidates the old refresh token. Claude Code itself
    also refreshes on its own schedule (IDE/CLI activity), so by the time
    Hermes notices an expired token, Claude Code may have already rotated it.
    POSTing our now-stale refresh token in that window races Claude Code and
    fails with ``invalid_grant``.

    So before refreshing, re-read the live credential sources. If Claude Code
    has already produced a valid token, adopt it and skip the POST entirely.
    Only fall back to refreshing ourselves when no fresh credential is found.
    """
    # Claude Code may have already refreshed — adopt its token rather than
    # racing it with our (possibly already-rotated) refresh token. The read,
    # decision, POST, and write-back all belong to the shared credentials
    # source, so hold the same path-keyed cross-process lock used by the pool.
    # Without this direct resolver path, two profiles can still spend one
    # single-use refresh token even though CredentialPool is serialized.
    try:
        from hermes_cli.auth import AUTH_LOCK_TIMEOUT_SECONDS, _auth_store_lock, env_float

        refresh_timeout_seconds = env_float(
            "HERMES_ANTHROPIC_REFRESH_TIMEOUT_SECONDS", 20
        )
        lock_timeout_seconds = max(
            float(AUTH_LOCK_TIMEOUT_SECONDS),
            float(refresh_timeout_seconds) + 5.0,
        )
        with _auth_store_lock(
            timeout_seconds=lock_timeout_seconds,
            target_path=claude_code_credentials_path(),
        ):
            # Only adopt when the live re-read produced a DIFFERENT token with
            # a real future expiry: re-adopting the same credential we were
            # just handed would be a no-op, and a 0/absent ``expiresAt`` means
            # "managed key / unknown expiry" (see is_claude_code_token_valid).
            current = read_claude_code_credentials()
            if current:
                current_token = current.get("accessToken", "")
                current_exp = current.get("expiresAt", 0) or 0
                if (
                    current_token
                    and current_token != creds.get("accessToken", "")
                    and current_exp > 0
                    and is_claude_code_token_valid(current)
                ):
                    logger.debug("Adopted Claude Code's already-refreshed OAuth token")
                    return current_token

            refresh_token = (
                (current or {}).get("refreshToken", "")
                or creds.get("refreshToken", "")
            )
            if not refresh_token:
                logger.debug("No refresh token available — cannot refresh")
                return None

            # Another process may have spent this refresh token and lost the
            # commit; its durable sidecar verdict is authoritative for the
            # shared source. POSTing it again would just burn the family into
            # ``invalid_grant``.
            if is_rotation_consumed_uncommitted(
                refresh_token, source_path=claude_code_credentials_path()
            ):
                logger.debug(
                    "Refresh token was already consumed by an uncommitted rotation "
                    "- refusing to replay it; re-run 'claude setup-token'"
                )
                return None

            try:
                refreshed = refresh_anthropic_oauth_pure(refresh_token, use_json=False)
            except Exception as e:
                logger.debug("Failed to refresh Claude Code token: %s", e)
                return None

            # The POST above already consumed ``refresh_token`` server-side.
            # Writing the replacement pair is the commit step of that
            # transaction, not a cache update: if it fails, the rotation is
            # unrecoverable and the pair still on disk is spent. Fail closed
            # rather than handing back an access token whose refresh half was
            # lost — reporting success here is what lets a later load replay
            # the consumed token and produce ``invalid_grant``.
            try:
                _write_claude_code_credentials(
                    refreshed["access_token"],
                    refreshed["refresh_token"],
                    refreshed["expires_at_ms"],
                )
            except Exception as e:
                logger.error(
                    "Anthropic OAuth refresh rotated the single-use token but could not "
                    "commit it to %s (%s) — treating the refresh as failed; "
                    "re-run 'claude setup-token' to reauthenticate",
                    claude_code_credentials_path(),
                    e,
                )
                # The POST already spent ``refresh_token`` server-side and the
                # replacement is gone.  The pre-rotation pair is still on disk,
                # so mark it: without this, source 5 re-reads it through the
                # pool and returns the consumed credential as a success.
                mark_rotation_consumed_uncommitted(
                    refresh_token,
                    creds.get("accessToken", ""),
                    (current or {}).get("accessToken", ""),
                    (current or {}).get("refreshToken", ""),
                    source_path=claude_code_credentials_path(),
                )
                return None

            logger.debug("Successfully refreshed Claude Code OAuth token")
            return refreshed["access_token"]
    except Exception as e:
        # Lock acquisition/read failures should preserve the resolver's
        # existing fail-soft contract rather than taking down agent startup.
        logger.debug("Failed to acquire Claude Code refresh lock: %s", e)
        return None


def _write_claude_code_credentials(
    access_token: str,
    refresh_token: str,
    expires_at_ms: int,
    *,
    scopes: Optional[list] = None,
) -> None:
    """Write refreshed credentials back to ~/.claude/.credentials.json.

    The optional *scopes* list (e.g. ``["user:inference", "user:profile", ...]``)
    is persisted so that Claude Code's own auth check recognises the credential
    as valid.  Claude Code >=2.1.81 gates on the presence of ``"user:inference"``
    in the stored scopes before it will use the token.

    Raises ``CredentialPersistError`` when the rotated pair does not reach the
    file. This write is the commit step of the refresh transaction, not a
    best-effort cache update: a swallowed failure leaves the consumed
    pre-rotation pair on disk to be re-seeded and replayed (see
    ``CredentialPersistError``).
    """
    cred_path = claude_code_credentials_path()
    try:
        # Read existing file to preserve other fields
        existing = {}
        if cred_path.exists():
            existing = json.loads(cred_path.read_text(encoding="utf-8"))

        oauth_data: Dict[str, Any] = {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expires_at_ms,
        }
        if scopes is not None:
            oauth_data["scopes"] = scopes
        elif "claudeAiOauth" in existing and "scopes" in existing["claudeAiOauth"]:
            # Preserve previously-stored scopes when the refresh response
            # does not include a scope field.
            oauth_data["scopes"] = existing["claudeAiOauth"]["scopes"]

        existing["claudeAiOauth"] = oauth_data

        cred_path.parent.mkdir(parents=True, exist_ok=True)
        # Per-process random suffix avoids collisions between concurrent
        # writers and stale leftovers from a prior crashed write.
        _tmp_cred = cred_path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
        try:
            # Create the temp file atomically at 0o600. The previous
            # write_text + post-replace chmod opened a TOCTOU window where
            # both the temp file and the destination briefly inherited the
            # process umask (commonly 0o644 = world-readable), exposing
            # Claude Code OAuth tokens to other local users between create
            # and chmod. Mirrors agent/google_oauth.py (#19673) and
            # tools/mcp_oauth.py (#21148). Parent dir (~/.claude/) is
            # owned by Claude Code itself, so we leave its mode alone.
            fd = os.open(
                str(_tmp_cred),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(_tmp_cred, cred_path)
        except OSError:
            try:
                _tmp_cred.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    except (OSError, IOError, ValueError) as e:
        # ValueError covers a corrupt existing file (JSONDecodeError): the
        # merge-read is part of the commit, so failing it means the rotated
        # pair never landed either.
        logger.error("Failed to write refreshed credentials to %s: %s", cred_path, e)
        raise CredentialPersistError(cred_path, e) from e


def _resolve_claude_code_token_from_credentials(creds: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve a token from Claude Code credential files, refreshing if needed."""
    creds = creds or read_claude_code_credentials()
    if creds and is_rotation_consumed_uncommitted(
        creds.get("accessToken", ""), source_path=claude_code_credentials_path()
    ):
        # This process already rotated this pair and failed to commit the
        # replacement.  The file still holds the spent copy; treating it as
        # usable is exactly the silent success this transaction fails closed
        # to prevent.
        logger.debug(
            "Claude Code credentials hold a rotated-but-uncommitted token - refusing"
        )
        return None
    if creds and is_claude_code_token_valid(creds):
        logger.debug("Using Claude Code credentials (auto-detected)")
        return creds["accessToken"]
    if creds:
        logger.debug("Claude Code credentials expired — attempting refresh")
        refreshed = _refresh_oauth_token(creds)
        if refreshed:
            return refreshed
        logger.debug("Token refresh failed — re-run 'claude setup-token' to reauthenticate")
    return None


def _prefer_refreshable_claude_code_token(env_token: str, creds: Optional[Dict[str, Any]]) -> Optional[str]:
    """Prefer Claude Code creds when a persisted env OAuth token would shadow refresh.

    Hermes historically persisted setup tokens into ANTHROPIC_TOKEN. That makes
    later refresh impossible because the static env token wins before we ever
    inspect Claude Code's refreshable credential file. If we have a refreshable
    Claude Code credential record, prefer it over the static env OAuth token.
    """
    if not env_token or not _is_oauth_token(env_token) or not isinstance(creds, dict):
        return None
    if not creds.get("refreshToken"):
        return None

    resolved = _resolve_claude_code_token_from_credentials(creds)
    if resolved and resolved != env_token:
        logger.debug(
            "Preferring Claude Code credential file over static env OAuth token so refresh can proceed"
        )
        return resolved
    return None


def _resolve_anthropic_pool_token() -> Optional[str]:
    """Return the first available Anthropic OAuth token from credential_pool.

    Read-only: enumerates with ``clear_expired=False, refresh=False`` so a bare
    token *resolve* (which runs from diagnostic/read-only call sites such as
    ``account_usage`` and ``hermes models``) never mutates ``~/.hermes/auth.json``
    or makes a network refresh call. Refresh-on-expiry is owned by the API call
    path's pool recovery, not the resolver.
    """
    try:
        from agent.credential_pool import AUTH_TYPE_OAUTH, load_pool
    except Exception:
        return None

    try:
        pool = load_pool("anthropic")
        # Enumerate read-only (clear_expired=False, refresh=False): never persist
        # to auth.json or trigger a network refresh from a bare resolve. select()
        # is deliberately NOT used — it runs clear_expired=True, refresh=True,
        # which would violate this read-only contract.
        entries, _pending = pool._available_entries(clear_expired=False, refresh=False)
    except Exception:
        logger.debug("Failed to read Anthropic credential_pool", exc_info=True)
        return None

    for entry in entries:
        if getattr(entry, "auth_type", None) != AUTH_TYPE_OAUTH:
            continue
        # access_token is a declared field but a persisted entry can carry an
        # explicit null (or a partially-written OAuth entry), so coerce before
        # strip — a bare None.strip() here would escape the try/excepts above
        # and crash the whole resolver, taking down the source #5 fallback too.
        # Matches the aux-client analog (auxiliary_client.py: str(key or "")).
        token = (getattr(entry, "access_token", None) or "").strip()
        if not token:
            continue
        # ``load_pool()`` re-seeds pool rows from the singleton files, so a
        # rotation that was consumed upstream but never committed comes back
        # here looking healthy.  Enumeration is deliberately read-only
        # (refresh=False), which means nothing on this path would otherwise
        # notice that the credential is spent.  Singleton-backed sources also
        # consult the durable sidecar registry: the failed commit may have
        # happened in a DIFFERENT process, whose process-local verdict this
        # interpreter never saw.
        entry_source_path = spent_rotation_source_path(getattr(entry, "source", None))
        if is_rotation_consumed_uncommitted(
            token, source_path=entry_source_path
        ) or is_rotation_consumed_uncommitted(
            getattr(entry, "refresh_token", None), source_path=entry_source_path
        ):
            logger.debug(
                "Skipping Anthropic pool entry %s: rotated-but-uncommitted credential",
                getattr(entry, "id", "?"),
            )
            continue
        return token

    return None


def resolve_anthropic_token() -> Optional[str]:
    """Resolve an Anthropic token from all available sources.

    Priority:
      1. ANTHROPIC_TOKEN env var (OAuth/setup token saved by Hermes)
      2. CLAUDE_CODE_OAUTH_TOKEN env var
      3. ANTHROPIC_API_KEY env var (explicit regular API key)
      4. Claude Code credentials (~/.claude.json or ~/.claude/.credentials.json)
         — with automatic refresh if expired and a refresh token is available
      5. Anthropic credential_pool OAuth entry (~/.hermes/auth.json)

    Returns the token string or None.
    """
    creds: Optional[Dict[str, Any]] = None
    creds_loaded = False

    def _read_creds() -> Optional[Dict[str, Any]]:
        nonlocal creds, creds_loaded
        if not creds_loaded:
            creds = read_claude_code_credentials()
            creds_loaded = True
        return creds

    # 1. Hermes-managed OAuth/setup token env var
    token = _getenv("ANTHROPIC_TOKEN").strip()
    if token:
        preferred = _prefer_refreshable_claude_code_token(token, _read_creds())
        if preferred:
            return preferred
        return token

    # 2. CLAUDE_CODE_OAUTH_TOKEN (used by Claude Code for setup-tokens)
    cc_token = _getenv("CLAUDE_CODE_OAUTH_TOKEN").strip()
    if cc_token:
        preferred = _prefer_refreshable_claude_code_token(cc_token, _read_creds())
        if preferred:
            return preferred
        return cc_token

    # 3. Regular API key. An explicit user-configured key must not be shadowed
    # by auto-discovered Claude Code or credential-pool OAuth credentials.
    api_key = _getenv("ANTHROPIC_API_KEY").strip()
    if api_key:
        return api_key

    # 4. Claude Code credential file
    resolved_claude_token = _resolve_claude_code_token_from_credentials(_read_creds())
    if resolved_claude_token:
        return resolved_claude_token

    # 5. Hermes credential_pool OAuth entry.
    resolved_pool_token = _resolve_anthropic_pool_token()
    if resolved_pool_token:
        return resolved_pool_token

    return None


def run_oauth_setup_token() -> Optional[str]:
    """Run 'claude setup-token' interactively and return the resulting token.

    Checks multiple sources after the subprocess completes:
      1. Claude Code credential files (may be written by the subprocess)
      2. CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_TOKEN env vars

    Returns the token string, or None if no credentials were obtained.
    Raises FileNotFoundError if the 'claude' CLI is not installed.
    """
    import shutil
    import subprocess

    claude_path = shutil.which("claude")
    if not claude_path:
        raise FileNotFoundError(
            "The 'claude' CLI is not installed. "
            "Install it with: npm install -g @anthropic-ai/claude-code"
        )

    # Run interactively — stdin/stdout/stderr inherited so the user can
    # complete the OAuth login prompt. Must keep inherited stdin; the TUI-EOF
    # concern does not apply to an interactive login the user explicitly
    # invokes.  noqa: subprocess-stdin
    try:
        subprocess.run([claude_path, "setup-token"])
    except (KeyboardInterrupt, EOFError):
        return None

    # Check if credentials were saved to Claude Code's config files
    creds = read_claude_code_credentials()
    if creds and is_claude_code_token_valid(creds):
        return creds["accessToken"]

    # Check env vars that may have been set
    for env_var in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_TOKEN"):
        val = _getenv(env_var).strip()
        if val:
            return val

    return None


# ── Hermes-native PKCE OAuth flow ────────────────────────────────────────
# Mirrors the flow used by Claude Code, pi-ai, and OpenCode.
# Stores credentials in ~/.hermes/.anthropic_oauth.json (our own file).

_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
# Anthropic migrated the OAuth token endpoint to platform.claude.com;
# console.anthropic.com now 404s. Callers should iterate _OAUTH_TOKEN_URLS
# (new host first, console fallback). _OAUTH_TOKEN_URL is kept as the primary
# for backward compatibility with existing imports and now points at the live host.
_OAUTH_TOKEN_URLS = [
    "https://platform.claude.com/v1/oauth/token",
    "https://console.anthropic.com/v1/oauth/token",
]
_OAUTH_TOKEN_URL = _OAUTH_TOKEN_URLS[0]
# User-Agent sent on the OAuth *token endpoint* (login exchange + refresh).
# Anthropic rate-limits (HTTP 429) any token-endpoint request whose UA starts
# with ``claude-code/`` — verified empirically against platform.claude.com:
# ``claude-code/2.1.200`` and ``Mozilla/5.0`` -> 429; ``axios/*``, ``node``,
# and SDK-style UAs -> 400 (reached code validation). The real Claude Code CLI
# exchanges the auth code with a bare axios client (``axios/<ver>``), NOT its
# ``claude-code/`` inference UA. We mirror that here. NOTE: the *inference* path
# (build_anthropic_kwargs) still uses the ``claude-code/`` UA + ``x-app: cli`` —
# that fingerprint is required there and is NOT throttled on the messages API.
_OAUTH_TOKEN_USER_AGENT = "axios/1.7.9"
_OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
_OAUTH_SCOPES = "org:create_api_key user:profile user:inference"
def _get_hermes_oauth_file() -> Path:
    return get_hermes_home() / ".anthropic_oauth.json"


def _root_hermes_oauth_file() -> Optional[Path]:
    """Global-root ``.anthropic_oauth.json`` when running inside a named profile.

    ``None`` in classic mode (profile == root). Used to commit a rotation of a
    grant the profile borrowed through the credential-pool root fallback.
    """
    try:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root()
        if root.resolve(strict=False) == get_hermes_home().resolve(strict=False):
            return None
        return root / ".anthropic_oauth.json"
    except Exception:
        return None


def _generate_pkce() -> tuple:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    import base64
    import hashlib
    import secrets

    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


def run_hermes_oauth_login_pure() -> Optional[Dict[str, Any]]:
    """Run Hermes-native OAuth PKCE flow and return credential state."""
    import secrets
    import time
    import webbrowser

    verifier, challenge = _generate_pkce()
    oauth_state = secrets.token_urlsafe(32)

    params = {
        "code": "true",
        "client_id": _OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _OAUTH_REDIRECT_URI,
        "scope": _OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": oauth_state,
    }
    from urllib.parse import urlencode

    auth_url = f"https://claude.ai/oauth/authorize?{urlencode(params)}"

    print()
    print("Authorize Hermes with your Claude Pro/Max subscription.")
    print()
    print("╭─ Claude Pro/Max Authorization ────────────────────╮")
    print("│                                                   │")
    print("│  Open this link in your browser:                  │")
    print("╰───────────────────────────────────────────────────╯")
    print()
    print(f"  {auth_url}")
    print()

    try:
        from hermes_cli.auth import _can_open_graphical_browser as _can_open_gui
    except Exception:
        _can_open_gui = lambda: True  # noqa: E731 — degrade to prior behavior

    if _can_open_gui():
        try:
            webbrowser.open(auth_url)
            print("  (Browser opened automatically)")
        except Exception:
            pass

    print()
    print("After authorizing, you'll see a code. Paste it below.")
    print()
    try:
        auth_code = input("Authorization code: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None

    if not auth_code:
        print("No code entered.")
        return None

    splits = auth_code.split("#")
    code = splits[0]
    received_state = splits[1] if len(splits) > 1 else ""

    # Validate state to prevent CSRF (RFC 6749 §10.12)
    if received_state != oauth_state:
        logger.warning("OAuth state mismatch — possible CSRF, aborting")
        return None

    try:
        import urllib.request

        exchange_data = json.dumps({
            "grant_type": "authorization_code",
            "client_id": _OAUTH_CLIENT_ID,
            "code": code,
            "state": received_state,
            "redirect_uri": _OAUTH_REDIRECT_URI,
            "code_verifier": verifier,
        }).encode()

        # Anthropic migrated the OAuth token endpoint to platform.claude.com;
        # console.anthropic.com now 404s. Try the new host first, then fall
        # back to console for older deployments (mirrors the refresh path).
        # UA is _OAUTH_TOKEN_USER_AGENT (a non-claude-code UA) — see the
        # constant's definition for why the token endpoint must not send
        # claude-code/ (429 UA-prefix block).
        result = None
        last_error = None
        for endpoint in _OAUTH_TOKEN_URLS:
            req = urllib.request.Request(
                endpoint,
                data=exchange_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": _OAUTH_TOKEN_USER_AGENT,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read().decode())
                break
            except Exception as exc:
                last_error = exc
                logger.debug("Anthropic token exchange failed at %s: %s", endpoint, exc)
                continue

        if result is None:
            raise last_error if last_error is not None else ValueError(
                "Anthropic token exchange failed"
            )
    except Exception as e:
        print(f"Token exchange failed: {e}")
        return None

    access_token = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")
    expires_in = result.get("expires_in", 3600)

    if not access_token:
        print("No access token in response.")
        return None

    expires_at_ms = int(time.time() * 1000) + (expires_in * 1000)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at_ms": expires_at_ms,
    }


def read_hermes_oauth_credentials() -> Optional[Dict[str, Any]]:
    """Read Hermes-managed OAuth credentials from ~/.hermes/.anthropic_oauth.json."""
    oauth_file = _get_hermes_oauth_file()
    if oauth_file.exists():
        try:
            data = json.loads(oauth_file.read_text(encoding="utf-8"))
            if data.get("accessToken"):
                return data
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read Hermes OAuth credentials: %s", e)
    return None


def _write_hermes_oauth_credentials(
    access_token: str,
    refresh_token: Optional[str],
    expires_at_ms: Optional[int],
    *,
    target: Optional[Path] = None,
) -> None:
    """Write refreshed hermes_pkce tokens back to ~/.hermes/.anthropic_oauth.json.

    ``target`` overrides the destination: a named profile that rotated a grant
    it BORROWED from the global root (credential-pool root fallback) must
    commit the new pair to the ROOT singleton, not create a forked copy under
    its own HERMES_HOME (#100339).

    Without this, a successful pool-level refresh of a ``hermes_pkce``-sourced
    entry is invisible to this singleton file. The next ``load_pool()`` call
    runs ``_seed_from_singletons()``, which reads the stale file and
    overwrites the freshly-rotated pool entry with the pre-refresh (and, for
    single-use Anthropic refresh tokens, already-consumed) token pair.

    Raises ``CredentialPersistError`` when the rotated pair does not reach the
    file, for the same reason ``_write_claude_code_credentials`` does: this is
    the commit step of the refresh transaction.
    """
    oauth_file = target if target is not None else _get_hermes_oauth_file()
    try:
        oauth_data = {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expires_at_ms,
        }
        oauth_file.parent.mkdir(parents=True, exist_ok=True)
        _tmp_oauth = oauth_file.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
        try:
            fd = os.open(
                str(_tmp_oauth),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(oauth_data, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(_tmp_oauth, oauth_file)
        except OSError:
            try:
                _tmp_oauth.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    except (OSError, IOError, ValueError) as e:
        logger.error(
            "Failed to write refreshed Hermes OAuth credentials to %s: %s", oauth_file, e
        )
        raise CredentialPersistError(oauth_file, e) from e

