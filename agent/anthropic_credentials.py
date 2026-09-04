"""Anthropic credential sources, OAuth flows, and token resolution.

``resolve_anthropic_token()`` order: ``ANTHROPIC_TOKEN`` / ``CLAUDE_CODE_OAUTH_TOKEN``,
``ANTHROPIC_API_KEY``, ``~/.claude/.credentials.json`` / macOS Keychain, then the
``auth.json`` credential pool. ``~/.hermes/.anthropic_oauth.json`` (Hermes PKCE) and
the Claude Code file are *singletons*: ``credential_pool._seed_from_singletons()``
re-reads them on every ``load_pool()``, so a failed write here is a failed refresh
(``CredentialPersistError``), not a cache miss.
"""

import base64
import contextlib
import functools
import hashlib
import json
import logging
import os
import platform
import secrets
import stat
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from agent.secret_scope import get_secret as _get_secret

logger = logging.getLogger(__name__)

_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
# platform.claude.com is the live token host; console.anthropic.com 404s but is kept as a fallback.
_OAUTH_TOKEN_URLS = [
    "https://platform.claude.com/v1/oauth/token", "https://console.anthropic.com/v1/oauth/token"
]
# Anthropic 429s token-endpoint requests whose UA starts with ``claude-code/`` (or Mozilla); the real CLI uses
# bare axios there. Inference (build_anthropic_kwargs) still needs claude-code/.
_OAUTH_TOKEN_USER_AGENT = "axios/1.7.9"
_OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
_OAUTH_SCOPES = "org:create_api_key user:profile user:inference"


def _getenv(name: str, default: str = "") -> str:
    """Profile-scoped os.getenv for credential reads (fail-closed on unscoped reads when multiplexing)."""
    val = _get_secret(name, default)
    return val if val is not None else default


def _first_env(*names: str) -> str:
    """First non-blank (stripped) value among *names*, else ''."""
    return next((v for v in (_getenv(n).strip() for n in names) if v), "")


def _is_oauth_token(key: str) -> bool:
    """True for Anthropic OAuth/setup tokens (sk-ant-*, eyJ JWTs, cc-); False for sk-ant-api* Console keys."""
    if not key or key.startswith("sk-ant-api"):
        return False
    return key.startswith(("sk-ant-", "eyJ", "cc-"))


class CredentialPersistError(RuntimeError):
    """A rotated single-use credential could not be durably committed. The refresh POST already spent the old
    refresh token, so a swallowed write failure leaves a consumed pair on disk that later replays as invalid_grant."""

    def __init__(self, path: Any, cause: BaseException) -> None:
        super().__init__(f"failed to durably persist rotated Anthropic credentials to {path}: {cause}")
        self.path = path


def _load_json_if_exists(path: Path, what: str) -> Optional[Any]:
    """Parsed JSON from *path*, or None when missing/unreadable/corrupt (debug-logged)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Failed to read %s: %s", what, e)
        return None


def _atomic_write_private_json(path: Path, payload: Any) -> None:
    """Write *payload* via a 0o600 O_EXCL temp file + fsync + os.replace: the token is never briefly umask-readable
    (write_text + chmod had a TOCTOU window); the random suffix avoids collisions with concurrent writers and
    crashed leftovers. The parent dir's mode is left alone (~/.claude/ is owned by Claude Code)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _commit_private_json(path: Path, payload: Any, what: str) -> None:
    """Atomic private write; any failure becomes ``CredentialPersistError`` (the commit step of a rotation)."""
    try:
        _atomic_write_private_json(path, payload)
    except (OSError, ValueError) as e:
        logger.error("Failed to write refreshed %s to %s: %s", what, path, e)
        raise CredentialPersistError(path, e) from e


# ── Spent-rotation registry: fingerprints of secrets whose refresh POST succeeded but whose replacement never
# reached its store. Two scopes: process-local (OrderedDict) and a durable sidecar next to the shared singleton
# file so OTHER processes fail closed too. Non-reversible digests; never cleared.
_SPENT_ROTATION_LOCK = threading.Lock()
_SPENT_ROTATION_FINGERPRINTS: "OrderedDict[str, None]" = OrderedDict()
_SPENT_ROTATION_MAX_TRACKED = 64
_SPENT_ROTATION_SIDECAR_COMMENT = (
    "Non-secret one-way fingerprints of Anthropic OAuth credentials whose rotation was "
    "consumed server-side but never durably committed. Written by Hermes so sibling "
    "processes sharing this credential source fail closed instead of replaying a spent "
    "single-use refresh token."
)


def _spent_rotation_sidecar_path(source_path: Path) -> Path:
    return source_path.with_name(source_path.name + ".hermes-spent-rotations.json")


def spent_rotation_source_path(source: Any) -> Optional[Path]:
    """Map a pool-entry source to the shared singleton file it borrows from (or None)."""
    getter = _SINGLETON_SOURCE_PATHS.get(source) if isinstance(source, str) else None
    return getter() if getter else None


def _read_spent_rotation_sidecar(source_path: Optional[Path]) -> set:
    if source_path is None:
        return set()
    try:
        raw = json.loads(_spent_rotation_sidecar_path(source_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    fingerprints = raw.get("fingerprints") if isinstance(raw, dict) else None
    return {fp for fp in fingerprints if isinstance(fp, str) and fp} if isinstance(fingerprints, list) else set()


def _append_spent_rotation_sidecar(source_path: Path, fingerprints: list) -> None:
    """Merge fingerprints into the sidecar (atomic replace; caller holds the path lock). Fail-soft: a sidecar
    write failure must never mask the process-local verdict."""
    sidecar = _spent_rotation_sidecar_path(source_path)
    try:
        merged = _read_spent_rotation_sidecar(source_path)
        merged.update(fingerprints)
        payload = json.dumps({
            "version": 1,
            "comment": _SPENT_ROTATION_SIDECAR_COMMENT,
            "fingerprints": sorted(merged)[-_SPENT_ROTATION_MAX_TRACKED * 4 :],
        }, indent=2)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        tmp = sidecar.with_name(sidecar.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, sidecar)
    except Exception:
        logger.debug("Failed to persist spent-rotation fingerprints to %s", sidecar, exc_info=True)


def _fingerprint(secret: Any) -> Optional[str]:
    from agent.credential_persistence import fingerprint_secret_value
    value = str(secret or "").strip()
    return fingerprint_secret_value(value) if value else None


def mark_rotation_consumed_uncommitted(*secrets: Any, source_path: Optional[Path] = None) -> None:
    """Record the pre-rotation pair of a refresh whose replacement never committed; with ``source_path`` the
    verdict is also persisted to that singleton's sidecar."""
    recorded = [fp for fp in map(_fingerprint, secrets) if fp]
    with _SPENT_ROTATION_LOCK:
        for fingerprint in recorded:
            _SPENT_ROTATION_FINGERPRINTS.pop(fingerprint, None)
            _SPENT_ROTATION_FINGERPRINTS[fingerprint] = None
            while len(_SPENT_ROTATION_FINGERPRINTS) > _SPENT_ROTATION_MAX_TRACKED:
                _SPENT_ROTATION_FINGERPRINTS.popitem(last=False)
    if recorded and source_path is not None:
        _append_spent_rotation_sidecar(source_path, recorded)


def is_rotation_consumed_uncommitted(secret: Any, *, source_path: Optional[Path] = None) -> bool:
    """True when *secret* belongs to a rotation that was spent but not committed."""
    fingerprint = _fingerprint(secret)
    if not fingerprint:
        return False
    with _SPENT_ROTATION_LOCK:
        if fingerprint in _SPENT_ROTATION_FINGERPRINTS:
            return True
    return fingerprint in _read_spent_rotation_sidecar(source_path)


# ── Claude Code credentials (Keychain / ~/.claude/.credentials.json) ──
# Only singleton-backed pool sources have a cross-process authority boundary.
_SINGLETON_SOURCE_PATHS = {
    "claude_code": lambda: claude_code_credentials_path(), "hermes_pkce": lambda: _get_hermes_oauth_file()
}


def _claude_oauth_record(data: Any, source: str) -> Optional[Dict[str, Any]]:
    """Normalise a ``{"claudeAiOauth": {...}}`` payload into our credential dict."""
    oauth_data = data.get("claudeAiOauth")
    access_token = oauth_data.get("accessToken", "") if isinstance(oauth_data, dict) else ""
    if not access_token:
        return None
    return {
        "accessToken": access_token, "refreshToken": oauth_data.get("refreshToken", ""),
        "expiresAt": oauth_data.get("expiresAt", 0), "source": source,
    }


def _read_claude_code_credentials_from_keychain() -> Optional[Dict[str, Any]]:
    """Read the "Claude Code-credentials" macOS Keychain entry (Claude Code >=2.1.114)."""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("Keychain: security command not available or timed out")
        return None
    if result.returncode != 0:
        logger.debug("Keychain: no entry found for 'Claude Code-credentials'")
        return None
    raw = result.stdout.strip()
    try:
        return _claude_oauth_record(json.loads(raw), "macos_keychain") if raw else None
    except json.JSONDecodeError:
        logger.debug("Keychain: credentials payload is not valid JSON")
        return None


def claude_code_credentials_path() -> Path:
    """Claude Code's shared OAuth file; every profile reads/writes this same path."""
    return Path.home() / ".claude" / ".credentials.json"


def _read_claude_code_credentials_from_file() -> Optional[Dict[str, Any]]:
    data = _load_json_if_exists(claude_code_credentials_path(), "~/.claude/.credentials.json")
    return _claude_oauth_record(data, "claude_code_credentials_file") if data is not None else None


def read_claude_code_credentials() -> Optional[Dict[str, Any]]:
    """Read refreshable Claude Code OAuth credentials (Keychain and/or file). When both exist: prefer the only
    non-expired one (Claude Code 2.1.x refreshes one source but not the other), else the later ``expiresAt`` so a
    refresh uses the freshest refreshToken. ~/.claude.json primaryApiKey is deliberately excluded."""
    kc_creds = _read_claude_code_credentials_from_keychain()
    file_creds = _read_claude_code_credentials_from_file()
    if not (kc_creds and file_creds):
        return kc_creds or file_creds
    kc_valid, file_valid = is_claude_code_token_valid(kc_creds), is_claude_code_token_valid(file_creds)
    if kc_valid != file_valid:
        return kc_creds if kc_valid else file_creds
    return kc_creds if (kc_creds.get("expiresAt", 0) or 0) >= (file_creds.get("expiresAt", 0) or 0) else file_creds


def is_claude_code_token_valid(creds: Dict[str, Any]) -> bool:
    """Non-expired access token (60s buffer); no expiresAt means managed key → valid if present."""
    expires_at = creds.get("expiresAt", 0)
    return int(time.time() * 1000) < (expires_at - 60_000) if expires_at else bool(creds.get("accessToken"))


# ── OAuth token endpoint ──


def _post_oauth_token(
    data: bytes, *, content_type: str, timeout: int, what: str, user_agent: str = _OAUTH_TOKEN_USER_AGENT
) -> Dict[str, Any]:
    """POST to the token endpoints in order; raise the last error if all fail."""
    import urllib.request
    last_error = None
    for endpoint in _OAUTH_TOKEN_URLS:
        req = urllib.request.Request(
            endpoint, data=data, method="POST", headers={"Content-Type": content_type, "User-Agent": user_agent}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            last_error = exc
            logger.debug("Anthropic token %s failed at %s: %s", what, endpoint, exc)
    raise last_error or ValueError(f"Anthropic token {what} failed")


def _oauth_token_state(result: Dict[str, Any], *, fallback_refresh_token: str = "") -> Dict[str, Any]:
    """Token-endpoint JSON -> ``{access_token, refresh_token, expires_at_ms}`` (expires_in defaults to 3600s)."""
    return {
        "access_token": result.get("access_token", ""),
        "refresh_token": result.get("refresh_token", fallback_refresh_token),
        "expires_at_ms": int(time.time() * 1000) + (result.get("expires_in", 3600) * 1000),
    }


def refresh_anthropic_oauth_pure(refresh_token: str, *, use_json: bool = False) -> Dict[str, Any]:
    """Refresh an Anthropic OAuth token without mutating local credential files."""
    import urllib.parse
    if not refresh_token:
        raise ValueError("refresh_token is required")
    payload = {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": _OAUTH_CLIENT_ID}
    encode, content_type = ((json.dumps, "application/json") if use_json
                            else (urllib.parse.urlencode, "application/x-www-form-urlencoded"))
    result = _post_oauth_token(encode(payload).encode(), content_type=content_type, timeout=10, what="refresh",
                               user_agent=_OAUTH_TOKEN_USER_AGENT)
    if not result.get("access_token"):
        raise ValueError("Anthropic refresh response was missing access_token")
    return _oauth_token_state(result, fallback_refresh_token=refresh_token)


def _refresh_oauth_token(creds: Dict[str, Any]) -> Optional[str]:
    """Refresh an expired Claude Code OAuth token, returning the new access token. Refresh tokens are single-use and
    Claude Code refreshes on its own schedule, so we first re-read the live sources and adopt an already-rotated
    token instead of racing it into ``invalid_grant``. Read, decision, POST and write-back share the pool's
    path-keyed cross-process lock (else two profiles can spend one refresh token)."""
    try:
        from hermes_cli.auth import AUTH_LOCK_TIMEOUT_SECONDS, _auth_store_lock, env_float
        refresh_timeout_seconds = env_float("HERMES_ANTHROPIC_REFRESH_TIMEOUT_SECONDS", 20)
        lock_timeout_seconds = max(float(AUTH_LOCK_TIMEOUT_SECONDS), float(refresh_timeout_seconds) + 5.0)
        cred_path = claude_code_credentials_path()
        with _auth_store_lock(timeout_seconds=lock_timeout_seconds, target_path=cred_path):
            # Adopt only a DIFFERENT token with a real future expiry (0/absent expiresAt = managed key/unknown).
            current = read_claude_code_credentials() or {}
            current_token = current.get("accessToken", "")
            if (current_token and current_token != creds.get("accessToken", "")
                    and (current.get("expiresAt", 0) or 0) > 0 and is_claude_code_token_valid(current)):
                logger.debug("Adopted Claude Code's already-refreshed OAuth token")
                return current_token

            refresh_token = current.get("refreshToken", "") or creds.get("refreshToken", "")
            if not refresh_token:
                logger.debug("No refresh token available — cannot refresh")
                return None
            # Another process may have spent this token and lost the commit; its sidecar verdict is authoritative.
            if is_rotation_consumed_uncommitted(refresh_token, source_path=cred_path):
                logger.debug("Refresh token was already consumed by an uncommitted rotation "
                             "- refusing to replay it; re-run 'claude setup-token'")
                return None
            try:
                refreshed = refresh_anthropic_oauth_pure(refresh_token, use_json=False)
            except Exception as e:
                logger.debug("Failed to refresh Claude Code token: %s", e)
                return None
            # The POST spent ``refresh_token``; this write is the commit step. On failure, fail closed and
            # mark the pre-rotation pair as spent.
            try:
                _write_claude_code_credentials(refreshed["access_token"], refreshed["refresh_token"], refreshed["expires_at_ms"])
            except Exception as e:
                logger.error(
                    "Anthropic OAuth refresh rotated the single-use token but could not "
                    "commit it to %s (%s) — treating the refresh as failed; "
                    "re-run 'claude setup-token' to reauthenticate",
                    cred_path, e,
                )
                mark_rotation_consumed_uncommitted(
                    refresh_token, creds.get("accessToken", ""), current.get("accessToken", ""),
                    current.get("refreshToken", ""), source_path=cred_path,
                )
                return None
            logger.debug("Successfully refreshed Claude Code OAuth token")
            return refreshed["access_token"]
    except Exception as e:
        # Lock/read failures keep the resolver's fail-soft contract.
        logger.debug("Failed to acquire Claude Code refresh lock: %s", e)
        return None


def _write_claude_code_credentials(
    access_token: str, refresh_token: str, expires_at_ms: int, *, scopes: Optional[list] = None
) -> None:
    """Commit refreshed credentials to ~/.claude/.credentials.json; ``CredentialPersistError`` on any failure (a
    corrupt existing file included). *scopes* (or the previously stored scopes) are persisted because Claude Code
    >=2.1.81 gates on ``"user:inference"`` being present."""
    cred_path = claude_code_credentials_path()
    try:
        existing = json.loads(cred_path.read_text(encoding="utf-8")) if cred_path.exists() else {}
    except (OSError, ValueError) as e:
        logger.error("Failed to write refreshed credentials to %s: %s", cred_path, e)
        raise CredentialPersistError(cred_path, e) from e
    oauth_data: Dict[str, Any] = {"accessToken": access_token, "refreshToken": refresh_token, "expiresAt": expires_at_ms}
    if scopes is not None:
        oauth_data["scopes"] = scopes
    elif "claudeAiOauth" in existing and "scopes" in existing["claudeAiOauth"]:
        oauth_data["scopes"] = existing["claudeAiOauth"]["scopes"]
    existing["claudeAiOauth"] = oauth_data
    _commit_private_json(cred_path, existing, "credentials")


# ── Resolution ──


def _resolve_claude_code_token_from_credentials(creds: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve a token from Claude Code credential files, refreshing if needed."""
    creds = creds or read_claude_code_credentials()
    if not creds:
        return None
    if is_rotation_consumed_uncommitted(creds.get("accessToken", ""), source_path=claude_code_credentials_path()):
        # The file still holds the spent pre-rotation copy of a failed commit.
        logger.debug("Claude Code credentials hold a rotated-but-uncommitted token - refusing")
        return None
    if is_claude_code_token_valid(creds):
        logger.debug("Using Claude Code credentials (auto-detected)")
        return creds["accessToken"]
    logger.debug("Claude Code credentials expired — attempting refresh")
    refreshed = _refresh_oauth_token(creds)
    if not refreshed:
        logger.debug("Token refresh failed — re-run 'claude setup-token' to reauthenticate")
    return refreshed or None


def _prefer_refreshable_claude_code_token(env_token: str, creds: Optional[Dict[str, Any]]) -> Optional[str]:
    """Prefer refreshable Claude Code creds over a static env OAuth token: Hermes historically persisted setup tokens
    into ANTHROPIC_TOKEN, and that static token would otherwise win before the refreshable file is inspected."""
    if not (env_token and _is_oauth_token(env_token) and isinstance(creds, dict) and creds.get("refreshToken")):
        return None
    resolved = _resolve_claude_code_token_from_credentials(creds)
    if resolved and resolved != env_token:
        logger.debug("Preferring Claude Code credential file over static env OAuth token so refresh can proceed")
        return resolved
    return None


def _resolve_anthropic_pool_token() -> Optional[str]:
    """First available Anthropic OAuth token from credential_pool, read-only: enumerates with ``clear_expired=False,
    refresh=False`` (never ``select()``) so diagnostic call sites (account_usage, ``hermes models``) never mutate
    auth.json or hit the network; refresh-on-expiry belongs to the API call path's pool recovery."""
    try:
        from agent.credential_pool import AUTH_TYPE_OAUTH, load_pool
        entries, _pending = load_pool("anthropic")._available_entries(clear_expired=False, refresh=False)
    except Exception:
        logger.debug("Failed to read Anthropic credential_pool", exc_info=True)
        return None
    for entry in entries:
        # access_token may be an explicit null on a persisted entry; None.strip() would crash the resolver.
        token = (getattr(entry, "access_token", None) or "").strip()
        if getattr(entry, "auth_type", None) != AUTH_TYPE_OAUTH or not token:
            continue
        # load_pool() re-seeds rows from the singleton files, so a spent-but-uncommitted rotation
        # (possibly from another process) looks healthy here.
        entry_source_path = spent_rotation_source_path(getattr(entry, "source", None))
        if any(
            is_rotation_consumed_uncommitted(secret, source_path=entry_source_path)
            for secret in (token, getattr(entry, "refresh_token", None))
        ):
            logger.debug("Skipping Anthropic pool entry %s: rotated-but-uncommitted credential", getattr(entry, "id", "?"))
            continue
        return token
    return None


def resolve_anthropic_token() -> Optional[str]:
    """Resolve an Anthropic token from all sources in priority order (see module docstring)."""
    _read_creds = functools.cache(read_claude_code_credentials)  # read the file at most once per resolve
    token = _first_env("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
    if token:
        return _prefer_refreshable_claude_code_token(token, _read_creds()) or token
    api_key = _first_env("ANTHROPIC_API_KEY")  # an explicit API key must not be shadowed by discovered OAuth creds
    if api_key:
        return api_key
    return _resolve_claude_code_token_from_credentials(_read_creds()) or _resolve_anthropic_pool_token()


def run_oauth_setup_token() -> Optional[str]:
    """Run 'claude setup-token' interactively; the resulting token or None. FileNotFoundError if no 'claude' CLI."""
    import shutil
    claude_path = shutil.which("claude")
    if not claude_path:
        raise FileNotFoundError("The 'claude' CLI is not installed. Install it with: npm install -g @anthropic-ai/claude-code")
    # Interactive: stdio inherited so the user can complete the OAuth prompt.  noqa: subprocess-stdin
    try:
        subprocess.run([claude_path, "setup-token"])
    except (KeyboardInterrupt, EOFError):
        return None
    creds = read_claude_code_credentials()
    if creds and is_claude_code_token_valid(creds):
        return creds["accessToken"]
    return _first_env("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_TOKEN") or None


# ── Hermes-native PKCE OAuth flow (~/.hermes/.anthropic_oauth.json); mirrors Claude Code / pi-ai / OpenCode ──


def _get_hermes_oauth_file() -> Path:
    return get_hermes_home() / ".anthropic_oauth.json"


def _root_hermes_oauth_file() -> Optional[Path]:
    """Global-root ``.anthropic_oauth.json`` inside a named profile (None in classic mode); used to commit a
    rotation of a grant the profile borrowed via the pool's root fallback."""
    try:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root()
        return None if root.resolve(strict=False) == get_hermes_home().resolve(strict=False) else root / ".anthropic_oauth.json"
    except Exception:
        return None


def _generate_pkce() -> tuple:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def run_hermes_oauth_login_pure() -> Optional[Dict[str, Any]]:
    """Run Hermes-native OAuth PKCE flow and return credential state."""
    import webbrowser
    from urllib.parse import urlencode
    verifier, challenge = _generate_pkce()
    oauth_state = secrets.token_urlsafe(32)
    params = {
        "code": "true", "client_id": _OAUTH_CLIENT_ID, "response_type": "code", "redirect_uri": _OAUTH_REDIRECT_URI,
        "scope": _OAUTH_SCOPES, "code_challenge": challenge, "code_challenge_method": "S256", "state": oauth_state,
    }
    auth_url = f"https://claude.ai/oauth/authorize?{urlencode(params)}"
    print("\n".join([
        "", "Authorize Hermes with your Claude Pro/Max subscription.", "",
        "╭─ Claude Pro/Max Authorization ────────────────────╮",
        "│                                                   │",
        "│  Open this link in your browser:                  │",
        "╰───────────────────────────────────────────────────╯",
        "", f"  {auth_url}", "",
    ]))
    try:
        from hermes_cli.auth import _can_open_graphical_browser as _can_open_gui
    except Exception:
        _can_open_gui = lambda: True  # noqa: E731 — degrade to prior behavior
    if _can_open_gui():
        with contextlib.suppress(Exception):
            webbrowser.open(auth_url)
            print("  (Browser opened automatically)")
    print("\nAfter authorizing, you'll see a code. Paste it below.\n")
    try:
        auth_code = input("Authorization code: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if not auth_code:
        print("No code entered.")
        return None
    splits = auth_code.split("#")
    code, received_state = splits[0], (splits[1] if len(splits) > 1 else "")
    if received_state != oauth_state:  # CSRF guard (RFC 6749 §10.12)
        logger.warning("OAuth state mismatch — possible CSRF, aborting")
        return None
    try:
        exchange_data = json.dumps({
            "grant_type": "authorization_code", "client_id": _OAUTH_CLIENT_ID, "code": code, "state": received_state,
            "redirect_uri": _OAUTH_REDIRECT_URI, "code_verifier": verifier,
        }).encode()
        result = _post_oauth_token(exchange_data, content_type="application/json", timeout=15, what="exchange")
    except Exception as e:
        print(f"Token exchange failed: {e}")
        return None
    if not result.get("access_token"):
        print("No access token in response.")
        return None
    return _oauth_token_state(result)


def read_hermes_oauth_credentials() -> Optional[Dict[str, Any]]:
    """Read Hermes-managed OAuth credentials from ~/.hermes/.anthropic_oauth.json."""
    data = _load_json_if_exists(_get_hermes_oauth_file(), "Hermes OAuth credentials")
    return data if data is not None and data.get("accessToken") else None


def _write_hermes_oauth_credentials(
    access_token: str, refresh_token: Optional[str], expires_at_ms: Optional[int], *, target: Optional[Path] = None
) -> None:
    """Commit refreshed hermes_pkce tokens to ~/.hermes/.anthropic_oauth.json (``CredentialPersistError`` on failure).
    ``target`` lets a named profile commit a grant it BORROWED from the global root back to the ROOT singleton
    instead of forking a copy under its own HERMES_HOME; without this write-through the next ``load_pool()``
    re-seeds the stale (consumed) pair from the file over the rotated pool entry."""
    _commit_private_json(
        target if target is not None else _get_hermes_oauth_file(),
        {"accessToken": access_token, "refreshToken": refresh_token, "expiresAt": expires_at_ms},
        "Hermes OAuth credentials",
    )
