"""Anthropic credential sources, OAuth flows, and token resolution.

``resolve_anthropic_token()`` order: ``ANTHROPIC_TOKEN`` / ``CLAUDE_CODE_OAUTH_TOKEN``,
``ANTHROPIC_API_KEY``, Hermes-owned OAuth grants in the ``auth.json`` credential
pool, then ``~/.claude/.credentials.json`` / macOS Keychain as a borrowed fallback.
``~/.hermes/.anthropic_oauth.json`` (Hermes PKCE) and
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
import re
import secrets
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write
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
# Claude Code's macOS Keychain entry (generic password). Hermes reads it
# (_read_claude_code_credentials_from_keychain) and, since #98334, mirrors the
# refresh write into it so the two stores stop diverging on a single-use rotation.
_CLAUDE_CODE_KEYCHAIN_SERVICE = "Claude Code-credentials"


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
    """0600-from-creation temp file + fsync + atomic replace (the token is never briefly umask-readable).
    The parent dir's mode is left alone (~/.claude/ is owned by Claude Code)."""
    atomic_json_write(path, payload, mode=0o600)


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
# Fingerprints of Claude Code refresh tokens the endpoint rejected terminally: the WARNING fires once per token
# per process and later attempts skip the POST (a re-login rotates the token, so a new one is tried normally).
_DEAD_REFRESH_TOKEN_FINGERPRINTS: set = set()
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


_KEYCHAIN_ATTR = r'(?:0x(?P<hex>[0-9A-Fa-f]+)\b.*|"(?P<text>.*)")'


def _decode_keychain_attr(match: Optional["re.Match[str]"]) -> str:
    """``security`` prints an attribute as ``"text"`` when it is plain printable ASCII and as
    ``0x<HEX>  "<octal-escaped echo>"`` otherwise; the quoted form is NOT escaped (an embedded
    ``"`` appears raw), so the text group must run to the last quote on the line."""
    if match is None:
        return ""
    if match.group("hex"):
        try:
            return bytes.fromhex(match.group("hex")).decode("utf-8")
        except ValueError:
            return ""
    return match.group("text") or ""


def _find_claude_code_keychain_item() -> Optional[tuple[str, Dict[str, Any]]]:
    """``(account, payload)`` of the ``Claude Code-credentials`` login Keychain item, or None.

    One ``find-generic-password -g`` call: attributes on stdout, ``password: …`` on stderr. The
    account matters because ``add-generic-password -U`` matches on account AND service — writing
    under another account would create a second item instead of updating the one Claude Code reads.
    """
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", _CLAUDE_CODE_KEYCHAIN_SERVICE, "-g"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    account = _decode_keychain_attr(re.search(r'^\s*"acct"<blob>=' + _KEYCHAIN_ATTR + r"\s*$", result.stdout, re.M))
    raw = _decode_keychain_attr(re.search(r"^password: " + _KEYCHAIN_ATTR + r"\s*$", result.stderr, re.M))
    if not account or not raw:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    return (account, payload) if isinstance(payload, dict) else None


def _read_claude_code_keychain_payload() -> Optional[Dict[str, Any]]:
    """Raw ``{"claudeAiOauth": {...}, ...}`` payload from the macOS Keychain, or None.

    Returns the full entry (not the normalised credential record) so a refresh
    write can merge the rotated token triple over the existing metadata
    (``subscriptionType`` / ``rateLimitTier`` / ``scopes``) instead of clobbering it.
    """
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", _CLAUDE_CODE_KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("Keychain: security command not available or timed out")
        return None
    if result.returncode != 0:
        logger.debug("Keychain: no entry found for %r", _CLAUDE_CODE_KEYCHAIN_SERVICE)
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Keychain: credentials payload is not valid JSON")
        return None
    return payload if isinstance(payload, dict) else None


def _keychain_mirror_command(account: str, payload: Dict[str, Any]) -> tuple[list[str], str]:
    """``(argv, stdin)`` that updates the Claude Code Keychain item with ``payload``.

    The command line goes to ``security -i`` on stdin, with the secret hex-encoded (``-X``):
    a bare ``-w`` prompts twice on /dev/tty when a terminal exists (hangs the CLI) and, with
    no terminal, reads only the first line and stores an EMPTY password when the confirmation
    read hits EOF — either way the live token must never sit on argv.
    """
    def quoted(value: str) -> str:  # the ``security -i`` tokenizer: double quotes, backslash escapes
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8").hex()
    line = f"add-generic-password -U -a {quoted(account)} -s {quoted(_CLAUDE_CODE_KEYCHAIN_SERVICE)} -X {encoded}\n"
    return ["security", "-i"], line


def _read_claude_code_credentials_from_keychain() -> Optional[Dict[str, Any]]:
    """Read the "Claude Code-credentials" macOS Keychain entry (Claude Code >=2.1.114)."""
    payload = _read_claude_code_keychain_payload()
    return _claude_oauth_record(payload, "macos_keychain") if payload else None


def claude_code_credentials_path() -> Path:
    """Claude Code's shared OAuth file; every profile reads/writes this same path. Honours ``CLAUDE_CONFIG_DIR``
    like the Claude CLI itself (blank = unset, as in ``hermes_cli.foreign_sessions``). The supported opt-out of
    borrowing the login is ``auth.adopt_external_logins: false`` in config.yaml."""
    override = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    root = Path(override).expanduser() if override else Path.home() / ".claude"
    return root / ".credentials.json"


def _read_claude_code_credentials_from_file() -> Optional[Dict[str, Any]]:
    data = _load_json_if_exists(claude_code_credentials_path(), "~/.claude/.credentials.json")
    return _claude_oauth_record(data, "claude_code_credentials_file") if data is not None else None


def read_claude_code_credentials() -> Optional[Dict[str, Any]]:
    """Read refreshable Claude Code OAuth credentials (Keychain and/or file). When both exist: prefer the only
    non-expired one (Claude Code 2.1.x refreshes one source but not the other), else the later ``expiresAt`` so a
    refresh uses the freshest refreshToken. ~/.claude.json primaryApiKey is deliberately excluded.

    This is the only reader of the borrowed login, so ``auth.adopt_external_logins: false`` is enforced here:
    every resolver, pool seed/sync and 401 refresher then sees "no Claude Code login" and never touches the file."""
    from agent.credential_sources import adopt_external_logins_enabled
    if not adopt_external_logins_enabled():
        return None
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


# OAuth ``error`` codes (RFC 6749 §5.2 + the provider's reuse detection) after which replaying the
# same refresh token can never succeed; only a fresh login recovers.
_OAUTH_GRANT_DEAD_CODES = frozenset({"invalid_grant", "invalid_token", "refresh_token_reused"})


class AnthropicOAuthError(ValueError):
    """Token endpoint rejected the request. ``code`` is the OAuth ``error`` field of the response body."""

    def __init__(self, status: int, code: str, description: str, *, what: str) -> None:
        self.status = status
        self.code = code
        detail = f" ({description})" if description else ""
        super().__init__(f"Anthropic token {what} failed: HTTP {status} {code or 'error'}{detail}")

    @property
    def relogin_required(self) -> bool:
        return self.status in (400, 401) and self.code in _OAUTH_GRANT_DEAD_CODES


def is_terminal_anthropic_refresh_error(exc: BaseException) -> bool:
    """True when retrying the same Anthropic refresh token cannot succeed (dead grant)."""
    return isinstance(exc, AnthropicOAuthError) and exc.relogin_required


def _oauth_http_error(exc: Any, *, what: str) -> AnthropicOAuthError:
    """``urllib.error.HTTPError`` -> structured error carrying the body's OAuth ``error`` code."""
    code, description = "", ""
    try:
        payload = json.loads(exc.read().decode() or "{}")
        code = str(payload.get("error") or "")
        description = str(payload.get("error_description") or "")
    except Exception:
        pass
    return AnthropicOAuthError(int(exc.code), code, description, what=what)


def _post_oauth_token(
    data: bytes, *, content_type: str, timeout: int, what: str, user_agent: str = _OAUTH_TOKEN_USER_AGENT
) -> Dict[str, Any]:
    """POST to the token endpoints in order; raise the last error if all fail."""
    import urllib.error
    import urllib.request
    last_error = None
    for endpoint in _OAUTH_TOKEN_URLS:
        req = urllib.request.Request(
            endpoint, data=data, method="POST", headers={"Content-Type": content_type, "User-Agent": user_agent}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            last_error = _oauth_http_error(exc, what=what)
            logger.debug("Anthropic token %s failed at %s: %s", what, endpoint, last_error)
            if last_error.relogin_required:
                break  # a dead grant is dead at every endpoint; do not replay it
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
                             "- refusing to replay it; run 'hermes auth add anthropic'")
                return None
            fingerprint = hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()[:32]
            if fingerprint in _DEAD_REFRESH_TOKEN_FINGERPRINTS:
                logger.debug("Claude Code refresh token was already rejected as terminally invalid - not replaying it")
                return None
            try:
                refreshed = refresh_anthropic_oauth_pure(refresh_token, use_json=False)
            except Exception as e:
                if is_terminal_anthropic_refresh_error(e):
                    _DEAD_REFRESH_TOKEN_FINGERPRINTS.add(fingerprint)
                    logger.warning(
                        "Claude Code OAuth refresh token is terminally invalid (%s); Hermes cannot use this "
                        "login. Run 'hermes auth add anthropic' to give Hermes its own login.", e)
                else:
                    logger.debug("Failed to refresh Claude Code token: %s", e)
                return None
            # The POST spent ``refresh_token``; this write is the commit step. On failure, fail closed and
            # mark the pre-rotation pair as spent.
            try:
                _write_claude_code_credentials(
                    refreshed["access_token"], refreshed["refresh_token"], refreshed["expires_at_ms"],
                    spent_refresh_token=refresh_token,
                )
            except Exception as e:
                logger.error(
                    "Anthropic OAuth refresh rotated the single-use token but could not "
                    "commit it to %s (%s) — treating the refresh as failed; "
                    "run 'hermes auth add anthropic' to give Hermes its own login",
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
    access_token: str, refresh_token: str, expires_at_ms: int, *, scopes: Optional[list] = None,
    spent_refresh_token: str = "",
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
    _mirror_claude_code_credentials_to_keychain(
        access_token, refresh_token, expires_at_ms, spent_refresh_token=spent_refresh_token)


def _merge_keychain_credential_payload(
    existing_payload: Dict[str, Any], access_token: str, refresh_token: str, expires_at_ms: int
) -> Dict[str, Any]:
    """Rotate the ``claudeAiOauth`` token triple over the existing Keychain payload,
    preserving its metadata (``subscriptionType`` / ``rateLimitTier`` / ``scopes``).

    Pure and host-agnostic so the merge semantics are unit-testable without a Keychain.
    """
    merged = dict(existing_payload)
    oauth = dict(existing_payload.get("claudeAiOauth") or {})
    oauth.update({"accessToken": access_token, "refreshToken": refresh_token, "expiresAt": expires_at_ms})
    merged["claudeAiOauth"] = oauth
    return merged


def _mirror_claude_code_credentials_to_keychain(
    access_token: str, refresh_token: str, expires_at_ms: int, *, spent_refresh_token: str
) -> None:
    """After a Hermes refresh, write the rotated pair into the Claude Code Keychain item too (#98334).

    Claude Code on macOS reads the login Keychain first. Refresh tokens are single-use, so a refresh
    that only updates the file leaves the Keychain holding a spent token and Claude Code logs itself
    out. Only the item that held the pair we just spent is updated — a different pair there means a
    different login (``CLAUDE_CONFIG_DIR``) or a rotation Claude Code already made, and clobbering it
    would be the bug in the other direction. Best-effort: never raises, never creates an item.
    """
    if platform.system() != "Darwin":
        return
    try:
        item = _find_claude_code_keychain_item()
        if item is None:
            return
        account, existing = item
        oauth = existing.get("claudeAiOauth")
        if not isinstance(oauth, dict) or oauth.get("refreshToken") != spent_refresh_token:
            logger.debug("Keychain mirror skipped: item does not hold the pair that was just rotated")
            return
        argv, line = _keychain_mirror_command(
            account, _merge_keychain_credential_payload(existing, access_token, refresh_token, expires_at_ms))
        result = subprocess.run(
            argv, input=line, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
    except Exception as e:  # the file commit already succeeded; a Keychain hiccup must not fail the rotation
        logger.debug("Keychain mirror skipped (%s)", e)
        return
    if result.returncode != 0:
        logger.debug("Keychain mirror failed (rc=%s): %s", result.returncode, (result.stderr or "").strip()[:200])


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
        logger.debug("Token refresh failed — run 'hermes auth add anthropic' to give Hermes its own login")
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


def _resolve_anthropic_pool_token(*, skip_borrowed: bool = False) -> Optional[str]:
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
        if skip_borrowed and entry.source == "claude_code":
            continue
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


def _available_anthropic_token(token: Optional[str], model: Optional[str]) -> Optional[str]:
    """Return *token* unless the pool holds an active cooldown for it on *model*.

    Only model-aware callers (the API-call paths) are gated: diagnostics that
    resolve a token without a model (usage display, model discovery) keep it.
    """
    if not token or not model:
        return token or None
    try:
        from agent.credential_pool import load_pool
        if load_pool("anthropic").token_is_blocked(token, model=model):
            return None
    except Exception:
        # Credential discovery must remain available when the pool store is
        # unavailable or malformed.
        logger.debug("Failed to check Anthropic model cooldown", exc_info=True)
    return token


def resolve_anthropic_token(*, model: Optional[str] = None) -> Optional[str]:
    """Resolve an Anthropic token from all sources in priority order (see module docstring).

    With *model*, a token the credential pool has benched for that model resolves to ``None``
    instead of being handed straight back to the caller that just saw it rate-limited."""
    _read_creds = functools.cache(read_claude_code_credentials)  # read the file at most once per resolve
    token = _first_env("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
    if token:
        return _available_anthropic_token(
            _prefer_refreshable_claude_code_token(token, _read_creds()) or token, model,
        )
    api_key = _first_env("ANTHROPIC_API_KEY")  # an explicit API key must not be shadowed by discovered OAuth creds
    if api_key:
        return _available_anthropic_token(api_key, model)
    # The pool's claude_code row mirrors the same externally owned refresh grant.
    return _available_anthropic_token(
        _resolve_anthropic_pool_token(skip_borrowed=True) or _resolve_claude_code_token_from_credentials(_read_creds()),
        model,
    )


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


def _generate_pkce() -> tuple:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def run_hermes_oauth_login_pure(open_browser: bool = True) -> Optional[Dict[str, Any]]:
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
    if open_browser and _can_open_gui():
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
    access_token: str, refresh_token: Optional[str], expires_at_ms: Optional[int],
) -> None:
    """Commit refreshed hermes_pkce tokens to ``<HERMES_HOME>/.anthropic_oauth.json`` (``CredentialPersistError``
    on failure); without it the next ``load_pool()`` re-seeds the stale (consumed) pair from the file over the
    rotated pool entry."""
    _commit_private_json(
        _get_hermes_oauth_file(),
        {"accessToken": access_token, "refreshToken": refresh_token, "expiresAt": expires_at_ms},
        "Hermes OAuth credentials",
    )
