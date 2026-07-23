"""Machine-wide Anthropic OAuth shared credential pool.

Canonical store: root ``auth.json`` under ``shared_credential_pools.anthropic``.
Scope marker: ``<root>/shared/anthropic_pool_scope.json``.

See the universal Anthropic OAuth pool design for the full contract.
This module is the single authoritative surface for shared-scope resolution,
mutation, and refresh. Callers must not reimplement bypasses around it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import hermes_cli.auth as auth_mod
from hermes_cli.auth import AuthError, _auth_store_lock
from hermes_constants import get_default_hermes_root, secure_parent_dir

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MARKER_VERSION = 1
PROVIDER = "anthropic"
SOURCE_HERMES_PKCE = "manual:hermes_pkce"
AUTH_TYPE_OAUTH = "oauth"
STRATEGY_FILL_FIRST = "fill_first"
GRANT_FINGERPRINT_PREFIX = "hmac-sha256:"
GRANT_HMAC_CONTEXT = b"hermes-anthropic-grant-v1\0"

ENDPOINT_PLATFORM = "platform_claude_v1"
ENDPOINT_CONSOLE = "console_anthropic_v1"
ENDPOINT_URLS = {
    ENDPOINT_PLATFORM: "https://platform.claude.com/v1/oauth/token",
    ENDPOINT_CONSOLE: "https://console.anthropic.com/v1/oauth/token",
}
URL_TO_ENDPOINT = {v: k for k, v in ENDPOINT_URLS.items()}

STATUS_OK = "ok"
STATUS_EXHAUSTED = "exhausted"
STATUS_DEAD = "dead"

REASON_RATE_LIMIT = "rate_limit"
REASON_BILLING = "billing"
REASON_AUTH_TERMINAL = "auth_terminal"
REASON_REFRESH_UNKNOWN = "refresh_outcome_unknown"
REASON_OTHER = "other_sanitized"

ATTEMPT_INFLIGHT = "inflight"
ATTEMPT_UNKNOWN = "unknown"
ATTEMPT_TERMINAL = "terminal"

REQUIRED_POOL_KEYS = (
    "schema_version",
    "revision",
    "strategy",
    "account_distinctness_attested",
    "account_distinctness_attested_at",
    "entries",
)
REQUIRED_ROW_KEYS = (
    "id",
    "provider",
    "auth_type",
    "source",
    "label",
    "grant_fingerprint",
    "token_generation",
    "oauth_token_endpoint",
    "access_token",
    "refresh_token",
    "expires_at_ms",
    "priority",
    "request_count",
    "last_status",
    "last_status_at",
    "last_error_code",
    "last_error_reason",
    "last_error_message",
    "last_error_reset_at",
    "last_refresh",
    "refresh_attempt",
)

OFFICIAL_ANTHROPIC_HOSTS = frozenset({"api.anthropic.com"})
OFFICIAL_PURPOSES = frozenset({"inference", "account_usage", "model_discovery"})

# Process-local epoch snapshot taken at first shared observation.
_startup_epoch: Optional[str] = None
_startup_epoch_lock = threading.Lock()
_inprocess_pool_lock = threading.RLock()
_active_leases: Dict[Tuple[str, int], int] = {}
_cached_clients_generation: int = 0


class SharedMutationCapability:
    """Unforgeable capability token — only constructed inside this module."""

    __slots__ = ("_token",)

    def __init__(self) -> None:
        self._token = object()


_SHARED_MUTATION_CAP = SharedMutationCapability()


def get_shared_mutation_capability() -> SharedMutationCapability:
    """Return the internal capability for explicit ``hermes auth ... --shared`` paths."""
    return _SHARED_MUTATION_CAP


def _require_capability(cap: Optional[SharedMutationCapability]) -> None:
    if cap is None or getattr(cap, "_token", None) is not _SHARED_MUTATION_CAP._token:
        raise AuthError(
            "Shared Anthropic pool mutation requires explicit CLI --shared capability. "
            "Use: hermes auth add/remove/reset/logout anthropic --shared "
            "(or hermes auth scope / backup / restore).",
            provider=PROVIDER,
            code="shared_mutation_forbidden",
        )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def root_hermes_path() -> Path:
    return get_default_hermes_root()


def root_auth_path() -> Path:
    return root_hermes_path() / "auth.json"


def shared_dir() -> Path:
    return root_hermes_path() / "shared"


def scope_marker_path() -> Path:
    return shared_dir() / "anthropic_pool_scope.json"


def grant_salt_path() -> Path:
    return shared_dir() / "anthropic_grant_salt"


def recovery_dir() -> Path:
    return shared_dir() / "recovery"


# ---------------------------------------------------------------------------
# Filesystem safety
# ---------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _path_is_safe_regular_file(path: Path, *, must_exist: bool) -> None:
    """Fail closed on symlink / wrong owner / group-world writable paths."""
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise AuthError(
                f"Refusing symlinked path: {path}",
                provider=PROVIDER,
                code="shared_path_unsafe",
            )
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode):
            raise AuthError(
                f"Path is not a regular file: {path}",
                provider=PROVIDER,
                code="shared_path_unsafe",
            )
        if st.st_nlink != 1:
            raise AuthError(
                f"Refusing hard-linked path: {path}",
                provider=PROVIDER,
                code="shared_path_unsafe",
            )
        if hasattr(os, "getuid"):
            if st.st_uid != os.getuid():
                raise AuthError(
                    f"Path not owned by current user: {path}",
                    provider=PROVIDER,
                    code="shared_path_unsafe",
                )
        mode = stat.S_IMODE(st.st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise AuthError(
                f"Path has group/world permissions: {path}",
                provider=PROVIDER,
                code="shared_path_unsafe",
            )
    elif must_exist:
        raise AuthError(
            f"Required path missing: {path}",
            provider=PROVIDER,
            code="shared_path_missing",
        )


def _ensure_owner_only_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    if path.is_symlink():
        raise AuthError(
            f"Refusing symlinked directory: {path}",
            provider=PROVIDER,
            code="shared_path_unsafe",
        )
    st = path.lstat()
    if not stat.S_ISDIR(st.st_mode):
        raise AuthError(
            f"Expected directory: {path}",
            provider=PROVIDER,
            code="shared_path_unsafe",
        )
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise AuthError(
            f"Directory not owned by current user: {path}",
            provider=PROVIDER,
            code="shared_path_unsafe",
        )
    mode = stat.S_IMODE(st.st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise AuthError(
            f"Directory has group/world permissions: {path}",
            provider=PROVIDER,
            code="shared_path_unsafe",
        )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Strict atomic write: temp + fsync + os.replace + dir fsync. No copy fallback."""
    if path.exists() or path.is_symlink():
        _path_is_safe_regular_file(path, must_exist=True)
    _ensure_owner_only_dir(path.parent)
    secure_parent_dir(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(tmp), flags, stat.S_IRUSR | stat.S_IWUSR)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            raise
        # Refuse replace onto a symlink target path.
        if path.is_symlink():
            raise AuthError(
                f"Refusing to replace symlinked path: {path}",
                provider=PROVIDER,
                code="shared_path_unsafe",
            )
        os.replace(str(tmp), str(path))
        _fsync_dir(path.parent)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, data)


# ---------------------------------------------------------------------------
# Scope marker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeState:
    mode: str  # "profile" | "shared"
    epoch: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


def read_scope_state() -> ScopeState:
    """Read and validate the scope marker. Fail closed on malformed/unsafe marker."""
    path = scope_marker_path()
    if not path.exists() and not path.is_symlink():
        return ScopeState(mode="profile")
    _path_is_safe_regular_file(path, must_exist=True)
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(path), flags)
        try:
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            raise
    except AuthError:
        raise
    except Exception as exc:
        raise AuthError(
            f"Malformed Anthropic shared scope marker: {exc}",
            provider=PROVIDER,
            code="shared_scope_corrupt",
        ) from exc
    if not isinstance(raw, dict):
        raise AuthError(
            "Malformed Anthropic shared scope marker: not an object",
            provider=PROVIDER,
            code="shared_scope_corrupt",
        )
    if raw.get("version") != MARKER_VERSION:
        raise AuthError(
            f"Unsupported scope marker version: {raw.get('version')!r}",
            provider=PROVIDER,
            code="shared_scope_corrupt",
        )
    if raw.get("scope") != "shared":
        raise AuthError(
            f"Invalid scope marker value: {raw.get('scope')!r}",
            provider=PROVIDER,
            code="shared_scope_corrupt",
        )
    epoch = raw.get("epoch")
    if not isinstance(epoch, str) or not epoch.strip():
        raise AuthError(
            "Scope marker missing epoch",
            provider=PROVIDER,
            code="shared_scope_corrupt",
        )
    return ScopeState(mode="shared", epoch=epoch.strip(), raw=raw)


def is_shared_scope_active() -> bool:
    return read_scope_state().mode == "shared"


def observe_startup_epoch() -> Optional[str]:
    """Snapshot the shared epoch for this process (idempotent)."""
    global _startup_epoch
    with _startup_epoch_lock:
        if _startup_epoch is not None:
            return _startup_epoch
        state = read_scope_state()
        if state.mode == "shared":
            _startup_epoch = state.epoch
        else:
            _startup_epoch = ""
        return _startup_epoch


def reset_startup_epoch_for_tests() -> None:
    global _startup_epoch
    with _startup_epoch_lock:
        _startup_epoch = None


def assert_epoch_unchanged() -> None:
    """Fail if scope changed since process startup (no hot-switch)."""
    observed = observe_startup_epoch()
    current = read_scope_state()
    if observed == "":
        # Started in profile mode.
        if current.mode == "shared":
            raise AuthError(
                "Anthropic shared scope changed; restart Hermes",
                provider=PROVIDER,
                code="shared_scope_changed",
            )
        return
    if current.mode != "shared" or current.epoch != observed:
        raise AuthError(
            "Anthropic shared scope changed; restart Hermes",
            provider=PROVIDER,
            code="shared_scope_changed",
        )


def write_scope_marker(*, epoch: Optional[str] = None) -> str:
    """Atomically publish shared scope marker (last step of enable)."""
    _ensure_owner_only_dir(shared_dir())
    epoch_val = epoch or str(uuid.uuid4())
    payload = {"version": MARKER_VERSION, "scope": "shared", "epoch": epoch_val}
    _atomic_write_json(scope_marker_path(), payload)
    return epoch_val


def remove_scope_marker() -> None:
    """Remove marker first (disable / logout). Idempotent."""
    path = scope_marker_path()
    if not path.exists() and not path.is_symlink():
        return
    _path_is_safe_regular_file(path, must_exist=True)
    path.unlink()
    _fsync_dir(path.parent)


# ---------------------------------------------------------------------------
# Grant salt + fingerprints
# ---------------------------------------------------------------------------


def ensure_grant_salt() -> bytes:
    path = grant_salt_path()
    if path.exists() or path.is_symlink():
        _path_is_safe_regular_file(path, must_exist=True)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(path), flags)
        try:
            with os.fdopen(fd, "rb") as handle:
                data = handle.read()
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            raise
        if len(data) < 16:
            raise AuthError(
                "Grant salt file is too short",
                provider=PROVIDER,
                code="shared_salt_corrupt",
            )
        return data
    salt = os.urandom(32)
    _ensure_owner_only_dir(shared_dir())
    _atomic_write_bytes(path, salt)
    return salt


def grant_fingerprint(initial_refresh_token: str) -> str:
    salt = ensure_grant_salt()
    digest = hmac.new(
        salt,
        GRANT_HMAC_CONTEXT + initial_refresh_token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{GRANT_FINGERPRINT_PREFIX}{digest}"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_rfc3339(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _normalize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isprintable() else " " for ch in (label or ""))
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        raise AuthError(
            "Shared Anthropic label must be 1-64 printable characters",
            provider=PROVIDER,
            code="shared_row_invalid",
        )
    if len(cleaned) > 64:
        cleaned = cleaned[:64].rstrip()
    return cleaned


def _validate_refresh_attempt(attempt: Any, token_generation: int) -> None:
    if attempt is None:
        return
    if not isinstance(attempt, dict):
        raise AuthError("refresh_attempt must be object or null", provider=PROVIDER, code="shared_row_invalid")
    required = {"attempt_id", "expected_generation", "started_at", "outcome"}
    if set(attempt.keys()) != required:
        raise AuthError("refresh_attempt has unexpected keys", provider=PROVIDER, code="shared_row_invalid")
    if not isinstance(attempt.get("attempt_id"), str) or not attempt["attempt_id"]:
        raise AuthError("refresh_attempt.attempt_id invalid", provider=PROVIDER, code="shared_row_invalid")
    if attempt.get("expected_generation") != token_generation:
        raise AuthError(
            "refresh_attempt.expected_generation must equal token_generation",
            provider=PROVIDER,
            code="shared_row_invalid",
        )
    if not _is_rfc3339(attempt.get("started_at")) or attempt.get("started_at") is None:
        raise AuthError("refresh_attempt.started_at invalid", provider=PROVIDER, code="shared_row_invalid")
    if attempt.get("outcome") not in {ATTEMPT_INFLIGHT, ATTEMPT_UNKNOWN, ATTEMPT_TERMINAL}:
        raise AuthError("refresh_attempt.outcome invalid", provider=PROVIDER, code="shared_row_invalid")


def validate_shared_row(row: Dict[str, Any], *, at_enrollment: bool = False) -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise AuthError("Shared row must be an object", provider=PROVIDER, code="shared_row_invalid")
    missing = [k for k in REQUIRED_ROW_KEYS if k not in row]
    if missing:
        raise AuthError(
            f"Shared row missing required fields: {', '.join(missing)}",
            provider=PROVIDER,
            code="shared_row_invalid",
        )
    extra = set(row.keys()) - set(REQUIRED_ROW_KEYS)
    if extra:
        raise AuthError(
            f"Shared row has forbidden fields: {', '.join(sorted(extra))}",
            provider=PROVIDER,
            code="shared_row_invalid",
        )
    try:
        uuid.UUID(str(row["id"]))
    except Exception as exc:
        raise AuthError("Shared row id must be a UUID", provider=PROVIDER, code="shared_row_invalid") from exc
    if row["provider"] != PROVIDER:
        raise AuthError("Shared row provider must be anthropic", provider=PROVIDER, code="shared_row_invalid")
    if row["auth_type"] != AUTH_TYPE_OAUTH:
        raise AuthError("Shared rows must be oauth", provider=PROVIDER, code="shared_row_invalid")
    if row["source"] != SOURCE_HERMES_PKCE:
        raise AuthError(
            "Shared rows must use source manual:hermes_pkce",
            provider=PROVIDER,
            code="shared_row_invalid",
        )
    label = _normalize_label(str(row["label"]))
    fp = row["grant_fingerprint"]
    if not isinstance(fp, str) or not fp.startswith(GRANT_FINGERPRINT_PREFIX) or len(fp) < 20:
        raise AuthError("Invalid grant_fingerprint", provider=PROVIDER, code="shared_row_invalid")
    if not isinstance(row["access_token"], str) or not row["access_token"].strip():
        raise AuthError("access_token required", provider=PROVIDER, code="shared_row_invalid")
    if not isinstance(row["refresh_token"], str) or not row["refresh_token"].strip():
        raise AuthError("refresh_token required", provider=PROVIDER, code="shared_row_invalid")
    if not isinstance(row["token_generation"], int) or row["token_generation"] < 1:
        raise AuthError("token_generation must be int >= 1", provider=PROVIDER, code="shared_row_invalid")
    if row["oauth_token_endpoint"] not in ENDPOINT_URLS:
        raise AuthError("oauth_token_endpoint invalid", provider=PROVIDER, code="shared_row_invalid")
    if not isinstance(row["expires_at_ms"], int) or row["expires_at_ms"] <= 0:
        raise AuthError("expires_at_ms must be positive int", provider=PROVIDER, code="shared_row_invalid")
    if at_enrollment and row["expires_at_ms"] <= int(time.time() * 1000):
        raise AuthError("expires_at_ms must be future at enrollment", provider=PROVIDER, code="shared_row_invalid")
    if not isinstance(row["priority"], int) or row["priority"] < 0:
        raise AuthError("priority must be int >= 0", provider=PROVIDER, code="shared_row_invalid")
    if not isinstance(row["request_count"], int) or row["request_count"] < 0:
        raise AuthError("request_count must be int >= 0", provider=PROVIDER, code="shared_row_invalid")

    status = row["last_status"]
    if status not in (None, STATUS_OK, STATUS_EXHAUSTED, STATUS_DEAD):
        raise AuthError("last_status invalid", provider=PROVIDER, code="shared_row_invalid")
    for ts_key in ("last_status_at", "last_error_reset_at"):
        val = row[ts_key]
        if val is not None and not isinstance(val, (int, float)):
            raise AuthError(f"{ts_key} must be number or null", provider=PROVIDER, code="shared_row_invalid")
    if row["last_error_code"] is not None and not isinstance(row["last_error_code"], int):
        raise AuthError("last_error_code must be int or null", provider=PROVIDER, code="shared_row_invalid")
    reason = row["last_error_reason"]
    if reason not in (
        None,
        REASON_RATE_LIMIT,
        REASON_BILLING,
        REASON_AUTH_TERMINAL,
        REASON_REFRESH_UNKNOWN,
        REASON_OTHER,
    ):
        raise AuthError("last_error_reason invalid", provider=PROVIDER, code="shared_row_invalid")
    msg = row["last_error_message"]
    if msg is not None:
        if not isinstance(msg, str) or len(msg) > 256:
            raise AuthError("last_error_message invalid", provider=PROVIDER, code="shared_row_invalid")
    if not _is_rfc3339(row["last_refresh"]):
        raise AuthError("last_refresh invalid", provider=PROVIDER, code="shared_row_invalid")

    _validate_refresh_attempt(row["refresh_attempt"], row["token_generation"])
    attempt = row["refresh_attempt"]
    outcome = attempt.get("outcome") if isinstance(attempt, dict) else None

    # Cross-field rules
    if status in (None, STATUS_OK):
        if any(
            row[k] is not None
            for k in ("last_error_code", "last_error_reason", "last_error_message", "last_error_reset_at")
        ):
            raise AuthError("ok/null status requires null error/reset state", provider=PROVIDER, code="shared_row_invalid")
        if attempt is not None and outcome != ATTEMPT_INFLIGHT:
            raise AuthError("ok/null status permits only null or inflight attempt", provider=PROVIDER, code="shared_row_invalid")
    elif status == STATUS_EXHAUSTED:
        if reason not in (REASON_RATE_LIMIT, REASON_BILLING):
            raise AuthError("exhausted requires rate_limit|billing", provider=PROVIDER, code="shared_row_invalid")
        if row["last_error_reset_at"] is None:
            raise AuthError("exhausted requires reset time", provider=PROVIDER, code="shared_row_invalid")
    elif status == STATUS_DEAD:
        if reason not in (REASON_AUTH_TERMINAL, REASON_REFRESH_UNKNOWN):
            raise AuthError("dead permits only auth_terminal|refresh_outcome_unknown", provider=PROVIDER, code="shared_row_invalid")
        if row["last_error_reset_at"] is not None:
            raise AuthError("dead must not have reset time", provider=PROVIDER, code="shared_row_invalid")
        if outcome == ATTEMPT_UNKNOWN and reason != REASON_REFRESH_UNKNOWN:
            raise AuthError("unknown attempt requires refresh_outcome_unknown", provider=PROVIDER, code="shared_row_invalid")
        if outcome == ATTEMPT_TERMINAL and reason != REASON_AUTH_TERMINAL:
            raise AuthError("terminal attempt requires auth_terminal", provider=PROVIDER, code="shared_row_invalid")

    out = dict(row)
    out["label"] = label
    return out


def validate_shared_pool(pool: Dict[str, Any], *, require_three: bool = False) -> Dict[str, Any]:
    if not isinstance(pool, dict):
        raise AuthError("Shared pool must be an object", provider=PROVIDER, code="shared_pool_invalid")
    missing = [k for k in REQUIRED_POOL_KEYS if k not in pool]
    if missing:
        raise AuthError(
            f"Shared pool missing fields: {', '.join(missing)}",
            provider=PROVIDER,
            code="shared_pool_invalid",
        )
    extra = set(pool.keys()) - set(REQUIRED_POOL_KEYS)
    if extra:
        raise AuthError(
            f"Shared pool has unknown fields: {', '.join(sorted(extra))}",
            provider=PROVIDER,
            code="shared_pool_invalid",
        )
    if pool.get("schema_version") != SCHEMA_VERSION:
        raise AuthError(
            f"Unsupported shared pool schema_version: {pool.get('schema_version')!r}",
            provider=PROVIDER,
            code="shared_pool_invalid",
        )
    if not isinstance(pool.get("revision"), int) or pool["revision"] < 1:
        raise AuthError("revision must be int >= 1", provider=PROVIDER, code="shared_pool_invalid")
    if pool.get("strategy") != STRATEGY_FILL_FIRST:
        raise AuthError("strategy must be fill_first", provider=PROVIDER, code="shared_pool_invalid")
    if not isinstance(pool.get("account_distinctness_attested"), bool):
        raise AuthError("account_distinctness_attested must be bool", provider=PROVIDER, code="shared_pool_invalid")
    if not _is_rfc3339(pool.get("account_distinctness_attested_at")):
        raise AuthError("account_distinctness_attested_at invalid", provider=PROVIDER, code="shared_pool_invalid")
    entries = pool.get("entries")
    if not isinstance(entries, list):
        raise AuthError("entries must be a list", provider=PROVIDER, code="shared_pool_invalid")
    validated = [validate_shared_row(e) for e in entries]
    ids = [e["id"] for e in validated]
    fps = [e["grant_fingerprint"] for e in validated]
    if len(ids) != len(set(ids)):
        raise AuthError("Duplicate shared row ids", provider=PROVIDER, code="shared_pool_invalid")
    if len(fps) != len(set(fps)):
        raise AuthError("Duplicate grant fingerprints", provider=PROVIDER, code="shared_pool_invalid")
    prios = sorted(e["priority"] for e in validated)
    if prios != list(range(len(validated))):
        raise AuthError(
            "priorities must be unique contiguous 0..n-1",
            provider=PROVIDER,
            code="shared_pool_invalid",
        )
    if require_three:
        if len(validated) != 3:
            raise AuthError(
                f"Shared scope requires exactly 3 OAuth grants (found {len(validated)})",
                provider=PROVIDER,
                code="shared_pool_count",
            )
        if not pool["account_distinctness_attested"]:
            raise AuthError(
                "Shared scope requires account distinctness attestation",
                provider=PROVIDER,
                code="shared_attestation_required",
            )
    validated.sort(key=lambda e: e["priority"])
    out = dict(pool)
    out["entries"] = validated
    return out


def empty_shared_pool() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": 1,
        "strategy": STRATEGY_FILL_FIRST,
        "account_distinctness_attested": False,
        "account_distinctness_attested_at": None,
        "entries": [],
    }


def new_shared_row(
    *,
    access_token: str,
    refresh_token: str,
    expires_at_ms: int,
    oauth_token_endpoint: str,
    label: str,
    priority: int,
    initial_refresh_token: Optional[str] = None,
) -> Dict[str, Any]:
    fp_source = initial_refresh_token or refresh_token
    row = {
        "id": str(uuid.uuid4()),
        "provider": PROVIDER,
        "auth_type": AUTH_TYPE_OAUTH,
        "source": SOURCE_HERMES_PKCE,
        "label": label,
        "grant_fingerprint": grant_fingerprint(fp_source),
        "token_generation": 1,
        "oauth_token_endpoint": oauth_token_endpoint,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at_ms": int(expires_at_ms),
        "priority": int(priority),
        "request_count": 0,
        "last_status": None,
        "last_status_at": None,
        "last_error_code": None,
        "last_error_reason": None,
        "last_error_message": None,
        "last_error_reset_at": None,
        "last_refresh": None,
        "refresh_attempt": None,
    }
    return validate_shared_row(row, at_enrollment=True)


# ---------------------------------------------------------------------------
# Strict root auth load/save
# ---------------------------------------------------------------------------


def _kernel_lock_available() -> bool:
    return auth_mod.fcntl is not None or auth_mod.msvcrt is not None


@contextmanager
def root_auth_lock(timeout_seconds: Optional[float] = None):
    if not _kernel_lock_available():
        raise AuthError(
            "Shared Anthropic scope requires a kernel cross-process lock primitive",
            provider=PROVIDER,
            code="shared_lock_unavailable",
        )
    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else max(float(auth_mod.AUTH_LOCK_TIMEOUT_SECONDS), 15.0)
    )
    with _auth_store_lock(timeout_seconds=timeout, target_path=root_auth_path()):
        yield


def load_root_auth_strict() -> Dict[str, Any]:
    """Load root auth.json without corruption-recovery empty-store fallback."""
    path = root_auth_path()
    if not path.exists() and not path.is_symlink():
        return {"version": getattr(auth_mod, "AUTH_STORE_VERSION", 1), "providers": {}}
    _path_is_safe_regular_file(path, must_exist=True)
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(path), flags)
        try:
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            raise
    except AuthError:
        raise
    except Exception as exc:
        raise AuthError(
            f"Root auth store unreadable/corrupt: {exc}",
            provider=PROVIDER,
            code="shared_root_auth_corrupt",
        ) from exc
    if not isinstance(raw, dict):
        raise AuthError(
            "Root auth store is not a JSON object",
            provider=PROVIDER,
            code="shared_root_auth_corrupt",
        )
    raw.setdefault("providers", {})
    if not isinstance(raw.get("providers"), dict):
        raise AuthError(
            "Root auth providers field corrupt",
            provider=PROVIDER,
            code="shared_root_auth_corrupt",
        )
    return raw


def save_root_auth_strict(auth_store: Dict[str, Any]) -> None:
    path = root_auth_path()
    auth_store = dict(auth_store)
    auth_store["version"] = getattr(auth_mod, "AUTH_STORE_VERSION", 1)
    auth_store["updated_at"] = _utc_now_rfc3339()
    _atomic_write_json(path, auth_store)


def get_shared_namespace(auth_store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    scp = auth_store.get("shared_credential_pools")
    if scp is None:
        return None
    if not isinstance(scp, dict):
        raise AuthError(
            "shared_credential_pools corrupt",
            provider=PROVIDER,
            code="shared_pool_invalid",
        )
    pool = scp.get(PROVIDER)
    if pool is None:
        return None
    return validate_shared_pool(pool)


def set_shared_namespace(auth_store: Dict[str, Any], pool: Optional[Dict[str, Any]]) -> None:
    if pool is None:
        scp = auth_store.get("shared_credential_pools")
        if isinstance(scp, dict):
            scp.pop(PROVIDER, None)
            if not scp:
                auth_store.pop("shared_credential_pools", None)
        return
    validated = validate_shared_pool(pool)
    scp = auth_store.setdefault("shared_credential_pools", {})
    if not isinstance(scp, dict):
        raise AuthError("shared_credential_pools corrupt", provider=PROVIDER, code="shared_pool_invalid")
    scp[PROVIDER] = validated


def has_dormant_or_active_shared_state() -> bool:
    """True when marker exists OR dormant shared namespace is present."""
    path = scope_marker_path()
    if path.exists() or path.is_symlink():
        return True
    try:
        store = load_root_auth_strict()
        scp = store.get("shared_credential_pools")
        if isinstance(scp, dict) and PROVIDER in scp:
            return True
    except AuthError:
        # Corrupt root while dormant still blocks generic backup.
        if root_auth_path().exists():
            return True
    return False


def load_shared_pool_for_management(*, require_active_three: bool = False) -> Dict[str, Any]:
    """Strict loader for management paths (profile or shared)."""
    with root_auth_lock():
        store = load_root_auth_strict()
        pool = get_shared_namespace(store)
        if pool is None:
            if require_active_three:
                raise AuthError(
                    "Shared Anthropic pool is empty/missing",
                    provider=PROVIDER,
                    code="shared_pool_empty",
                )
            return empty_shared_pool()
        if require_active_three:
            return validate_shared_pool(pool, require_three=True)
        return pool


# ---------------------------------------------------------------------------
# Official target predicate + resolution context
# ---------------------------------------------------------------------------


def is_official_anthropic_oauth_target(
    provider: Any = None,
    purpose: Any = "inference",
    api_mode: Any = None,
    base_url: Any = None,
) -> bool:
    """True only for official native Anthropic OAuth targets."""
    prov = str(provider or "").strip().lower()
    if prov and prov != PROVIDER:
        return False
    purpose_s = str(purpose or "inference").strip().lower()
    if purpose_s not in OFFICIAL_PURPOSES:
        return False

    raw_url = str(base_url or "").strip()
    if not raw_url:
        host = "api.anthropic.com"
        scheme = "https"
        port = None
        userinfo = False
        fragment = False
    else:
        parsed = urlparse(raw_url)
        if parsed.scheme and parsed.scheme.lower() != "https":
            return False
        if parsed.username or parsed.password:
            return False
        if parsed.fragment:
            return False
        host = (parsed.hostname or "").lower()
        scheme = (parsed.scheme or "https").lower()
        port = parsed.port
        userinfo = bool(parsed.username or parsed.password)
        fragment = bool(parsed.fragment)
        # Deceptive suffixes / path tricks — host must be exact.
        if not host:
            return False

    if scheme != "https":
        return False
    if userinfo or fragment:
        return False
    if host not in OFFICIAL_ANTHROPIC_HOSTS:
        return False
    if port is not None and port != 443:
        return False

    if purpose_s == "inference":
        mode = str(api_mode or "").strip().lower()
        # Empty / default / anthropic messages modes are OK; chat_completions is not native.
        if mode in ("", "anthropic", "anthropic_messages", "messages", "native"):
            return True
        if mode in ("chat_completions", "codex_responses", "openai"):
            return False
        # Unknown modes: only accept if clearly anthropic-ish.
        return "anthropic" in mode or mode == "messages"
    return True


@dataclass
class AnthropicCredentialContext:
    """Authoritative credential resolution result (does not leak tokens via repr)."""

    access_token: str
    row_id: str
    token_generation: int
    source: str = "shared_pool"
    auth_type: str = AUTH_TYPE_OAUTH
    label: str = ""
    expires_at_ms: Optional[int] = None
    pool_revision: int = 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AnthropicCredentialContext(row_id={self.row_id!r}, "
            f"token_generation={self.token_generation}, source={self.source!r})"
        )


def _row_is_selectable(row: Dict[str, Any], now: float) -> bool:
    if row.get("refresh_attempt") is not None:
        return False
    status = row.get("last_status")
    if status == STATUS_DEAD:
        return False
    if status == STATUS_EXHAUSTED:
        reset_at = row.get("last_error_reset_at")
        if reset_at is None:
            return False
        if float(reset_at) > now:
            return False
        # Elapsed exhaustion — will be normalized under lock before selection.
    return True


def _normalize_elapsed_exhaustion(pool: Dict[str, Any], now: float) -> bool:
    changed = False
    for row in pool["entries"]:
        if row.get("last_status") != STATUS_EXHAUSTED:
            continue
        reset_at = row.get("last_error_reset_at")
        if reset_at is not None and float(reset_at) <= now:
            row["last_status"] = None
            row["last_status_at"] = None
            row["last_error_code"] = None
            row["last_error_reason"] = None
            row["last_error_message"] = None
            row["last_error_reset_at"] = None
            changed = True
    return changed


def _abandon_inflight_attempts(pool: Dict[str, Any]) -> bool:
    """Convert abandoned inflight attempts to unknown/dead under root lock."""
    changed = False
    now_ts = time.time()
    for row in pool["entries"]:
        attempt = row.get("refresh_attempt")
        if not isinstance(attempt, dict):
            continue
        if attempt.get("outcome") != ATTEMPT_INFLIGHT:
            continue
        row["refresh_attempt"] = {
            "attempt_id": attempt["attempt_id"],
            "expected_generation": row["token_generation"],
            "started_at": attempt.get("started_at") or _utc_now_rfc3339(),
            "outcome": ATTEMPT_UNKNOWN,
        }
        row["last_status"] = STATUS_DEAD
        row["last_status_at"] = now_ts
        row["last_error_code"] = None
        row["last_error_reason"] = REASON_REFRESH_UNKNOWN
        row["last_error_message"] = "abandoned inflight refresh"
        row["last_error_reset_at"] = None
        changed = True
    return changed


def select_fill_first(pool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    now = time.time()
    for row in sorted(pool["entries"], key=lambda r: r["priority"]):
        if _row_is_selectable(row, now):
            return row
    return None


def resolve_shared_anthropic_credential(
    *,
    provider: Any = PROVIDER,
    purpose: str = "inference",
    api_mode: Any = None,
    base_url: Any = None,
    refresh_if_needed: bool = True,
) -> AnthropicCredentialContext:
    """Authoritative shared-scope resolution. Raises AuthError on failure."""
    if not is_official_anthropic_oauth_target(provider, purpose, api_mode, base_url):
        raise AuthError(
            "Not an official Anthropic OAuth target",
            provider=PROVIDER,
            code="not_official_target",
        )
    assert_epoch_unchanged()
    with _inprocess_pool_lock:
        with root_auth_lock(_refresh_lock_timeout() if refresh_if_needed else None):
            store = load_root_auth_strict()
            pool = get_shared_namespace(store)
            if pool is None:
                raise AuthError(
                    "Shared Anthropic pool is empty",
                    provider=PROVIDER,
                    code="shared_pool_empty",
                )
            pool = validate_shared_pool(pool, require_three=True)
            changed = _abandon_inflight_attempts(pool)
            changed = _normalize_elapsed_exhaustion(pool, time.time()) or changed
            if changed:
                pool["revision"] = int(pool["revision"]) + 1
                set_shared_namespace(store, pool)
                save_root_auth_strict(store)
                pool = validate_shared_pool(pool, require_three=True)

            row = select_fill_first(pool)
            if row is None:
                raise AuthError(
                    "No available Anthropic shared credentials",
                    provider=PROVIDER,
                    code="shared_pool_exhausted",
                )

            needs_refresh = (
                refresh_if_needed
                and isinstance(row.get("expires_at_ms"), int)
                and row["expires_at_ms"] <= int(time.time() * 1000) + 60_000
            )
            if needs_refresh:
                row = _commit_refresh_locked(store, pool, row, force=True)
                pool = get_shared_namespace(store)
                assert pool is not None
                pool = validate_shared_pool(pool, require_three=True)

            return AnthropicCredentialContext(
                access_token=row["access_token"],
                row_id=row["id"],
                token_generation=int(row["token_generation"]),
                label=str(row.get("label") or ""),
                expires_at_ms=row.get("expires_at_ms"),
                pool_revision=int(pool["revision"]),
            )


def resolve_anthropic_credential_authoritative(
    *,
    provider: Any = PROVIDER,
    purpose: str = "inference",
    api_mode: Any = None,
    base_url: Any = None,
    explicit_api_key: Optional[str] = None,
    allow_legacy: bool = True,
) -> Optional[str]:
    """Single gate used by all native Anthropic consumers.

    While shared scope is active and the target is official, ignores explicit
    keys, env vars, Claude Code files, and profile pools. Returns access token
    string or None (legacy mode only).
    """
    target = is_official_anthropic_oauth_target(provider, purpose, api_mode, base_url)
    try:
        shared = is_shared_scope_active()
    except AuthError:
        # Malformed marker — fail closed for official targets.
        if target:
            raise
        shared = False

    if shared and target:
        ctx = resolve_shared_anthropic_credential(
            provider=provider,
            purpose=purpose,
            api_mode=api_mode,
            base_url=base_url,
        )
        return ctx.access_token

    if not allow_legacy:
        return None

    # Legacy path: do not use shared dormant rows.
    if explicit_api_key and str(explicit_api_key).strip():
        return str(explicit_api_key).strip()
    return None  # caller continues with existing resolve_anthropic_token


# ---------------------------------------------------------------------------
# Leases (generation-keyed)
# ---------------------------------------------------------------------------


def acquire_lease(row_id: str, token_generation: int) -> None:
    key = (row_id, int(token_generation))
    with _inprocess_pool_lock:
        _active_leases[key] = _active_leases.get(key, 0) + 1


def release_lease(row_id: str, token_generation: int) -> None:
    key = (row_id, int(token_generation))
    with _inprocess_pool_lock:
        cur = _active_leases.get(key, 0)
        if cur <= 1:
            _active_leases.pop(key, None)
        else:
            _active_leases[key] = cur - 1


def lease_count(row_id: str, token_generation: int) -> int:
    return _active_leases.get((row_id, int(token_generation)), 0)


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


def _refresh_lock_timeout() -> float:
    # Anthropic HTTP timeout in refresh_anthropic_oauth_pure is 10s; margin +5.
    return max(float(auth_mod.AUTH_LOCK_TIMEOUT_SECONDS), 10.0 + 5.0)


def _post_refresh(
    refresh_token: str,
    endpoint_id: str,
    *,
    transport: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """POST once to the row's canonical endpoint. No endpoint fallback."""
    if transport is not None:
        return transport(refresh_token=refresh_token, endpoint_id=endpoint_id)

    import urllib.error
    import urllib.parse
    import urllib.request

    url = ENDPOINT_URLS[endpoint_id]
    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    data = json.dumps(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "hermes-agent-shared-oauth/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        err = AuthError(
            f"Anthropic refresh HTTP {exc.code}",
            provider=PROVIDER,
            code="refresh_http_error",
            relogin_required=exc.code in (400, 401),
        )
        err.http_status = exc.code  # type: ignore[attr-defined]
        err.body_snippet = body  # type: ignore[attr-defined]
        raise err from exc
    except Exception as exc:
        err = AuthError(
            f"Anthropic refresh transport failure: {type(exc).__name__}",
            provider=PROVIDER,
            code="refresh_transport",
        )
        err.ambiguous = True  # type: ignore[attr-defined]
        raise err from exc

    access = result.get("access_token") or ""
    refresh = result.get("refresh_token") or ""
    expires_in = result.get("expires_in")
    if not isinstance(access, str) or not access.strip():
        raise AuthError("malformed refresh response: access_token", provider=PROVIDER, code="refresh_malformed")
    if not isinstance(refresh, str) or not refresh.strip():
        raise AuthError("malformed refresh response: refresh_token", provider=PROVIDER, code="refresh_malformed")
    if not isinstance(expires_in, (int, float)) or expires_in <= 0:
        raise AuthError("malformed refresh response: expires_in", provider=PROVIDER, code="refresh_malformed")
    return {
        "access_token": access.strip(),
        "refresh_token": refresh.strip(),
        "expires_at_ms": int(time.time() * 1000) + int(expires_in) * 1000,
    }


def _commit_refresh_locked(
    store: Dict[str, Any],
    pool: Dict[str, Any],
    row: Dict[str, Any],
    *,
    force: bool,
    expected_generation: Optional[int] = None,
    transport: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Caller already holds root auth lock + in-process pool lock."""
    # Re-find row by id
    entries = {e["id"]: e for e in pool["entries"]}
    current = entries.get(row["id"])
    if current is None:
        raise AuthError("Shared credential disappeared", provider=PROVIDER, code="shared_row_missing")

    _abandon_inflight_attempts(pool)

    # Re-validate after abandon
    current = next(e for e in pool["entries"] if e["id"] == row["id"])
    if current.get("refresh_attempt") is not None:
        outcome = current["refresh_attempt"].get("outcome")
        if outcome in (ATTEMPT_UNKNOWN, ATTEMPT_TERMINAL):
            raise AuthError(
                "Shared credential requires re-auth",
                provider=PROVIDER,
                code="shared_reauth_required",
                relogin_required=True,
            )

    exp_gen = expected_generation if expected_generation is not None else row.get("token_generation")
    if exp_gen is not None and current["token_generation"] != exp_gen:
        # Disk already rotated — adopt.
        return current

    if not force and current["expires_at_ms"] > int(time.time() * 1000) + 60_000:
        return current

    attempt_id = str(uuid.uuid4())
    current["refresh_attempt"] = {
        "attempt_id": attempt_id,
        "expected_generation": current["token_generation"],
        "started_at": _utc_now_rfc3339(),
        "outcome": ATTEMPT_INFLIGHT,
    }
    pool["revision"] = int(pool["revision"]) + 1
    set_shared_namespace(store, validate_shared_pool(pool, require_three=len(pool["entries"]) == 3))
    save_root_auth_strict(store)

    refresh_token = current["refresh_token"]
    endpoint_id = current["oauth_token_endpoint"]
    try:
        refreshed = _post_refresh(refresh_token, endpoint_id, transport=transport)
    except AuthError as exc:
        code = getattr(exc, "code", "") or ""
        http_status = getattr(exc, "http_status", None)
        body = (getattr(exc, "body_snippet", "") or "").lower()
        terminal = (
            code in {"invalid_grant", "refresh_http_error"}
            and (
                "invalid_grant" in body
                or "revoked" in body
                or http_status in (400, 401)
            )
        )
        # After intent commit, ambiguous/errors → unknown; terminal invalid_grant → terminal.
        if terminal and ("invalid_grant" in body or "revoked" in body):
            current["refresh_attempt"] = {
                "attempt_id": attempt_id,
                "expected_generation": current["token_generation"],
                "started_at": current["refresh_attempt"]["started_at"],
                "outcome": ATTEMPT_TERMINAL,
            }
            current["last_status"] = STATUS_DEAD
            current["last_status_at"] = time.time()
            current["last_error_reason"] = REASON_AUTH_TERMINAL
            current["last_error_message"] = "invalid_grant"
            current["last_error_reset_at"] = None
        else:
            current["refresh_attempt"] = {
                "attempt_id": attempt_id,
                "expected_generation": current["token_generation"],
                "started_at": current["refresh_attempt"]["started_at"],
                "outcome": ATTEMPT_UNKNOWN,
            }
            current["last_status"] = STATUS_DEAD
            current["last_status_at"] = time.time()
            current["last_error_reason"] = REASON_REFRESH_UNKNOWN
            current["last_error_message"] = "refresh outcome unknown"
            current["last_error_reset_at"] = None
        pool["revision"] = int(pool["revision"]) + 1
        set_shared_namespace(store, validate_shared_pool(pool, require_three=len(pool["entries"]) == 3))
        save_root_auth_strict(store)
        raise

    current["access_token"] = refreshed["access_token"]
    current["refresh_token"] = refreshed["refresh_token"]
    current["expires_at_ms"] = refreshed["expires_at_ms"]
    current["token_generation"] = int(current["token_generation"]) + 1
    current["last_refresh"] = _utc_now_rfc3339()
    current["refresh_attempt"] = None
    current["last_status"] = None
    current["last_status_at"] = None
    current["last_error_code"] = None
    current["last_error_reason"] = None
    current["last_error_message"] = None
    current["last_error_reset_at"] = None
    pool["revision"] = int(pool["revision"]) + 1
    set_shared_namespace(store, validate_shared_pool(pool, require_three=len(pool["entries"]) == 3))
    save_root_auth_strict(store)
    global _cached_clients_generation
    _cached_clients_generation = current["token_generation"]
    return current


def commit_refresh(
    row_id: str,
    *,
    expected_generation: int,
    transport: Optional[Callable[..., Dict[str, Any]]] = None,
    capability: Optional[SharedMutationCapability] = None,
) -> Dict[str, Any]:
    """Public refresh entry (also used by runtime). Capability optional for runtime refresh."""
    # Runtime refresh is allowed without management capability; management
    # mutations still require it.
    with _inprocess_pool_lock:
        with root_auth_lock(_refresh_lock_timeout()):
            assert_epoch_unchanged()
            store = load_root_auth_strict()
            pool = get_shared_namespace(store)
            if pool is None:
                raise AuthError("Shared pool empty", provider=PROVIDER, code="shared_pool_empty")
            pool = validate_shared_pool(pool, require_three=is_shared_scope_active())
            row = next((e for e in pool["entries"] if e["id"] == row_id), None)
            if row is None:
                raise AuthError("row not found", provider=PROVIDER, code="shared_row_missing")
            return _commit_refresh_locked(
                store,
                pool,
                row,
                force=True,
                expected_generation=expected_generation,
                transport=transport,
            )


# ---------------------------------------------------------------------------
# Transactional mutations (management)
# ---------------------------------------------------------------------------


def append_row(
    row: Dict[str, Any],
    *,
    capability: SharedMutationCapability,
) -> Dict[str, Any]:
    _require_capability(capability)
    if is_shared_scope_active():
        raise AuthError(
            "Cannot add while shared scope is active; switch to profile scope first",
            provider=PROVIDER,
            code="shared_add_while_active",
        )
    validated = validate_shared_row(row, at_enrollment=True)
    with _inprocess_pool_lock:
        with root_auth_lock():
            store = load_root_auth_strict()
            pool = get_shared_namespace(store) or empty_shared_pool()
            if len(pool["entries"]) >= 3:
                raise AuthError(
                    "Shared Anthropic pool already has 3 grants",
                    provider=PROVIDER,
                    code="shared_pool_full",
                )
            fps = {e["grant_fingerprint"] for e in pool["entries"]}
            if validated["grant_fingerprint"] in fps:
                # Idempotent by fingerprint — return existing.
                existing = next(
                    e for e in pool["entries"] if e["grant_fingerprint"] == validated["grant_fingerprint"]
                )
                return existing
            validated["priority"] = len(pool["entries"])
            # Renumber is automatic via append order.
            pool["entries"].append(validated)
            pool["account_distinctness_attested"] = False
            pool["account_distinctness_attested_at"] = None
            pool["revision"] = int(pool.get("revision") or 0) + 1
            if pool["revision"] < 1:
                pool["revision"] = 1
            set_shared_namespace(store, validate_shared_pool(pool))
            save_root_auth_strict(store)
            return validated


def remove_row(
    target: str,
    *,
    capability: SharedMutationCapability,
) -> None:
    _require_capability(capability)
    if is_shared_scope_active():
        raise AuthError(
            "Active shared remove is rejected; switch to profile scope first",
            provider=PROVIDER,
            code="shared_remove_while_active",
        )
    with _inprocess_pool_lock:
        with root_auth_lock():
            store = load_root_auth_strict()
            pool = get_shared_namespace(store)
            if pool is None or not pool["entries"]:
                raise AuthError("No shared rows to remove", provider=PROVIDER, code="shared_row_missing")
            match_idx = None
            for i, e in enumerate(pool["entries"]):
                if e["id"] == target or e["label"] == target or str(i) == str(target) or str(i + 1) == str(target):
                    match_idx = i
                    break
            if match_idx is None:
                raise AuthError(f"Shared row not found: {target}", provider=PROVIDER, code="shared_row_missing")
            pool["entries"].pop(match_idx)
            for i, e in enumerate(sorted(pool["entries"], key=lambda r: r["priority"])):
                e["priority"] = i
            pool["entries"].sort(key=lambda r: r["priority"])
            pool["account_distinctness_attested"] = False
            pool["account_distinctness_attested_at"] = None
            pool["revision"] = int(pool["revision"]) + 1
            set_shared_namespace(store, validate_shared_pool(pool))
            save_root_auth_strict(store)


def reset_statuses(*, capability: SharedMutationCapability) -> int:
    """Clear recoverable exhaustion only. Returns count cleared."""
    _require_capability(capability)
    cleared = 0
    with _inprocess_pool_lock:
        with root_auth_lock():
            store = load_root_auth_strict()
            pool = get_shared_namespace(store)
            if pool is None:
                return 0
            for row in pool["entries"]:
                if row.get("last_status") != STATUS_EXHAUSTED:
                    continue
                if row.get("refresh_attempt") is not None:
                    continue
                if row.get("last_error_reason") not in (REASON_RATE_LIMIT, REASON_BILLING):
                    continue
                row["last_status"] = None
                row["last_status_at"] = None
                row["last_error_code"] = None
                row["last_error_reason"] = None
                row["last_error_message"] = None
                row["last_error_reset_at"] = None
                cleared += 1
            if cleared:
                pool["revision"] = int(pool["revision"]) + 1
                set_shared_namespace(store, validate_shared_pool(pool))
                save_root_auth_strict(store)
    return cleared


def patch_status(
    row_id: str,
    *,
    expected_generation: int,
    last_status: Optional[str],
    last_error_code: Optional[int] = None,
    last_error_reason: Optional[str] = None,
    last_error_message: Optional[str] = None,
    last_error_reset_at: Optional[float] = None,
    capability: Optional[SharedMutationCapability] = None,
) -> None:
    """Runtime status patch (generation-guarded). No capability required for runtime."""
    with _inprocess_pool_lock:
        with root_auth_lock():
            try:
                assert_epoch_unchanged()
            except AuthError:
                # Status patches during scope change: drop silently? Spec says
                # fail new requests. Fail here.
                raise
            store = load_root_auth_strict()
            pool = get_shared_namespace(store)
            if pool is None:
                return
            row = next((e for e in pool["entries"] if e["id"] == row_id), None)
            if row is None:
                return
            if row["token_generation"] != expected_generation:
                # Stale callback — adopt disk, no mutation.
                return
            now = time.time()
            row["last_status"] = last_status
            row["last_status_at"] = now if last_status is not None else None
            row["last_error_code"] = last_error_code
            row["last_error_reason"] = last_error_reason
            if last_error_message is not None:
                row["last_error_message"] = str(last_error_message)[:256]
            else:
                row["last_error_message"] = None
            row["last_error_reset_at"] = last_error_reset_at
            # Validate combination
            validate_shared_row(row)
            pool["revision"] = int(pool["revision"]) + 1
            require_three = is_shared_scope_active()
            set_shared_namespace(store, validate_shared_pool(pool, require_three=require_three))
            save_root_auth_strict(store)


def enable_shared_scope(*, attest_distinct_accounts: bool, capability: SharedMutationCapability) -> str:
    _require_capability(capability)
    if not attest_distinct_accounts:
        raise AuthError(
            "Enabling shared scope requires --attest-distinct-accounts "
            "(operator attestation that three browser sessions used different Anthropic accounts; "
            "Hermes cannot machine-verify account identity)",
            provider=PROVIDER,
            code="shared_attestation_required",
        )
    # Already valid shared → revalidate, no write.
    try:
        state = read_scope_state()
        if state.mode == "shared":
            with root_auth_lock():
                store = load_root_auth_strict()
                pool = get_shared_namespace(store)
                validate_shared_pool(pool or {}, require_three=True)
            return state.epoch or ""
    except AuthError as exc:
        if getattr(exc, "code", "") == "shared_scope_corrupt":
            raise
        # Fall through if pool invalid while marker present → exit 1
        if is_shared_scope_active():
            raise

    with _inprocess_pool_lock:
        with root_auth_lock():
            store = load_root_auth_strict()
            pool = get_shared_namespace(store)
            if pool is None:
                raise AuthError(
                    "Stage exactly three OAuth grants with: hermes auth add anthropic --type oauth --shared",
                    provider=PROVIDER,
                    code="shared_pool_empty",
                )
            pool["account_distinctness_attested"] = True
            pool["account_distinctness_attested_at"] = _utc_now_rfc3339()
            pool["revision"] = int(pool["revision"]) + 1
            set_shared_namespace(store, validate_shared_pool(pool, require_three=True))
            save_root_auth_strict(store)
            # Marker last.
            epoch = write_scope_marker()
            return epoch


def disable_shared_scope(*, capability: SharedMutationCapability) -> None:
    _require_capability(capability)
    # Marker first.
    remove_scope_marker()


def clear_shared_namespace(*, capability: SharedMutationCapability) -> None:
    _require_capability(capability)
    with _inprocess_pool_lock:
        with root_auth_lock():
            # Ensure marker already gone for logout; if still present remove.
            if scope_marker_path().exists() or scope_marker_path().is_symlink():
                remove_scope_marker()
            store = load_root_auth_strict()
            set_shared_namespace(store, None)
            save_root_auth_strict(store)


def repair_malformed_marker(*, yes: bool, capability: SharedMutationCapability) -> Path:
    _require_capability(capability)
    path = scope_marker_path()
    if not path.exists() and not path.is_symlink():
        raise AuthError("No scope marker to repair", provider=PROVIDER, code="shared_marker_missing")
    if not yes and not sys_stdin_is_tty():
        raise AuthError(
            "repair requires --yes on non-TTY",
            provider=PROVIDER,
            code="shared_repair_needs_yes",
        )
    with root_auth_lock():
        _ensure_owner_only_dir(recovery_dir())
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = recovery_dir() / f"{ts}-anthropic-scope.json"
        # Read raw bytes without following if possible
        raw = path.read_bytes() if path.exists() else b""
        _atomic_write_bytes(backup, raw)
        # Remove marker by inode check
        _path_is_safe_regular_file(path, must_exist=True)
        st = path.lstat()
        path.unlink()
        _fsync_dir(path.parent)
        logger.info("Repaired Anthropic scope marker; forensic backup id=%s nlink_was=%s", backup.name, st.st_nlink)
        return backup


def sys_stdin_is_tty() -> bool:
    try:
        return bool(os.isatty(0))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Redacted listing
# ---------------------------------------------------------------------------


def redact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "label": row["label"],
        "priority": row["priority"],
        "token_generation": row["token_generation"],
        "last_status": row["last_status"],
        "last_error_reason": row["last_error_reason"],
        "expires_at_ms": row["expires_at_ms"],
        "oauth_token_endpoint": row["oauth_token_endpoint"],
        "grant_fingerprint_prefix": row["grant_fingerprint"][:20] + "…",
        "has_refresh_attempt": row.get("refresh_attempt") is not None,
    }


def list_redacted(*, require_active: bool = False) -> Dict[str, Any]:
    state = read_scope_state()
    with root_auth_lock():
        store = load_root_auth_strict()
        pool = get_shared_namespace(store)
        if state.mode == "shared":
            pool = validate_shared_pool(pool or {}, require_three=True)
        elif pool is not None:
            pool = validate_shared_pool(pool)
        else:
            pool = empty_shared_pool()
    return {
        "scope": state.mode,
        "epoch": state.epoch,
        "revision": pool["revision"],
        "strategy": pool["strategy"],
        "account_distinctness_attested": pool["account_distinctness_attested"],
        "entries": [redact_row(e) for e in pool["entries"]],
        "root_auth": str(root_auth_path()),
    }


# ---------------------------------------------------------------------------
# Backup / restore (shared)
# ---------------------------------------------------------------------------

BACKUP_ARCHIVE_VERSION = 1


def create_shared_backup(output: Path, *, capability: SharedMutationCapability) -> Path:
    _require_capability(capability)
    output = Path(output)
    if not output.is_absolute():
        raise AuthError("backup output must be an absolute path", provider=PROVIDER, code="shared_backup_path")
    import tarfile
    import io

    with root_auth_lock():
        store = load_root_auth_strict()
        pool = get_shared_namespace(store)
        marker = None
        try:
            st = read_scope_state()
            if st.mode == "shared" and st.raw:
                marker = st.raw
        except AuthError:
            marker = None
        salt = None
        sp = grant_salt_path()
        if sp.exists():
            _path_is_safe_regular_file(sp, must_exist=True)
            salt = sp.read_bytes().hex()
        # Auth data before marker in manifest.
        manifest = {
            "version": BACKUP_ARCHIVE_VERSION,
            "created_at": _utc_now_rfc3339(),
            "shared_credential_pools": {"anthropic": pool} if pool else {},
            "grant_salt_hex": salt,
            "marker": marker,
        }
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        _ensure_owner_only_dir(output.parent)
        tmp = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
        try:
            with tarfile.open(tmp, "w:gz") as tar:
                info = tarfile.TarInfo(name="manifest.json")
                info.size = len(payload)
                info.mode = 0o600
                tar.addfile(info, io.BytesIO(payload))
            os.replace(str(tmp), str(output))
            try:
                output.chmod(0o600)
            except OSError:
                pass
            _fsync_dir(output.parent)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
    return output


def restore_shared_backup(input_path: Path, *, yes: bool, capability: SharedMutationCapability) -> None:
    _require_capability(capability)
    if not yes:
        raise AuthError("restore requires --yes", provider=PROVIDER, code="shared_restore_needs_yes")
    input_path = Path(input_path)
    if not input_path.is_absolute():
        raise AuthError("restore input must be absolute", provider=PROVIDER, code="shared_backup_path")
    _path_is_safe_regular_file(input_path, must_exist=True)
    import tarfile
    import tempfile

    with tarfile.open(input_path, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) != 1 or members[0].name != "manifest.json":
            raise AuthError("malformed shared backup archive", provider=PROVIDER, code="shared_backup_corrupt")
        extracted = tar.extractfile(members[0])
        if extracted is None:
            raise AuthError("malformed shared backup archive", provider=PROVIDER, code="shared_backup_corrupt")
        manifest = json.loads(extracted.read().decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != BACKUP_ARCHIVE_VERSION:
        raise AuthError("unsupported backup version", provider=PROVIDER, code="shared_backup_corrupt")

    with _inprocess_pool_lock:
        with root_auth_lock():
            # Marker first removal.
            if scope_marker_path().exists() or scope_marker_path().is_symlink():
                remove_scope_marker()
            store = load_root_auth_strict()
            scp = manifest.get("shared_credential_pools") or {}
            pool = scp.get("anthropic")
            if pool is not None:
                set_shared_namespace(store, validate_shared_pool(pool))
            else:
                set_shared_namespace(store, None)
            save_root_auth_strict(store)
            salt_hex = manifest.get("grant_salt_hex")
            if isinstance(salt_hex, str) and salt_hex:
                _atomic_write_bytes(grant_salt_path(), bytes.fromhex(salt_hex))
            marker = manifest.get("marker")
            if isinstance(marker, dict) and marker.get("scope") == "shared":
                # Publish marker last.
                _atomic_write_json(scope_marker_path(), marker)


def refuse_generic_backup_if_shared() -> None:
    if has_dormant_or_active_shared_state():
        raise AuthError(
            "Anthropic shared pool state detected. "
            "Use `hermes auth backup anthropic --shared --output <path>` "
            "(and matching restore). Generic backup/import refused.",
            provider=PROVIDER,
            code="shared_blocks_generic_backup",
        )


# ---------------------------------------------------------------------------
# Gateway PID discovery (best-effort)
# ---------------------------------------------------------------------------


def discover_live_gateway_pids() -> List[int]:
    """Best-effort discovery of live gateway PIDs (not a safety boundary)."""
    pids: List[int] = []
    try:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root()
        candidates = [root / "gateway.pid"]
        profiles = root / "profiles"
        if profiles.is_dir():
            for child in profiles.iterdir():
                if child.is_dir():
                    candidates.append(child / "gateway.pid")
        for pid_path in candidates:
            if not pid_path.exists() or pid_path.is_symlink():
                continue
            try:
                raw = json.loads(pid_path.read_text(encoding="utf-8"))
                pid = raw.get("pid") if isinstance(raw, dict) else None
                if isinstance(pid, int) and pid > 0:
                    os.kill(pid, 0)
                    pids.append(pid)
            except Exception:
                continue
    except Exception:
        pass
    return pids


def require_no_live_gateways_for_scope_change() -> None:
    pids = discover_live_gateway_pids()
    if pids:
        raise AuthError(
            f"Stop discoverable Hermes gateways before changing Anthropic scope "
            f"(live PIDs: {pids}). Process discovery is best-effort; epoch preflight remains mandatory.",
            provider=PROVIDER,
            code="shared_gateways_live",
        )
