"""1Password (`op` CLI) secret source.

Resolve provider credentials from 1Password ``op://vault/item/field``
references at process startup so they don't have to live in plaintext in
``~/.hermes/.env``.

Design summary
--------------

* Users map environment-variable names to official 1Password secret
  references in ``secrets.onepassword.env``::

      secrets:
        onepassword:
          enabled: true
          env:
            OPENAI_API_KEY: "op://Private/OpenAI/api key"
            ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"

* After ``.env`` loads, each reference is resolved with a single
  ``op read -- <reference>`` call and injected into ``os.environ`` (the
  same point in startup as the Bitwarden source).
* Authentication is whatever the user's ``op`` CLI already uses — a
  service-account token (``OP_SERVICE_ACCOUNT_TOKEN``) for headless boxes,
  or a desktop/interactive session (``OP_SESSION_*``).  Hermes never
  authenticates on the user's behalf; it shells out to an already-trusted,
  already-authenticated CLI.
* Failures NEVER block startup.  A missing ``op`` binary, expired auth, a
  bad reference, or a permission error each surface a one-line warning and
  Hermes continues with whatever credentials ``.env`` already had.

The atomic-write / ``0600`` / TTL cache mechanics are shared with the other
backends via :mod:`agent.secret_sources._cache` — successful, complete pulls
are cached in-process and on disk under ``<hermes_home>/cache/op_cache.enc.json``
so back-to-back short-lived ``hermes`` invocations don't re-shell ``op`` for
every reference.  The disk cache is **encrypted-only** (AES-GCM, keyed off the
auth material; never plaintext).  A legacy plaintext ``op_cache.json`` from an
older Hermes is re-encrypted and removed on first read.  The disk file holds
only resolved secret *values*; auth material is fingerprinted, never stored.
With ``cache_ttl_seconds: 0`` there is no disk persistence at all (memory-only)
— plaintext is never an option.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.secret_sources._cache import (
    CachedFetch,
    DiskCache,
    FetchResult,
    is_valid_env_name,
)
from agent.secret_sources.base import ErrorKind, SecretSource
from agent.secret_sources.base import get_source_environment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# How long to wait for a single `op read`, in seconds.
_OP_RUN_TIMEOUT = 30

# Default env var the official `op` CLI reads for service-account auth.  Users
# can point `service_account_token_env` at a different name; we always export
# the value to the child as OP_SERVICE_ACCOUNT_TOKEN, which is what `op` itself
# looks for.
_DEFAULT_TOKEN_ENV = "OP_SERVICE_ACCOUNT_TOKEN"

# ANSI stripping for `op` diagnostics we surface uses the shared
# tools.ansi_strip.strip_ansi (full ECMA-48: CSI, OSC, DCS/SOS/PM/APC,
# C1) so a control sequence can't reposition the cursor or hide text
# after a redaction marker.

# Env vars the `op` child actually needs.  We build a minimal allowlisted env
# rather than copying all of os.environ (which, post-dotenv, holds every
# provider credential) into the child — tighter blast radius if `op` or
# anything it execs ever misbehaves.  OP_SESSION_* and the token are added
# dynamically in _op_child_env().
_OP_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "SystemRoot",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "OP_ACCOUNT",
    "OP_CONNECT_HOST",
    "OP_CONNECT_TOKEN",
    # Lets a user skip op's desktop-app integration probe (which can hang with
    # no timeout on a wedged desktop container) and go straight to token auth.
    "OP_LOAD_DESKTOP_APP_SETTINGS",
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# In-process cache.  The key folds in str(home_path) so a HERMES_HOME switch
# inside one long-lived process (e.g. the gateway) can't return another
# profile's secrets from L1.  The disk layer omits home from its serialized
# key because the file already lives under the home dir (see _disk_key_str).
_CacheKey = Tuple[str, str, str, str]  # (auth_fp, account, home, refs_fp)
_CACHE: Dict[_CacheKey, CachedFetch] = {}

# Disk cache is ENCRYPTED-ONLY.  The plaintext ``op_cache.json`` written by
# older Hermes versions stored every resolved secret value at rest — the same
# vulnerability class the BWS series (#77008) eliminated.  Values now persist
# only as AES-GCM ciphertext (``op_cache.enc.json``), keyed off the same auth
# material that feeds the cache-key fingerprint; the raw token never touches
# disk.  ``cache_ttl_seconds: 0`` disables disk persistence entirely
# (memory-only) — plaintext is never an option.
_DISK_CACHE_BASENAME = "op_cache.json"
_ENCRYPTED_CACHE_BASENAME = "op_cache.enc.json"
_ENCRYPTED_CACHE_VERSION = 1
_ENCRYPTED_CACHE_INFO = b"hermes-onepassword-encrypted-cache-v1"


def _disk_key_str(cache_key: _CacheKey) -> str:
    """Serialize a cache key for on-disk storage, omitting home_path.

    The disk file is already partitioned by home (it lives under
    ``<home>/cache/``), so the path provides the home dimension; folding
    it into the key string too would be redundant.
    """
    auth_fp, account, _home, refs_fp = cache_key
    return f"{auth_fp}|{account}|{refs_fp}"


_DISK_CACHE: DiskCache = DiskCache(
    _DISK_CACHE_BASENAME, key_serializer=_disk_key_str
)


def _disk_cache_path(home_path: Optional[Path] = None) -> Path:
    """Path to the legacy plaintext on-disk cache (removed on migration)."""
    return _DISK_CACHE.path(home_path)


def _encrypted_disk_cache_path(home_path: Optional[Path] = None) -> Path:
    """Path to the encrypted on-disk cache."""
    from agent.secret_sources._cache import resolve_cache_home

    return resolve_cache_home(home_path) / "cache" / _ENCRYPTED_CACHE_BASENAME


def _b64e(raw: bytes) -> str:
    """Base64-encode bytes for JSON payload storage."""
    import base64

    return base64.b64encode(raw).decode("ascii")


def _b64d(text: str) -> bytes:
    """Base64-decode a JSON payload field."""
    import base64

    return base64.b64decode(text)


def _auth_key_material(token_env: str) -> str:
    """Return the key-derivation material for the encrypted cache.

    The same auth material that feeds ``_auth_fingerprint`` — the
    service-account token plus OP_ACCOUNT / OP_CONNECT / OP_SESSION_*
    identity vars.  The raw value is used to derive the AES key and is
    never persisted or logged.
    """
    source_env = get_source_environment()
    parts: List[str] = [
        f"token={source_env.get(token_env, '')}",
        f"account={source_env.get('OP_ACCOUNT', '')}",
        f"connect_host={source_env.get('OP_CONNECT_HOST', '')}",
        f"connect_token={source_env.get('OP_CONNECT_TOKEN', '')}",
    ]
    for key in sorted(source_env):
        if key.startswith("OP_SESSION_"):
            parts.append(f"{key}={source_env[key]}")
    return "\n".join(parts)


def _derive_encrypted_cache_key(auth_material: str, salt: bytes) -> bytes:
    """Derive the local cache encryption key from the auth material."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=_ENCRYPTED_CACHE_INFO,
    ).derive(auth_material.encode("utf-8"))


def _write_encrypted_disk_cache(
    *,
    cache_key: _CacheKey,
    auth_material: str,
    entry: CachedFetch,
    home_path: Optional[Path] = None,
) -> bool:
    """Persist an encrypted cache entry atomically.

    Best-effort by design: cache write failure must never block a fresh
    ``op read``.  The raw auth material is not stored; it only derives the
    AES key.  A successful write removes the legacy plaintext cache so
    stale secret values cannot remain on disk.

    Returns True when the encrypted write (and legacy-plaintext removal)
    completed; False when a non-fatal error was swallowed.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    path = _encrypted_disk_cache_path(home_path)
    try:
        cache_dir = path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(cache_dir, 0o700)
        except OSError:
            pass
        salt = os.urandom(16)
        nonce = os.urandom(12)
        serialized_key = _disk_key_str(cache_key)
        key = _derive_encrypted_cache_key(auth_material, salt)
        plaintext = json.dumps(
            {"secrets": entry.secrets, "fetched_at": entry.fetched_at},
            separators=(",", ":"),
        ).encode("utf-8")
        ciphertext = AESGCM(key).encrypt(
            nonce, plaintext, serialized_key.encode("utf-8")
        )
        payload = {
            "version": _ENCRYPTED_CACHE_VERSION,
            "key": serialized_key,
            "salt": _b64e(salt),
            "nonce": _b64e(nonce),
            "ciphertext": _b64e(ciphertext),
        }
        fd, tmp = tempfile.mkstemp(
            prefix=".op_cache_enc_", suffix=".tmp", dir=str(cache_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
            # A successful encrypted write completes migration; remove the
            # legacy plaintext cache so stale secrets cannot remain on disk.
            try:
                _disk_cache_path(home_path).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
            return True
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:  # noqa: BLE001 — best-effort cache only
        return False


def _read_encrypted_disk_cache(
    *,
    cache_key: _CacheKey,
    auth_material: str,
    max_age_seconds: float,
    home_path: Optional[Path] = None,
) -> Optional[CachedFetch]:
    """Return a decrypted encrypted-cache entry if it matches and is in-window."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if max_age_seconds <= 0:
        return None
    path = _encrypted_disk_cache_path(home_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        serialized_key = _disk_key_str(cache_key)
        if payload.get("version") != _ENCRYPTED_CACHE_VERSION:
            return None
        if payload.get("key") != serialized_key:
            return None
        salt = _b64d(str(payload.get("salt", "")))
        nonce = _b64d(str(payload.get("nonce", "")))
        ciphertext = _b64d(str(payload.get("ciphertext", "")))
        key = _derive_encrypted_cache_key(auth_material, salt)
        raw = AESGCM(key).decrypt(
            nonce, ciphertext, serialized_key.encode("utf-8")
        )
        inner = json.loads(raw.decode("utf-8"))
        if not isinstance(inner, dict):
            return None
        secrets = inner.get("secrets")
        inner_fetched_at = inner.get("fetched_at")
        if not isinstance(secrets, dict) or not isinstance(inner_fetched_at, (int, float)):
            return None
        entry_age = time.time() - float(inner_fetched_at)
        if entry_age < 0 or entry_age > max_age_seconds:
            return None
        typed = {
            k: v for k, v in secrets.items()
            if isinstance(k, str) and isinstance(v, str)
        }
        return CachedFetch(secrets=typed, fetched_at=float(inner_fetched_at))
    except Exception:  # noqa: BLE001 — cache miss on parse/decrypt/I/O errors
        return None


def _remove_legacy_plaintext_cache(home_path: Optional[Path] = None) -> None:
    """Delete the legacy plaintext op_cache.json if it exists.

    Best-effort: never raises.  Called on the memory-only path so a
    pre-upgrade plaintext cache does not survive the encrypted-only
    policy.
    """
    try:
        _disk_cache_path(home_path).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Reference validation + fingerprinting
# ---------------------------------------------------------------------------


def _validate_references(
    references: Optional[Dict[str, str]],
) -> Tuple[Dict[str, str], List[str]]:
    """Return ``(valid_refs, warnings)`` from an ``env`` mapping.

    A reference is kept only if its target env-var name is a valid POSIX
    name and the value is a stripped ``op://…`` reference string.  Everything
    else produces a warning and is dropped (never fatal).
    """
    valid: Dict[str, str] = {}
    warnings: List[str] = []
    for name, ref in (references or {}).items():
        if not is_valid_env_name(name):
            warnings.append(f"Skipping {name!r}: not a valid env-var name")
            continue
        if not isinstance(ref, str):
            warnings.append(f"Skipping {name!r}: reference is not a string")
            continue
        cleaned = ref.strip()
        if not cleaned.startswith("op://"):
            warnings.append(
                f"Skipping {name!r}: {ref!r} is not an op:// secret reference"
            )
            continue
        valid[name] = cleaned
    return valid, warnings


def _auth_fingerprint(token_env: str) -> str:
    """SHA-256 prefix over the auth material `op` would use.

    Folds in the service-account token, ``OP_ACCOUNT``, the 1Password Connect
    ``OP_CONNECT_HOST``/``OP_CONNECT_TOKEN``, and *all* ``OP_SESSION_*`` vars
    (the names `op` actually exports for interactive sessions —
    ``OP_SESSION_<account_shorthand>``).  Signing out and into a different
    identity therefore changes the cache key, so a value cached under a
    previous identity is never served under a new one.  Never logged or
    displayed; the raw token never leaves this hash.
    """
    source_env = get_source_environment()
    parts: List[str] = [
        f"token={source_env.get(token_env, '')}",
        f"account={source_env.get('OP_ACCOUNT', '')}",
        f"connect_host={source_env.get('OP_CONNECT_HOST', '')}",
        f"connect_token={source_env.get('OP_CONNECT_TOKEN', '')}",
    ]
    for key in sorted(source_env):
        if key.startswith("OP_SESSION_"):
            parts.append(f"{key}={source_env[key]}")
    material = "\n".join(parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _refs_fingerprint(references: Dict[str, str]) -> str:
    """SHA-256 prefix over the configured name→reference mapping."""
    material = "\n".join(f"{name}={references[name]}" for name in sorted(references))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def find_op(binary_path: str = "") -> Optional[Path]:
    """Resolve a usable ``op`` binary, or None.

    When ``binary_path`` is set it is used verbatim and PATH is NOT consulted
    — pinning an absolute path is a way to avoid trusting whatever ``op`` shows
    up first on ``PATH``.  A pinned-but-missing path returns None (the caller
    surfaces a clear error) rather than silently falling back.
    """
    if binary_path:
        pinned = Path(binary_path)
        if pinned.exists() and os.access(pinned, os.X_OK):
            return pinned
        return None
    found = shutil.which("op")
    return Path(found) if found else None


# ---------------------------------------------------------------------------
# `op read` invocation
# ---------------------------------------------------------------------------


def _scrub(text: str) -> str:
    """Remove ANSI control sequences and trim, for safe message surfacing."""
    from tools.ansi_strip import strip_ansi

    # strip_ansi removes well-formed sequences; drop any stray lone ESC too.
    return strip_ansi(text).replace("\x1b", "").strip()


def _op_child_env(token_value: str) -> Dict[str, str]:
    """Build a minimal allowlisted environment for the ``op`` child process."""
    source_env = get_source_environment()
    env: Dict[str, str] = {}
    for key in _OP_ENV_ALLOWLIST:
        val = source_env.get(key)
        if val is not None:
            env[key] = val
    # Desktop / interactive session credentials.
    for key, val in source_env.items():
        if key.startswith("OP_SESSION_"):
            env[key] = val
    # `op` reads OP_SERVICE_ACCOUNT_TOKEN regardless of which env var the user
    # configured Hermes to source it from, so normalize to that name here.
    if token_value:
        env["OP_SERVICE_ACCOUNT_TOKEN"] = token_value
    env["NO_COLOR"] = "1"
    return env


def _run_op_read(
    op: Path,
    reference: str,
    *,
    account: str = "",
    token_value: str = "",
) -> str:
    """Resolve a single ``op://`` reference to its value.

    Raises :class:`RuntimeError` on any failure — including a ``returncode 0``
    with empty output, which would otherwise silently clobber a good
    ``.env``/shell credential with ``""``.
    """
    cmd: List[str] = [str(op), "read"]
    if account:
        cmd += ["--account", account]
    # `--` terminates option parsing so a reference can never be mis-parsed as
    # an `op` flag even if validation is ever loosened.
    cmd += ["--", reference]

    try:
        proc = subprocess.run(  # noqa: S603 — op path is user-trusted, argv list
            cmd,
            env=_op_child_env(token_value),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_OP_RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"op read timed out after {_OP_RUN_TIMEOUT}s for {reference!r}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke op: {exc}") from exc

    if proc.returncode != 0:
        err = _scrub(proc.stderr or "")[:200]
        if err:
            raise RuntimeError(f"op read failed for {reference!r}: {err}")
        raise RuntimeError(
            f"op read exited {proc.returncode} for {reference!r}"
        )

    # `op` appends a trailing newline; strip only that so a value with
    # intentional internal/edge spaces survives.  But a value that is empty or
    # whitespace-only is treated as empty: applying it would silently clobber a
    # good .env/shell credential with effectively nothing.
    value = (proc.stdout or "").rstrip("\r\n")
    if not value.strip():
        raise RuntimeError(f"op read returned an empty value for {reference!r}")
    return value


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_onepassword_secrets(
    *,
    references: Dict[str, str],
    account: str = "",
    token_env: str = _DEFAULT_TOKEN_ENV,
    binary: Optional[Path] = None,
    binary_path: str = "",
    use_cache: bool = True,
    cache_ttl_seconds: float = 300,
    home_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Resolve ``references`` (name → ``op://…``) to ``(secrets, warnings)``.

    Raises :class:`RuntimeError` only when no ``op`` binary is available — a
    fatal "can't fetch anything" condition.  Per-reference failures (expired
    auth, bad reference, empty value) are collected as warnings and the
    reference is dropped, so one bad entry never sinks the rest.

    Only a complete, error-free pull is cached, so a transient auth failure
    isn't frozen in for the whole TTL window.
    """
    valid, warnings = _validate_references(references)
    if not valid:
        return {}, warnings

    token_value = get_source_environment().get(token_env, "").strip()
    auth_material = _auth_key_material(token_env)
    cache_key: _CacheKey = (
        _auth_fingerprint(token_env),
        account or "",
        str(home_path) if home_path is not None else "",
        _refs_fingerprint(valid),
    )

    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return dict(cached.secrets), warnings
        # Encrypted-only storage policy: the disk cache is ALWAYS AES-GCM
        # encrypted.  With encryption disabled there is NO disk read at all
        # — plaintext is never consulted.  A legacy plaintext op_cache.json
        # from an older Hermes is re-encrypted and removed on the way out.
        disk_cached = _read_encrypted_disk_cache(
            cache_key=cache_key,
            auth_material=auth_material,
            max_age_seconds=cache_ttl_seconds,
            home_path=home_path,
        )
        if disk_cached is None:
            disk_cached = _migrate_legacy_plaintext_cache(
                cache_key=cache_key,
                auth_material=auth_material,
                max_age_seconds=cache_ttl_seconds,
                home_path=home_path,
            )
        if disk_cached is not None:
            # Promote into L1 so later fetches in this process skip the disk read.
            _CACHE[cache_key] = disk_cached
            return dict(disk_cached.secrets), warnings

        # Memory-only mode (cache_ttl_seconds <= 0): neither disk read above
        # can return an entry (both short-circuit on a non-positive TTL), so a
        # pre-upgrade plaintext op_cache.json would otherwise survive untouched.
        # Sweep it — plaintext must not survive the encrypted-only policy even
        # when disk persistence is disabled.
        if cache_ttl_seconds <= 0:
            _remove_legacy_plaintext_cache(home_path)

    op = binary or find_op(binary_path)
    if op is None:
        raise RuntimeError(
            "op CLI not found.  Install the 1Password CLI "
            "(https://developer.1password.com/docs/cli/get-started/) or set "
            "secrets.onepassword.binary_path to its absolute location."
        )

    secrets: Dict[str, str] = {}
    read_errors = 0
    for name in sorted(valid):
        try:
            secrets[name] = _run_op_read(
                op, valid[name], account=account, token_value=token_value
            )
        except RuntimeError as exc:
            warnings.append(str(exc))
            read_errors += 1

    if use_cache and not read_errors and secrets:
        entry = CachedFetch(secrets=dict(secrets), fetched_at=time.time())
        _CACHE[cache_key] = entry
        # Memory-only (ttl <= 0): L1 only — never persist to disk.
        if cache_ttl_seconds > 0:
            _write_encrypted_disk_cache(
                cache_key=cache_key,
                auth_material=auth_material,
                entry=entry,
                home_path=home_path,
            )

    return secrets, warnings


def _migrate_legacy_plaintext_cache(
    *,
    cache_key: _CacheKey,
    auth_material: str,
    max_age_seconds: float,
    home_path: Optional[Path] = None,
) -> Optional[CachedFetch]:
    """Re-encrypt a legacy plaintext cache entry and return it.

    Older Hermes versions persisted the cross-process cache as plaintext
    ``op_cache.json``.  With the encrypted-only policy, a leftover
    plaintext file is migrated on first read: the entry is read from the
    legacy DiskCache (honoring ``max_age_seconds``), written to the
    encrypted cache — whose writer removes the plaintext file on success
    (see ``_write_encrypted_disk_cache``) — and returned to the caller so
    the cached values are served without a live fetch.

    Returns None when there is no legacy plaintext entry for ``cache_key``
    (or it is outside ``max_age_seconds``); the caller then falls through
    to a live fetch.  Best-effort by design: a migration failure is
    treated as a plain cache miss and never raises.
    """
    try:
        legacy = _DISK_CACHE.read(cache_key, max_age_seconds, home_path)
        if legacy is None:
            return None
        if _write_encrypted_disk_cache(
            cache_key=cache_key,
            auth_material=auth_material,
            entry=legacy,
            home_path=home_path,
        ):
            return legacy
        return None
    except Exception:  # noqa: BLE001 — migration must never block startup
        return None


# ---------------------------------------------------------------------------
# Public entry point — called from hermes_cli.env_loader
# ---------------------------------------------------------------------------


def apply_onepassword_secrets(
    *,
    enabled: bool,
    env: Optional[Dict[str, str]] = None,
    account: str = "",
    service_account_token_env: str = _DEFAULT_TOKEN_ENV,
    binary_path: str = "",
    override_existing: bool = True,
    cache_ttl_seconds: float = 300,
    home_path: Optional[Path] = None,
) -> FetchResult:
    """Resolve configured ``op://`` references and set them on ``os.environ``.

    Called by ``load_hermes_dotenv()`` after the .env files have loaded.
    Intentionally defensive — any failure returns a :class:`FetchResult` with
    ``error`` set (or surfaces warnings); it never raises.

    Parameters mirror the ``secrets.onepassword.*`` config keys so the caller
    can splat the dict in.  References that are already satisfied by the
    current environment (when ``override_existing`` is false) are skipped
    *before* fetching, so ``op`` is never invoked for a value that would be
    discarded.
    """
    result = FetchResult()

    if not enabled:
        return result

    valid, warnings = _validate_references(env)
    result.warnings.extend(warnings)

    # Skip-before-fetch: never resolve a reference we'd only throw away.
    refs_to_fetch: Dict[str, str] = {}
    for name, ref in valid.items():
        if name == service_account_token_env:
            # Never let a resolved secret clobber the very token used to auth.
            result.skipped.append(name)
            continue
        if not override_existing and os.environ.get(name):
            result.skipped.append(name)
            continue
        refs_to_fetch[name] = ref

    if not refs_to_fetch:
        return result

    binary = find_op(binary_path)
    result.binary_path = binary
    if binary is None:
        if binary_path:
            result.error = (
                f"secrets.onepassword.binary_path ({binary_path!r}) is not an "
                "executable op binary."
            )
        else:
            result.error = (
                "secrets.onepassword.enabled is true but the op CLI was not "
                "found on PATH.  Install it "
                "(https://developer.1password.com/docs/cli/get-started/) or set "
                "secrets.onepassword.binary_path."
            )
        return result

    try:
        secrets, fetch_warnings = fetch_onepassword_secrets(
            references=refs_to_fetch,
            account=account,
            token_env=service_account_token_env,
            binary=binary,
            cache_ttl_seconds=cache_ttl_seconds,
            home_path=home_path,
        )
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(fetch_warnings)

    for name, value in secrets.items():
        # The token-var and override guards already filtered refs_to_fetch, but
        # re-check defensively in case the fetch layer ever returns extras.
        if name == service_account_token_env:
            if name not in result.skipped:
                result.skipped.append(name)
            continue
        if not override_existing and os.environ.get(name):
            if name not in result.skipped:
                result.skipped.append(name)
            continue
        os.environ[name] = value
        result.applied.append(name)

    return result


# ---------------------------------------------------------------------------
# SecretSource adapter — the registry-facing wrapper around this module.
# ---------------------------------------------------------------------------


class OnePasswordSource(SecretSource):
    """1Password as a registered secret source.

    Thin adapter over the module's fetch machinery.  ``fetch()`` only
    *fetches* — precedence, override semantics, conflict warnings, and
    the ``os.environ`` writes are the orchestrator's job
    (see ``agent.secret_sources.registry.apply_all``).

    1Password is a **mapped** source: the user explicitly binds each env
    var to an ``op://`` reference under ``secrets.onepassword.env``, so
    its claims outrank bulk sources (e.g. a Bitwarden project dump) on
    contested vars.
    """

    name = "onepassword"
    label = "1Password"
    shape = "mapped"
    scheme = "op"

    def override_existing(self, cfg: dict) -> bool:
        # Default True: an explicit VAR→op:// binding is the strongest
        # user intent there is — leaving a stale .env line in place
        # should not silently defeat it (same rotation rationale as
        # Bitwarden).
        return bool(isinstance(cfg, dict) and cfg.get("override_existing", True))

    def protected_env_vars(self, cfg: dict):
        token_env = _DEFAULT_TOKEN_ENV
        if isinstance(cfg, dict):
            token_env = str(cfg.get("service_account_token_env") or token_env)
        return frozenset({token_env})

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Master switch", "default": False},
            "env": {
                "description": "Map of ENV_VAR -> op://vault/item/field reference",
                "default": {},
            },
            "account": {
                "description": "op --account shorthand (empty = default account)",
                "default": "",
            },
            "service_account_token_env": {
                "description": "Env var holding the service-account token "
                               "(unset = desktop/interactive session)",
                "default": _DEFAULT_TOKEN_ENV,
            },
            "binary_path": {
                "description": "Pin the op binary (empty = resolve via PATH)",
                "default": "",
            },
            "cache_ttl_seconds": {
                "description": "Disk+memory cache TTL; 0 disables",
                "default": 300,
            },
            "override_existing": {
                "description": "Resolved values overwrite .env/shell values",
                "default": True,
            },
        }

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        cfg = cfg if isinstance(cfg, dict) else {}
        result = FetchResult()

        env_map = cfg.get("env")
        valid, warnings = _validate_references(
            env_map if isinstance(env_map, dict) else None
        )
        result.warnings.extend(warnings)
        if not valid:
            if not warnings:
                result.error = (
                    "secrets.onepassword.enabled is true but the env: map is "
                    "empty.  Add ENV_VAR: op://vault/item/field entries."
                )
                result.error_kind = ErrorKind.NOT_CONFIGURED
            return result

        binary_path = str(cfg.get("binary_path") or "")
        binary = find_op(binary_path)
        result.binary_path = binary
        if binary is None:
            if binary_path:
                result.error = (
                    f"secrets.onepassword.binary_path ({binary_path!r}) is "
                    "not an executable op binary."
                )
            else:
                result.error = (
                    "secrets.onepassword.enabled is true but the op CLI was "
                    "not found on PATH.  Install it "
                    "(https://developer.1password.com/docs/cli/get-started/) "
                    "or set secrets.onepassword.binary_path."
                )
            result.error_kind = ErrorKind.BINARY_MISSING
            return result

        try:
            ttl = float(cfg.get("cache_ttl_seconds", 300))
        except (TypeError, ValueError):
            ttl = 300.0

        try:
            secrets, fetch_warnings = fetch_onepassword_secrets(
                references=valid,
                account=str(cfg.get("account") or ""),
                token_env=str(
                    cfg.get("service_account_token_env") or _DEFAULT_TOKEN_ENV
                ),
                binary=binary,
                cache_ttl_seconds=ttl,
                home_path=home_path,
            )
        except RuntimeError as exc:
            result.error = str(exc)
            result.error_kind = _classify_op_error(str(exc))
            return result

        result.secrets = secrets
        result.warnings.extend(fetch_warnings)
        return result

    def remediation(self, kind, cfg: dict) -> str:
        if kind in (ErrorKind.AUTH_FAILED, ErrorKind.AUTH_EXPIRED):
            token_env = _DEFAULT_TOKEN_ENV
            if isinstance(cfg, dict):
                token_env = str(cfg.get("service_account_token_env") or token_env)
            return (
                "Run `hermes secrets onepassword token` to paste a fresh "
                f"service-account token ({token_env}), or `op signin` for an "
                "interactive session."
            )
        if kind == ErrorKind.BINARY_MISSING:
            return (
                "Install the 1Password CLI "
                "(https://developer.1password.com/docs/cli/get-started/) or "
                "set secrets.onepassword.binary_path."
            )
        return super().remediation(kind, cfg)


def _classify_op_error(message: str) -> ErrorKind:
    """Best-effort mapping of op failure text onto the shared taxonomy."""
    lowered = message.lower()
    if "timed out" in lowered:
        return ErrorKind.TIMEOUT
    if "not found on path" in lowered or "not an executable" in lowered \
            or "failed to invoke" in lowered:
        return ErrorKind.BINARY_MISSING
    if any(tok in lowered for tok in ("unauthorized", "not signed in",
                                      "session expired", "authentication",
                                      "401", "403")):
        return ErrorKind.AUTH_FAILED
    if "empty value" in lowered:
        return ErrorKind.EMPTY_VALUE
    if any(tok in lowered for tok in ("network", "connection", "resolve host",
                                      "dns")):
        return ErrorKind.NETWORK
    return ErrorKind.INTERNAL


# ---------------------------------------------------------------------------
# Test hook — used by hermetic tests to flush the cache between cases.
# ---------------------------------------------------------------------------


def clear_caches(home_path: Optional[Path] = None) -> None:
    """Drop in-process AND disk caches.

    Used after a token rotation (`hermes secrets onepassword token`) so
    the next startup resolves fresh with the new credential instead of
    serving values cached under the old token's fingerprint.
    """
    _CACHE.clear()
    _DISK_CACHE.clear(home_path)
    try:
        _encrypted_disk_cache_path(home_path).unlink()
    except (FileNotFoundError, OSError):
        pass


def _reset_cache_for_tests(home_path: Optional[Path] = None) -> None:
    """Clear in-process AND disk caches.

    Tests can pass ``home_path`` to scope the disk cleanup to a tmpdir.
    Without it we fall back to the same default resolution as the writer.
    """
    clear_caches(home_path)
