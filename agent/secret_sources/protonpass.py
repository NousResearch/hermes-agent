"""Proton Pass (`pass-cli`) secret source.

Resolve provider credentials from Proton Pass ``pass://vault/item/field``
references at process startup so they don't have to live in plaintext in
``~/.hermes/.env``.

Design summary
--------------

* Users map environment-variable names to Proton Pass secret references in
  ``secrets.protonpass.env``::

      secrets:
        protonpass:
          enabled: true
          env:
            OPENAI_API_KEY: "pass://Private/OpenAI/api key"
            ANTHROPIC_API_KEY: "pass://Private/Anthropic/credential"

* After ``.env`` loads, each reference is resolved and injected into
  ``os.environ`` (the same point in startup as the Bitwarden / 1Password
  sources).  Resolution shells out to ``pass-cli run --no-masking -- printenv
  <NAME>`` with the reference placed in the child's environment: ``run``
  substitutes the ``pass://`` URI for the real value before exec'ing the
  command, and ``--no-masking`` keeps that value from being replaced with
  ``<concealed by Proton Pass>`` on stdout.  This uses only documented
  behaviour and sidesteps the decorated/uncertain output of ``item view``.
* Authentication uses Proton Pass's persistent session model.  Unlike the
  1Password CLI's per-invocation service-account token, ``pass-cli`` logs in
  once (``pass-cli login``, which reads ``PROTON_PASS_PERSONAL_ACCESS_TOKEN``)
  and stores a session in a platform-specific keyring.  We rely on an existing
  session when one is present and only attempt a login — using the configured
  personal-access-token env var — when a resolve fails for an auth-shaped
  reason.  Hermes never downloads ``pass-cli``.
* Failures NEVER block startup.  A missing ``pass-cli`` binary, a login
  failure, a bad reference, or an empty value each surface a one-line warning
  and Hermes continues with whatever credentials ``.env`` already had.

The atomic-write / ``0600`` / TTL cache mechanics are kept self-contained in
this module (mirroring ``bitwarden.py`` on ``main``) so the backend merges
without depending on the shared ``agent.secret_sources._cache`` substrate
introduced in the open 1Password PR (#36896); it can be rebased onto that
substrate once it lands.  The disk file holds only resolved secret *values*;
auth material is fingerprinted, never stored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# How long to wait for a single `pass-cli` subprocess, in seconds.
_PASS_RUN_TIMEOUT = 30

# Default env var the `pass-cli login` command reads for non-interactive
# personal-access-token auth.  Users can point `personal_access_token_env` at a
# different name; we always export the value to the child as
# PROTON_PASS_PERSONAL_ACCESS_TOKEN, which is what `pass-cli` itself looks for.
_DEFAULT_TOKEN_ENV = "PROTON_PASS_PERSONAL_ACCESS_TOKEN"

# Env var name `pass-cli` reads its personal access token from.
_PASS_TOKEN_ENV = "PROTON_PASS_PERSONAL_ACCESS_TOKEN"

# Internal env var we set the pass:// reference under before invoking
# `pass-cli run -- <python> -c <echo script>`.  `run` resolves the reference
# found in this variable and the wrapped command echoes the resolved value
# back to us.  We wrap the *current* Python interpreter rather than a shell
# builtin like `printenv` (POSIX-only) so resolution works identically on
# Windows, macOS, and Linux.
_RESOLVE_SENTINEL = "HERMES_PASS_CLI_RESOLVE"

# Echo script run under the current interpreter: write the (now-resolved)
# sentinel value to stdout verbatim, with no added newline so secrets with
# meaningful trailing whitespace survive.
_ECHO_SCRIPT = (
    "import os,sys;sys.stdout.write(os.environ.get(%r,''))" % _RESOLVE_SENTINEL
)

# `run` masks resolved secrets in child stdout/stderr by default; we pass
# --no-masking to read the value, so this marker should never appear, but we
# treat it as an empty/failed resolve defensively.
_CONCEALED_MARKER = "<concealed by Proton Pass>"

# Strip whole ANSI CSI sequences (colour, cursor moves, line erases) from any
# `pass-cli` diagnostic we surface — not just the lone ESC byte — so a control
# sequence can't reposition the cursor or hide text after a redaction marker.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# stderr fragments that indicate an auth/session problem rather than a bad
# reference — these are the cases where attempting a `pass-cli login` and
# retrying is worthwhile.
_AUTH_ERROR_HINTS = (
    "not logged in",
    "not signed in",
    "no session",
    "session expired",
    "session has expired",
    "unauthorized",
    "unauthenticated",
    "authentication",
    "please log in",
    "please login",
    "log in first",
)

# Env vars the `pass-cli` child actually needs.  We build a minimal allowlisted
# env rather than copying all of os.environ (which, post-dotenv, holds every
# provider credential) into the child — tighter blast radius if `pass-cli` or
# anything it execs ever misbehaves.  HOME / XDG_* / DBUS are needed so the
# persistent session keyring is reachable; the token is added only for login.
_PASS_ENV_ALLOWLIST = (
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
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
)


# ---------------------------------------------------------------------------
# Result + cache dataclasses (self-contained; mirror bitwarden.py on main)
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """Outcome of a single Proton Pass pull.

    ``error`` is set only for *fatal* conditions (nothing was fetched);
    non-fatal issues go into ``warnings`` so the values that did resolve are
    still applied.  ``ok`` is the convenience inverse of ``error``.
    """

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)   # set into os.environ
    skipped: List[str] = field(default_factory=list)   # already set / protected
    warnings: List[str] = field(default_factory=list)  # non-fatal issues
    error: Optional[str] = None                        # fatal: nothing fetched
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class _CachedFetch:
    secrets: Dict[str, str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


def is_valid_env_name(name: str) -> bool:
    """Return True if ``name`` is a usable POSIX environment-variable name.

    Must be non-empty, start with a letter or underscore, and contain only
    alphanumerics and underscores.  Used to drop secret names that couldn't be
    exported (e.g. ``"has spaces"`` or ``"1LEADING_DIGIT"``).
    """
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# In-process cache.  The key folds in str(home_path) so a HERMES_HOME switch
# inside one long-lived process (e.g. the gateway) can't return another
# profile's secrets from L1.
_CacheKey = Tuple[str, str, str]  # (auth_fp, home, refs_fp)
_CACHE: Dict[_CacheKey, _CachedFetch] = {}

_DISK_CACHE_BASENAME = "protonpass_cache.json"


def _disk_cache_path(home_path: Optional[Path] = None) -> Path:
    """Path to the on-disk cache (exposed for tests and direct callers)."""
    if home_path is None:
        home_path = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return home_path / "cache" / _DISK_CACHE_BASENAME


def _disk_key_str(cache_key: _CacheKey) -> str:
    """Serialize a cache key for on-disk storage, omitting home_path.

    The disk file is already partitioned by home (it lives under
    ``<home>/cache/``), so the path provides the home dimension.
    """
    auth_fp, _home, refs_fp = cache_key
    return f"{auth_fp}|{refs_fp}"


def _read_disk_cache(
    cache_key: _CacheKey, ttl_seconds: float, home_path: Optional[Path] = None
) -> Optional[_CachedFetch]:
    """Return a fresh cached entry for ``cache_key`` from disk, or None.

    Best-effort: any I/O or parse error, a key mismatch, or a stale entry all
    return None so the caller re-fetches.
    """
    if ttl_seconds <= 0:
        return None
    path = _disk_cache_path(home_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("key") != _disk_key_str(cache_key):
        return None
    secrets = payload.get("secrets")
    fetched_at = payload.get("fetched_at")
    if not isinstance(secrets, dict) or not isinstance(fetched_at, (int, float)):
        return None
    typed: Dict[str, str] = {
        k: v for k, v in secrets.items() if isinstance(k, str) and isinstance(v, str)
    }
    entry = _CachedFetch(secrets=typed, fetched_at=float(fetched_at))
    if not entry.is_fresh(ttl_seconds):
        return None
    return entry


def _write_disk_cache(
    cache_key: _CacheKey,
    entry: _CachedFetch,
    ttl_seconds: float,
    home_path: Optional[Path] = None,
) -> None:
    """Persist ``entry`` for ``cache_key`` atomically at mode ``0600``.

    No-op when ``ttl_seconds <= 0`` (caching genuinely off) or on any I/O error
    — the next invocation just re-fetches.
    """
    if ttl_seconds <= 0:
        return
    path = _disk_cache_path(home_path)
    try:
        cache_dir = path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        # mkdir's mode is umask-subject; chmod the dir to 0700 so cache
        # metadata isn't exposed if HERMES_HOME is ever made traversable.
        try:
            os.chmod(cache_dir, 0o700)
        except OSError:
            pass
        payload = {
            "key": _disk_key_str(cache_key),
            "secrets": entry.secrets,
            "fetched_at": entry.fetched_at,
        }
        fd, tmp = tempfile.mkstemp(
            prefix=".protonpass_cache_", suffix=".tmp", dir=str(cache_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        pass  # best-effort — a disk-cache miss next invocation is fine


# ---------------------------------------------------------------------------
# Reference validation + fingerprinting
# ---------------------------------------------------------------------------


def _validate_references(
    references: Optional[Dict[str, str]],
) -> Tuple[Dict[str, str], List[str]]:
    """Return ``(valid_refs, warnings)`` from an ``env`` mapping.

    A reference is kept only if its target env-var name is a valid POSIX name
    and the value is a stripped ``pass://…`` reference string.  Everything else
    produces a warning and is dropped (never fatal).
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
        if not cleaned.startswith("pass://"):
            warnings.append(
                f"Skipping {name!r}: {ref!r} is not a pass:// secret reference"
            )
            continue
        valid[name] = cleaned
    return valid, warnings


def _auth_fingerprint(token_env: str) -> str:
    """SHA-256 prefix over the auth material a login would use.

    Folds in the configured personal-access-token value.  Rotating the token to
    a different identity therefore changes the cache key, so a value cached
    under a previous identity is never served under a new one.  The session
    itself lives in an external keyring (not in os.environ), so it can't be
    fingerprinted here — the token is the stable identity proxy.  Never logged
    or displayed; the raw token never leaves this hash.
    """
    material = f"token={os.environ.get(token_env, '')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _refs_fingerprint(references: Dict[str, str]) -> str:
    """SHA-256 prefix over the configured name→reference mapping."""
    material = "\n".join(f"{name}={references[name]}" for name in sorted(references))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def find_pass_cli(binary_path: str = "") -> Optional[Path]:
    """Resolve a usable ``pass-cli`` binary, or None.

    When ``binary_path`` is set it is used verbatim and PATH is NOT consulted —
    pinning an absolute path is a way to avoid trusting whatever ``pass-cli``
    shows up first on ``PATH``.  A pinned-but-missing path returns None (the
    caller surfaces a clear error) rather than silently falling back.
    """
    if binary_path:
        pinned = Path(binary_path)
        if pinned.exists() and os.access(pinned, os.X_OK):
            return pinned
        return None
    found = shutil.which("pass-cli")
    return Path(found) if found else None


# ---------------------------------------------------------------------------
# `pass-cli` invocation
# ---------------------------------------------------------------------------


def _scrub(text: str) -> str:
    """Remove ANSI control sequences and trim, for safe message surfacing."""
    return _ANSI_CSI_RE.sub("", text).replace("\x1b", "").strip()


def _looks_like_auth_error(stderr: str) -> bool:
    low = stderr.lower()
    return any(hint in low for hint in _AUTH_ERROR_HINTS)


def _pass_child_env(token_value: str = "") -> Dict[str, str]:
    """Build a minimal allowlisted environment for the ``pass-cli`` child.

    ``token_value`` is injected as PROTON_PASS_PERSONAL_ACCESS_TOKEN only for
    login; resolve calls pass an empty token and rely on the persistent
    session.
    """
    env: Dict[str, str] = {}
    for key in _PASS_ENV_ALLOWLIST:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    if token_value:
        env[_PASS_TOKEN_ENV] = token_value
    env["NO_COLOR"] = "1"
    return env


class _AuthError(RuntimeError):
    """A resolve failed for an auth/session reason — a login retry may help."""


def _pass_login(pass_cli: Path, token_value: str) -> None:
    """Establish a Proton Pass session from a personal access token.

    Raises :class:`RuntimeError` on failure.  Runs fully non-interactively:
    the token is passed via the child environment (never argv, so it stays out
    of the process list) and stdin is closed so a missing/invalid token fails
    fast instead of blocking on an interactive prompt.
    """
    if not token_value:
        raise RuntimeError(
            "no personal access token available to log in (set the env var named "
            "by secrets.protonpass.personal_access_token_env)"
        )
    cmd = [str(pass_cli), "login"]
    try:
        proc = subprocess.run(  # noqa: S603 — pass-cli path is user-trusted, argv list
            cmd,
            env=_pass_child_env(token_value),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PASS_RUN_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"pass-cli login timed out after {_PASS_RUN_TIMEOUT}s"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke pass-cli: {exc}") from exc

    if proc.returncode != 0:
        err = _scrub(proc.stderr or "")[:200]
        raise RuntimeError(err or f"pass-cli login exited {proc.returncode}")


def _run_pass_resolve(pass_cli: Path, reference: str) -> str:
    """Resolve a single ``pass://`` reference to its value.

    Uses ``pass-cli run --no-masking -- <python> -c <echo script>`` with the
    reference placed in the child env: ``run`` substitutes the resolved value
    into the env var and the wrapped interpreter echoes it back.  Raises
    :class:`_AuthError` on an auth-shaped failure (so the caller can attempt a
    login + retry) and :class:`RuntimeError` on any other failure — including a
    ``returncode 0`` with empty/concealed output, which would otherwise
    silently clobber a good ``.env``/shell credential.
    """
    cmd = [
        str(pass_cli),
        "run",
        "--no-masking",
        "--",
        sys.executable,
        "-c",
        _ECHO_SCRIPT,
    ]
    child_env = _pass_child_env()
    child_env[_RESOLVE_SENTINEL] = reference

    try:
        proc = subprocess.run(  # noqa: S603 — pass-cli path is user-trusted, argv list
            cmd,
            env=child_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PASS_RUN_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"pass-cli run timed out after {_PASS_RUN_TIMEOUT}s for {reference!r}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke pass-cli: {exc}") from exc

    if proc.returncode != 0:
        err = _scrub(proc.stderr or "")[:200]
        msg = (
            f"pass-cli run failed for {reference!r}: {err}"
            if err
            else f"pass-cli run exited {proc.returncode} for {reference!r}"
        )
        if _looks_like_auth_error(err):
            raise _AuthError(msg)
        raise RuntimeError(msg)

    # The echo script writes the value verbatim with no added newline, so we
    # take stdout as-is — a value with meaningful internal/trailing whitespace
    # survives intact.  An empty/whitespace-only or still-concealed value is
    # treated as empty: applying it would silently clobber a good credential
    # with effectively nothing.
    value = proc.stdout or ""
    if not value.strip() or value.strip() == _CONCEALED_MARKER:
        raise RuntimeError(f"pass-cli returned an empty value for {reference!r}")
    return value


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _resolve_batch(
    pass_cli: Path, references: Dict[str, str]
) -> Tuple[Dict[str, str], List[str], bool]:
    """Resolve ``references`` once.  Returns ``(secrets, warnings, auth_failed)``.

    ``auth_failed`` is True if any reference failed for an auth/session reason,
    signalling the caller to try a login and re-run the batch.
    """
    secrets: Dict[str, str] = {}
    warnings: List[str] = []
    auth_failed = False
    for name in sorted(references):
        try:
            secrets[name] = _run_pass_resolve(pass_cli, references[name])
        except _AuthError as exc:
            auth_failed = True
            warnings.append(str(exc))
        except RuntimeError as exc:
            warnings.append(str(exc))
    return secrets, warnings, auth_failed


def fetch_protonpass_secrets(
    *,
    references: Dict[str, str],
    token_env: str = _DEFAULT_TOKEN_ENV,
    binary: Optional[Path] = None,
    binary_path: str = "",
    use_cache: bool = True,
    cache_ttl_seconds: float = 300,
    home_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Resolve ``references`` (name → ``pass://…``) to ``(secrets, warnings)``.

    Raises :class:`RuntimeError` only when no ``pass-cli`` binary is available —
    a fatal "can't fetch anything" condition.  Per-reference failures (bad
    reference, empty value, persistent auth failure) are collected as warnings
    and the reference is dropped, so one bad entry never sinks the rest.

    On an auth-shaped failure we attempt a single ``pass-cli login`` using the
    configured token env var and re-resolve the still-missing references.

    Only a complete, error-free pull is cached, so a transient auth failure
    isn't frozen in for the whole TTL window.
    """
    valid, warnings = _validate_references(references)
    if not valid:
        return {}, warnings

    cache_key: _CacheKey = (
        _auth_fingerprint(token_env),
        str(home_path) if home_path is not None else "",
        _refs_fingerprint(valid),
    )

    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return dict(cached.secrets), warnings
        disk_cached = _read_disk_cache(cache_key, cache_ttl_seconds, home_path)
        if disk_cached is not None:
            # Promote into L1 so later fetches in this process skip the disk read.
            _CACHE[cache_key] = disk_cached
            return dict(disk_cached.secrets), warnings

    pass_cli = binary or find_pass_cli(binary_path)
    if pass_cli is None:
        raise RuntimeError(
            "pass-cli not found.  Install the Proton Pass CLI "
            "(https://protonpass.github.io/pass-cli/) or set "
            "secrets.protonpass.binary_path to its absolute location."
        )

    secrets, batch_warnings, auth_failed = _resolve_batch(pass_cli, valid)

    # One login + retry of the still-missing references on an auth failure.
    if auth_failed and len(secrets) < len(valid):
        token_value = os.environ.get(token_env, "").strip()
        try:
            _pass_login(pass_cli, token_value)
            remaining = {n: r for n, r in valid.items() if n not in secrets}
            retry_secrets, retry_warnings, _ = _resolve_batch(pass_cli, remaining)
            secrets.update(retry_secrets)
            # Replace the auth warnings with the retry's outcome — anything still
            # missing after a successful login is a real, reportable failure.
            batch_warnings = retry_warnings
        except RuntimeError as exc:
            batch_warnings.append(f"pass-cli login failed: {exc}")

    warnings.extend(batch_warnings)

    # Cache only a complete, error-free pull.
    if use_cache and len(secrets) == len(valid) and not batch_warnings and secrets:
        entry = _CachedFetch(secrets=dict(secrets), fetched_at=time.time())
        _CACHE[cache_key] = entry
        _write_disk_cache(cache_key, entry, cache_ttl_seconds, home_path)

    return secrets, warnings


# ---------------------------------------------------------------------------
# Public entry point — called from hermes_cli.env_loader
# ---------------------------------------------------------------------------


def apply_protonpass_secrets(
    *,
    enabled: bool,
    env: Optional[Dict[str, str]] = None,
    personal_access_token_env: str = _DEFAULT_TOKEN_ENV,
    binary_path: str = "",
    override_existing: bool = True,
    cache_ttl_seconds: float = 300,
    home_path: Optional[Path] = None,
) -> FetchResult:
    """Resolve configured ``pass://`` references and set them on ``os.environ``.

    Called by ``load_hermes_dotenv()`` after the .env files have loaded.
    Intentionally defensive — any failure returns a :class:`FetchResult` with
    ``error`` set (or surfaces warnings); it never raises.

    Parameters mirror the ``secrets.protonpass.*`` config keys so the caller
    can splat the dict in.  References already satisfied by the current
    environment (when ``override_existing`` is false) are skipped *before*
    fetching, so ``pass-cli`` is never invoked for a value that would be
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
        if name == personal_access_token_env:
            # Never let a resolved secret clobber the very token used to auth.
            result.skipped.append(name)
            continue
        if not override_existing and os.environ.get(name):
            result.skipped.append(name)
            continue
        refs_to_fetch[name] = ref

    if not refs_to_fetch:
        return result

    binary = find_pass_cli(binary_path)
    result.binary_path = binary
    if binary is None:
        if binary_path:
            result.error = (
                f"secrets.protonpass.binary_path ({binary_path!r}) is not an "
                "executable pass-cli binary."
            )
        else:
            result.error = (
                "secrets.protonpass.enabled is true but pass-cli was not found "
                "on PATH.  Install it (https://protonpass.github.io/pass-cli/) "
                "or set secrets.protonpass.binary_path."
            )
        return result

    try:
        secrets, fetch_warnings = fetch_protonpass_secrets(
            references=refs_to_fetch,
            token_env=personal_access_token_env,
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
        if name == personal_access_token_env:
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
# Test hook — used by hermetic tests to flush the cache between cases.
# ---------------------------------------------------------------------------


def _reset_cache_for_tests(home_path: Optional[Path] = None) -> None:
    """Clear in-process AND disk caches.

    Tests can pass ``home_path`` to scope the disk cleanup to a tmpdir.
    Without it we fall back to the same default resolution as the writer.
    """
    _CACHE.clear()
    try:
        _disk_cache_path(home_path).unlink()
    except (FileNotFoundError, OSError):
        pass
