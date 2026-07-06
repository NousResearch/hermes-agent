"""KeePassXC (``keepassxc-cli``) secret source integration.

Hermes pulls API keys from a KeePassXC database at process startup so they
don't have to live in plaintext in ``~/.hermes/.env``.

Design summary
--------------

* The user provides the path to their ``.kdbx`` database and a mapping from
  env-var names to KeePassXC entry paths (e.g.
  ``{"OPENAI_API_KEY": "Dev/OpenAI"}``).  These mirror the
  ``secrets.keepassxc.*`` config keys.
* Database credentials are supplied by one of:

  * the ``KEEPASSXC_PASSWORD`` env var (configurable via
    ``password_env``);
  * a key file passed with ``key_file``;
  * ``--no-password`` for unprotected databases.

  When a password is available it is piped into ``keepassxc-cli`` via
  stdin; the binary never receives it as a command-line argument.
* One ``keepassxc-cli show`` subprocess is spawned per entry.  The CLI
  reads the database each time, so the fetch is naturally bounded and
  stateless.  We still cache the returned secrets in-process and on disk
  for ``cache_ttl_seconds`` so back-to-back ``hermes`` invocations don't
  repeatedly prompt the user for a database unlock.
* Failures never block Hermes startup.  Missing binary, wrong password,
  missing entry, etc. are recorded as warnings (or a single fatal ``error``
  on the :class:`FetchResult`) and the process continues with whatever
  credentials ``.env`` already had.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# How long to wait for a keepassxc-cli subprocess, in seconds.
_KP_RUN_TIMEOUT = 15

# In-process cache so repeated load_hermes_dotenv() calls (CLI startup,
# gateway hot-reload, test suites) don't re-fetch from KeePassXC.
_CacheKey = Tuple[str, str]  # (resolved_db_path, credentials_fingerprint)
_CACHE: Dict[_CacheKey, "_CachedFetch"] = {}

# Disk-persisted cache so back-to-back CLI invocations (e.g.
# `hermes chat -q ...` called from scripts, cron, the gateway forking new
# agents) don't each pay the KeePassXC unlock tax. The in-process _CACHE
# above only saves repeated fetches WITHIN one process; this saves repeated
# fetches ACROSS processes.
#
# Layout: one JSON object per cache key, written atomically with mode 0600 in
# <hermes_home>/cache/keepassxc_cache.json. The file holds only the secret
# VALUES, never the database password. It's plaintext-equivalent to
# ~/.hermes/.env (which we already accept) but kept out of the .env file so
# users editing it won't accidentally commit KeePassXC-sourced secrets.
_DISK_CACHE_BASENAME = "keepassxc_cache.json"


def _disk_cache_path(home_path: Optional[Path] = None) -> Path:
    """Return the disk cache path under hermes_home/cache/.

    `home_path` is what `load_hermes_dotenv()` already resolved; falling back
    to `$HERMES_HOME` / `~/.hermes` keeps direct callers working too.
    """
    if home_path is None:
        home_path = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return home_path / "cache" / _DISK_CACHE_BASENAME


def _cache_key_str(cache_key: _CacheKey) -> str:
    """Serialize a cache key to a stable string for JSON storage."""
    db_path, cred_fp = cache_key
    return f"{db_path}|{cred_fp}"


def _read_disk_cache(
    cache_key: _CacheKey,
    ttl_seconds: float,
    home_path: Optional[Path] = None,
) -> Optional["_CachedFetch"]:
    """Return a cached entry from disk if fresh, else None.

    Best-effort: any I/O or parse error returns None and we re-fetch.
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
    if payload.get("key") != _cache_key_str(cache_key):
        return None
    secrets = payload.get("secrets")
    fetched_at = payload.get("fetched_at")
    if not isinstance(secrets, dict) or not isinstance(fetched_at, (int, float)):
        return None
    # Coerce all values to strings — JSON allows numbers but env vars need strings
    typed_secrets: Dict[str, str] = {
        k: v for k, v in secrets.items() if isinstance(k, str) and isinstance(v, str)
    }
    entry = _CachedFetch(secrets=typed_secrets, fetched_at=float(fetched_at))
    if not entry.is_fresh(ttl_seconds):
        return None
    return entry


def _write_disk_cache(
    cache_key: _CacheKey,
    entry: "_CachedFetch",
    home_path: Optional[Path] = None,
) -> None:
    """Persist a cache entry to disk atomically with mode 0600.

    Best-effort: any I/O error is swallowed (the next invocation will just
    re-fetch). We never want disk cache failures to break startup.
    """
    path = _disk_cache_path(home_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": _cache_key_str(cache_key),
            "secrets": entry.secrets,
            "fetched_at": entry.fetched_at,
        }
        # Write to a temp file in the same directory and atomic-rename.
        # tempfile honors os.umask, so we explicitly chmod 0600 before rename.
        fd, tmp = tempfile.mkstemp(
            prefix=".keepassxc_cache_", suffix=".tmp", dir=str(path.parent)
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
        pass  # best-effort — disk cache miss on next invocation is fine


@dataclass
class _CachedFetch:
    secrets: Dict[str, str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """Outcome of a single KeePassXC pull."""

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)   # set into os.environ
    skipped: List[str] = field(default_factory=list)   # already set, not overridden
    warnings: List[str] = field(default_factory=list)  # non-fatal issues
    error: Optional[str] = None                        # fatal: nothing was fetched
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def find_keepassxc_cli() -> Optional[Path]:
    """Return a path to a usable ``keepassxc-cli`` binary, or None.

    Resolution order:
      1. ``KEEPASSXC_CLI_PATH`` env var (full path to the binary).
      2. ``shutil.which("keepassxc-cli")`` (system PATH).

    Unlike the Bitwarden source we do NOT auto-install: keepassxc-cli is a
    system package and should be installed by the platform package manager.
    """
    env_path = os.getenv("KEEPASSXC_CLI_PATH", "").strip()
    if env_path:
        candidate = Path(env_path)
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate

    system = shutil.which("keepassxc-cli")
    if system:
        return Path(system)

    return None


# ---------------------------------------------------------------------------
# Credentials fingerprint for cache keying
# ---------------------------------------------------------------------------


def _credentials_fingerprint(
    password: str, key_file: str
) -> str:
    """Build a stable, opaque cache-key component from the credentials.

    We never log or display this value.  It includes the password and, when a
    key file is present, the key file contents so that rotating either
    credential invalidates the cache.
    """
    h = hashlib.sha256()
    h.update(password.encode("utf-8"))
    if key_file:
        try:
            with open(key_file, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except OSError:
            # If the key file can't be read we still hash the path so the
            # cache key changes when the path changes.
            h.update(key_file.encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Secret fetch + apply
# ---------------------------------------------------------------------------


def _run_keepassxc_show(
    binary: Path,
    db_path: Path,
    entry_path: str,
    password: str,
    key_file: str,
    no_password: bool,
) -> str:
    """Return the Password attribute for a single KeePassXC entry.

    Raises :class:`RuntimeError` for fatal conditions (wrong password,
    binary invocation failure).  Returns the password string on success.
    """
    cmd = [str(binary), "show", "-a", "Password", "-s", str(db_path), entry_path]
    if no_password:
        cmd.insert(2, "--no-password")
    if key_file:
        cmd.insert(2, "-k")
        cmd.insert(3, key_file)

    try:
        proc = subprocess.run(  # noqa: S603 — keepassxc-cli path is trusted
            cmd,
            input=password,
            capture_output=True,
            text=True,
            timeout=_KP_RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"keepassxc-cli timed out after {_KP_RUN_TIMEOUT}s fetching {entry_path!r}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"failed to invoke keepassxc-cli for {entry_path!r}: {exc}"
        ) from exc

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().replace("\x1b", "")
        lower_err = err.lower()
        # Check "entry not found" BEFORE "password" — stderr often contains
        # "Enter password to unlock" even when the real error is a missing entry.
        # keepassxc-cli says "Could not find entry with path X" (not "entry not found").
        if "not found" in lower_err or "could not find" in lower_err:
            raise RuntimeError(
                f"KeePassXC entry not found: {entry_path!r}"
            )
        if "credentials" in lower_err or "password" in lower_err:
            raise RuntimeError(
                f"KeePassXC credentials failed for {entry_path!r}: {err[:200]}"
            )
        raise RuntimeError(
            f"keepassxc-cli exited {proc.returncode} for {entry_path!r}: {err[:200]}"
        )

    return proc.stdout.strip()


def _is_valid_env_name(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def fetch_keepassxc_secrets(
    *,
    db_path: str,
    password: str,
    key_file: str = "",
    no_password: bool = False,
    binary: Optional[Path] = None,
    mappings: Optional[Dict[str, str]] = None,
    cache_ttl_seconds: float = 300,
    use_cache: bool = True,
    home_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Pull the mapped secrets from a KeePassXC database.

    Returns ``(secrets_dict, warnings_list)``.

    ``mappings`` is a dict of ``env_var_name → entry_path`` where
    ``entry_path`` is the KeePassXC entry path as displayed in the
    application (e.g. ``"Dev/OpenAI"``).  ``env_var_name`` must be a valid
    shell env-var name.

    Caching is a two-layer LRU: an in-process dict (for hot-reload paths
    inside one process) and a disk-persisted JSON file under
    ``<hermes_home>/cache/keepassxc_cache.json`` (for back-to-back CLI
    invocations).  Both share the same TTL.  Pass ``home_path`` so disk
    cache lookups find the right directory in tests / non-standard installs;
    otherwise we fall back to ``$HERMES_HOME`` / ``~/.hermes``.

    Raises :class:`RuntimeError` for fatal conditions (missing binary,
    missing database, wrong password).  Per-entry failures (entry not
    found, timeout, etc.) are returned as warnings instead of raising.
    """
    if not db_path:
        raise RuntimeError("KeePassXC db_path is empty")
    db = Path(db_path).expanduser().resolve()
    if not db.exists():
        raise RuntimeError(f"KeePassXC database not found: {db}")

    mappings = mappings or {}
    if not mappings:
        raise RuntimeError("KeePassXC mappings are empty")

    if binary is None:
        binary = find_keepassxc_cli()
    if binary is None:
        raise RuntimeError(
            "keepassxc-cli binary not available. Install KeePassXC and ensure "
            "`keepassxc-cli` is on PATH, or set KEEPASSXC_CLI_PATH to the binary."
        )

    if not no_password and not password and not key_file:
        # This is reachable if the caller passes an empty password and no
        # key file. keepassxc-cli would prompt interactively, which would
        # hang in non-interactive contexts; fail fast instead.
        raise RuntimeError(
            "KeePassXC password is empty and no key_file or no_password option provided"
        )

    cache_key = (str(db), _credentials_fingerprint(password, key_file))
    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return cached.secrets, []
        # L2: disk cache.
        disk_cached = _read_disk_cache(cache_key, cache_ttl_seconds, home_path)
        if disk_cached is not None:
            _CACHE[cache_key] = disk_cached
            return disk_cached.secrets, []

    secrets: Dict[str, str] = {}
    warnings: List[str] = []

    for env_name, entry_path in mappings.items():
        if not _is_valid_env_name(env_name):
            warnings.append(
                f"Skipping KeePassXC mapping {env_name!r}: not a valid env-var name"
            )
            continue

        try:
            value = _run_keepassxc_show(
                binary=binary,
                db_path=db,
                entry_path=entry_path,
                password=password,
                key_file=key_file,
                no_password=no_password,
            )
        except subprocess.TimeoutExpired:
            warnings.append(
                f"Timed out reading KeePassXC entry {entry_path!r} for {env_name}"
            )
            continue
        except RuntimeError as exc:
            msg = str(exc)
            if "entry not found" in msg:
                warnings.append(
                    f"KeePassXC entry {entry_path!r} not found for {env_name}; skipped"
                )
            elif "credentials failed" in msg:
                # Re-raise wrong-password errors so the public entry point can
                # record them as the fatal FetchResult.error.
                raise
            else:
                warnings.append(
                    f"Failed to read KeePassXC entry {entry_path!r} for {env_name}: {msg}"
                )
            continue

        if not value:
            warnings.append(
                f"KeePassXC entry {entry_path!r} has no Password attribute for {env_name}"
            )
            continue

        secrets[env_name] = value

    entry = _CachedFetch(secrets=secrets, fetched_at=time.time())
    _CACHE[cache_key] = entry
    if use_cache:
        _write_disk_cache(cache_key, entry, home_path)

    return secrets, warnings


# ---------------------------------------------------------------------------
# Public entry point — called from hermes_cli.env_loader
# ---------------------------------------------------------------------------


def apply_keepassxc_secrets(
    *,
    enabled: bool,
    db_path: str,
    password_env: str = "KEEPASSXC_PASSWORD",
    key_file: str = "",
    no_password: bool = False,
    mappings: Optional[Dict[str, str]] = None,
    override_existing: bool = False,
    cache_ttl_seconds: float = 300,
    home_path: Optional[Path] = None,
) -> FetchResult:
    """Pull secrets from KeePassXC and set them on ``os.environ``.

    This is the function ``load_hermes_dotenv()`` calls after the .env
    files have loaded.  It is intentionally defensive — any failure
    returns a :class:`FetchResult` with ``error`` set; it never raises.

    Parameters mirror the ``secrets.keepassxc.*`` config keys so the
    caller can just splat the dict in.
    """
    result = FetchResult()

    if not enabled:
        return result

    if not db_path:
        result.error = "secrets.keepassxc.db_path is empty"
        return result

    mappings = mappings or {}
    if not mappings:
        result.error = "secrets.keepassxc.mappings is empty"
        return result

    password = os.environ.get(password_env, "").strip()
    if not no_password and not password and not key_file:
        result.error = (
            f"secrets.keepassxc.enabled is true but {password_env} is not set "
            "and no key_file or no_password option is provided"
        )
        return result

    binary = find_keepassxc_cli()
    result.binary_path = binary
    if binary is None:
        result.error = (
            "keepassxc-cli binary not available. Install KeePassXC and ensure "
            "`keepassxc-cli` is on PATH, or set KEEPASSXC_CLI_PATH to the binary."
        )
        return result

    try:
        secrets, warnings = fetch_keepassxc_secrets(
            db_path=db_path,
            password=password,
            key_file=key_file,
            no_password=no_password,
            binary=binary,
            mappings=mappings,
            cache_ttl_seconds=cache_ttl_seconds,
            home_path=home_path,
        )
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(warnings)

    for key, value in secrets.items():
        if not override_existing and os.environ.get(key):
            result.skipped.append(key)
            continue
        os.environ[key] = value
        result.applied.append(key)

    return result


# ---------------------------------------------------------------------------
# Test hook — used by hermetic tests to flush the cache between cases.
# ---------------------------------------------------------------------------


def _reset_cache_for_tests(home_path: Optional[Path] = None) -> None:
    """Clear in-process AND disk caches.

    Tests can pass ``home_path`` to scope the disk cleanup to a tmpdir.
    Without it we fall back to the same default resolution as the cache
    writer itself.
    """
    _CACHE.clear()
    try:
        _disk_cache_path(home_path).unlink()
    except (FileNotFoundError, OSError):
        pass
