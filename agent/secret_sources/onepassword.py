"""1Password CLI (``op``) secret-reference integration.

Hermes can resolve environment-variable-shaped credentials from 1Password
at process startup so API keys do not have to live in plaintext in
``~/.hermes/.env``. Users map env var names to 1Password secret references
(``op://vault/item/field``) in ``config.yaml``; Hermes shells out to the
official ``op`` CLI and injects the resolved values into ``os.environ``.

Design summary
--------------

* Hermes never stores 1Password item values in config.yaml or .env.
* The integration is opt-in via ``secrets.onepassword.enabled``.
* ``~/.hermes/.env`` and shell exports load first; by default 1Password can
  then override them so rotations in the vault take effect on next startup.
* Successful reads are cached in-process and on disk under
  ``<hermes_home>/cache/op_cache.json`` for ``cache_ttl_seconds`` so
  short-lived Hermes commands do not repeatedly pay ``op`` startup/auth cost.
* When ``override_existing`` is false, values that already exist in the
  environment are skipped before contacting ``op``.
* Failures NEVER block Hermes startup. Missing ``op``, locked desktop app,
  expired session, bad references, etc. all return a ``FetchResult`` error
  and the caller continues with whatever credentials were already available.
* This module does not auto-install ``op``. The official 1Password CLI has
  platform-specific install/sign-in flows, desktop-app integration, and
  service-account options; Hermes only consumes an already configured CLI.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from agent.secret_sources.disk_cache import (
    disk_cache_path,
    read_secret_cache,
    write_secret_cache,
)

logger = logging.getLogger(__name__)

_OP_RUN_TIMEOUT = 30

# Cache key: (token/session fingerprint, account, sorted refs).  We include a
# fingerprint of common 1Password auth env vars so changing service-account or
# session credentials does not reuse values fetched under a previous identity.
_CacheKey = Tuple[str, str, Tuple[Tuple[str, str], ...]]
_CACHE: Dict[_CacheKey, "_CachedFetch"] = {}

# Disk-persisted cache so short-lived Hermes processes (gateway workers,
# shell one-shots, cron jobs) do not each pay one `op read` subprocess per
# configured secret. Values are plaintext-equivalent to ~/.hermes/.env, so the
# file is kept under <hermes_home>/cache, written atomically, and chmod 0600.
# The cache key includes only fingerprints / references, never token values.
_DISK_CACHE_BASENAME = "op_cache.json"


def _disk_cache_path(home_path: Optional[Path] = None) -> Path:
    """Return the 1Password cache path under hermes_home/cache/."""

    return disk_cache_path(_DISK_CACHE_BASENAME, home_path)


def _cache_key_str(cache_key: _CacheKey) -> str:
    """Serialize a cache key to a stable string for JSON storage."""

    auth_fp, account, refs = cache_key
    refs_material = "\0".join(f"{name}={ref}" for name, ref in refs)
    refs_fp = hashlib.sha256(refs_material.encode("utf-8")).hexdigest()[:16]
    return f"{auth_fp}|{account}|{refs_fp}"


def _read_disk_cache(
    cache_key: _CacheKey,
    ttl_seconds: float,
    home_path: Optional[Path] = None,
) -> Optional["_CachedFetch"]:
    """Return a fresh disk cache entry, or None on miss/error/stale."""

    if ttl_seconds <= 0:
        return None
    cached = read_secret_cache(
        cache_basename=_DISK_CACHE_BASENAME,
        cache_key=_cache_key_str(cache_key),
        ttl_seconds=ttl_seconds,
        home_path=home_path,
    )
    if cached is None:
        return None
    secrets, fetched_at = cached
    return _CachedFetch(secrets=secrets, fetched_at=fetched_at)


def _write_disk_cache(
    cache_key: _CacheKey,
    entry: "_CachedFetch",
    home_path: Optional[Path] = None,
) -> None:
    """Persist a cache entry atomically with mode 0600; best effort only."""

    write_secret_cache(
        cache_basename=_DISK_CACHE_BASENAME,
        cache_key=_cache_key_str(cache_key),
        secrets=entry.secrets,
        fetched_at=entry.fetched_at,
        temp_prefix=".op_cache_",
        home_path=home_path,
    )


@dataclass
class _CachedFetch:
    secrets: Dict[str, str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


@dataclass
class FetchResult:
    """Outcome of applying 1Password references into ``os.environ``."""

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def find_op() -> Optional[Path]:
    """Return a path to the 1Password CLI, or ``None`` if unavailable."""

    found = shutil.which("op")
    return Path(found) if found else None


def fetch_onepassword_secrets(
    *,
    references: Mapping[str, str],
    binary: Optional[Path] = None,
    account: str = "",
    cache_ttl_seconds: float = 300,
    use_cache: bool = True,
    service_account_token_env: str = "OP_SERVICE_ACCOUNT_TOKEN",
    home_path: Optional[Path] = None,
) -> tuple[Dict[str, str], list[str]]:
    """Resolve configured 1Password references with ``op read``.

    ``references`` maps environment-variable names to 1Password secret
    references such as ``op://Private/OpenAI API key/credential``.

    Raises :class:`RuntimeError` for fatal conditions. The env-loader path
    catches these and warns without blocking Hermes startup.
    """

    valid_refs, warnings = _validate_references(references)
    if not valid_refs:
        return {}, warnings

    account = (account or "").strip()
    cache_key = (
        _auth_fingerprint(service_account_token_env),
        account,
        tuple(sorted(valid_refs.items())),
    )
    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return dict(cached.secrets), warnings
        disk_cached = _read_disk_cache(cache_key, cache_ttl_seconds, home_path)
        if disk_cached is not None:
            _CACHE[cache_key] = disk_cached
            return dict(disk_cached.secrets), warnings

    op = binary or find_op()
    if op is None:
        raise RuntimeError(
            "1Password CLI `op` is not available on PATH. Install it from "
            "https://developer.1password.com/docs/cli/get-started/ and sign in, "
            "or set OP_SERVICE_ACCOUNT_TOKEN for a service account."
        )

    secrets: Dict[str, str] = {}
    read_errors: list[str] = []
    for env_var, ref in valid_refs.items():
        try:
            secrets[env_var] = _run_op_read(
                op,
                ref,
                account=account,
                service_account_token_env=service_account_token_env,
            )
        except RuntimeError as exc:
            read_errors.append(f"{env_var}: {exc}")

    if read_errors:
        warnings.extend(read_errors)
        if not secrets:
            raise RuntimeError("; ".join(read_errors))

    entry = _CachedFetch(secrets=dict(secrets), fetched_at=time.time())
    _CACHE[cache_key] = entry
    if use_cache:
        _write_disk_cache(cache_key, entry, home_path)
    return secrets, warnings


def apply_onepassword_secrets(
    *,
    enabled: bool,
    references: Optional[Mapping[str, str]] = None,
    override_existing: bool = False,
    cache_ttl_seconds: float = 300,
    account: str = "",
    service_account_token_env: str = "OP_SERVICE_ACCOUNT_TOKEN",
    home_path: Optional[Path] = None,
) -> FetchResult:
    """Resolve 1Password references and set them on ``os.environ``.

    Parameters mirror ``secrets.onepassword.*`` config keys so the caller can
    pass values directly from config.yaml. The function is intentionally
    defensive: it returns errors in the result object and never raises.
    """

    result = FetchResult()
    if not enabled:
        return result

    references = references or {}
    if not references:
        result.error = (
            "secrets.onepassword.enabled is true but no references are configured. "
            "Add entries under secrets.onepassword.env."
        )
        return result

    valid_refs, validation_warnings = _validate_references(references)
    result.warnings.extend(validation_warnings)

    refs_to_fetch: Dict[str, str] = {}
    for key, ref in valid_refs.items():
        if key == service_account_token_env:
            # Avoid clobbering the bootstrap credential used by op service
            # accounts if someone maps it accidentally.
            result.skipped.append(key)
            continue
        if not override_existing and os.environ.get(key):
            # Performance fast path: if the caller explicitly wants existing
            # env/.env values to win, do not contact `op` for values we will
            # discard anyway. This matters for shells/services that keep most
            # credentials exported but have 1Password configured as fallback.
            result.skipped.append(key)
            continue
        refs_to_fetch[key] = ref

    if not refs_to_fetch:
        if result.skipped:
            return result
        if result.warnings:
            result.error = "No valid 1Password references were resolved. " + "; ".join(
                result.warnings
            )
            return result
        return result

    binary = find_op()
    result.binary_path = binary

    try:
        secrets, warnings = fetch_onepassword_secrets(
            references=refs_to_fetch,
            binary=binary,
            account=account,
            cache_ttl_seconds=cache_ttl_seconds,
            service_account_token_env=service_account_token_env,
            home_path=home_path,
        )
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(warnings)
    if not secrets and warnings:
        result.error = "No valid 1Password references were resolved. " + "; ".join(warnings)
        return result

    for key, value in secrets.items():
        os.environ[key] = value
        result.applied.append(key)

    return result


def _validate_references(references: Mapping[str, str]) -> tuple[Dict[str, str], list[str]]:
    valid: Dict[str, str] = {}
    warnings: list[str] = []
    for key, ref in references.items():
        if not isinstance(key, str) or not _is_valid_env_name(key):
            warnings.append(f"Skipping reference for {key!r}: not a valid env-var name")
            continue
        if not isinstance(ref, str) or not ref.strip():
            warnings.append(f"Skipping {key}: 1Password reference is empty")
            continue
        ref = ref.strip()
        if not ref.startswith("op://"):
            warnings.append(f"Skipping {key}: 1Password reference must start with 'op://'")
            continue
        valid[key] = ref
    return valid, warnings


def _run_op_read(
    op: Path,
    reference: str,
    *,
    account: str = "",
    service_account_token_env: str = "OP_SERVICE_ACCOUNT_TOKEN",
) -> str:
    cmd = [str(op), "read", reference]
    if account:
        cmd.extend(["--account", account])
    env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("OP_")
        or key in {"PATH", "HOME", "USER", "USERNAME", "SystemRoot"}
    }
    env["NO_COLOR"] = "1"
    if service_account_token_env != "OP_SERVICE_ACCOUNT_TOKEN":
        custom_token = os.environ.get(service_account_token_env)
        if custom_token:
            env["OP_SERVICE_ACCOUNT_TOKEN"] = custom_token
    try:
        proc = subprocess.run(  # noqa: S603 — op path is trusted
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_OP_RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"op timed out after {_OP_RUN_TIMEOUT}s reading {reference!r}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke op: {exc}") from exc

    if proc.returncode != 0:
        err = (proc.stderr or "").strip().replace("\x1b", "")
        if not err:
            err = f"op exited with status {proc.returncode}"
        raise RuntimeError(f"op read failed for {reference!r}: {err[:200]}")

    # `op read` prints a trailing newline in normal terminal use. Preserve all
    # other characters in the secret value.
    value = (proc.stdout or "").rstrip("\r\n")
    if value == "":
        raise RuntimeError(f"op read returned an empty value for {reference!r}")
    return value


def _auth_fingerprint(service_account_token_env: str = "OP_SERVICE_ACCOUNT_TOKEN") -> str:
    names = {"OP_SERVICE_ACCOUNT_TOKEN", service_account_token_env, "OP_ACCOUNT"}
    names.update(name for name in os.environ if name.startswith("OP_SESSION_"))
    values = [os.environ.get(name, "") for name in sorted(names)]
    if not any(values):
        return ""
    material = "\0".join(values)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _is_valid_env_name(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def _reset_cache_for_tests(home_path: Optional[Path] = None) -> None:
    """Clear in-process and default disk cache for hermetic tests."""

    _CACHE.clear()
    try:
        _disk_cache_path(home_path).unlink()
    except OSError:
        pass
