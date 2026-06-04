"""1Password CLI (`op`) secret-source integration.

Hermes can resolve 1Password secret references at process startup and inject
those values into ``os.environ`` so secrets do not have to live in plaintext in
``~/.hermes/.env``.

Design summary
--------------

* Authentication uses a 1Password Service Account token stored in the env var
  named by ``secrets.onepassword.service_account_token_env`` (default:
  ``OP_SERVICE_ACCOUNT_TOKEN``). This is the one bootstrap secret.
* The config maps environment variable names to 1Password secret references,
  e.g. ``TELEGRAM_BOT_TOKEN: op://Hermes/TELEGRAM_BOT_TOKEN/password``.
* Fetching is subprocess-driven through ``op read <secret-ref>`` so Hermes does
  not need a heavyweight SDK dependency.
* Failures never block Hermes startup. The caller receives a FetchResult with a
  short error/warning and Hermes continues with whatever credentials were
  already present.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_OP_RUN_TIMEOUT = 20
_CacheKey = Tuple[str, str]  # (token_fingerprint, stable mapping json)
_CACHE: Dict[_CacheKey, "_CachedFetch"] = {}


@dataclass
class _CachedFetch:
    secrets: Dict[str, str]
    warnings: List[str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


@dataclass
class FetchResult:
    """Outcome of a single 1Password pull."""

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def find_op() -> Optional[Path]:
    """Return the 1Password CLI path, if available on PATH."""
    found = shutil.which("op")
    return Path(found) if found else None


def _token_fingerprint(token: str) -> str:
    """SHA-256 prefix used as a cache key — never logged, never displayed."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _mapping_key(mapping: Dict[str, str]) -> str:
    return json.dumps(mapping, sort_keys=True, separators=(",", ":"))


def _is_valid_env_name(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def _normalize_mapping(mapping: object) -> Tuple[Dict[str, str], List[str]]:
    warnings: List[str] = []
    normalized: Dict[str, str] = {}
    if not isinstance(mapping, dict):
        return {}, ["secrets.onepassword.mapping must be a mapping of ENV_VAR to op:// reference"]

    for raw_key, raw_ref in mapping.items():
        key = str(raw_key).strip()
        ref = str(raw_ref).strip()
        if not _is_valid_env_name(key):
            warnings.append(f"Skipping mapping {key!r}: not a valid env-var name")
            continue
        if not ref.startswith("op://"):
            warnings.append(f"Skipping {key}: secret reference must start with op://")
            continue
        normalized[key] = ref
    return normalized, warnings


def fetch_onepassword_secrets(
    *,
    service_account_token: str,
    mapping: Dict[str, str],
    binary: Optional[Path] = None,
    cache_ttl_seconds: float = 300,
    use_cache: bool = True,
) -> Tuple[Dict[str, str], List[str]]:
    """Resolve configured 1Password secret references.

    Returns ``(secrets_dict, warnings_list)``. Raises ``RuntimeError`` for fatal
    setup/auth/CLI problems. Caller controls whether the values are applied to
    the process environment.
    """
    if not service_account_token:
        raise RuntimeError("1Password service account token is empty")

    normalized, warnings = _normalize_mapping(mapping)
    if not normalized:
        return {}, warnings or ["No valid 1Password secret mappings configured"]

    cache_key = (_token_fingerprint(service_account_token), _mapping_key(normalized))
    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return cached.secrets, list(cached.warnings)

    op_bin = binary or find_op()
    if op_bin is None:
        raise RuntimeError(
            "op binary not available — install 1Password CLI (`brew install --cask "
            "1password-cli`) and re-run `hermes secrets onepassword status`."
        )

    secrets: Dict[str, str] = {}
    run_warnings: List[str] = list(warnings)
    for key, ref in normalized.items():
        value = _run_op_read(op_bin, service_account_token, ref, env_name=key)
        secrets[key] = value

    _CACHE[cache_key] = _CachedFetch(
        secrets=secrets,
        warnings=run_warnings,
        fetched_at=time.time(),
    )
    return secrets, run_warnings


def _run_op_read(
    op_bin: Path,
    service_account_token: str,
    ref: str,
    *,
    env_name: Optional[str] = None,
) -> str:
    target = env_name or "secret reference"
    env = os.environ.copy()
    env["OP_SERVICE_ACCOUNT_TOKEN"] = service_account_token
    env.setdefault("NO_COLOR", "1")
    try:
        proc = subprocess.run(  # noqa: S603 — op path is trusted
            [str(op_bin), "read", ref],
            env=env,
            capture_output=True,
            text=True,
            timeout=_OP_RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"op timed out after {_OP_RUN_TIMEOUT}s reading {target}") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke op: {exc}") from exc

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().replace("\x1b", "")
        raise RuntimeError(f"op read failed for {target}: {err[:240]}")

    return (proc.stdout or "").rstrip("\r\n")


def apply_onepassword_secrets(
    *,
    enabled: bool,
    service_account_token_env: str = "OP_SERVICE_ACCOUNT_TOKEN",
    mapping: Optional[Dict[str, str]] = None,
    override_existing: bool = False,
    cache_ttl_seconds: float = 300,
) -> FetchResult:
    """Pull 1Password references and set them on ``os.environ`` defensively."""
    result = FetchResult()
    if not enabled:
        return result

    token = os.environ.get(service_account_token_env, "").strip()
    if not token:
        result.error = (
            f"secrets.onepassword.enabled is true but {service_account_token_env} "
            "is not set. Store your 1Password Service Account token in .env."
        )
        return result

    normalized, warnings = _normalize_mapping(mapping or {})
    result.warnings.extend(warnings)
    if not normalized:
        result.error = (
            "secrets.onepassword.mapping is empty. Add ENV_VAR: op://... references "
            "or run `hermes secrets onepassword setup --map ENV=op://...`."
        )
        return result

    binary = find_op()
    result.binary_path = binary
    if binary is None:
        result.error = (
            "op binary not available. Install 1Password CLI with "
            "`brew install --cask 1password-cli`."
        )
        return result

    try:
        secrets, fetch_warnings = fetch_onepassword_secrets(
            service_account_token=token,
            mapping=normalized,
            binary=binary,
            cache_ttl_seconds=cache_ttl_seconds,
        )
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(fetch_warnings)

    for key, value in secrets.items():
        if key == service_account_token_env:
            result.skipped.append(key)
            continue
        if not override_existing and os.environ.get(key):
            result.skipped.append(key)
            continue
        os.environ[key] = value
        result.applied.append(key)

    return result


def _reset_cache_for_tests() -> None:
    _CACHE.clear()
