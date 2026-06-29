"""1Password CLI (`op`) integration for Hermes secret sources.

Hermes can pull API keys from 1Password at process startup so provider
credentials do not have to live in plaintext in ``~/.hermes/.env``.

Design summary
--------------

* Users keep a small references file (default:
  ``~/.hermes/secrets/1password.env``) that maps environment variable
  names to 1Password secret references, for example::

      OPENROUTER_API_KEY=op://Private/OpenRouter API Key/credential

* The only optional bootstrap secret is ``OP_SERVICE_ACCOUNT_TOKEN`` (or a
  configured alternative) in ``~/.hermes/.env`` / the shell.  Users can
  also rely on an already authenticated 1Password app / CLI session.
* Fetching is subprocess-driven via ``op read <reference>``.  We do not
  depend on a Python SDK and we do not print secret values.
* Failures NEVER block Hermes startup.  Missing ``op``, unauthenticated
  sessions, invalid references, etc. all emit warnings and continue with
  credentials already present in the environment.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_OP_RUN_TIMEOUT = 20
_CacheKey = Tuple[str, str]  # (env_file_fingerprint, op_path)
_CACHE: Dict[_CacheKey, "_CachedFetch"] = {}


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


@dataclass
class _CachedFetch:
    secrets: Dict[str, str]
    warnings: List[str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


def find_op(*, op_path: str = "") -> Optional[Path]:
    """Return a usable ``op`` binary path, or None.

    Resolution order:
      1. explicit ``op_path`` from config / CLI
      2. ``shutil.which("op")`` / ``op.exe`` on PATH
    """
    configured = str(op_path or "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
        return None

    system = shutil.which(_platform_binary_name()) or shutil.which("op")
    if system:
        return Path(system)
    return None


def _platform_binary_name() -> str:
    return "op.exe" if os.name == "nt" else "op"


def _is_valid_env_name(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def _resolve_env_file(env_file: str, home_path: Optional[Path] = None) -> Path:
    raw = str(env_file or "").strip() or "~/.hermes/secrets/1password.env"
    home = home_path or Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    if raw == "~/.hermes" or raw.startswith("~/.hermes/"):
        suffix = raw.removeprefix("~/.hermes").lstrip("/")
        return home / suffix
    if raw.startswith("~"):
        return Path(raw).expanduser()
    path = Path(raw)
    if path.is_absolute():
        return path
    return home / path


def parse_reference_file(path: Path) -> Tuple[Dict[str, str], List[str]]:
    """Parse an env-style file of ``NAME=op://...`` references.

    Plaintext-looking values are skipped with a warning.  This keeps the
    native 1Password backend honest: the file is for references, not final
    secret values.
    """
    references: Dict[str, str] = {}
    warnings: List[str] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise RuntimeError(f"1Password references file not readable: {path}: {exc}") from exc

    for line_no, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            warnings.append(f"Skipping line {line_no}: expected NAME=op:// reference")
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not _is_valid_env_name(name):
            warnings.append(f"Skipping line {line_no}: {name!r} is not a valid env-var name")
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not value.startswith("op://"):
            warnings.append(
                f"Skipping {name}: value is not a 1Password secret reference (op://...)"
            )
            continue
        references[name] = value
    return references, warnings


def _fingerprint_references(references: Dict[str, str], env_file: Path) -> str:
    h = hashlib.sha256()
    h.update(str(env_file.resolve()).encode("utf-8", errors="replace"))
    for name in sorted(references):
        h.update(b"\0")
        h.update(name.encode("utf-8"))
        h.update(b"=")
        h.update(references[name].encode("utf-8"))
    return h.hexdigest()[:24]


def fetch_onepassword_secrets(
    *,
    env_file: str = "~/.hermes/secrets/1password.env",
    binary: Optional[Path] = None,
    op_path: str = "",
    cache_ttl_seconds: float = 300,
    use_cache: bool = True,
    home_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Resolve all ``op://`` references in ``env_file`` via ``op read``.

    Returns ``(secrets_dict, warnings_list)``.  Secret values are never
    included in warnings or exceptions.
    """
    path = _resolve_env_file(env_file, home_path)
    if not path.exists():
        raise RuntimeError(
            f"1Password references file does not exist: {path}. "
            "Run `hermes secrets onepassword setup`."
        )

    references, warnings = parse_reference_file(path)
    if not references:
        return {}, warnings + ["1Password references file contains no op:// entries"]

    op = binary or find_op(op_path=op_path)
    if op is None:
        raise RuntimeError(
            "1Password CLI `op` is not available. Install it from "
            "https://developer.1password.com/docs/cli/get-started/ "
            "or set secrets.onepassword.op_path."
        )

    cache_key = (_fingerprint_references(references, path), str(op))
    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return cached.secrets, cached.warnings

    secrets: Dict[str, str] = {}
    for name, reference in references.items():
        try:
            secrets[name] = _op_read(op, reference)
        except RuntimeError as exc:
            warnings.append(f"Skipping {name}: {exc}")
            continue

    entry = _CachedFetch(secrets=secrets, warnings=list(warnings), fetched_at=time.time())
    _CACHE[cache_key] = entry
    return secrets, warnings


def _op_read(op: Path, reference: str) -> str:
    try:
        proc = subprocess.run(  # noqa: S603 — op path is either user configured or PATH-resolved
            [str(op), "read", reference],
            capture_output=True,
            text=True,
            timeout=_OP_RUN_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"op timed out after {_OP_RUN_TIMEOUT}s") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke op: {exc}") from exc

    if proc.returncode != 0:
        err = _sanitize_op_error(proc.stderr or proc.stdout or "")
        raise RuntimeError(f"op exited {proc.returncode}: {err}")
    return (proc.stdout or "").rstrip("\r\n")


def _sanitize_op_error(text: str) -> str:
    """Keep op errors useful without echoing references or credential-like data."""
    cleaned = text.strip().replace("\x1b", "")
    for token in cleaned.split():
        if token.startswith("op://"):
            cleaned = cleaned.replace(token, "op://[REDACTED]")
    return cleaned[:200] or "unknown error"


def apply_onepassword_secrets(
    *,
    enabled: bool,
    env_file: str = "~/.hermes/secrets/1password.env",
    service_account_token_env: str = "OP_SERVICE_ACCOUNT_TOKEN",
    override_existing: bool = False,
    cache_ttl_seconds: float = 300,
    op_path: str = "",
    home_path: Optional[Path] = None,
) -> FetchResult:
    """Pull secrets from 1Password and set them on ``os.environ``.

    Defensive by design: returns a :class:`FetchResult` and never raises,
    because external secret sources must not block Hermes startup.
    """
    result = FetchResult()
    if not enabled:
        return result

    binary = find_op(op_path=op_path)
    result.binary_path = binary
    if binary is None:
        result.error = (
            "1Password CLI `op` is not available. Run `hermes secrets "
            "onepassword setup` after installing op."
        )
        return result

    # service_account_token_env is optional: `op` can also use an already
    # authenticated desktop-app / CLI session.  If the env var is present,
    # subprocesses inherit it via os.environ.
    _ = service_account_token_env

    try:
        secrets, warnings = fetch_onepassword_secrets(
            env_file=env_file,
            binary=binary,
            op_path=op_path,
            cache_ttl_seconds=cache_ttl_seconds,
            home_path=home_path,
        )
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(warnings)

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
    """Clear in-process cache for hermetic tests."""
    _CACHE.clear()
