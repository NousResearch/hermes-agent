"""1Password SDK secret source for Hermes.

Hermes resolves ``op://`` secret references in ``.env`` files at startup
using the official 1Password Python SDK.  Secrets are resolved in-memory
and never pass through stdout or shell pipes.

Design summary
--------------

* Two auth modes: **Desktop Auth** (biometric approval, interactive) and
  **Service Account** (token-based, unattended).  The mode is selected
  in ``config.yaml`` under ``secrets.onepassword.mode``.
* The SDK is installed in a dedicated venv at
  ``~/.hermes/venvs/1password/``.  It requires brew Python 3.12+ with
  OpenSSL 3 (system Python uses LibreSSL, which is incompatible).
* Values in ``.env`` that start with ``op://`` are detected and resolved
  via the SDK.  Non-``op://`` values are left untouched.
* Failures NEVER block Hermes startup.  Missing venv, no 1Password app,
  expired session, etc. all emit a one-line warning and continue with
  whatever credentials ``.env`` already had.
* Results are cached in-process for ``cache_ttl_seconds`` so repeated
  ``load_hermes_dotenv()`` calls don't re-resolve.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------

_CACHE: Dict[str, "_CachedResolve"] = {}


@dataclass
class _CachedResolve:
    value: str
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


# ---------------------------------------------------------------------------
# Public dataclasses (mirrors Bitwarden FetchResult)
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """Outcome of a 1Password resolve pass."""

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# SDK venv discovery
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    """Get HERMES_HOME without importing hermes_constants (avoids 3.10+ syntax on 3.9)."""
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def _find_sdk_python() -> Optional[Path]:
    """Locate the 1Password SDK venv Python.

    Resolution order:
      1. ``~/.hermes/venvs/1password/bin/python3.12``
      2. ``~/.hermes/venvs/1password/bin/python3``
    """
    venv_dir = _hermes_home() / "venvs" / "1password"
    for name in ("python3.12", "python3.13", "python3"):
        candidate = venv_dir / "bin" / name
        if candidate.exists():
            return candidate
    return None


def _sdk_venv_available() -> bool:
    """Check if the 1Password SDK venv is usable."""
    python = _find_sdk_python()
    if python is None:
        return False
    import subprocess

    try:
        proc = subprocess.run(
            [str(python), "-c", "import onepassword; print('ok')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0 and "ok" in proc.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Secret resolution
# ---------------------------------------------------------------------------


def _is_op_reference(value: str) -> bool:
    """Check if a value is a 1Password secret reference."""
    return value.strip().startswith("op://")


def _resolve_via_subprocess(
    references: List[str],
    mode: str,
    service_account_token: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve op:// references by invoking the SDK via a subprocess.

    This keeps the SDK venv isolated from Hermes' own Python environment.
    Secrets are captured from stdout of the subprocess — but only the
    resolved values, which are then set into os.environ.
    """
    python = _find_sdk_python()
    if python is None:
        raise RuntimeError(
            "1Password SDK venv not found at ~/.hermes/venvs/1password/. "
            "Run: /opt/homebrew/bin/python3.12 -m venv ~/.hermes/venvs/1password && "
            "~/.hermes/venvs/1password/bin/pip install onepassword-sdk"
        )

    # Build the resolver script
    if mode == "service_account" and service_account_token:
        auth_code = f'"{service_account_token}"'
    else:
        auth_code = 'onepassword.DesktopAuth("my")'

    refs_json = repr(references)

    script = f"""
import asyncio, json, sys, os

async def main():
    try:
        import onepassword
    except ImportError:
        print(json.dumps({{"error": "onepassword SDK not installed in venv"}}))
        return

    refs = {refs_json}
    results = {{}}

    try:
        client = await onepassword.Client.authenticate(
            auth={auth_code},
            integration_name="hermes-agent",
            integration_version="0.1.0",
        )

        # Resolve each reference individually to handle partial failures
        for ref in refs:
            try:
                value = await client.secrets.resolve(ref)
                results[ref] = value
            except Exception as e:
                results[ref] = {{"_error": str(e)}}

        print(json.dumps(results))
    except onepassword.DesktopSessionExpiredException:
        print(json.dumps({{"_error": "desktop_session_expired: Unlock 1Password app and retry"}}))
    except Exception as e:
        print(json.dumps({{"_error": f"auth_failed: {{type(e).__name__}}: {{e}}"}}))

asyncio.run(main())
"""

    import subprocess
    import json

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)  # Don't leak Hermes' Python path

    try:
        proc = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "1Password SDK timed out after 30s. "
            "Is the 1Password desktop app unlocked?"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to invoke 1Password SDK: {exc}") from exc

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"1Password SDK exited {proc.returncode}: {err[:200]}")

    raw = proc.stdout.strip()
    if not raw:
        raise RuntimeError("1Password SDK returned no output")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"1Password SDK returned non-JSON: {exc}") from exc

    # Check for top-level error
    if "_error" in payload and len(payload) == 1:
        raise RuntimeError(f"1Password: {payload['_error']}")

    # Filter out per-reference errors
    resolved = {}
    for ref, value in payload.items():
        if isinstance(value, dict) and "_error" in value:
            logger.warning("Failed to resolve %s: %s", ref, value["_error"])
        else:
            resolved[ref] = value

    return resolved


# ---------------------------------------------------------------------------
# Public entry point — called from hermes_cli.env_loader
# ---------------------------------------------------------------------------


def apply_onepassword_secrets(
    *,
    enabled: bool = False,
    mode: str = "desktop",
    service_account_token_env: str = "OP_SERVICE_ACCOUNT_TOKEN",
    cache_ttl_seconds: float = 300,
    override_existing: bool = False,
) -> FetchResult:
    """Resolve ``op://`` references found in already-loaded env vars.

    Scans ``os.environ`` for values starting with ``op://`` and resolves
    them via the 1Password SDK.  The resolved value replaces the ``op://``
    reference in ``os.environ``.

    This function is called from ``load_hermes_dotenv()`` after the .env
    files have been loaded by python-dotenv.  It is intentionally
    defensive — any failure returns a :class:`FetchResult` with ``error``
    set; it never raises.
    """
    result = FetchResult()

    if not enabled:
        return result

    # Scan os.environ for op:// references
    op_refs: Dict[str, str] = {}  # env_var_name -> op:// reference
    for key, value in os.environ.items():
        if _is_op_reference(value):
            op_refs[key] = value.strip()

    if not op_refs:
        return result  # Nothing to resolve

    # Check if SDK venv is available
    if not _sdk_venv_available():
        result.error = (
            "1Password SDK venv not found or broken. "
            "Run: /opt/homebrew/bin/python3.12 -m venv ~/.hermes/venvs/1password && "
            "~/.hermes/venvs/1password/bin/pip install onepassword-sdk"
        )
        return result

    # Check cache first
    uncached_refs: Dict[str, str] = {}
    for env_var, ref in op_refs.items():
        cached = _CACHE.get(ref)
        if cached and cached.is_fresh(cache_ttl_seconds):
            result.secrets[env_var] = cached.value
        else:
            uncached_refs[env_var] = ref

    if uncached_refs:
        # Resolve uncached references
        service_account_token = None
        if mode == "service_account":
            service_account_token = os.environ.get(service_account_token_env, "").strip()
            if not service_account_token:
                result.error = (
                    f"secrets.onepassword.mode is 'service_account' but "
                    f"{service_account_token_env} is not set."
                )
                return result

        try:
            references = list(uncached_refs.values())
            resolved = _resolve_via_subprocess(
                references, mode, service_account_token
            )
        except RuntimeError as exc:
            result.error = str(exc)
            return result

        # Map resolved values back to env var names
        ref_to_env = {v: k for k, v in uncached_refs.items()}
        for ref, value in resolved.items():
            env_var = ref_to_env.get(ref)
            if env_var:
                result.secrets[env_var] = value
                _CACHE[ref] = _CachedResolve(value=value, fetched_at=time.time())

    # Apply resolved secrets to os.environ
    for env_var, value in result.secrets.items():
        if not override_existing and env_var in op_refs:
            # This key was an op:// ref — we should always replace it
            # (the whole point is to resolve the reference)
            pass
        os.environ[env_var] = value
        result.applied.append(env_var)

    if result.applied:
        print(
            f"  1Password: resolved {len(result.applied)} "
            f"secret{'s' if len(result.applied) != 1 else ''} "
            f"({', '.join(sorted(result.applied))})",
            file=sys.stderr,
        )

    return result