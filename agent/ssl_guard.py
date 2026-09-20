"""Preventive SSL CA certificate checks — catch broken CA bundle paths before
OpenAI/httpx turns them into an opaque ``FileNotFoundError``."""

from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path

from agent.errors import SSLConfigurationError
from utils import is_truthy_value

logger = logging.getLogger(__name__)

_CA_BUNDLE_ENV_VARS = ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
_REPAIR_HINT = (
    "Repair: run `hermes doctor --fix` (auto-reinstalls certifi), or "
    "manually: python -m pip install --force-reinstall certifi openai httpx\n"
    "If you configured a custom corporate CA bundle, fix or unset the broken CA bundle environment variable."
)


def _ssl_err(message: str) -> SSLConfigurationError:
    """Create a consistent, user-actionable SSL configuration error."""
    return SSLConfigurationError(f"{message}\n{_REPAIR_HINT}")


def _validate_bundle_path(label: str, value: str, *, require_substantial: bool = False) -> None:
    path = Path(value).expanduser()
    if not path.exists():
        raise _ssl_err(f"{label} points to a missing CA bundle: {value}")
    if not path.is_file():
        raise _ssl_err(f"{label} does not point to a CA bundle file: {value}")
    if require_substantial and path.stat().st_size < 1024:
        raise _ssl_err(f"{label} at {value} appears corrupted (too small)")
    try:
        ctx = ssl.create_default_context(cafile=str(path))
    except Exception as exc:
        raise _ssl_err(f"{label} CA bundle at {value} cannot be loaded: {exc}") from exc
    try:
        loaded_certs = ctx.get_ca_certs()
    except NotImplementedError:  # truststore-backed SSLContext (Windows) lacks get_ca_certs(); loading validated it
        return
    if not loaded_certs:
        raise _ssl_err(f"{label} CA bundle at {value} did not load any certificates")


def verify_ca_bundle() -> None:
    """Raise SSLConfigurationError when a CA-bundle env var points at a bad path or certifi's ``cacert.pem``
    is missing/corrupt."""
    if is_truthy_value(os.getenv("HERMES_SKIP_SSL_GUARD", "")):
        logger.debug("SSL CA bundle guard skipped via HERMES_SKIP_SSL_GUARD")
        return
    for env_var in _CA_BUNDLE_ENV_VARS:
        if value := os.getenv(env_var):
            _validate_bundle_path(env_var, value)
    try:
        import certifi
    except Exception as exc:
        raise _ssl_err(f"certifi is not importable: {exc}") from exc
    _validate_bundle_path("certifi", str(certifi.where()), require_substantial=True)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def verify_ca_bundle_with_fallback() -> None:
    """Backward-compatible wrapper for older call sites.

    The old PR name mentioned a platform fallback, but allowing startup with a
    broken certifi bundle still leaves httpx/OpenAI and requests call sites
    failing later. Keep the wrapper name but enforce the same check.
    """
    verify_ca_bundle()
# ---- END PLUGIN-COMPAT ----
