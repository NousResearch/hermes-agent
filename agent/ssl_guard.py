"""Preventive SSL CA certificate checks — catch broken CA bundle paths before
OpenAI/httpx turns them into an opaque ``FileNotFoundError``."""

from __future__ import annotations

import errno
import logging
import os
import ssl
from pathlib import Path
from typing import NoReturn

from agent.errors import SSLConfigurationError
from utils import is_truthy_value

logger = logging.getLogger(__name__)

_CA_BUNDLE_ENV_VARS = ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
_REPAIR_HINT = (
    "Repair: run `hermes doctor --fix` (auto-reinstalls certifi), or "
    "manually: python -m pip install --force-reinstall certifi openai httpx\n"
    "If you configured a custom corporate CA bundle, fix or unset the broken CA bundle environment variable."
)
_FD_EXHAUSTION_HINT = (
    "This process has run out of file descriptors (EMFILE). The CA bundle file "
    "is probably intact — reinstalling certifi will not help.\n"
    "Repair: restart the Hermes process that hit the limit. For Hermes Desktop "
    "connected over SSH, quit and reopen the app (or kill the remote "
    "`hermes serve --isolated` process so Desktop respawns it). "
    "Long-lived backends that keep leaking fds should be restarted after "
    "`hermes update`; raising nofile only delays the outage."
)


def is_fd_exhaustion_error(exc: BaseException) -> bool:
    """True when *exc* (or its cause chain) is EMFILE/ENFILE / "too many open files".

    ``ssl.create_default_context(cafile=...)`` needs a spare fd to open the
    bundle. When the process is already at RLIMIT_NOFILE the OSError is wrapped
    as "CA bundle cannot be loaded", which used to send operators at
    ``hermes doctor --fix`` / a certifi reinstall. See #88033.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in (errno.EMFILE, errno.ENFILE):
            return True
        text = str(current).lower()
        if "too many open files" in text or "emfile" in text or "[errno 24]" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def _ssl_err(message: str, *, fd_exhausted: bool = False) -> SSLConfigurationError:
    """Create a consistent, user-actionable SSL configuration error."""
    hint = _FD_EXHAUSTION_HINT if fd_exhausted else _REPAIR_HINT
    return SSLConfigurationError(f"{message}\n{hint}")


def _reraise_bundle_load_error(label: str, value: str, exc: BaseException) -> NoReturn:
    raise _ssl_err(
        f"{label} CA bundle at {value} cannot be loaded: {exc}",
        fd_exhausted=is_fd_exhaustion_error(exc),
    ) from exc


def _validate_bundle_path(label: str, value: str, *, require_substantial: bool = False) -> None:
    try:
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
            _reraise_bundle_load_error(label, value, exc)
        try:
            loaded_certs = ctx.get_ca_certs()
        except NotImplementedError:  # truststore-backed SSLContext (Windows) lacks get_ca_certs(); loading validated it
            return
        if not loaded_certs:
            raise _ssl_err(f"{label} CA bundle at {value} did not load any certificates")
    except SSLConfigurationError:
        raise
    except OSError as exc:
        _reraise_bundle_load_error(label, value, exc)


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
        raise _ssl_err(f"certifi is not importable: {exc}", fd_exhausted=is_fd_exhaustion_error(exc)) from exc
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
