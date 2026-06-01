"""Preventive SSL CA certificate guard for Hermes Agent.

This module provides an early fail-fast check to detect corrupted or missing
certifi CA bundles before any network client is initialized.
"""

import logging
import os
import platform
import ssl
from pathlib import Path

import certifi

from agent.errors import SSLConfigurationError

logger = logging.getLogger(__name__)


def _ssl_err(message: str) -> SSLConfigurationError:
    """Helper to create a consistent error with remediation hint."""
    return SSLConfigurationError(message + "\nRun: pip install -e .")


def verify_ca_bundle() -> None:
    """Verify that the certifi CA bundle is valid and loadable.

    Raises:
        SSLConfigurationError: If the bundle is missing, empty, or cannot be
            used to create a working SSLContext.
    """
    if os.getenv("HERMES_SKIP_SSL_GUARD"):
        logger.debug("SSL guard skipped via HERMES_SKIP_SSL_GUARD")
        return

    ca_bundle = str(certifi.where())
    bundle_path = Path(ca_bundle)

    if not bundle_path.exists():
        raise _ssl_err(f"certifi CA bundle not found at {ca_bundle}")

    if bundle_path.stat().st_size < 1024:
        raise _ssl_err(f"certifi CA bundle at {ca_bundle} appears corrupted (too small)")

    try:
        ctx = ssl.create_default_context(cafile=ca_bundle)
    except Exception as exc:
        raise _ssl_err(
            f"CA certificate bundle at {ca_bundle} cannot be loaded: {exc}"
        ) from exc

    # Paranoid check + macOS fallback
    if not ctx.get_ca_certs():
        try:
            fallback = ssl.create_default_context()
            if not fallback.get_ca_certs():
                raise _ssl_err(
                    f"CA certificate bundle at {ca_bundle} is empty and "
                    "no system CA certificates are available."
                )
            logger.debug(
                "certifi bundle at %s is empty but system CA store is ok", ca_bundle
            )
        except Exception:
            raise


def verify_ca_bundle_with_fallback() -> None:
    """Verify CA bundle with macOS paranoid fallback.

    On macOS, if certifi fails but the system trust store works,
    we allow startup (some corporate proxies / MDM setups break certifi).
    The fallback only applies to "empty/unloadable" cases, not to
    completely missing files.
    """
    try:
        verify_ca_bundle()
    except SSLConfigurationError as e:
        if platform.system() == "Darwin" and "not found" not in str(e).lower():
            try:
                context = ssl.create_default_context()
                if context.get_ca_certs():
                    logger.warning(
                        "certifi bundle invalid but macOS system trust store works. "
                        "Proceeding with reduced security."
                    )
                    return
            except Exception:
                pass
        raise
