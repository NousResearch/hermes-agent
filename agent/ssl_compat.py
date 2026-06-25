"""SSL/TLS compatibility layer for Hermes Agent.

Auto-configures SSL on platforms where the system CA bundle is outdated
(macOS 10.13/High Sierra, Windows corporate networks with proxy-inspected certs).

On macOS with certifi installed, sets ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE``
so that all stdlib ``ssl``, ``requests``, and ``httpx`` callers transparently use
the modern CA bundle from certifi.  Respects user-explicit env vars and a
``tls`` config section so the user can override or disable verification.

Usage:
    from agent.ssl_compat import setup_ssl_compat
    setup_ssl_compat()  # call once at process startup
"""

from __future__ import annotations

import logging
import os
import platform
import ssl
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Env vars that the ``ssl`` stdlib module / ``requests`` / ``httpx`` respect.
# We set all three so every callsite is covered regardless of which library it uses.
_CA_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HERMES_CA_BUNDLE")


def _detect_macos_high_sierra() -> bool:
    """Return True if running on macOS High Sierra (10.13) or older."""
    if sys.platform != "darwin":
        return False
    try:
        release = platform.mac_ver()[0]
        if release:
            major_minor = tuple(int(x) for x in release.split(".")[:2])
            return major_minor <= (10, 13)
    except (ValueError, TypeError):
        pass
    return False


def _certifi_path() -> Optional[str]:
    """Return certifi's CA bundle path, or None if not installed."""
    try:
        import certifi  # noqa: F811

        return certifi.where()
    except ImportError:
        return None


def _any_ca_env_var_set() -> bool:
    """Return True if any known CA env var is already set to an existing file."""
    for var in _CA_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val and os.path.isfile(val):
            return True
    return False


# =============================================================================
# Public API
# =============================================================================


def setup_ssl_compat(config: Optional[dict] = None) -> None:
    """One-time process-wide SSL setup.

    Should be called at process startup, before any network calls are made.
    Safe to call multiple times — subsequent calls are no-ops once env vars
    have been set.

    Parameters
    ----------
    config:
        Optional ``tls`` subsection of the agent config.  Recognised keys:

        ``ca_bundle``
            Path to a custom CA bundle PEM file.
        ``insecure``
            ``True`` to disable certificate verification entirely.

    Environment variables (checked in order of precedence):

    1. ``HERMES_CA_BUNDLE`` (Hermes convention — highest precedence)
    2. ``SSL_CERT_FILE`` (stdlib / httpx convention)
    3. ``REQUESTS_CA_BUNDLE`` (requests convention)

    If none of these are set:

    - On **macOS High Sierra (10.13) or older**: automatically uses
      ``certifi.where()`` if certifi is installed, and logs a warning
      suggesting the user ``pip install certifi`` if it isn't.
    - On **other platforms**: does nothing — the system CA bundle is
      presumed adequate.
    """
    # --- Guard: already configured? ---
    if _any_ca_env_var_set():
        logger.debug("ssl_compat: CA env var already set; skipping setup")
        return

    # --- Respect config overrides ---
    tls_cfg = (config or {}).get("tls") or {}
    ca_bundle: Optional[str] = tls_cfg.get("ca_bundle")
    insecure: bool = bool(tls_cfg.get("insecure", False))

    if insecure:
        # The caller is expected to pass verify=False on each client.
        # We do NOT set env vars here because that would turn off
        # verification for ALL requests, including ones the user
        # might not expect.
        logger.info("ssl_compat: TLS verification disabled via config")
        return

    # --- Custom CA bundle from config ---
    if ca_bundle:
        ca_path = os.path.expanduser(ca_bundle)
        if os.path.isfile(ca_path):
            for var in _CA_ENV_VARS:
                os.environ[var] = ca_path
            logger.info("ssl_compat: using custom CA bundle from %s", ca_path)
            return
        logger.warning(
            "ssl_compat: configured ca_bundle %r not found; falling back", ca_bundle
        )

    # --- Auto-detect: macOS 10.13 + certifi ---
    is_old_macos = _detect_macos_high_sierra()
    certifi_path = _certifi_path()

    if is_old_macos and certifi_path:
        for var in _CA_ENV_VARS:
            os.environ[var] = certifi_path
        logger.info(
            "ssl_compat: macOS 10.13 detected — pinned CA bundle to certifi (%s)",
            certifi_path,
        )
        return

    if is_old_macos and not certifi_path:
        logger.warning(
            "ssl_compat: macOS 10.13 detected but certifi is not installed. "
            "Run `pip install certifi` to fix SSL certificate verification errors."
        )

    # --- certifi available on any platform ---
    if certifi_path and not _any_ca_env_var_set():
        # Set env vars so that all subsequent requests/httpx calls benefit.
        # This is safe on any platform — certifi is a well-maintained bundle
        # that's strictly newer than most system bundles.
        for var in _CA_ENV_VARS:
            os.environ[var] = certifi_path
        logger.debug(
            "ssl_compat: pinned CA bundle to certifi (%s)", certifi_path
        )


def resolve_requests_verify(config: Optional[dict] = None) -> bool | str:
    """Resolve SSL verify param for the ``requests`` library.

    Precedence:
    1. ``tls.insecure`` config → ``False``
    2. ``tls.ca_bundle`` config → path (if file exists)
    3. Known CA env vars → path (if file exists)
    4. certifi fallback → ``certifi.where()``
    5. Default → ``True`` (defer to requests built-in)
    """
    tls_cfg = (config or {}).get("tls") or {}

    if tls_cfg.get("insecure", False):
        return False

    ca_bundle: Optional[str] = tls_cfg.get("ca_bundle")
    if ca_bundle:
        ca_path = os.path.expanduser(ca_bundle)
        if os.path.isfile(ca_path):
            return ca_path

    for var in _CA_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val and os.path.isfile(val):
            return val

    certifi_path = _certifi_path()
    if certifi_path:
        return certifi_path

    return True


def resolve_httpx_verify(config: Optional[dict] = None) -> bool | ssl.SSLContext:
    """Resolve SSL verify param for the ``httpx`` library.

    Returns an ``ssl.SSLContext`` when a custom CA bundle is available,
    ``False`` when verification is disabled, or ``True`` for the default.
    """
    tls_cfg = (config or {}).get("tls") or {}

    if tls_cfg.get("insecure", False):
        return False

    ca_bundle: Optional[str] = tls_cfg.get("ca_bundle")
    if ca_bundle:
        ca_path = os.path.expanduser(ca_bundle)
        if os.path.isfile(ca_path):
            return ssl.create_default_context(cafile=ca_path)

    for var in _CA_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val and os.path.isfile(val):
            return ssl.create_default_context(cafile=val)

    certifi_path = _certifi_path()
    if certifi_path:
        return ssl.create_default_context(cafile=certifi_path)

    return True
