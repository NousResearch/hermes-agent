"""SSL CA certificate diagnostic guard.

Pre-flight check that runs before any HTTP client (OpenAI, Anthropic,
etc.) is constructed. If an explicit CA bundle is configured via the
environment, or if certifi's default bundle is missing/empty/unloadable,
the agent fails fast with an actionable error instead of entering a
crash-loop for every incoming message.
"""

from __future__ import annotations

import logging
import os
import ssl
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_CA_BUNDLE_ENV_VARS = ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")


@dataclass(frozen=True)
class _CABundleCandidate:
    path: str
    source: str


def _ssl_err(message: str, *, instruction: str | None = None):
    """Build an :class:`agent.errors.SSLConfigurationError`."""
    from agent.errors import SSLConfigurationError

    return SSLConfigurationError(message, instruction=instruction)


def _instruction_for_source(source: str) -> str | None:
    if source in _CA_BUNDLE_ENV_VARS:
        return (
            f"Unset {source} or point it at a valid PEM CA bundle, then restart Hermes.\n"
            "Doc:  docs/rca-ssl-cacert-post-git-pull.md"
        )
    return None


def _resolve_ca_bundle() -> _CABundleCandidate:
    """Return the CA bundle path Hermes should validate.

    Explicit environment variables win over certifi so invalid operator
    configuration is caught before HTTP clients start up. If no explicit
    bundle is configured, fall back to certifi's bundled cacert.pem.
    """
    for env_var in _CA_BUNDLE_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return _CABundleCandidate(path=value, source=env_var)

    try:
        import certifi  # noqa: SC100
    except ImportError:
        raise _ssl_err(
            "The 'certifi' package is not installed. This usually means the "
            "virtual environment is stale after a git pull."
        )
    return _CABundleCandidate(path=certifi.where(), source="certifi")


def check_ssl_ca_bundle() -> None:
    """Verify the configured CA certificate bundle is loadable.

    Raises :class:`~agent.errors.SSLConfigurationError` when the bundle
    is missing, empty, or corrupt, so the caller can surface a clear
    remediation message instead of an opaque ``RuntimeError``.
    """
    candidate = _resolve_ca_bundle()
    ca_bundle = candidate.path
    source = candidate.source

    if not ca_bundle or not os.path.isfile(ca_bundle) or os.path.getsize(ca_bundle) == 0:
        raise _ssl_err(
            f"CA certificate bundle from {source} is missing or empty: {ca_bundle}.",
            instruction=_instruction_for_source(source),
        )

    # Try to load the bundle into an SSL context — this is the operation
    # that actually fails when certifi/env configuration is stale or broken.
    try:
        ctx = ssl.create_default_context(cafile=ca_bundle)
    except Exception as exc:
        raise _ssl_err(
            f"CA certificate bundle from {source} at {ca_bundle} cannot be loaded: {exc}.",
            instruction=_instruction_for_source(source),
        )

    # Paranoid check: ensure at least one certificate was parsed.
    # On macOS the system trust store may still work even without
    # certifi, so if this check fails for the default certifi path we fall
    # back to a system-only context before declaring the environment broken.
    if not ctx.get_ca_certs():
        if source != "certifi":
            raise _ssl_err(
                f"CA certificate bundle from {source} at {ca_bundle} did not load "
                "any CA certificates.",
                instruction=_instruction_for_source(source),
            )
        try:
            fallback = ssl.create_default_context()
            if not fallback.get_ca_certs():
                raise _ssl_err(
                    f"CA certificate bundle from {source} at {ca_bundle} is empty and "
                    "no system CA certificates are available."
                )
            logger.debug(
                "certifi bundle at %s is empty but system CA store is ok", ca_bundle
            )
        except Exception:
            raise  # re-raise whatever _ssl_err produced


# Re-export so tests can patch
try:
    from agent.errors import SSLConfigurationError
except ImportError:
    SSLConfigurationError = Exception  # type: ignore[misc,assignment]
