"""Process-wide CA bundle selection for Hermes entry points."""

from __future__ import annotations

import logging
import os
import ssl
import sys

logger = logging.getLogger(__name__)


def _certifi_bundle() -> str | None:
    try:
        import certifi
    except ImportError:
        return None

    bundle = certifi.where()
    return bundle if os.path.exists(bundle) else None


def ensure_ssl_certs() -> None:
    """Set ``SSL_CERT_FILE`` when Hermes must choose the process default."""
    configured_cert = os.environ.get("SSL_CERT_FILE")
    if configured_cert:
        if os.path.exists(configured_cert):
            return
        logger.warning(
            "Ignoring stale SSL_CERT_FILE=%r because the path does not exist",
            configured_cert,
        )
        os.environ.pop("SSL_CERT_FILE", None)

    certifi_bundle = _certifi_bundle()
    if sys.platform == "darwin" and certifi_bundle:
        os.environ["SSL_CERT_FILE"] = certifi_bundle
        return

    paths = ssl.get_default_verify_paths()
    for candidate in (paths.cafile, paths.openssl_cafile):
        if candidate and os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

    if certifi_bundle:
        os.environ["SSL_CERT_FILE"] = certifi_bundle
        return

    for candidate in (
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
        "/etc/ssl/ca-bundle.pem",
        "/etc/ssl/cert.pem",
        "/etc/pki/tls/cert.pem",
        "/usr/local/etc/openssl@1.1/cert.pem",
        "/opt/homebrew/etc/openssl@1.1/cert.pem",
    ):
        if os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return


__all__ = ["ensure_ssl_certs"]
