"""TLS verification helpers extracted from ``hermes_cli.auth``."""

from __future__ import annotations

import logging
import os
import ssl
import sys
from typing import Any, Dict, Optional

from utils import is_truthy_value

logger = logging.getLogger("hermes_cli.auth")

# =============================================================================
# TLS verification helper
# =============================================================================

def _default_verify() -> bool | ssl.SSLContext:
    """Platform-aware default SSL verify for httpx clients.

    On macOS with Homebrew Python, the system OpenSSL cannot locate the
    system trust store and valid public certs fail verification. When
    certifi is importable we pin its bundle explicitly; elsewhere we
    defer to httpx's built-in default (certifi via its own dependency).
    Mirrors the weixin fix in 3a0ec1d93.
    """
    if sys.platform == "darwin":
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
    return True


def _resolve_verify(
    *,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    auth_state: Optional[Dict[str, Any]] = None,
) -> bool | ssl.SSLContext:
    tls_state = auth_state.get("tls") if isinstance(auth_state, dict) else {}
    tls_state = tls_state if isinstance(tls_state, dict) else {}

    effective_insecure = (
        is_truthy_value(insecure, default=False) if insecure is not None
        else is_truthy_value(tls_state.get("insecure", False), default=False)
    )
    effective_ca = (
        ca_bundle
        or tls_state.get("ca_bundle")
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
    )

    if effective_insecure:
        return False
    if effective_ca:
        ca_path = str(effective_ca)
        if not os.path.isfile(ca_path):
            logger.warning(
                "CA bundle path does not exist: %s — falling back to default certificates",
                ca_path,
            )
            return _default_verify()
        return ssl.create_default_context(cafile=ca_path)
    return _default_verify()


__all__ = ("_default_verify", "_resolve_verify")
