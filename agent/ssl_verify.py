"""TLS verify resolution for httpx/OpenAI provider clients."""

from __future__ import annotations

import logging
import os
import ssl
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CA_BUNDLE_ENV_VARS = ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
_INSECURE_STRINGS = {"false", "0", "no", "off"}
_CA_CONTEXTS: dict[str, ssl.SSLContext] = {}
_CA_CONTEXTS_LOCK = threading.Lock()


def _context_for_ca_bundle(ca_path: str) -> ssl.SSLContext:
    """One ``SSLContext`` per CA bundle path, process-wide.

    ``ssl.create_default_context(cafile=...)`` parses the whole bundle each call, and httpx
    transport sharing keys on context identity — so a per-agent context cost one parsed bundle
    AND one private connection pool per agent (and per delegated child). An ``SSLContext`` is
    safe to share across connections.
    """
    with _CA_CONTEXTS_LOCK:
        ctx = _CA_CONTEXTS.get(ca_path)
        if ctx is None:
            ctx = ssl.create_default_context(cafile=ca_path)
            _CA_CONTEXTS[ca_path] = ctx
        return ctx


def resolve_httpx_verify(*, ca_bundle: Optional[str] = None, ssl_verify: Any = None, base_url: str = "") -> bool | ssl.SSLContext:
    """Resolve httpx ``verify``: ``ssl_verify: false`` > explicit ``ca_bundle`` >
    CA-bundle env vars > ``True`` (certifi default). ``base_url`` only feeds the warning."""
    if ssl_verify is False or (isinstance(ssl_verify, str) and ssl_verify.strip().lower() in _INSECURE_STRINGS):
        logger.warning(
            "TLS certificate verification DISABLED (ssl_verify: false) for %s — "
            "this is intended for local development only and is unsafe on any "
            "network you do not fully control.",
            base_url or "a custom provider endpoint",
        )
        return False

    effective_ca = (ca_bundle or "").strip() or next(
        (v for v in (os.getenv(var, "").strip() for var in _CA_BUNDLE_ENV_VARS) if v), "",
    )
    if effective_ca:
        ca_path = str(Path(effective_ca).expanduser())
        if os.path.isfile(ca_path):
            return _context_for_ca_bundle(ca_path)
        logger.warning("CA bundle path does not exist: %s — falling back to default certificates", effective_ca)
    return True
