"""TLS verify resolution for httpx/OpenAI provider clients."""

from __future__ import annotations

import logging
import os
import ssl
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _coerce_insecure(ssl_verify: Any) -> bool:
    if ssl_verify is False:
        return True
    if isinstance(ssl_verify, str) and ssl_verify.strip().lower() in {"false", "0", "no", "off"}:
        return True
    return False


_CA_CONTEXTS: dict[str, ssl.SSLContext] = {}
_CA_CONTEXTS_LOCK = threading.Lock()


def _context_for_ca_bundle(ca_path: str) -> ssl.SSLContext:
    """One ``SSLContext`` per CA bundle path, process-wide.

    ``ssl.create_default_context(cafile=...)`` parses the whole bundle each
    call. Every AIAgent (and every delegated child) resolves verify for its
    own client, so an env/config CA bundle used to cost one parsed context —
    and, because sharing keys on context identity, one private connection
    pool — per agent. An ``SSLContext`` is safe to share across connections.
    """
    with _CA_CONTEXTS_LOCK:
        ctx = _CA_CONTEXTS.get(ca_path)
        if ctx is None:
            ctx = ssl.create_default_context(cafile=ca_path)
            _CA_CONTEXTS[ca_path] = ctx
        return ctx


def resolve_httpx_verify(
    *,
    ca_bundle: Optional[str] = None,
    ssl_verify: Any = None,
    base_url: str = "",
) -> bool | ssl.SSLContext:
    """Resolve httpx ``verify`` for provider HTTP clients.

    Priority:
    1. ``ssl_verify: false`` — disable verification (local dev only)
    2. explicit ``ca_bundle`` (per-provider ``ssl_ca_cert`` config field)
    3. ``HERMES_CA_BUNDLE``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``,
       ``CURL_CA_BUNDLE`` env vars
    4. ``True`` (httpx/certifi default)

    ``base_url`` is used only for the insecure-mode warning message.
    """
    if _coerce_insecure(ssl_verify):
        logger.warning(
            "TLS certificate verification DISABLED (ssl_verify: false) for %s — "
            "this is intended for local development only and is unsafe on any "
            "network you do not fully control.",
            base_url or "a custom provider endpoint",
        )
        return False

    effective_ca = (
        (ca_bundle or "").strip()
        or os.getenv("HERMES_CA_BUNDLE", "").strip()
        or os.getenv("SSL_CERT_FILE", "").strip()
        or os.getenv("REQUESTS_CA_BUNDLE", "").strip()
        or os.getenv("CURL_CA_BUNDLE", "").strip()
    )
    if effective_ca:
        ca_path = str(Path(effective_ca).expanduser())
        if os.path.isfile(ca_path):
            return _context_for_ca_bundle(ca_path)
        logger.warning(
            "CA bundle path does not exist: %s — falling back to default certificates",
            effective_ca,
        )
    return True
