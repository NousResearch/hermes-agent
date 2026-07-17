"""TLS verify resolution for httpx/OpenAI provider clients."""

from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _coerce_insecure(ssl_verify: Any) -> bool:
    if ssl_verify is False:
        return True
    if isinstance(ssl_verify, str) and ssl_verify.strip().lower() in {"false", "0", "no", "off"}:
        return True
    return False


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
            return ssl.create_default_context(cafile=ca_path)
        logger.warning(
            "CA bundle path does not exist: %s — falling back to default certificates",
            effective_ca,
        )
    return True


def resolve_requests_verify(
    *,
    ca_bundle: Optional[str] = None,
    ssl_verify: Any = None,
    base_url: str = "",
) -> bool | str:
    """Resolve ``verify=`` for ``requests``-library metadata/pricing probes.

    Mirrors :func:`resolve_httpx_verify` for the provider-scoped TLS case but
    returns a path or ``bool`` instead of an :class:`ssl.SSLContext`, because
    ``requests.get(verify=...)`` accepts only a filesystem path or a bool —
    not an ``SSLContext``.

    This closes the split described in issue #66544: per-provider
    ``ssl_ca_cert`` / ``ssl_verify`` reached the httpx inference clients but
    not the ``requests``-based ``/models`` metadata and pricing probes in
    ``agent.model_metadata``. With this resolver, a custom endpoint's
    provider-scoped CA applies to those probes too.

    Priority:
    1. ``ssl_verify: false`` — disable verification (local dev only)
    2. explicit ``ca_bundle`` (per-provider ``ssl_ca_cert`` config field)
    3. ``HERMES_CA_BUNDLE``, ``REQUESTS_CA_BUNDLE``, ``SSL_CERT_FILE``,
       ``CURL_CA_BUNDLE`` env vars (first existing file wins; non-existent
       paths are skipped so a typo'd higher-priority var does not mask a
       valid lower-priority one)
    4. ``True`` (``requests``/certifi default)
    """
    if _coerce_insecure(ssl_verify):
        logger.warning(
            "TLS certificate verification DISABLED (ssl_verify: false) for %s — "
            "this is intended for local development only and is unsafe on any "
            "network you do not fully control.",
            base_url or "a custom provider endpoint",
        )
        return False

    if ca_bundle:
        ca_path = str(Path(ca_bundle).expanduser())
        if os.path.isfile(ca_path):
            return ca_path
        logger.warning(
            "CA bundle path does not exist: %s — falling back to env/default certificates",
            ca_bundle,
        )

    for env_var in ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        val = os.getenv(env_var)
        if val and os.path.isfile(val):
            return val
    return True
