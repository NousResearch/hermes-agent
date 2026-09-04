"""TLS trust for Hermes. One authority, one trust source.

Trust comes from the platform verifier via ``truststore``: CryptoAPI on
Windows, Security.framework on macOS, and OpenSSL's own store (with a
distro candidate sweep) on Linux. That is the machine's real answer to
"is this chain valid", so a corporate MITM root installed by MDM, an
internal CA, and a locked-down NixOS box all work with no configuration.

``install_truststore()`` runs once at process start and patches
``ssl.SSLContext`` process-wide, so every stack that builds a default
context inherits it — httpx, requests/urllib3, aiohttp, AND the stdlib
``urllib.request`` call sites (the llama.cpp engine download among them)
that a certifi-only or requests-only approach never reached.

The only thing above the platform store is EXPLICIT PER-PROVIDER CONFIG:
``ssl_ca_cert`` (a self-signed or internal endpoint's bundle) and
``ssl_verify: false`` (local development). Those are deliberate
statements about one endpoint, not an ambient guess about the machine.
"""

from __future__ import annotations

import logging
import ssl
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_installed: bool | None = None


def install_truststore() -> bool:
    """Point every default SSLContext at the OS trust store. Idempotent.

    Returns True when the platform verifier is in force. A False return
    means truststore could not load (an unsupported platform, or a
    stripped payload); TLS still works off OpenSSL's compiled-in paths,
    it just won't see certificates the OS trusts.

    Call before the first HTTPS client is constructed.
    """
    global _installed
    if _installed is not None:
        return _installed
    try:
        import truststore

        truststore.inject_into_ssl()
        _installed = True
        logger.debug("TLS trust: platform store (truststore)")
    except Exception as exc:  # noqa: BLE001 — never break startup over TLS setup
        _installed = False
        logger.warning(
            "truststore unavailable (%s); falling back to OpenSSL's default "
            "trust paths. Certificates trusted only by the OS store — a "
            "corporate root, for instance — will not verify.",
            exc,
        )
    return _installed


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


def resolve_httpx_verify(
    *,
    ca_bundle: Optional[str] = None,
    ssl_verify: Any = None,
    base_url: str = "",
) -> bool | ssl.SSLContext:
    """Resolve ``verify`` for an HTTP client.

    1. ``ssl_verify: false`` — verification off (local development only)
    2. explicit ``ca_bundle`` (the provider's ``ssl_ca_cert``) — that
       bundle INSTEAD of the platform store, for an endpoint whose chain
       the machine has no reason to trust
    3. ``True`` — the platform store, via the process-wide install

    ``base_url`` only labels the insecure-mode warning.
    """
    install_truststore()

    if _coerce_insecure(ssl_verify):
        logger.warning(
            "TLS certificate verification DISABLED (ssl_verify: false) for %s — "
            "this is intended for local development only and is unsafe on any "
            "network you do not fully control.",
            base_url or "a custom provider endpoint",
        )
        return False

    effective_ca = (ca_bundle or "").strip()
    if effective_ca:
        path = Path(effective_ca).expanduser()
        if path.is_file():
            # An explicit bundle REPLACES the platform store for this client.
            # inject_into_ssl() rebinds ssl.SSLContext to truststore's class,
            # which ignores loaded CA files and verifies against the OS store
            # anyway — so build from the ORIGINAL class, which truststore
            # keeps for exactly this purpose. Without it a pinned
            # ssl_ca_cert silently means "trust the machine" instead.
            #
            # Do NOT assign verify_mode/check_hostname here: the stock
            # class's setters resolve ``super(SSLContext, SSLContext)``
            # against the PATCHED module global and recurse to the stack
            # limit. PROTOCOL_TLS_CLIENT already implies CERT_REQUIRED and
            # check_hostname, so there is nothing to set.
            from truststore._ssl_constants import _original_SSLContext

            ctx = _original_SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.load_verify_locations(cafile=str(path))
            return ctx
        logger.warning(
            "ssl_ca_cert path does not exist: %s — using the OS trust store instead",
            effective_ca,
        )
    return True


def resolve_requests_verify(
    *,
    ca_bundle: Optional[str] = None,
    ssl_verify: Any = None,
    base_url: str = "",
) -> bool | str:
    """The same decision for ``requests``, which takes a path, not a context.

    With the platform store installed process-wide, ``True`` already means
    "verify against the OS store" here — urllib3 builds its context through
    the patched ``ssl.SSLContext``.
    """
    install_truststore()

    if _coerce_insecure(ssl_verify):
        logger.warning(
            "TLS certificate verification DISABLED (ssl_verify: false) for %s — "
            "this is intended for local development only and is unsafe on any "
            "network you do not fully control.",
            base_url or "a custom provider endpoint",
        )
        return False

    bundle = (ca_bundle or "").strip()
    if bundle:
        path = Path(bundle).expanduser()
        if path.is_file():
            return str(path)
        logger.warning(
            "ssl_ca_cert path does not exist: %s — using the OS trust store instead",
            bundle,
        )
    return True
