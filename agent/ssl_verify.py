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
        try:
            _repair_stdlib_context_setters()
        except Exception:  # noqa: BLE001 — private ssl internals; never break startup over it
            logger.debug("stdlib SSLContext setter repair skipped", exc_info=True)
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


_CA_CONTEXTS: dict[str | None, ssl.SSLContext] = {}
_CA_CONTEXTS_LOCK = threading.Lock()


def _stdlib_ssl_context_class() -> type[ssl.SSLContext]:
    """The un-injected stdlib ``ssl.SSLContext``.

    An explicit bundle must REPLACE OS trust, and truststore's context falls
    back to the OS verifier whenever the loaded bundle rejects a chain — so
    the bundle context has to be built from the stdlib class. Once injected
    (here, or by pm.launch/pm.worker before this module loads) the only
    handle on it is truststore's own saved reference; when nothing is
    injected — truststore absent or not installed — ``ssl.SSLContext`` is
    already the stdlib class and truststore must not be imported at all.
    """
    if ssl.SSLContext.__module__ == "ssl":
        return ssl.SSLContext
    from truststore._ssl_constants import _original_SSLContext

    return _original_SSLContext


# CPython 3.14 implements these ``SSLContext`` property setters as
# ``super(SSLContext, SSLContext).<prop>.__set__`` — the name resolves through the ``ssl``
# module global, which ``inject_into_ssl()`` repoints at truststore's subclass.
_STDLIB_CONTEXT_SETTERS = (
    "minimum_version", "maximum_version", "options", "verify_flags", "verify_mode", "_msg_callback",
)


def _repair_stdlib_context_setters() -> None:
    """Keep the pre-injection stdlib ``SSLContext`` usable after ``inject_into_ssl()``.

    Every setter in ``_STDLIB_CONTEXT_SETTERS`` resolves ``SSLContext`` through the module
    global, which the injection repoints at truststore's subclass. On the pre-injection
    class — still referenced by anything that captured it first, as botocore does when it
    binds urllib3's alias into ``botocore.httpsession`` at its import time — ``super()``
    then walks back to the very same property and recurses until ``RecursionError``:
    botocore's vendored ``create_urllib3_context`` dies on ``context.options |= options``
    (#126808). Rebind those setters to the C-level descriptors the ``super()`` call was
    reaching for. No-op on ≤3.13, where ``SSLContext`` is the C type itself.
    """
    original = _stdlib_ssl_context_class()
    if original is ssl.SSLContext:
        return  # nothing was injected — the stdlib class is already in force
    for name in _STDLIB_CONTEXT_SETTERS:
        prop = original.__dict__.get(name)
        c_descriptor = ssl._SSLContext.__dict__.get(name)
        if isinstance(prop, property) and c_descriptor is not None:
            setattr(original, name, property(prop.fget, c_descriptor.__set__))


def _shared_context(ca_path: str | None) -> ssl.SSLContext:
    """A stable context identity lets clients reuse the existing transport pool."""
    with _CA_CONTEXTS_LOCK:
        ctx = _CA_CONTEXTS.get(ca_path)
        if ctx is None:
            if ca_path is not None:
                # PROTOCOL_TLS_CLIENT sets hostname checking and CERT_REQUIRED;
                # assigning the original class's properties after injection recurses.
                ctx = _stdlib_ssl_context_class()(ssl.PROTOCOL_TLS_CLIENT)
                ctx.load_verify_locations(cafile=ca_path)
            else:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                if not _installed:
                    ctx.load_default_certs()
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
            return _shared_context(str(path.resolve()))
        logger.warning(
            "ssl_ca_cert path does not exist: %s — using the OS trust store instead",
            effective_ca,
        )
    # HTTPX reads CA env vars before the injected verifier gets control. Pass
    # the platform context directly so stale paths cannot break construction;
    # proxy environment handling remains enabled.
    import os

    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return _shared_context(None)
    return True
