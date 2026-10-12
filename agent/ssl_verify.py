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

An exported ``SSL_CERT_FILE`` bundle is trusted ON TOP of the platform
store; only EXPLICIT PER-PROVIDER CONFIG replaces that trust:
``ssl_ca_cert`` (a self-signed or internal endpoint's bundle) and
``ssl_verify: false`` (local development). Those are deliberate
statements about one endpoint, not an ambient guess about the machine.
"""

from __future__ import annotations

import logging
import os
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
    except Exception as exc:
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


_CA_CONTEXTS: dict[tuple[str | None, bool], ssl.SSLContext] = {}
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


def _shared_context(ca_path: str | None, union: bool = False) -> ssl.SSLContext:
    """A stable context identity lets clients reuse the existing transport pool.

    ``union`` adds ``ca_path`` to the platform store (the ``SSL_CERT_FILE``
    contract) instead of replacing it (a provider ``ssl_ca_cert``).
    """
    with _CA_CONTEXTS_LOCK:
        ctx = _CA_CONTEXTS.get((ca_path, union))
        if ctx is None:
            if ca_path is not None and not union:
                # PROTOCOL_TLS_CLIENT sets hostname checking and CERT_REQUIRED;
                # assigning the original class's properties after injection recurses.
                ctx = _stdlib_ssl_context_class()(ssl.PROTOCOL_TLS_CLIENT)
            else:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                if not _installed:
                    ctx.load_default_certs()
            if ca_path is not None:
                ctx.load_verify_locations(cafile=ca_path)
            _CA_CONTEXTS[(ca_path, union)] = ctx
        return ctx


def platform_ssl_context() -> ssl.SSLContext:
    """The process-wide client context (cheap after the first call).

    An existing ``SSL_CERT_FILE`` bundle is trusted ON TOP of the platform store,
    as it was for httpx ``verify=True``; a missing or stale path falls back to
    the platform store alone.
    """
    install_truststore()
    ca = os.environ.get("SSL_CERT_FILE", "").strip()
    if ca and Path(ca).is_file():
        return _shared_context(str(Path(ca).resolve()), union=True)
    return _shared_context(None)


def certifi_bundle_path() -> Optional[str]:
    """certifi's CA bundle path, re-derived when a rotation invalidated it.

    ``certifi.where()`` builds its path from the imported module's ``__file__``, so a
    process that outlives an install-environment rotation — the package manager creates
    a new environment directory and deletes the old one — keeps pointing into the
    deleted directory for the rest of its life, and every TLS client it builds
    afterwards fails. Re-executing the module re-derives ``__file__`` from the live
    ``sys.path``; when even that finds nothing, the platform store is the answer and
    never an exception.

    Returns a bundle path that exists on disk, else ``None``.
    """
    try:
        import certifi
    except ImportError:
        return None
    stale = certifi.where()
    if Path(stale).is_file():
        return stale
    recovered = _reload_certifi_bundle()
    if recovered is not None:
        logger.warning(
            "TLS trust: certifi's CA bundle %s no longer exists — the install "
            "environment it belongs to was replaced under this process; re-resolved "
            "to %s. TLS clients built before this point used the dead path.",
            stale,
            recovered,
        )
        return recovered
    logger.warning(
        "TLS trust: certifi's CA bundle %s no longer exists and certifi could not be "
        "re-resolved; using the platform trust store for the rest of this process.",
        stale,
    )
    return None


def _reload_certifi_bundle() -> Optional[str]:
    """Re-execute the imported certifi module so its path follows the live ``sys.path``.

    ``None`` when certifi was never imported or its package is gone from every
    ``sys.path`` entry — the caller then falls back to the platform store.
    """
    import importlib
    import sys

    module = sys.modules.get("certifi")
    if module is None:
        return None
    # The replaced directory is gone; a finder holding a cached listing of it would
    # still resolve the old package.
    importlib.invalidate_caches()
    try:
        path = importlib.reload(module).where()
    except Exception as exc:  # the reload re-executes third-party module code
        logger.debug("certifi could not be re-resolved: %s", exc, exc_info=True)
        return None
    return path if path and Path(path).is_file() else None


def requests_verify() -> bool | str:
    """``verify`` for a ``requests`` call in this process.

    ``requests`` resolves ``verify=True`` through ``DEFAULT_CA_BUNDLE_PATH``, a string
    frozen when ``requests`` was imported — the one certifi consumer a module reload
    cannot reach, so a rotated environment leaves every ``requests`` call raising
    ``OSError: Could not find a suitable TLS CA certificate bundle``. The live bundle
    (certifi's, re-derived; else OpenSSL's own default) keeps those calls working;
    ``True`` only when nothing resolves, which is requests' own default.
    """
    bundle = certifi_bundle_path()
    if bundle:
        return bundle
    defaults = ssl.get_default_verify_paths()
    for candidate in (defaults.cafile, defaults.capath):
        if candidate and Path(candidate).exists():
            return candidate
    return True


def _default_verify() -> bool | ssl.SSLContext:
    """``verify`` for the platform default path.

    httpx resolves ``verify=True`` through ``certifi.where()`` at client construction,
    which raises ``FileNotFoundError`` on a bundle path a rotation deleted. Hand the
    platform context over instead when no bundle can be found; a re-derivable bundle
    keeps ``True`` — the module reload already repaired certifi process-wide.
    """
    return True if certifi_bundle_path() else _shared_context(None)


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
    3. ``True`` — the platform store, via the process-wide install, or the
       platform context when the certifi bundle this process resolved at
       import no longer exists (a replaced install environment)

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
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return _shared_context(None)
    return _default_verify()
