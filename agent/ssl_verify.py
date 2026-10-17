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

import contextlib
import logging
import os
import ssl
import threading
import weakref
from pathlib import Path
from typing import Any, Iterator, Optional

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
        _memoize_openssl_bundle_reload()
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


_openssl_bundle_state: weakref.WeakKeyDictionary[ssl.SSLContext, tuple] = weakref.WeakKeyDictionary()


def _openssl_trust_signature() -> tuple:
    """Identity of the CA sources truststore's OpenSSL backend would load.

    ``cafile``/``capath`` come from ``ssl.get_default_verify_paths()`` (env
    vars or OpenSSL's compiled-in paths); the stat tuple notices a bundle
    file being replaced behind our back.
    """
    defaults = ssl.get_default_verify_paths()
    parts: list[Any] = [defaults.cafile, defaults.capath]
    for path in (defaults.cafile, defaults.capath):
        if path:
            try:
                st = os.stat(path)
            except OSError:
                parts.append(None)
            else:
                parts.append((st.st_mtime_ns, st.st_size, st.st_ino, st.st_dev))
        else:
            parts.append(None)
    return tuple(parts)


def _memoize_openssl_bundle_reload() -> None:
    """Stop truststore's OpenSSL backend re-parsing the system CA bundle per connection.

    On Linux, ``truststore._openssl._configure_context`` runs
    ``ctx.set_default_verify_paths()`` on the same inner context for every
    ``wrap_socket``/``wrap_bio``, so each new TLS connection (Telegram
    heartbeat probes, provider clients, webhooks) pays a full bundle re-read
    — on the event loop for async clients. Loading is additive and
    idempotent, so a context configured against an unchanged bundle needs
    nothing more; only a bundle file actually changing triggers a reload.
    The platform verifier backends (macOS, Windows) have no such reload and
    are left alone, as is any truststore layout we don't recognize.
    """
    try:
        from truststore import _api

        original = _api._configure_context
        if getattr(original, "__module__", None) != "truststore._openssl":
            return

        @contextlib.contextmanager
        def _configure_context_once(ctx: ssl.SSLContext) -> Iterator[None]:
            signature = _openssl_trust_signature()
            # Unlocked on purpose: a racing first load merely repeats an
            # additive load; a lock here would serialize connection setup.
            if _openssl_bundle_state.get(ctx) == signature:
                yield
                return
            with original(ctx):
                _openssl_bundle_state[ctx] = signature
                yield

        _api._configure_context = _configure_context_once
        logger.debug("TLS trust: OpenSSL bundle load memoized per context")
    except Exception:
        # Private-API drift: the reload cost returns, correctness never changes.
        logger.debug("truststore OpenSSL bundle memo not installed", exc_info=True)


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
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return _shared_context(None)
    return True
