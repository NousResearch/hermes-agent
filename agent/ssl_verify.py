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

import hashlib
import logging
import os
import ssl
import tempfile
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


def _new_client_context(ca_path: str | None, union: bool = False) -> ssl.SSLContext:
    """A FRESH client context — never a shared one.

    Anything that loads a client certificate onto a context mutates it for every
    client holding the same object, so certificate-bearing contexts are built
    here and cached under their own key by :func:`mtls_context`.
    """
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
    return ctx


def _shared_context(ca_path: str | None, union: bool = False) -> ssl.SSLContext:
    """A stable context identity lets clients reuse the existing transport pool.

    ``union`` adds ``ca_path`` to the platform store (the ``SSL_CERT_FILE``
    contract) instead of replacing it (a provider ``ssl_ca_cert``).
    """
    with _CA_CONTEXTS_LOCK:
        ctx = _CA_CONTEXTS.get((ca_path, union))
        if ctx is None:
            ctx = _new_client_context(ca_path, union)
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


_STALE_BUNDLE_OUTCOMES: dict[str, Optional[str]] = {}
_STALE_BUNDLE_LOCK = threading.Lock()


def certifi_bundle_path() -> Optional[str]:
    """certifi's CA bundle path, re-derived when a rotation invalidated it.

    ``certifi.where()`` resolves its path once, against the ``certifi`` package on the
    ``sys.path`` of the moment it was first called, and caches it in ``certifi.core``:
    a process that outlives an install-environment rotation — the package manager
    creates a new environment directory and deletes the old one — keeps pointing into
    the deleted directory for the rest of its life, and every TLS client it builds
    afterwards fails. Re-importing the package against the live ``sys.path`` repairs it
    process-wide; when even that finds nothing, the platform store is the answer and
    never an exception.

    The repair, and its warning, happen ONCE per dead path: a gateway builds clients
    per turn, per cron run and per model-metadata fetch, so an un-cached reload here
    would re-execute third-party module code and re-log on every client it builds.

    Returns a bundle path that exists on disk, else ``None``.
    """
    try:
        import certifi
    except ImportError:
        return None
    stale = certifi.where()
    if Path(stale).is_file():
        return stale
    return _recover_bundle_once(stale)


def _recover_bundle_once(stale: str) -> Optional[str]:
    """Re-derive ``stale`` at most once, and report it at most once.

    A cached outcome is reused while it still holds (a re-derived path that exists, or
    an unrecoverable one); a later rotation dead-path is a new key and is repaired on
    its own.
    """
    with _STALE_BUNDLE_LOCK:
        if stale in _STALE_BUNDLE_OUTCOMES:
            cached = _STALE_BUNDLE_OUTCOMES[stale]
            if cached is None or Path(cached).is_file():
                return cached
        recovered = _reload_certifi_bundle()
        _STALE_BUNDLE_OUTCOMES[stale] = recovered
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
    """Re-import certifi from the live ``sys.path`` so its ``where()`` follows it.

    ``importlib.reload`` on its own does NOT re-derive the path with the real package:
    ``certifi.where`` is ``certifi.core.where`` re-exported by ``__init__``, and ``core``
    caches the resolved path in a module global, so re-executing ``__init__`` re-binds
    ``where`` from the still-loaded ``core`` — the replaced directory, cached a second
    time. Only dropping the package and its submodules from ``sys.modules`` makes both
    files re-execute against the live ``sys.path``. They are put back when the import
    fails, so the process is never left without the ``certifi`` it already had.

    ``None`` when certifi was never imported or its package is gone from every
    ``sys.path`` entry — the caller then falls back to the platform store.
    """
    import importlib
    import sys

    if "certifi" not in sys.modules:
        return None
    # The replaced directory is gone; a finder holding a cached listing of it would
    # still resolve the old package.
    importlib.invalidate_caches()
    dropped = {
        name: sys.modules.pop(name)
        for name in [n for n in sys.modules if n == "certifi" or n.startswith("certifi.")]
    }
    try:
        path = importlib.import_module("certifi").where()
    except Exception as exc:  # the import re-executes third-party module code
        sys.modules.update(dropped)
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
    ``True`` only when nothing resolves, which is requests' own default. A healthy
    install keeps ``True`` too: requests reads ``REQUESTS_CA_BUNDLE``/``CURL_CA_BUNDLE``
    only for ``verify=True``, and an explicit path would silently bypass a user's
    proxy CA.
    """
    from requests.utils import DEFAULT_CA_BUNDLE_PATH

    if os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE"):
        return True
    if Path(DEFAULT_CA_BUNDLE_PATH).is_file():
        return True
    bundle = certifi_bundle_path()
    if bundle:
        return bundle
    defaults = ssl.get_default_verify_paths()
    for candidate in (defaults.cafile, defaults.capath):
        if candidate and Path(candidate).exists():
            return candidate
    # Windows keeps its roots in the system store with no OpenSSL file, and ``True``
    # would send requests back to the dead frozen path: hand it the store as a bundle.
    return _platform_store_bundle() or True


_PLATFORM_STORE_BUNDLE: Optional[str] = None


def _platform_store_bundle() -> Optional[str]:
    """The OS store's CA certificates written once to a PEM file under HERMES_HOME."""
    global _PLATFORM_STORE_BUNDLE
    if _PLATFORM_STORE_BUNDLE and Path(_PLATFORM_STORE_BUNDLE).is_file():
        return _PLATFORM_STORE_BUNDLE
    ctx = _stdlib_ssl_context_class()(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_default_certs()
    certs = ctx.get_ca_certs(binary_form=True)
    if not certs:
        return None
    from hermes_constants import get_hermes_home
    from utils import atomic_write_text

    path = get_hermes_home() / "cache" / "platform-ca-bundle.pem"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, "".join(ssl.DER_cert_to_PEM_cert(der) for der in certs))
    except OSError as exc:  # a full or read-only home must not replace the TLS fallback with a disk error
        logger.debug("platform CA bundle not written: %s", exc, exc_info=True)
        return None
    _PLATFORM_STORE_BUNDLE = str(path)
    return _PLATFORM_STORE_BUNDLE


def _default_verify() -> bool | ssl.SSLContext:
    """``verify`` for the platform default path.

    httpx resolves ``verify=True`` through ``certifi.where()`` at client construction,
    which raises ``FileNotFoundError`` on a bundle path a rotation deleted. Hand the
    platform context over instead when no bundle can be found; a re-derivable bundle
    keeps ``True`` — the module reload already repaired certifi process-wide.
    """
    return True if certifi_bundle_path() else _shared_context(None)


# ---------------------------------------------------------------------------
# Mutual TLS — client certificates (``agent.mtls``)
# ---------------------------------------------------------------------------

ClientCert = tuple[str, Optional[str], Optional[str]]  # (cert path, key path, password)


class MtlsConfigError(RuntimeError):
    """``agent.mtls`` is configured but cannot be honoured."""


_MTLS_CONTEXTS: dict[tuple, ssl.SSLContext] = {}
_MTLS_WARNED: set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key in _MTLS_WARNED:
        return
    _MTLS_WARNED.add(key)
    logger.warning(message, *args)


def mtls_settings() -> dict[str, Any]:
    """The ``agent.mtls`` block; ``{}`` when absent or unreadable.

    Read through the config cache (no disk hit after the first load), so a config
    edit needs a restart to take effect — same contract as every other TLS setting.
    """
    try:
        from hermes_cli.config import load_config_readonly

        block = (load_config_readonly().get("agent") or {}).get("mtls")
    except Exception:
        logger.debug("agent.mtls settings unreadable", exc_info=True)
        return {}
    return block if isinstance(block, dict) else {}


def _expand_mtls_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return str(Path(value.strip()).expanduser().resolve())


def _mtls_password(block: dict[str, Any]) -> Optional[str]:
    """The passphrase, read from the env var named by ``password_env``.

    Never from a config value: config.yaml is copied into backups and bug reports.
    """
    env_name = block.get("password_env")
    if not isinstance(env_name, str) or not env_name.strip():
        return None
    return os.environ.get(env_name.strip()) or None


def _is_pkcs12(path: str) -> bool:
    return Path(path).suffix.lower() in {".p12", ".pfx", ".pkcs12"}


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.chmod(str(path), 0o600)


def _pkcs12_pem_paths(p12_path: str, password: Optional[str]) -> tuple[str, str]:
    """Unwrap a PKCS#12 bundle into a private PEM pair and return their paths.

    ``ssl.SSLContext.load_cert_chain`` reads files only, so a bundle has to touch
    disk. The pair lands under ``HERMES_HOME/cache/mtls`` (dir 0700, files 0600),
    keyed by the bundle's path/size/mtime, so rotating the bundle re-extracts it.
    """
    bundle = Path(p12_path)
    try:
        stat = bundle.stat()
    except OSError as exc:
        raise MtlsConfigError(f"agent.mtls.cert_file is not readable: {p12_path!r} ({exc})") from exc
    stamp = f"{p12_path}:{stat.st_size}:{stat.st_mtime_ns}:{bool(password)}"
    digest = hashlib.sha256(stamp.encode("utf-8", "surrogateescape")).hexdigest()[:32]
    try:
        from hermes_constants import get_hermes_home

        out_dir = Path(get_hermes_home()) / "cache" / "mtls"
    except Exception:  # no HERMES_HOME resolvable: keep the pair process-local
        out_dir = Path(tempfile.mkdtemp(prefix="hermes-mtls-"))
    cert_out, key_out = out_dir / f"{digest}-cert.pem", out_dir / f"{digest}-key.pem"
    if cert_out.is_file() and key_out.is_file():
        return str(cert_out), str(key_out)

    try:
        from cryptography.hazmat.primitives.serialization import (
            Encoding, NoEncryption, PrivateFormat, pkcs12,
        )
    except ImportError as exc:  # cryptography is a core pin; a stripped install may lack it
        raise MtlsConfigError(
            f"PKCS#12 client certificate {p12_path!r} needs the 'cryptography' package: {exc}"
        ) from exc
    try:
        key, cert, extra = pkcs12.load_key_and_certificates(
            bundle.read_bytes(), password.encode("utf-8") if password else None
        )
    except Exception as exc:
        raise MtlsConfigError(
            f"PKCS#12 bundle {p12_path!r} could not be opened ({exc}); set the passphrase "
            "env var named by agent.mtls.password_env"
        ) from exc
    if key is None or cert is None:
        raise MtlsConfigError(
            f"PKCS#12 bundle {p12_path!r} holds no client certificate + private key pair"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(out_dir, 0o700)
    except OSError:
        pass
    _write_private(key_out, key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
    chain = cert.public_bytes(Encoding.PEM) + b"".join(
        item.public_bytes(Encoding.PEM) for item in (extra or ())
    )
    _write_private(cert_out, chain)
    return str(cert_out), str(key_out)


def client_cert_spec(block: Optional[dict[str, Any]] = None) -> Optional[ClientCert]:
    """Resolve ``agent.mtls`` into ``(cert, key, password)``; None when unconfigured.

    PEM, combined PEM (cert + key in one file) and PKCS#12 bundles are all accepted.
    Raises :class:`MtlsConfigError` when the block IS configured but unusable — the
    caller decides whether that is fatal (MCP per-server) or a warning (global).
    """
    settings = mtls_settings() if block is None else block
    cert = _expand_mtls_path(settings.get("cert_file"))
    if not cert:
        return None
    if not Path(cert).is_file():
        raise MtlsConfigError(f"agent.mtls.cert_file does not exist: {cert!r}")
    password = _mtls_password(settings)
    if _is_pkcs12(cert):
        cert_pem, key_pem = _pkcs12_pem_paths(cert, password)
        return (cert_pem, key_pem, None)
    key = _expand_mtls_path(settings.get("key_file"))
    if key and not Path(key).is_file():
        raise MtlsConfigError(f"agent.mtls.key_file does not exist: {key!r}")
    return (cert, key or None, password)


def _coerce_cert_spec(
    client_cert: Any, client_key: Any, client_password: Any, base_url: str
) -> Optional[ClientCert]:
    """An endpoint's own ``cert=`` shape (path, or 2/3-tuple) → a normalised spec."""
    if not client_cert:
        return None
    if isinstance(client_cert, (tuple, list)):
        parts = list(client_cert)
        if len(parts) not in (2, 3):
            raise MtlsConfigError(
                f"{base_url or 'endpoint'}: client_cert list form must have 2 or 3 elements"
            )
        cert, key = _expand_mtls_path(parts[0]), _expand_mtls_path(parts[1])
        password = parts[2] if len(parts) == 3 else None
    else:
        cert = _expand_mtls_path(client_cert)
        key = _expand_mtls_path(client_key) if client_key else ""
        password = client_password or None
    if not cert or not Path(cert).is_file():
        raise MtlsConfigError(f"{base_url or 'endpoint'}: client_cert not found at {cert!r}")
    if key and not Path(key).is_file():
        raise MtlsConfigError(f"{base_url or 'endpoint'}: client_key not found at {key!r}")
    return (cert, key or None, password)


def _global_cert_spec(base_url: str) -> Optional[ClientCert]:
    """Global ``agent.mtls`` certificate, warned-and-skipped when unusable.

    A broken global identity must not take the agent down: every provider would
    fail to build a client. The warning names the setting, and connections carry
    on without a client certificate.
    """
    try:
        return client_cert_spec()
    except MtlsConfigError as exc:
        _warn_once(
            "mtls-unusable",
            "mTLS client certificate unusable (%s) — connecting to %s WITHOUT a "
            "client certificate. Fix agent.mtls (or `hermes config check`).",
            exc,
            base_url or "provider endpoints",
        )
        return None


def _global_ca_path(base_url: str) -> Optional[str]:
    """``agent.mtls.ca_cert`` — the fallback trust root for mTLS endpoints."""
    block = mtls_settings()
    ca = _expand_mtls_path(block.get("ca_cert"))
    if not ca:
        return None
    if not Path(ca).is_file():
        _warn_once(
            "mtls-ca-missing",
            "agent.mtls.ca_cert does not exist: %s — trusting the OS store for %s",
            ca,
            base_url or "provider endpoints",
        )
        return None
    return ca


def _resolve_ca_path(ca_bundle: Optional[str]) -> Optional[str]:
    """The explicit CA bundle's path, or None when unset/unreadable.

    Only an EXPLICIT bundle is resolved here. The ambient ``SSL_CERT_FILE`` contract is
    the trust-only path's business (``resolve_httpx_verify`` keeps handing httpx the
    platform context instead), and a certificate-bearing context must not quietly pick
    up a different trust root than the same client would without mTLS.
    """
    effective = (ca_bundle or "").strip()
    if not effective:
        return None
    path = Path(effective).expanduser()
    if path.is_file():
        return str(path.resolve())
    logger.warning(
        "ssl_ca_cert path does not exist: %s — using the OS trust store instead",
        effective,
    )
    return None


def mtls_context(
    cert: ClientCert,
    *,
    ca_path: Optional[str] = None,
    union: bool = False,
    insecure: bool = False,
) -> ssl.SSLContext:
    """A client context carrying ``cert``, cached by identity so pools stay stable.

    Certificate-bearing contexts are NEVER the shared trust contexts: adding a
    chain to one of those would present this client's identity on every other
    connection that already holds it.
    """
    key = ("mtls", ca_path, union, insecure, cert)
    with _CA_CONTEXTS_LOCK:
        ctx = _MTLS_CONTEXTS.get(key)
        if ctx is None:
            if insecure:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            else:
                ctx = _new_client_context(ca_path, union)
            cert_path, key_path, password = cert
            try:
                ctx.load_cert_chain(cert_path, key_path, password)
            except Exception as exc:
                raise MtlsConfigError(
                    f"client certificate {cert_path!r} could not be loaded ({exc})"
                ) from exc
            _MTLS_CONTEXTS[key] = ctx
        return ctx


def mtls_verify_for(base_url: str = "", *, ca_bundle: Optional[str] = None) -> Optional[ssl.SSLContext]:
    """The global mTLS ``verify`` context for an endpoint, or None when unconfigured.

    For stacks that take ``verify`` and ``cert`` as separate arguments (httpx
    transports, MCP): the certificate rides on the context, so a caller that also
    passes ``cert=`` gets the same identity twice — harmless, but prefer
    :func:`client_cert_spec` alongside a trust-only context in that case.
    """
    spec = _global_cert_spec(base_url)
    if spec is None:
        return None
    ca_path = _resolve_ca_path(ca_bundle) or _global_ca_path(base_url)
    return mtls_context(spec, ca_path=ca_path)


def ca_verify_for(label: str = "") -> Optional[ssl.SSLContext]:
    """Trust-only context for ``agent.mtls.ca_cert``, or None when unset.

    Nothing about a client identity here: this is what an endpoint that passes
    ``cert=`` separately (httpx ``AsyncHTTPTransport``, MCP) needs for a private
    signer without every server repeating the bundle path.
    """
    ca_path = _global_ca_path(label)
    return _shared_context(ca_path) if ca_path else None


def resolve_httpx_verify(
    *,
    ca_bundle: Optional[str] = None,
    ssl_verify: Any = None,
    base_url: str = "",
    client_cert: Any = None,
    client_key: Any = None,
    client_password: Any = None,
) -> bool | ssl.SSLContext:
    """Resolve ``verify`` for an HTTP client.

    1. ``ssl_verify: false`` — verification off (local development only)
    2. explicit ``ca_bundle`` (the provider's ``ssl_ca_cert``) — that
       bundle INSTEAD of the platform store, for an endpoint whose chain
       the machine has no reason to trust
    3. ``True`` — the platform store, via the process-wide install, or the
       platform context when the certifi bundle this process resolved at
       import no longer exists (a replaced install environment)

    When a client certificate is configured — per endpoint (``client_cert``, an
    httpx ``cert=`` shape) or globally under ``agent.mtls`` — the returned
    context also presents it, because ``verify`` is the only TLS object every
    client stack here (OpenAI SDK, Gemini native, auxiliary, Anthropic,
    model-metadata, MCP) agrees on. ``ca_bundle`` still wins over the global
    ``agent.mtls.ca_cert``: an endpoint's own trust statement is more specific
    than the machine-wide one.

    ``base_url`` only labels the insecure-mode warning.
    """
    install_truststore()

    insecure = _coerce_insecure(ssl_verify)
    cert = _coerce_cert_spec(client_cert, client_key, client_password, base_url)
    global_spec = _global_cert_spec(base_url) if cert is None else None
    if cert is None:
        cert = global_spec

    ca_path = _resolve_ca_path(ca_bundle)
    if cert is not None:
        if ca_path is None and global_spec is not None:
            ca_path = _global_ca_path(base_url)
        if insecure:
            logger.warning(
                "mTLS client certificate configured but TLS verification is DISABLED "
                "for %s — presenting the certificate without verifying the server. "
                "This is intended for local development only.",
                base_url or "a custom endpoint",
            )
        return mtls_context(cert, ca_path=ca_path, insecure=insecure)

    if insecure:
        logger.warning(
            "TLS certificate verification DISABLED (ssl_verify: false) for %s — "
            "this is intended for local development only and is unsafe on any "
            "network you do not fully control.",
            base_url or "a custom provider endpoint",
        )
        return False

    if ca_path:
        return _shared_context(ca_path)
    # HTTPX reads CA env vars before the injected verifier gets control. Pass
    # the platform context directly so stale paths cannot break construction;
    # proxy environment handling remains enabled.
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return _shared_context(None)
    return _default_verify()
