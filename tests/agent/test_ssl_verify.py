"""TLS trust contract: the OS certificate store, with explicit config on top.

These are behavior contracts, not snapshots — they assert WHERE trust comes
from and that explicit per-provider settings still beat it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from agent.ssl_verify import resolve_httpx_verify
from datetime import UTC


@pytest.fixture
def no_ca_env(monkeypatch):
    """An operator's SSL_CERT_FILE flips resolve_httpx_verify() from
    ``True`` to a shared platform context. Pin the env for this fallback
    assertion rather than inheriting a developer shell's values."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)


def test_missing_explicit_bundle_falls_back_to_the_platform_store(tmp_path, caplog, no_ca_env):
    missing = str(tmp_path / "nope.pem")

    assert resolve_httpx_verify(ca_bundle=missing) is True
    assert "does not exist" in caplog.text


def test_missing_bundle_under_a_cert_env_var_shares_the_platform_context(tmp_path, monkeypatch):
    """With SSL_CERT_FILE exported the platform store is handed over as one
    shared context (httpx would otherwise read the env var itself); a missing
    bundle must land on that same object, not a second pool. The long-lived
    clients' platform_ssl_context() honours the bundle ON TOP of the platform
    store (as httpx verify=True did), cached once."""
    import ssl

    import certifi

    from agent import ssl_verify

    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    platform = resolve_httpx_verify()
    assert platform is not True
    assert resolve_httpx_verify(ca_bundle=str(tmp_path / "nope.pem")) is platform

    bundle_ctx = ssl_verify.platform_ssl_context()
    assert bundle_ctx is ssl_verify.platform_ssl_context()
    assert bundle_ctx is not platform
    assert type(bundle_ctx) is ssl.SSLContext  # truststore-injected: bundle + OS store
    assert bundle_ctx is ssl_verify._CA_CONTEXTS[(str(Path(certifi.where()).resolve()), True)]


@pytest.mark.parametrize("value", [False, "false", "0", "no", "off", "FALSE"])
def test_insecure_disables_verification(value):
    assert resolve_httpx_verify(ssl_verify=value) is False


def test_insecure_beats_an_explicit_bundle():
    import certifi

    assert resolve_httpx_verify(ca_bundle=certifi.where(), ssl_verify=False) is False


def test_truststore_failure_degrades_to_openssl_defaults():
    import subprocess
    import sys

    child = subprocess.run([sys.executable, "-c", """
import builtins, ssl
original = ssl.SSLContext
real_import = builtins.__import__
def no_truststore(name, *args, **kwargs):
    if name == 'truststore':
        raise ImportError('unavailable fixture')
    return real_import(name, *args, **kwargs)
builtins.__import__ = no_truststore
from agent.ssl_verify import install_truststore, resolve_httpx_verify
assert install_truststore() is False
assert install_truststore() is False
assert ssl.SSLContext is original
import httpx
with httpx.Client(verify=resolve_httpx_verify()) as client:
    ctx = client._transport._pool._ssl_context
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
"""], capture_output=True, text=True, timeout=30, check=False)
    assert child.returncode == 0, child.stderr
    assert "truststore unavailable" in child.stderr


@contextmanager
def _private_https_server(tmp_path: Path) -> Iterator[tuple[Any, Path]]:
    """A TLS server no public certificate store has ever seen.

    Yields ``(server, ca)``: the running server, and the PEM of the CA that signed
    its certificate — the only bundle that can trust it.
    """
    from datetime import datetime, timedelta
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from ipaddress import ip_address
    import ssl
    import threading

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Private provider")])
    now = datetime.now(UTC)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip_address("127.0.0.1"))]), critical=False)
            .sign(key, hashes.SHA256()))
    ca = tmp_path / "provider-ca.pem"
    private = tmp_path / "provider.key"
    ca.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"private provider")

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # Earlier tests may have injected truststore; its wrapper is client-only.
    from agent.ssl_verify import _stdlib_ssl_context_class
    context = _stdlib_ssl_context_class()(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(ca, private)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, ca
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _child_env(tmp_path: Path) -> dict[str, str]:
    """A subprocess environment with no inherited TLS overrides."""
    import os

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home), NO_PROXY="127.0.0.1")
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        env.pop(key, None)
    return env


def test_explicit_provider_ca_replaces_platform_trust_on_real_https(tmp_path):
    """A private endpoint trusts only its provider CA, never a global fallback."""
    import subprocess
    import sys

    with _private_https_server(tmp_path) as (server, ca):
        script = """
import httpx, sys
from agent.ssl_verify import resolve_httpx_verify
url, ca = sys.argv[1:]
try:
    httpx.get(url, verify=resolve_httpx_verify(), timeout=5)
except httpx.ConnectError:
    pass
else:
    raise AssertionError('unconfigured platform store trusted private provider')
response = httpx.get(url, verify=resolve_httpx_verify(ca_bundle=ca), timeout=5)
assert response.content == b'private provider', response
"""
        child = subprocess.run(
            [sys.executable, "-c", script, f"https://127.0.0.1:{server.server_port}/", str(ca)],
            env=_child_env(tmp_path), capture_output=True, text=True, timeout=30, check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr


def test_explicit_bundle_works_without_truststore():
    """A provider ``ssl_ca_cert`` on an interpreter without truststore
    (3.11–3.13 bridge installs) must still yield a verifying context built
    on that bundle, not an import error at client construction."""
    import subprocess
    import sys

    child = subprocess.run([sys.executable, "-c", """
import sys, ssl, certifi
sys.modules['truststore'] = None
from agent.ssl_verify import resolve_httpx_verify
ctx = resolve_httpx_verify(ca_bundle=certifi.where())
assert isinstance(ctx, ssl.SSLContext), ctx
assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
assert ctx.cert_store_stats()['x509_ca'] > 0
"""], capture_output=True, text=True, timeout=30, check=False)
    assert child.returncode == 0, child.stderr


# ---------------------------------------------------------------------------
# An install environment replaced under a live process: certifi's path dies
# ---------------------------------------------------------------------------


class _Rotation(NamedTuple):
    """One install-environment rotation, as the live process sees it."""

    module: Any
    """The certifi imported from the environment that was then deleted."""
    stale_package: Path
    """``env-a/certifi`` — gone from disk, still the path certifi reports."""
    live_package: Path | None
    """``env-b/certifi`` when a replacement was installed, else ``None``."""


def _stand_in_certifi_env(root: Path, name: str, bundle: bytes) -> Path:
    """A byte-for-byte copy of the INSTALLED certifi package.

    The stand-in has to be the real package, not a file that recomputes its own path
    on every call: certifi's ``where`` lives in ``certifi.core`` and caches the
    resolved path in a module global, and clearing that caching is exactly what a
    repair has to do. A single-file stand-in passes while the real package stays
    broken.
    """
    import shutil

    import certifi as installed

    package = root / name / "certifi"
    shutil.copytree(Path(installed.__file__).parent, package,
                    ignore=shutil.ignore_patterns("__pycache__"))
    (package / "cacert.pem").write_bytes(bundle)  # a real PEM: contexts must load it
    return package


def _client_ssl_context(client):
    """The context httpx built for this client's connection pool."""
    from typing import cast

    return cast(Any, client._transport)._pool._ssl_context


def _rotated_certifi(tmp_path: Path, monkeypatch, *, replacement: bool) -> _Rotation:
    """Delete the environment directory the imported certifi points into.

    The package manager's rotation without a package manager: certifi is imported
    from ``env-a``, then that directory is removed. ``replacement`` adds the
    identical package the manager would have installed as ``env-b``.
    """
    import importlib
    import shutil
    import sys

    import certifi as real_certifi

    bundle = Path(real_certifi.where()).read_bytes()  # a real PEM: contexts must load it
    stale = _stand_in_certifi_env(tmp_path, "env-a", bundle)
    live = None
    if replacement:
        live = _stand_in_certifi_env(tmp_path, "env-b", bundle)
        monkeypatch.syspath_prepend(str(live.parent))
    monkeypatch.syspath_prepend(str(stale.parent))
    for name in [n for n in sys.modules if n == "certifi" or n.startswith("certifi.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.invalidate_caches()
    stand_in = importlib.import_module("certifi")
    monkeypatch.setitem(sys.modules, "certifi", stand_in)
    assert Path(stand_in.where()).parent == stale
    shutil.rmtree(stale)
    assert not Path(stand_in.where()).exists()
    return _Rotation(stand_in, stale, live)


def _ssl_verify_warnings(caplog) -> list:
    return [record for record in caplog.records if record.name == "agent.ssl_verify"]


def test_replaced_install_environment_re_resolves_certifi_in_place(tmp_path, monkeypatch, caplog, no_ca_env):
    """A process that outlives an environment rotation holds certifi's path into the
    deleted directory, and httpx builds every SSL context through it. The package is
    re-imported against the live ``sys.path``, so client construction succeeds again
    for this process and every other certifi consumer in it."""
    import logging
    import ssl
    import sys

    import httpx

    rotation = _rotated_certifi(tmp_path, monkeypatch, replacement=True)
    stale_core = sys.modules["certifi.core"]

    with caplog.at_level(logging.WARNING, logger="agent.ssl_verify"):
        verify = resolve_httpx_verify()

    assert verify is True  # httpx's own default, now backed by a bundle that exists
    with httpx.Client(verify=verify) as client:  # the construction that raised FileNotFoundError
        assert _client_ssl_context(client).verify_mode == ssl.CERT_REQUIRED
    # The repair reaches certifi itself, not just this call's return value: httpx
    # resolves verify=True through certifi.where() at every construction.
    assert rotation.live_package is not None
    live_bundle = Path(sys.modules["certifi"].where())
    assert sys.modules["certifi"] is not rotation.module
    assert sys.modules["certifi.core"] is not stale_core
    assert live_bundle.is_file() and live_bundle.parent.resolve() == rotation.live_package.resolve()
    live_core_file = Path(str(getattr(sys.modules["certifi.core"], "__file__", "")))
    assert live_core_file.parent.resolve() == rotation.live_package.resolve()
    logged = _ssl_verify_warnings(caplog)
    assert len(logged) == 1 and "re-resolved" in logged[0].getMessage()
    assert str(rotation.stale_package) in logged[0].getMessage()


def test_rotation_repair_and_warning_happen_once(tmp_path, monkeypatch, caplog, no_ca_env):
    """A gateway builds a TLS client per turn, per cron run and per model-metadata
    fetch: the repair re-executes third-party module code, so it — and the single
    warning an operator needs — must not repeat per client."""
    import logging

    from agent import ssl_verify

    _rotated_certifi(tmp_path, monkeypatch, replacement=True)
    reloads: list[int] = []
    real_reload = ssl_verify._reload_certifi_bundle

    def counting_reload():
        reloads.append(1)
        return real_reload()

    monkeypatch.setattr(ssl_verify, "_reload_certifi_bundle", counting_reload)

    with caplog.at_level(logging.WARNING, logger="agent.ssl_verify"):
        for _ in range(5):
            assert ssl_verify.resolve_httpx_verify() is True
            ssl_verify.requests_verify()

    assert len(reloads) == 1
    assert len(_ssl_verify_warnings(caplog)) == 1


def test_unrecoverable_bundle_hands_over_the_platform_context(tmp_path, monkeypatch, caplog, no_ca_env):
    """Nothing left to re-resolve: the platform store is handed over instead of
    raising, which is the difference between a degraded process and a fleet-wide
    TLS outage that only a restart clears."""
    import logging
    import ssl

    import httpx

    from agent import ssl_verify

    _rotated_certifi(tmp_path, monkeypatch, replacement=False)
    attempts: list[int] = []

    def unrecoverable() -> None:
        attempts.append(1)  # the answer this stand-in gives is None

    monkeypatch.setattr(ssl_verify, "_reload_certifi_bundle", unrecoverable)

    verify: Any = None
    with caplog.at_level(logging.WARNING, logger="agent.ssl_verify"):
        for _ in range(5):
            verify = resolve_httpx_verify()

    assert isinstance(verify, ssl.SSLContext)
    assert len(attempts) == 1  # a dead path is not re-probed per client
    logged = _ssl_verify_warnings(caplog)
    assert len(logged) == 1 and "could not be re-resolved" in logged[0].getMessage()
    with httpx.Client(verify=verify) as client:
        assert _client_ssl_context(client).verify_mode == ssl.CERT_REQUIRED


def test_requests_verify_hands_over_a_bundle_that_exists(tmp_path, monkeypatch, no_ca_env):
    """``requests`` freezes its default bundle path when it is imported, so no reload
    can reach it: its call sites ask for a live bundle instead."""
    from agent import ssl_verify

    rotation = _rotated_certifi(tmp_path, monkeypatch, replacement=True)

    verify = ssl_verify.requests_verify()

    assert rotation.live_package is not None
    assert Path(str(verify)).resolve() == (rotation.live_package / "cacert.pem").resolve()
    assert Path(str(verify)).is_file()


def test_requests_verify_never_hands_over_the_deleted_bundle(tmp_path, monkeypatch, no_ca_env):
    """With nothing re-resolvable the answer is OpenSSL's own bundle, or requests'
    own default — never the path the rotation deleted."""
    from agent import ssl_verify

    rotation = _rotated_certifi(tmp_path, monkeypatch, replacement=False)
    dead = rotation.module.where()
    monkeypatch.setattr(ssl_verify, "_reload_certifi_bundle", lambda: None)

    verify = ssl_verify.requests_verify()

    assert str(verify) != dead
    assert verify is True or Path(str(verify)).exists()


def test_rotated_process_keeps_trusting_the_bundle_it_was_configured_with(tmp_path):
    """End to end over real TLS: a process whose install environment was replaced
    still builds a client that trusts the CA it was configured with.

    The private server is trusted by the rotated bundle and by nothing else, so a
    process that fell back to the platform store — or that kept pointing into the
    deleted directory — cannot complete this request.
    """
    import subprocess
    import sys

    with _private_https_server(tmp_path) as (server, ca):
        bundle = ca.read_bytes()
        stale = _stand_in_certifi_env(tmp_path, "env-a", bundle)
        live = _stand_in_certifi_env(tmp_path, "env-b", bundle)
        script = """
import shutil, sys
from pathlib import Path
url, env_a, env_b = sys.argv[1:4]
sys.path.insert(0, env_a)
import certifi
assert Path(certifi.where()).parent.resolve() == Path(env_a, 'certifi').resolve(), certifi.where()
sys.path.insert(0, env_b)
sys.path.remove(env_a)
shutil.rmtree(env_a)
import httpx
from agent.ssl_verify import resolve_httpx_verify
response = httpx.get(url, verify=resolve_httpx_verify(), timeout=5)
assert response.content == b'private provider', response.content
live = Path(sys.modules['certifi'].where())
assert live.is_file() and live.parent.resolve() == Path(env_b, 'certifi').resolve(), live
"""
        child = subprocess.run(
            [sys.executable, "-c", script, f"https://127.0.0.1:{server.server_port}/",
             str(stale.parent), str(live.parent)],
            env=_child_env(tmp_path), capture_output=True, text=True, timeout=60, check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr
