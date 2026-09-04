"""TLS trust contract: the OS certificate store, with explicit config on top.

These are behavior contracts, not snapshots — they assert WHERE trust comes
from and that explicit per-provider settings still beat it.
"""

import ssl

import pytest

from agent import ssl_verify
from agent.ssl_verify import (
    install_truststore,
    resolve_httpx_verify,
    resolve_requests_verify,
)


@pytest.fixture(autouse=True)
def _reset_install_cache():
    ssl_verify._installed = None
    yield
    ssl_verify._installed = None


def test_install_truststore_puts_the_platform_verifier_in_charge():
    assert install_truststore() is True
    # Every default context built anywhere in the process is now
    # platform-verified — including stdlib urllib call sites.
    assert ssl.SSLContext.__module__.startswith("truststore")


def test_install_truststore_is_idempotent():
    assert install_truststore() is True
    assert install_truststore() is True


def test_default_verify_defers_to_the_platform_store():
    # True means "verify normally", which after the install means the OS
    # store. No CA bundle path is threaded through.
    assert resolve_httpx_verify() is True
    assert resolve_requests_verify() is True


def test_env_ca_bundle_vars_no_longer_steer_trust(monkeypatch, tmp_path):
    """The old SSL_CERT_FILE/REQUESTS_CA_BUNDLE ladders are gone.

    Trust is the machine's, and a stale env var pointing at a bundle (or at
    nothing) must not silently redirect or break verification — the exact
    failure the removed gateway/auth/urllib ladders each hand-rolled a
    workaround for.
    """
    import certifi

    for var in ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        monkeypatch.setenv(var, certifi.where())
    assert resolve_httpx_verify() is True

    for var in ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        monkeypatch.setenv(var, str(tmp_path / "does-not-exist.pem"))
    assert resolve_httpx_verify() is True


def test_explicit_ca_bundle_replaces_the_platform_store():
    """A pinned ssl_ca_cert must verify against THAT bundle only.

    Tripwire: truststore's injected SSLContext ignores load_verify_locations
    and verifies against the OS store regardless, so this context has to come
    from the original class. If a truststore bump moves
    ``_ssl_constants._original_SSLContext``, this fails here instead of
    silently degrading a pinned corporate bundle into "trust the machine".
    """
    import certifi
    from truststore._ssl_constants import _original_SSLContext

    ctx = resolve_httpx_verify(ca_bundle=certifi.where())

    # NOT isinstance(ctx, ssl.SSLContext): that name is the patched class
    # now, and this context deliberately is not one.
    assert isinstance(ctx, _original_SSLContext)
    assert not type(ctx).__module__.startswith("truststore")
    # PROTOCOL_TLS_CLIENT's own defaults — asserted, not assigned: setting
    # them on this context recurses forever through the patched module
    # global (super(SSLContext, SSLContext) resolves to truststore's class).
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    # Introspectable precisely because it is NOT truststore-backed.
    assert len(ctx.get_ca_certs()) > 0


def test_requests_explicit_ca_bundle_is_returned_as_a_path():
    import certifi

    assert resolve_requests_verify(ca_bundle=certifi.where()) == certifi.where()


def test_missing_explicit_bundle_falls_back_to_the_platform_store(tmp_path, caplog):
    missing = str(tmp_path / "nope.pem")

    assert resolve_httpx_verify(ca_bundle=missing) is True
    assert resolve_requests_verify(ca_bundle=missing) is True
    assert "does not exist" in caplog.text


@pytest.mark.parametrize("value", [False, "false", "0", "no", "off", "FALSE"])
def test_insecure_disables_verification(value):
    assert resolve_httpx_verify(ssl_verify=value) is False
    assert resolve_requests_verify(ssl_verify=value) is False


def test_insecure_beats_an_explicit_bundle():
    import certifi

    assert resolve_httpx_verify(ca_bundle=certifi.where(), ssl_verify=False) is False


def test_truststore_failure_degrades_to_openssl_defaults(monkeypatch, caplog):
    """No truststore (stripped env, odd platform) must not break startup."""
    import builtins

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "truststore":
            raise ImportError("no truststore here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert install_truststore() is False
    assert "truststore unavailable" in caplog.text
    # Still resolves to a usable verify value rather than raising.
    monkeypatch.undo()
    assert resolve_httpx_verify() is True
