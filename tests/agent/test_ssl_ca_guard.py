"""Tests for the preventive SSL CA bundle guard."""

import errno
from pathlib import Path

import certifi
import pytest

from agent.errors import SSLConfigurationError
from agent.ssl_guard import is_fd_exhaustion_error, verify_ca_bundle


def test_healthy_bundle_passes(monkeypatch):
    """A real, non-empty certifi bundle must verify without raising."""
    for key in ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        monkeypatch.delenv(key, raising=False)
    bundle = Path(certifi.where())
    assert bundle.exists()
    assert bundle.stat().st_size > 1024
    verify_ca_bundle()


def test_empty_certifi_bundle_raises_ssl_error(monkeypatch, tmp_path):
    """Empty file is treated as a corrupted bundle."""
    fake = tmp_path / "empty.pem"
    fake.write_bytes(b"")
    monkeypatch.setattr(certifi, "where", lambda: str(fake))
    with pytest.raises(SSLConfigurationError) as exc:
        verify_ca_bundle()
    assert "too small" in str(exc.value).lower()


@pytest.mark.parametrize("env_var", ["HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"])
def test_missing_explicit_ca_bundle_env_raises_before_httpx(monkeypatch, tmp_path, env_var):
    """Bad CA-bundle env vars should be reported before OpenAI/httpx init."""
    fake = tmp_path / "missing.pem"
    monkeypatch.setenv(env_var, str(fake))
    with pytest.raises(SSLConfigurationError) as exc:
        verify_ca_bundle()
    message = str(exc.value)
    assert env_var in message
    assert str(fake) in message
    assert "force-reinstall" in message


def test_truststore_get_ca_certs_not_implemented_is_accepted(monkeypatch, tmp_path):
    """A truststore-backed SSLContext (Windows OS trust store) raises
    NotImplementedError from get_ca_certs(). The guard must accept the
    already-loaded bundle rather than fail.

    Regression for the empty-message ``Failed to initialize OpenAI client:``
    seen on every fresh agent init on Windows (str(NotImplementedError()) == "").
    """
    from agent import ssl_guard

    bundle = tmp_path / "bundle.pem"
    bundle.write_text(
        "-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )

    class _TruststoreLikeContext:
        def get_ca_certs(self, binary_form=False):  # noqa: ARG002 - mirror ssl API
            raise NotImplementedError()

    # create_default_context(cafile=...) loads the bundle fine; only the
    # post-load introspection is unsupported under truststore.
    monkeypatch.setattr(
        ssl_guard.ssl, "create_default_context", lambda *a, **k: _TruststoreLikeContext()
    )
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))

    # Must not raise on the explicit env bundle nor the certifi check.
    verify_ca_bundle()


def _write_tiny_pem(path: Path) -> Path:
    path.write_text(
        "-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )
    return path


def test_emfile_loading_bundle_does_not_recommend_certifi_reinstall(monkeypatch, tmp_path):
    """FD exhaustion while opening the CA file is not a broken certifi install.

    Long-lived `hermes serve --isolated` (Desktop SSH) hits RLIMIT_NOFILE and
    then agent init fails in ssl_guard with a message that used to tell the
    operator to run `hermes doctor --fix`. That reinstall cannot open the
    bundle either and hides the real leak (#88033).
    """
    from agent import ssl_guard

    bundle = _write_tiny_pem(tmp_path / "bundle.pem")
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))

    def _boom(*a, **k):
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(ssl_guard.ssl, "create_default_context", _boom)
    with pytest.raises(SSLConfigurationError) as exc:
        verify_ca_bundle()
    message = str(exc.value)
    assert "Too many open files" in message
    assert "force-reinstall" not in message
    assert "hermes doctor --fix" not in message
    assert "file descriptor" in message.lower()
    assert "serve --isolated" in message
    assert is_fd_exhaustion_error(exc.value)


def test_emfile_on_stat_is_classified_as_fd_exhaustion(monkeypatch, tmp_path):
    """exists()/stat() can also raise EMFILE before the bundle is opened."""
    from pathlib import Path as PathType

    bundle = _write_tiny_pem(tmp_path / "bundle.pem")
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))

    real_exists = PathType.exists

    def _exists(self):
        if self == bundle or self.resolve() == bundle.resolve():
            raise OSError(errno.EMFILE, "Too many open files")
        return real_exists(self)

    monkeypatch.setattr(PathType, "exists", _exists)
    with pytest.raises(SSLConfigurationError) as exc:
        verify_ca_bundle()
    assert "force-reinstall" not in str(exc.value)
    assert is_fd_exhaustion_error(exc.value)


@pytest.mark.parametrize(
    "exc",
    [
        OSError(errno.EMFILE, "Too many open files"),
        OSError(errno.ENFILE, "Too many open files in system"),
        SSLConfigurationError("CA bundle cannot be loaded: [Errno 24] Too many open files"),
    ],
)
def test_is_fd_exhaustion_error_detects_wrapped_and_raw(exc):
    assert is_fd_exhaustion_error(exc)
