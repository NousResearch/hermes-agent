"""Tests for BitwardenLoginBackend environment variable allowlist (#120512)."""

from agent.vault_backends.bitwarden import BitwardenLoginBackend


def test_bitwarden_backend_env_forwards_tls_ca_certs(monkeypatch):
    """Bitwarden child env keeps NODE_EXTRA_CA_CERTS and standard TLS CA certs for private/self-hosted instances (#120512)."""
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/path/to/custom-ca.pem")
    monkeypatch.setenv("SSL_CERT_FILE", "/path/to/ssl-cert.pem")
    monkeypatch.setenv("SSL_CERT_DIR", "/etc/ssl/certs")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/path/to/requests-ca.pem")

    backend = BitwardenLoginBackend({"enabled": True})
    env = backend._env(None)

    assert env.get("NODE_EXTRA_CA_CERTS") == "/path/to/custom-ca.pem"
    assert env.get("SSL_CERT_FILE") == "/path/to/ssl-cert.pem"
    assert env.get("SSL_CERT_DIR") == "/etc/ssl/certs"
    assert env.get("REQUESTS_CA_BUNDLE") == "/path/to/requests-ca.pem"
