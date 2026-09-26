"""Probe-specific policy edges; real TLS handshakes live in test_ssl_transport_contracts."""
import ssl
from unittest.mock import MagicMock, patch

import pytest

from agent.model_metadata_http import resolve_verify
from hermes_cli.models import _custom_provider_ssl_context

_BASE = "https://relay.example.invalid/v1"


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.mark.parametrize("case", ["no-url", "unmatched", "missing-ca", "config-error"])
@pytest.mark.parametrize("ambient", [False, True])
def test_probe_policy_edges(clean_env, case, ambient):
    if ambient:
        clean_env.setenv("SSL_CERT_FILE", "/does/not/exist.pem")
    providers = MagicMock(return_value=[{
        "name": "test", "base_url": _BASE if case != "unmatched" else "https://other.invalid",
        "ssl_ca_cert": "/missing-provider.pem",
    }])
    if case == "config-error":
        providers.side_effect = RuntimeError("config unavailable")
    url = "" if case == "no-url" else _BASE
    with patch("hermes_cli.config.get_compatible_custom_providers", providers):
        verify = resolve_verify(url)
        context = _custom_provider_ssl_context(url)
        assert context is (verify if ambient else None)
    if ambient:
        assert verify.verify_mode == ssl.CERT_REQUIRED and verify.check_hostname
    else:
        assert verify is True
    if not url:
        providers.assert_not_called()


def test_public_endpoint_calls_seam_without_ssl_context_kwarg(clean_env):
    """A public endpoint must not pass ssl_context to the call seam.

    Regression guard: threading ssl_context unconditionally broke existing
    call-seam mocks whose signature is ``(req, timeout=...)``. The probe
    must keep the original 2-arg call shape when no per-provider override
    applies, so a strict 2-arg mock still works.
    """
    import hermes_cli.models as models

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"data": [{"id": "local-model"}]}'

    calls = []

    def _strict_two_arg(req, timeout=5.0):
        calls.append(req.full_url)
        return _Resp()

    with patch(
        "hermes_cli.config.get_compatible_custom_providers",
        return_value=[],
    ), patch.object(
        models, "_urlopen_model_catalog_request", side_effect=_strict_two_arg
    ):
        probe = models.probe_api_models("key", "http://localhost:8000", timeout=1)

    assert probe["models"] == ["local-model"]
    assert calls == ["http://localhost:8000/models"]
