"""Custom-provider TLS settings must reach the /models and pricing probes.

Regression coverage for provider-scoped ``ssl_ca_cert`` / ``ssl_verify`` being
ignored by the discovery/pricing probes. Two probe families share the root
cause and are covered here:

* ``requests``-based endpoint metadata / pricing probe
  (``agent.model_metadata.fetch_endpoint_model_metadata`` via
  ``_resolve_requests_verify``).
* ``urllib``-based ``/models`` catalog discovery probe
  (``hermes_cli.models.probe_api_models`` via ``_custom_provider_ssl_context``).

Both previously resolved TLS from process-wide env vars only, so a custom
endpoint whose chain verifies against the provider's configured bundle (but not
``SSL_CERT_FILE``) logged a spurious CERTIFICATE_VERIFY_FAILED on every probe
even though the chat client succeeded.

No network I/O: real CA-bundle stand-in files via ``tmp_path`` plus a patched
provider list and a patched request seam.
"""

from __future__ import annotations

import ssl
import urllib.error
from unittest.mock import MagicMock, patch

import certifi
import pytest

from agent.model_metadata import _resolve_requests_verify
from hermes_cli.models import _custom_provider_ssl_context

_CA_ENV_VARS = (
    "HERMES_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "CURL_CA_BUNDLE",
)

_BASE = "https://relay.example.invalid/v1"


@pytest.fixture
def clean_env(monkeypatch):
    """Clear the CA env vars so each test starts from a known state."""
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def bundle_file(tmp_path):
    path = tmp_path / "provider-ca.pem"
    path.write_text("-----BEGIN CERTIFICATE-----\nstub\n-----END CERTIFICATE-----\n")
    return str(path)


@pytest.fixture
def real_ca():
    """A real, parseable CA bundle on disk.

    ``ssl.create_default_context(cafile=...)`` parses the file eagerly, so the
    urllib context path needs a genuine bundle rather than a stub. The
    ``requests`` path only stores the path string (parsed lazily by requests at
    call time), so it can use the ``bundle_file`` stub.
    """
    return certifi.where()


def _providers(base_url, **tls):
    entry = {"name": "relay", "base_url": base_url}
    entry.update(tls)
    return [entry]


class TestResolveRequestsVerifyProviderScoped:
    """``_resolve_requests_verify(base_url)`` — the requests probe path."""

    def test_provider_ca_used_for_matching_base_url(self, clean_env, bundle_file):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_ca_cert=bundle_file),
        ):
            assert _resolve_requests_verify(_BASE) == bundle_file

    def test_provider_ca_overrides_env_ssl_cert_file(self, clean_env, tmp_path, bundle_file):
        env_bundle = tmp_path / "env-ca.pem"
        env_bundle.write_text("stub")
        clean_env.setenv("SSL_CERT_FILE", str(env_bundle))
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_ca_cert=bundle_file),
        ):
            assert _resolve_requests_verify(_BASE) == bundle_file

    def test_provider_ssl_verify_false_disables(self, clean_env):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_verify=False),
        ):
            assert _resolve_requests_verify(_BASE) is False

    def test_no_base_url_does_not_consult_config(self, clean_env, bundle_file):
        """Existing callers pass no base_url — env-only behavior, no config read."""
        clean_env.setenv("HERMES_CA_BUNDLE", bundle_file)
        probe = MagicMock(return_value=[])
        with patch("hermes_cli.config.get_compatible_custom_providers", probe):
            assert _resolve_requests_verify() == bundle_file
        probe.assert_not_called()

    def test_unmatched_base_url_falls_through_to_env(self, clean_env, bundle_file):
        clean_env.setenv("REQUESTS_CA_BUNDLE", bundle_file)
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers("https://other.example.invalid/v1", ssl_ca_cert="/nope.pem"),
        ):
            assert _resolve_requests_verify(_BASE) == bundle_file

    def test_unmatched_base_url_no_env_returns_true(self, clean_env):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=[],
        ):
            assert _resolve_requests_verify(_BASE) is True

    def test_provider_ca_missing_file_falls_through_to_env(self, clean_env, bundle_file):
        clean_env.setenv("SSL_CERT_FILE", bundle_file)
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_ca_cert="/does/not/exist.pem"),
        ):
            assert _resolve_requests_verify(_BASE) == bundle_file

    def test_config_lookup_failure_falls_through_to_env(self, clean_env, bundle_file):
        clean_env.setenv("SSL_CERT_FILE", bundle_file)
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            side_effect=RuntimeError("config boom"),
        ):
            assert _resolve_requests_verify(_BASE) == bundle_file


class TestCustomProviderSSLContext:
    """``_custom_provider_ssl_context`` — the urllib /models discovery path."""

    def test_returns_verifying_context_with_provider_ca(self, real_ca):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_ca_cert=real_ca),
        ):
            ctx = _custom_provider_ssl_context(_BASE)
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_ssl_verify_false_returns_unverified_context(self):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_verify=False),
        ):
            ctx = _custom_provider_ssl_context(_BASE)
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.check_hostname is False
        assert ctx.verify_mode == ssl.CERT_NONE

    def test_no_base_url_returns_none(self):
        assert _custom_provider_ssl_context("") is None

    def test_unmatched_returns_none(self):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=[],
        ):
            assert _custom_provider_ssl_context(_BASE) is None

    def test_missing_ca_file_returns_none(self):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_ca_cert="/does/not/exist.pem"),
        ):
            assert _custom_provider_ssl_context(_BASE) is None

    def test_config_lookup_failure_returns_none(self):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            side_effect=RuntimeError("config boom"),
        ):
            assert _custom_provider_ssl_context(_BASE) is None


class TestMetadataProbeThreadsProviderCA:
    """End-to-end: the requests metadata probe carries the provider CA to the wire."""

    def test_fetch_endpoint_model_metadata_uses_provider_ca(self, clean_env, bundle_file):
        import agent.model_metadata as mm

        captured = {}

        def fake_get(url, headers=None, timeout=None, verify=None):
            captured["verify"] = verify
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"data": []}
            return resp

        mm._endpoint_model_metadata_cache.clear()
        mm._endpoint_model_metadata_cache_time.clear()
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_ca_cert=bundle_file),
        ), patch.object(mm.requests, "get", side_effect=fake_get):
            mm.fetch_endpoint_model_metadata(_BASE, force_refresh=True)

        assert captured["verify"] == bundle_file

    def test_public_endpoint_keeps_env_default(self, clean_env):
        import agent.model_metadata as mm

        captured = {}

        def fake_get(url, headers=None, timeout=None, verify=None):
            captured["verify"] = verify
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"data": []}
            return resp

        mm._endpoint_model_metadata_cache.clear()
        mm._endpoint_model_metadata_cache_time.clear()
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=[],
        ), patch.object(mm.requests, "get", side_effect=fake_get):
            mm.fetch_endpoint_model_metadata(_BASE, force_refresh=True)

        assert captured["verify"] is True


class TestCatalogProbeThreadsSSLContext:
    """End-to-end: the urllib catalog probe carries the provider SSL context."""

    def test_probe_api_models_passes_ssl_context(self, clean_env, real_ca):
        import hermes_cli.models as models

        captured = {}

        def fake_open(req, *, timeout, ssl_context=None):
            captured["ssl_context"] = ssl_context
            raise urllib.error.URLError("stop after capture")

        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_ca_cert=real_ca),
        ), patch.object(models, "open_credentialed_url", side_effect=fake_open):
            models.probe_api_models(None, _BASE, timeout=1)

        assert isinstance(captured["ssl_context"], ssl.SSLContext)
        assert captured["ssl_context"].verify_mode == ssl.CERT_REQUIRED

    def test_probe_api_models_public_endpoint_uses_default_policy(self, clean_env):
        import hermes_cli.models as models

        captured = {}

        def fake_open(req, *, timeout, ssl_context=None):
            captured["ssl_context"] = ssl_context
            raise urllib.error.URLError("stop after capture")

        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=[],
        ), patch.object(models, "open_credentialed_url", side_effect=fake_open):
            models.probe_api_models(None, _BASE, timeout=1)

        assert captured["ssl_context"] is None

    def test_public_endpoint_calls_seam_without_ssl_context_kwarg(self, clean_env):
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
