"""Custom-provider TLS settings must reach the /models and pricing probes.

Regression coverage for provider-scoped ``ssl_ca_cert`` / ``ssl_verify`` being
ignored by the discovery/pricing probes. Two probe families share the root
cause and are covered here:

* HTTPX endpoint metadata / pricing probe
  (``agent.model_metadata.fetch_endpoint_model_metadata`` via
  ``resolve_verify``).
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

from contextlib import contextmanager
import ssl
import urllib.error
from unittest.mock import MagicMock, patch

import certifi
import pytest

from agent.model_metadata_http import resolve_verify
from agent.ssl_verify import resolve_httpx_verify
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
    from pathlib import Path
    path.write_bytes(Path(certifi.where()).read_bytes())
    return str(path)


@pytest.fixture
def real_ca():
    """Both probe transports parse this actual bundle before connecting."""
    return certifi.where()


def _providers(base_url, **tls):
    entry = {"name": "relay", "base_url": base_url}
    entry.update(tls)
    return [entry]


class TestResolveVerifyProviderScoped:
    """Provider configuration resolves before the probe's transport is built."""

    def test_provider_ca_used_for_matching_base_url(self, clean_env, bundle_file):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_ca_cert=bundle_file),
        ):
            assert resolve_verify(_BASE) is resolve_httpx_verify(ca_bundle=bundle_file)

    def test_provider_ca_overrides_env_ssl_cert_file(self, clean_env, tmp_path, bundle_file):
        env_bundle = tmp_path / "env-ca.pem"
        env_bundle.write_text("stub")
        clean_env.setenv("SSL_CERT_FILE", str(env_bundle))
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_ca_cert=bundle_file),
        ):
            assert resolve_verify(_BASE) is resolve_httpx_verify(ca_bundle=bundle_file)

    def test_provider_ssl_verify_false_disables(self, clean_env):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_verify=False),
        ):
            assert resolve_verify(_BASE) is False

    def test_no_base_url_does_not_consult_config(self, clean_env, bundle_file):
        """Existing callers pass no base_url — no config read, and env never steers trust."""
        clean_env.setenv("HERMES_CA_BUNDLE", bundle_file)
        probe = MagicMock(return_value=[])
        with patch("hermes_cli.config.get_compatible_custom_providers", probe):
            assert resolve_verify() is True
        probe.assert_not_called()

    def test_unmatched_base_url_env_ignored_returns_true(self, clean_env, bundle_file):
        """An unmatched base_url and an ambient CA env var both leave trust at the OS store."""
        clean_env.setenv("REQUESTS_CA_BUNDLE", bundle_file)
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers("https://other.example.invalid/v1", ssl_ca_cert="/nope.pem"),
        ):
            assert resolve_verify(_BASE) is True

    def test_unmatched_base_url_no_env_returns_true(self, clean_env):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=[],
        ):
            assert resolve_verify(_BASE) is True

    def test_provider_ca_missing_file_keeps_platform_trust(self, clean_env):
        """A configured bundle that does not exist warns and verifies against the OS store —
        an ambient ``SSL_CERT_FILE`` does not become the fallback authority."""
        clean_env.setenv("SSL_CERT_FILE", "/does/not/matter.pem")
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_ca_cert="/does/not/exist.pem"),
        ):
            context = resolve_verify(_BASE)
            assert type(context).__module__.startswith("truststore")
            assert context.verify_mode == ssl.CERT_REQUIRED

    def test_config_lookup_failure_keeps_platform_trust(self, clean_env):
        """A config crash must not silently swap in an ambient env var as trust authority."""
        clean_env.setenv("SSL_CERT_FILE", "/does/not/matter.pem")
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            side_effect=RuntimeError("config boom"),
        ):
            context = resolve_verify(_BASE)
            assert type(context).__module__.startswith("truststore")
            assert context.verify_mode == ssl.CERT_REQUIRED


class TestCustomProviderSSLContext:
    """``_custom_provider_ssl_context`` — the urllib /models discovery path."""

    def test_returns_verifying_context_with_provider_ca(self, real_ca):
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_providers(_BASE, ssl_ca_cert=real_ca),
        ):
            ctx = _custom_provider_ssl_context(_BASE)
        assert ctx is resolve_httpx_verify(ca_bundle=real_ca)
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
    @pytest.mark.parametrize("pinned", [False, True])
    def test_provider_policy_reaches_the_transport(self, clean_env, real_ca, pinned):
        import httpx
        from agent import model_metadata as mm
        from agent import model_metadata_http
        from agent.ssl_verify import resolve_httpx_verify

        @contextmanager
        def response(*args, **kwargs):
            yield httpx.Response(200, json={"data": []}, request=httpx.Request("GET", args[0]))

        mm._endpoint_model_metadata_cache.clear()
        mm._endpoint_model_metadata_cache_time.clear()
        providers = _providers(_BASE, ssl_ca_cert=real_ca) if pinned else []
        with patch("hermes_cli.config.get_compatible_custom_providers", return_value=providers), \
             patch.object(model_metadata_http, "stream", side_effect=response) as stream:
            assert mm.fetch_endpoint_model_metadata(_BASE, force_refresh=True) == {}
        expected = resolve_httpx_verify(ca_bundle=real_ca) if pinned else True
        assert stream.call_args.kwargs["verify"] is expected


class TestCatalogProbeThreadsSSLContext:
    """End-to-end: the urllib catalog probe carries the provider SSL context."""

    @pytest.fixture(autouse=True)
    def _clear_probe_neg_cache(self):
        import hermes_cli.models as models

        models._probe_neg_cache.clear()
        yield
        models._probe_neg_cache.clear()

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

        assert captured["ssl_context"] is resolve_httpx_verify(ca_bundle=real_ca)
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
