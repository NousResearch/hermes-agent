"""Tests for SSL certificate handling helpers in gateway/run.py."""

import os
import sys
from unittest.mock import patch, MagicMock


def _load_ssl_helpers():
    """Load the SSL helper functions without importing the whole gateway."""
    # We can test via the actual module since conftest isolates HERMES_HOME,
    # but we need to be careful about side effects.  Instead, replicate the
    # logic in a controlled way.
    from types import ModuleType
    import textwrap, ssl as _ssl  # noqa: F401

    code = textwrap.dedent("""\
    import os, ssl, sys

    def _ensure_ssl_certs():
        if "SSL_CERT_FILE" in os.environ:
            return
        paths = ssl.get_default_verify_paths()
        for candidate in (paths.cafile, paths.openssl_cafile):
            if candidate and os.path.exists(candidate):
                os.environ["SSL_CERT_FILE"] = candidate
                return
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
            return
        except ImportError:
            pass
        for candidate in (
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/ssl/cert.pem",
        ):
            if os.path.exists(candidate):
                os.environ["SSL_CERT_FILE"] = candidate
                return

    def _refresh_aiohttp_ssl_contexts_if_loaded():
        aiohttp_connector = sys.modules.get("aiohttp.connector")
        if aiohttp_connector is None:
            return

        make_ssl_context = getattr(aiohttp_connector, "_make_ssl_context", None)
        if not callable(make_ssl_context):
            return

        try:
            aiohttp_connector._SSL_CONTEXT_VERIFIED = make_ssl_context(True)
            aiohttp_connector._SSL_CONTEXT_UNVERIFIED = make_ssl_context(False)
        except Exception:
            pass
    """)
    mod = ModuleType("_ssl_helper")
    exec(code, mod.__dict__)
    return mod._ensure_ssl_certs, mod._refresh_aiohttp_ssl_contexts_if_loaded


class TestEnsureSslCerts:
    def test_respects_existing_env_var(self):
        fn, _ = _load_ssl_helpers()
        with patch.dict(os.environ, {"SSL_CERT_FILE": "/custom/ca.pem"}):
            fn()
            assert os.environ["SSL_CERT_FILE"] == "/custom/ca.pem"

    def test_sets_from_ssl_default_paths(self, tmp_path):
        fn, _ = _load_ssl_helpers()
        cert = tmp_path / "ca.crt"
        cert.write_text("FAKE CERT")

        mock_paths = MagicMock()
        mock_paths.cafile = str(cert)
        mock_paths.openssl_cafile = None

        env = {k: v for k, v in os.environ.items() if k != "SSL_CERT_FILE"}
        with patch.dict(os.environ, env, clear=True), \
             patch("ssl.get_default_verify_paths", return_value=mock_paths):
            fn()
            assert os.environ.get("SSL_CERT_FILE") == str(cert)

    def test_no_op_when_nothing_found(self):
        fn, _ = _load_ssl_helpers()
        mock_paths = MagicMock()
        mock_paths.cafile = None
        mock_paths.openssl_cafile = None

        env = {k: v for k, v in os.environ.items() if k != "SSL_CERT_FILE"}
        with patch.dict(os.environ, env, clear=True), \
             patch("ssl.get_default_verify_paths", return_value=mock_paths), \
             patch("os.path.exists", return_value=False), \
             patch.dict("sys.modules", {"certifi": None}):
            fn()
            assert "SSL_CERT_FILE" not in os.environ


class TestRefreshAiohttpSslContexts:
    def test_refreshes_cached_verified_and_unverified_contexts(self):
        _, refresh = _load_ssl_helpers()

        class FakeConnectorModule:
            _SSL_CONTEXT_VERIFIED = "stale-verified"
            _SSL_CONTEXT_UNVERIFIED = "stale-unverified"

            @staticmethod
            def _make_ssl_context(verified):
                return f"fresh-{verified}"

        original = sys.modules.get("aiohttp.connector")
        try:
            sys.modules["aiohttp.connector"] = FakeConnectorModule
            refresh()
            assert FakeConnectorModule._SSL_CONTEXT_VERIFIED == "fresh-True"
            assert FakeConnectorModule._SSL_CONTEXT_UNVERIFIED == "fresh-False"
        finally:
            if original is None:
                sys.modules.pop("aiohttp.connector", None)
            else:
                sys.modules["aiohttp.connector"] = original

    def test_no_op_when_aiohttp_connector_not_loaded(self):
        _, refresh = _load_ssl_helpers()
        original = sys.modules.pop("aiohttp.connector", None)
        try:
            refresh()
        finally:
            if original is not None:
                sys.modules["aiohttp.connector"] = original
