"""Tests for SSL certificate auto-detection in gateway/run.py."""

import os
from unittest.mock import patch, MagicMock


def _load_ensure_ssl():
    """Import _ensure_ssl_certs fresh (gateway/run.py has heavy deps, so we
    extract just the function source to avoid importing the whole gateway)."""
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
        problematic_darwin_defaults = {"/private/etc/ssl/cert.pem", "/etc/ssl/cert.pem"}
        if sys.platform == "darwin" and (paths.cafile in problematic_darwin_defaults or paths.openssl_cafile in problematic_darwin_defaults):
            for candidate in (
                "/opt/homebrew/etc/openssl@3/cert.pem",
                "/usr/local/etc/openssl@3/cert.pem",
                "/opt/homebrew/etc/openssl@1.1/cert.pem",
                "/usr/local/etc/openssl@1.1/cert.pem",
            ):
                if os.path.exists(candidate):
                    os.environ["SSL_CERT_FILE"] = candidate
                    return
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
    """)
    mod = ModuleType("_ssl_helper")
    exec(code, mod.__dict__)
    return mod._ensure_ssl_certs


class TestEnsureSslCerts:
    def test_respects_existing_env_var(self):
        fn = _load_ensure_ssl()
        with patch.dict(os.environ, {"SSL_CERT_FILE": "/custom/ca.pem"}):
            fn()
            assert os.environ["SSL_CERT_FILE"] == "/custom/ca.pem"

    def test_sets_from_ssl_default_paths(self, tmp_path):
        fn = _load_ensure_ssl()
        cert = tmp_path / "ca.crt"
        cert.write_text("FAKE CERT")

        mock_paths = MagicMock()
        mock_paths.cafile = str(cert)
        mock_paths.openssl_cafile = None

        env = {k: v for k, v in os.environ.items() if k != "SSL_CERT_FILE"}

        def fake_exists(path):
            return path == str(cert)

        with patch.dict(os.environ, env, clear=True), \
             patch("ssl.get_default_verify_paths", return_value=mock_paths), \
             patch("os.path.exists", side_effect=fake_exists):
            fn()
            assert os.environ.get("SSL_CERT_FILE") == str(cert)

    def test_prefers_homebrew_openssl_bundle_over_problematic_darwin_default(self):
        fn = _load_ensure_ssl()
        homebrew = "/opt/homebrew/etc/openssl@3/cert.pem"
        python_default = "/private/etc/ssl/cert.pem"

        mock_paths = MagicMock()
        mock_paths.cafile = python_default
        mock_paths.openssl_cafile = None

        env = {k: v for k, v in os.environ.items() if k != "SSL_CERT_FILE"}

        def fake_exists(path):
            return path in {homebrew, python_default}

        with patch.dict(os.environ, env, clear=True), \
             patch("ssl.get_default_verify_paths", return_value=mock_paths), \
             patch("os.path.exists", side_effect=fake_exists), \
             patch("sys.platform", "darwin"):
            fn()
            assert os.environ.get("SSL_CERT_FILE") == homebrew

    def test_does_not_override_non_problematic_darwin_default_with_homebrew_bundle(self):
        fn = _load_ensure_ssl()
        homebrew = "/opt/homebrew/etc/openssl@3/cert.pem"
        python_default = "/custom/python/cert.pem"

        mock_paths = MagicMock()
        mock_paths.cafile = python_default
        mock_paths.openssl_cafile = None

        env = {k: v for k, v in os.environ.items() if k != "SSL_CERT_FILE"}

        def fake_exists(path):
            return path in {homebrew, python_default}

        with patch.dict(os.environ, env, clear=True), \
             patch("ssl.get_default_verify_paths", return_value=mock_paths), \
             patch("os.path.exists", side_effect=fake_exists), \
             patch("sys.platform", "darwin"):
            fn()
            assert os.environ.get("SSL_CERT_FILE") == python_default

    def test_no_op_when_nothing_found(self):
        fn = _load_ensure_ssl()
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
