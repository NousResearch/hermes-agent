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
        _macos_system_ca = {"/etc/ssl/cert.pem", "/private/etc/ssl/cert.pem"}
        for candidate in (paths.cafile, paths.openssl_cafile):
            if not candidate or not os.path.exists(candidate):
                continue
            if sys.platform == "darwin":
                try:
                    if os.path.realpath(candidate) in _macos_system_ca:
                        continue
                except OSError:
                    if candidate in _macos_system_ca:
                        continue
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

    def test_macos_skips_frozen_system_bundle_for_certifi(self, tmp_path):
        """#100414: /etc/ssl/cert.pem always exists on macOS and is incomplete."""
        fn = _load_ensure_ssl()
        certifi_path = str(tmp_path / "cacert.pem")
        tmp_path.joinpath("cacert.pem").write_text("dummy", encoding="utf-8")
        fake_paths = MagicMock(cafile="/etc/ssl/cert.pem", openssl_cafile="/etc/ssl/cert.pem")
        fake_certifi = MagicMock()
        fake_certifi.where.return_value = certifi_path
        with patch.dict(os.environ, {}, clear=False), \
             patch("sys.platform", "darwin"), \
             patch("ssl.get_default_verify_paths", return_value=fake_paths), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.realpath", side_effect=lambda p: p), \
             patch.dict("sys.modules", {"certifi": fake_certifi}):
            os.environ.pop("SSL_CERT_FILE", None)
            fn()
            assert os.environ["SSL_CERT_FILE"] == certifi_path


