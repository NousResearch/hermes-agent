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
    import os, ssl

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


