"""Gateway imports must not replace the platform trust store with a bundled CA file."""

import os
import subprocess
import sys


def test_gateway_import_does_not_export_a_certifi_bundle(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home))
    env.pop("SSL_CERT_FILE", None)
    env.pop("SSL_CERT_DIR", None)
    # Mask only platform CA-file discovery. certifi remains installed, so the
    # old gateway fallback would publish certifi.where() process-wide.
    script = """
import os, ssl
original = ssl.get_default_verify_paths
ssl.get_default_verify_paths = lambda: original()._replace(cafile=None, openssl_cafile=None)
import gateway.run
assert 'SSL_CERT_FILE' not in os.environ, os.environ.get('SSL_CERT_FILE')
"""
    child = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True,
        text=True, timeout=60, check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr


def test_gateway_import_preserves_explicit_certificate_override(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    bundle = tmp_path / "operator.pem"
    bundle.write_text("operator-provided bundle")
    env = os.environ.copy()
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home), SSL_CERT_FILE=str(bundle))
    script = "import os, gateway.run; assert os.environ['SSL_CERT_FILE'] == os.environ['EXPECTED_CA']"
    env["EXPECTED_CA"] = str(bundle)
    child = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True,
        text=True, timeout=60, check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
