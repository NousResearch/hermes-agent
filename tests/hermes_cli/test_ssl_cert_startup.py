"""Behavior coverage for CLI certificate initialization order."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.macos_only
def test_cli_startup_selects_certifi_before_https_context(tmp_path):
    env = os.environ.copy()
    for name in (
        "HERMES_CA_BUNDLE",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        env.pop(name, None)
    env["HOME"] = str(tmp_path)
    env["HERMES_HOME"] = str(tmp_path / ".hermes")

    code = """
import hashlib
import http.client
import os
from pathlib import Path
import urllib.request

import certifi
import hermes_cli.main

selected = Path(os.environ["SSL_CERT_FILE"]).resolve()
expected_path = Path(certifi.where()).resolve()
assert selected == expected_path

handler = next(
    item
    for item in urllib.request.build_opener().handlers
    if type(item) is urllib.request.HTTPSHandler
)
context = handler._context
if context is None:
    context = http.client.HTTPSConnection("example.invalid")._context

expected_context = __import__("ssl").create_default_context(cafile=str(expected_path))
fingerprints = lambda value: {
    hashlib.sha256(cert).digest() for cert in value.get_ca_certs(binary_form=True)
}
assert fingerprints(context) == fingerprints(expected_context)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
