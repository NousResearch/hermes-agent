"""Checks for the optional-mcps/socialrobot catalog entry manifest."""

import re
from pathlib import Path

import pytest

MANIFEST = (
    Path(__file__).resolve().parents[2] / "optional-mcps" / "socialrobot" / "manifest.yaml"
)


@pytest.fixture(scope="module")
def manifest_text():
    assert MANIFEST.exists(), f"missing {MANIFEST}"
    return MANIFEST.read_text()


def test_manifest_required_fields(manifest_text):
    assert re.search(r"^manifest_version:\s*1\s*$", manifest_text, re.MULTILINE)
    assert re.search(r"^name:\s*socialrobot\s*$", manifest_text, re.MULTILINE)
    assert re.search(r"^description:\s*.+\.\s*$", manifest_text, re.MULTILINE)
    assert re.search(r"^source:\s*https://socialrobot\.io/mcp\s*$", manifest_text, re.MULTILINE)


def test_transport_is_remote_http(manifest_text):
    assert re.search(r"^transport:\s*$", manifest_text, re.MULTILINE)
    assert re.search(r"^\s+type:\s*http\s*$", manifest_text, re.MULTILINE)
    assert re.search(
        r"^\s+url:\s*https://socialrobot\.io/api/mcp\s*$", manifest_text, re.MULTILINE
    )


def test_auth_is_native_oauth(manifest_text):
    assert re.search(r"^auth:\s*$", manifest_text, re.MULTILINE)
    assert re.search(r"^\s+type:\s*oauth\s*$", manifest_text, re.MULTILINE)


def test_no_secrets_in_manifest(manifest_text):
    assert "sk-" not in manifest_text
    assert "api_key:" not in manifest_text
    assert re.search(r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}", manifest_text) is None


def test_post_install_present(manifest_text):
    assert "post_install:" in manifest_text
    assert "hermes mcp configure socialrobot" in manifest_text
