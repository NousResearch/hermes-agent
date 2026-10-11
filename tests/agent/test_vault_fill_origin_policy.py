"""Execute the vault mutation boundary with valid and opaque document origins."""
import json
import shutil
import subprocess

import pytest

from agent.vault_login_classifier import build_fill_js


@pytest.mark.parametrize(("expected", "effective", "location_origin", "filled"), [
    ("null", "null", "null", 0),
    ("https://idp.test", "null", "https://idp.test", 0),
    ("file://host", "file://host", "file://host", 0),
    ("not-an-origin", "not-an-origin", "not-an-origin", 0),
    ("https://idp.test", "https://idp.test", "https://idp.test", 1),
    ("http://idp.test:8080", "http://idp.test:8080", "http://idp.test:8080", 1),
    ("https://[::1]:8443", "https://[::1]:8443", "https://[::1]:8443", 1),
])
def test_mutation_requires_supported_nonopaque_document(expected, effective, location_origin, filled):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the generated mutation script")
    expression = build_fill_js([{"index": 0, "token": "current-password", "value": "canary"}],
                               expected_origin=expected, expected_top_level_origin="https://rp.test", nonce="test")
    setup = f"""
const el = {{type: 'password', value: '', focus() {{}}, dispatchEvent() {{}}, removeAttribute() {{}}}};
global.self = {{origin: {json.dumps(effective)}}};
global.window = {{location: {{origin: {json.dumps(location_origin)}, ancestorOrigins: ['https://rp.test']}}}};
window.top = {{}};
global.document = {{querySelector: () => el, querySelectorAll: () => [el]}};
global.HTMLInputElement = function() {{}};
global.Event = function() {{}};
global.InputEvent = function() {{}};
const result = JSON.parse({expression});
console.log(JSON.stringify({{result, value: el.value}}));
"""
    result = subprocess.run([node, "-"], input=setup, text=True, capture_output=True, check=True)
    observed = json.loads(result.stdout)
    assert observed["result"].get("filled", 0) == filled
    assert observed["value"] == ("canary" if filled else "")
