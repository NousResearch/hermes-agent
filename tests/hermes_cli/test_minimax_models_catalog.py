"""Anthropic-wire live model discovery probes ``{base}/v1/models`` (#152406).

An ``anthropic_messages`` base URL is the SDK root (the SDK posts to ``{base}/v1/messages``), so
the generic ``{base}/models`` probe hit ``api.minimax.io/anthropic/models`` (404 on both MiniMax
hosts) and live discovery silently returned nothing: every new MiniMax release stayed invisible.
"""

import json

import pytest

from providers import get_provider_profile


def _record_probe(monkeypatch):
    seen = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"data": [{"id": "MiniMax-M3.1-Flash"}, {"id": "MiniMax-M3"}]}).encode()

    def _open(req, timeout=0):
        seen.append(req.full_url)
        return _Response()

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", _open)
    return seen


@pytest.mark.parametrize(("provider", "expected_url"), [
    ("minimax", "https://api.minimax.io/anthropic/v1/models"),
    ("minimax-cn", "https://api.minimaxi.com/anthropic/v1/models"),
    ("minimax-oauth", "https://api.minimax.io/anthropic/v1/models"),
])
def test_anthropic_wire_profile_probes_v1_models(monkeypatch, provider, expected_url):
    seen = _record_probe(monkeypatch)

    models = get_provider_profile(provider).fetch_models(api_key="probe-key", timeout=1.5)

    assert seen == [expected_url]
    assert models[:1] == ["MiniMax-M3.1-Flash"]


@pytest.mark.parametrize(("custom_base", "expected_url"), [
    ("https://proxy.internal/anthropic", "https://proxy.internal/anthropic/v1/models"),
    ("https://proxy.internal/anthropic/v1/", "https://proxy.internal/anthropic/v1/models"),
])
def test_custom_anthropic_base_probes_the_sdk_catalog_path(monkeypatch, custom_base, expected_url):
    """A user proxy on the Anthropic wire is probed where the SDK would talk to it; never ``/v1/v1``."""
    seen = _record_probe(monkeypatch)

    get_provider_profile("minimax").fetch_models(base_url=custom_base, api_key="k", timeout=1.5)

    assert seen == [expected_url]
