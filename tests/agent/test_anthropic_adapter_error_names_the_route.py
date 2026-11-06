"""The missing-SDK error names the route that needs it, not a provider the operator may not use (#144749).

`build_anthropic_client` is the single entry point for every `anthropic_messages` route, including
`base_url`s pointing at GLM / Kimi / a proxy. Naming "the Anthropic provider" sent those operators
looking at the wrong dependency.
"""

import pytest

from agent import anthropic_adapter


@pytest.fixture
def no_sdk(monkeypatch):
    monkeypatch.setattr(anthropic_adapter, "_get_anthropic_sdk", lambda: None)
    monkeypatch.setattr(anthropic_adapter, "_anthropic_install_error", None)


def test_a_proxy_route_names_its_own_base_url(no_sdk):
    with pytest.raises(ImportError) as excinfo:
        anthropic_adapter.build_anthropic_client("key", base_url="https://proxy.example/v1")

    message = str(excinfo.value)
    assert "proxy.example" in message
    assert "the Anthropic provider" not in message


def test_the_native_route_names_the_public_endpoint(no_sdk):
    with pytest.raises(ImportError) as excinfo:
        anthropic_adapter.build_anthropic_client("key")

    assert "api.anthropic.com" in str(excinfo.value)
