"""The custom profile hands Ollama endpoints the native ``/api/chat`` client.

``CustomProfile.create_client()`` is the provider-supplied client hook that
``create_openai_client`` consults before its built-in branches. It returns an
``OllamaNativeClient`` only for an endpoint that answers ``/api/version`` as Ollama, so
vLLM, llama.cpp and LM Studio (also ``provider: custom``) keep the standard client. These
tests drive the real entry point, as ``tests/agent/test_provider_client_seam.py`` does.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.ollama_native_adapter import OllamaNativeClient

_OLLAMA_URL = "http://localhost:11434/v1"
_PROBE = "agent.ollama_native_adapter.is_native_ollama_base_url"


def _agent(provider: str):
    return SimpleNamespace(
        provider=provider,
        _client_log_context=lambda: "",
        _build_keepalive_http_client=lambda *a, **k: None,
    )


def _build(provider: str, base_url: str, **extra):
    from agent.agent_runtime_helpers import create_openai_client

    kwargs = {"api_key": "ollama", "base_url": base_url, **extra}
    return create_openai_client(_agent(provider), kwargs, reason="t", shared=False)


@pytest.mark.parametrize("provider", ["custom", "ollama", "custom:ollama-2"])
def test_an_ollama_endpoint_gets_the_native_client(provider):
    """Bare ``custom``, its aliases and named ``custom:<name>`` routes all resolve to the
    custom profile, so a second Ollama instance is not left on ``/v1``."""
    with patch(_PROBE, return_value=True):
        client = _build(provider, _OLLAMA_URL)
    assert isinstance(client, OllamaNativeClient)
    assert client.base_url == "http://localhost:11434"


@pytest.mark.parametrize("provider", ["custom", "custom:vllm"])
def test_a_non_ollama_endpoint_keeps_the_standard_client(provider):
    from openai import OpenAI

    with patch(_PROBE, return_value=False):
        client = _build(provider, "http://localhost:8000/v1")
    assert isinstance(client, OpenAI)


def test_the_probe_carries_the_endpoint_auth_and_tls_trust():
    """The core pops ``ssl_ca_cert`` / ``ssl_verify`` before the hook runs. Without the
    endpoint's TLS trust the probe fails against a private CA or a self-signed
    certificate and native routing never engages for those deployments. The entry
    is read from a real ``config.yaml`` in the test's ``HERMES_HOME``."""
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(
        f"custom_providers:\n  - name: lan\n    base_url: {_OLLAMA_URL}\n    ssl_verify: false\n",
        encoding="utf-8",
    )
    with patch(_PROBE, return_value=False) as probe:
        _build("custom:lan", _OLLAMA_URL, default_headers={"X-Gateway": "t"}, default_query={"token": "q"})
    probe.assert_called_once()
    kwargs = probe.call_args.kwargs
    assert kwargs["verify"] is False
    assert kwargs["api_key"] == "ollama"
    assert kwargs["default_headers"] == {"X-Gateway": "t"}
    assert kwargs["default_query"] == {"token": "q"}


def test_no_base_url_never_probes():
    from providers import get_provider_profile

    with patch(_PROBE) as probe:
        assert get_provider_profile("custom").create_client(api_key="k", base_url="") is None
    probe.assert_not_called()
