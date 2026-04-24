"""Regression guard: _create_openai_client must skip TCP keepalives for local endpoints.

Issue #14916: TCP keepalive socket options (SO_KEEPALIVE, TCP_KEEPIDLE, etc.)
 injected via a custom httpx.HTTPTransport cause HTTP 502 errors when
 connecting to local LLM servers (llama.cpp, Ollama, vLLM) on localhost
 or 127.0.0.1.  Local servers typically do not expect or handle these
 socket options, and the resulting connection behaviour manifests as
 ``HTTP 502: Error code: 502``.

The fix: before injecting the keepalive-enabled httpx.Client, check whether
 the configured base_url points to a local endpoint.  If so, skip the
 keepalive injection and let OpenAI use its default transport.

This test pins that local endpoints do NOT receive an injected http_client,
 while remote endpoints still do (so the dead-peer detection from #10324
 continues to work for cloud providers).
"""
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(base_url: str):
    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _make_fake_openai_factory(constructed: list):
    """Return a fake ``OpenAI`` class that records kwargs of every construction."""

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            constructed.append(kwargs)

        def close(self):
            pass

    return _FakeOpenAI


class TestLocalEndpointSkipsKeepalive:
    """Local endpoints must not receive the keepalive http_client injection."""

    def test_localhost_skips_http_client(self):
        constructed = []
        fake_openai = _make_fake_openai_factory(constructed)
        agent = _make_agent("http://localhost:1234/v1")

        with patch("run_agent.OpenAI", fake_openai):
            agent._create_openai_client(
                agent._client_kwargs, reason="test", shared=False
            )

        assert len(constructed) == 1
        assert constructed[0].get("http_client") is None, (
            "Localhost endpoint should NOT receive keepalive http_client injection. "
            "This causes HTTP 502 with local LLM servers (#14916)."
        )

    def test_127_0_0_1_skips_http_client(self):
        constructed = []
        fake_openai = _make_fake_openai_factory(constructed)
        agent = _make_agent("http://127.0.0.1:8080/v1")

        with patch("run_agent.OpenAI", fake_openai):
            agent._create_openai_client(
                agent._client_kwargs, reason="test", shared=False
            )

        assert len(constructed) == 1
        assert constructed[0].get("http_client") is None, (
            "127.0.0.1 endpoint should NOT receive keepalive http_client injection."
        )

    def test_0_0_0_0_skips_http_client(self):
        constructed = []
        fake_openai = _make_fake_openai_factory(constructed)
        agent = _make_agent("http://0.0.0.0:5000/v1")

        with patch("run_agent.OpenAI", fake_openai):
            agent._create_openai_client(
                agent._client_kwargs, reason="test", shared=False
            )

        assert len(constructed) == 1
        assert constructed[0].get("http_client") is None, (
            "0.0.0.0 endpoint should NOT receive keepalive http_client injection."
        )


class TestRemoteEndpointKeepsKeepalive:
    """Remote endpoints must continue to receive the keepalive http_client injection."""

    def test_openrouter_gets_http_client(self):
        constructed = []
        fake_openai = _make_fake_openai_factory(constructed)
        agent = _make_agent("https://openrouter.ai/api/v1")

        with patch("run_agent.OpenAI", fake_openai):
            agent._create_openai_client(
                agent._client_kwargs, reason="test", shared=False
            )

        assert len(constructed) == 1
        assert constructed[0].get("http_client") is not None, (
            "Remote endpoint (openrouter) SHOULD receive keepalive http_client "
            "injection for dead-peer detection (#10324)."
        )

    def test_cloud_provider_gets_http_client(self):
        constructed = []
        fake_openai = _make_fake_openai_factory(constructed)
        agent = _make_agent("https://api.openai.com/v1")

        with patch("run_agent.OpenAI", fake_openai):
            agent._create_openai_client(
                agent._client_kwargs, reason="test", shared=False
            )

        assert len(constructed) == 1
        assert constructed[0].get("http_client") is not None, (
            "Remote endpoint (OpenAI) SHOULD receive keepalive http_client "
            "injection for dead-peer detection (#10324)."
        )


class TestExplicitHttpClientPreserved:
    """If caller explicitly passes http_client, it must be preserved regardless of endpoint."""

    def test_explicit_http_client_preserved_for_localhost(self):
        constructed = []
        fake_openai = _make_fake_openai_factory(constructed)
        agent = _make_agent("http://localhost:1234/v1")
        explicit_client = MagicMock()

        with patch("run_agent.OpenAI", fake_openai):
            agent._create_openai_client(
                {**agent._client_kwargs, "http_client": explicit_client},
                reason="test",
                shared=False,
            )

        assert len(constructed) == 1
        assert constructed[0].get("http_client") is explicit_client, (
            "Explicitly provided http_client should be preserved even for localhost."
        )
