"""End-to-end test: a real config.yaml setting for follow_redirects flows
through the real ``load_config_readonly`` + ``get_provider_follow_redirects``
chain to the constructed LLM API client.

Mirrors the AGENTS.md rubric: "E2E validation, not just green unit mocks.
For anything touching resolution chains, config propagation, security
boundaries, remote backends, or file/network I/O, exercise the real path
with real imports against a temp ``HERMES_HOME``."
"""
from unittest.mock import patch

import httpx
import pytest

from run_agent import AIAgent


def _write_config(home, body):
    (home / "config.yaml").write_text(body)


def _make_agent(provider, base_url):
    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider=provider,
        model="gpt-5.4",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


@patch("run_agent.OpenAI")
def test_e2e_follow_redirects_via_real_config_loading(
    mock_openai, tmp_path, monkeypatch
):
    """A real ``config.yaml`` with ``providers.<id>.follow_redirects: true``
    produces an ``httpx.Client`` with ``follow_redirects=True``."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        "providers:\n"
        "  redirect-provider:\n"
        "    follow_redirects: true\n",
    )

    from hermes_cli.config import load_config_readonly
    # Sanity: the resolver must observe the real config we just wrote.
    from hermes_cli.timeouts import get_provider_follow_redirects
    assert get_provider_follow_redirects("redirect-provider", "gpt-5.4") is True
    # And the unrelated provider stays at the default.
    assert get_provider_follow_redirects("openai-codex", "gpt-5.4") is False
    # (cache is fine — we wrote the file before the first call)

    agent = _make_agent("redirect-provider", "https://api.example.com/v1")
    agent._create_openai_client(
        {"api_key": "test-key", "base_url": "https://api.example.com/v1"},
        reason="test",
        shared=False,
    )
    forwarded = mock_openai.call_args.kwargs
    http_client = forwarded.get("http_client")
    assert isinstance(http_client, httpx.Client)
    assert http_client.follow_redirects is True
    http_client.close()


@patch("run_agent.OpenAI")
def test_e2e_unset_provider_stays_off(mock_openai, tmp_path, monkeypatch):
    """An unrelated config.yaml entry does NOT enable redirects for other
    providers — the per-provider scoping must work end-to-end."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        "providers:\n"
        "  redirect-provider:\n"
        "    follow_redirects: true\n",
    )

    agent = _make_agent("openai-codex", "https://api.openai.com/v1")
    agent._create_openai_client(
        {"api_key": "test-key", "base_url": "https://api.openai.com/v1"},
        reason="test",
        shared=False,
    )
    forwarded = mock_openai.call_args.kwargs
    http_client = forwarded.get("http_client")
    assert isinstance(http_client, httpx.Client)
    assert http_client.follow_redirects is False
    http_client.close()


@patch("run_agent.OpenAI")
def test_e2e_no_config_at_all_keeps_default_off(mock_openai, tmp_path, monkeypatch):
    """Fresh ``HERMES_HOME`` with no config.yaml → follow_redirects stays off."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Deliberately do not write any config.yaml — get_provider_follow_redirects
    # must tolerate missing files and return False.

    agent = _make_agent("redirect-provider", "https://api.example.com/v1")
    agent._create_openai_client(
        {"api_key": "test-key", "base_url": "https://api.example.com/v1"},
        reason="test",
        shared=False,
    )
    forwarded = mock_openai.call_args.kwargs
    http_client = forwarded.get("http_client")
    assert isinstance(http_client, httpx.Client)
    assert http_client.follow_redirects is False
    http_client.close()
