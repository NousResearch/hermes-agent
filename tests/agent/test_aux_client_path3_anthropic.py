"""Tests for the explicit_base_url (path 3) anthropic_messages fix (#16254).

Before this fix, ``resolve_provider_client`` with ``explicit_base_url`` set
always built a plain OpenAI client regardless of ``api_mode``, silently sending
Anthropic-format bodies to the OpenAI endpoint.  The fix mirrors the identical
dispatch that path 2 (named custom providers) has had since #15059.

Also tests the ``key_env`` fix in ``_resolve_task_provider_model``: auxiliary
task configs that specify ``key_env`` instead of ``api_key`` must resolve the
env var, matching the behaviour of ``providers:`` dict entries.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY",
                "ANTHROPIC_TOKEN", "ZENMUX_API_KEY", "TEST_KEY_ENV"):
        monkeypatch.delenv(key, raising=False)


# ── Path 3: explicit_base_url + api_mode=anthropic_messages ─────────────────

def _make_anthropic_mock():
    """Return a fake Anthropic client and the patcher."""
    fake_client = MagicMock(name="anthropic_real_client")
    patcher = patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_client,
    )
    return fake_client, patcher


def test_explicit_base_url_anthropic_messages_returns_anthropic_client():
    """Path 3: api_mode=anthropic_messages must return AnthropicAuxiliaryClient."""
    from agent.auxiliary_client import (
        AnthropicAuxiliaryClient,
        resolve_provider_client,
    )

    fake_raw, patcher = _make_anthropic_mock()
    with patcher:
        client, model = resolve_provider_client(
            "custom",
            model="moonshotai/kimi-k2.6",
            explicit_base_url="https://zenmux.ai/api/anthropic",
            explicit_api_key="sk-test-key",
            api_mode="anthropic_messages",
        )

    assert isinstance(client, AnthropicAuxiliaryClient), (
        f"Expected AnthropicAuxiliaryClient, got {type(client).__name__}.  "
        "Path 3 is silently routing to OpenAI-wire when api_mode=anthropic_messages "
        "is set — this is the #16254 regression."
    )
    assert model == "moonshotai/kimi-k2.6"


def test_explicit_base_url_no_api_mode_returns_openai_client():
    """Path 3 without api_mode must still return a plain OpenAI-wire client."""
    from agent.auxiliary_client import AnthropicAuxiliaryClient, resolve_provider_client

    client, model = resolve_provider_client(
        "custom",
        model="gpt-4o-mini",
        explicit_base_url="https://my-openai-proxy.example.com/v1",
        explicit_api_key="sk-test",
        api_mode=None,
    )
    assert not isinstance(client, AnthropicAuxiliaryClient), (
        "Without api_mode, path 3 should return an OpenAI-wire client."
    )


def test_explicit_base_url_anthropic_sdk_missing_falls_back_to_openai(monkeypatch):
    """If anthropic SDK absent, path 3 falls back gracefully to OpenAI-wire."""
    from agent.auxiliary_client import AnthropicAuxiliaryClient, resolve_provider_client

    # Simulate ImportError from build_anthropic_client
    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        side_effect=ImportError("anthropic not installed"),
    ):
        client, model = resolve_provider_client(
            "custom",
            model="test-model",
            explicit_base_url="https://anthropic-gateway.example.com",
            explicit_api_key="sk-x",
            api_mode="anthropic_messages",
        )

    assert not isinstance(client, AnthropicAuxiliaryClient), (
        "On ImportError, path 3 must fall back to OpenAI-wire rather than raise."
    )


def test_explicit_base_url_anthropic_messages_async_returns_async_wrapper():
    """Path 3 + async_mode=True must wrap in AsyncAnthropicAuxiliaryClient."""
    from agent.auxiliary_client import (
        AsyncAnthropicAuxiliaryClient,
        resolve_provider_client,
    )

    _, patcher = _make_anthropic_mock()
    with patcher:
        client, _ = resolve_provider_client(
            "custom",
            model="claude-3-haiku",
            explicit_base_url="https://zenmux.ai/api/anthropic",
            explicit_api_key="sk-z",
            api_mode="anthropic_messages",
            async_mode=True,
        )

    assert isinstance(client, AsyncAnthropicAuxiliaryClient)


# ── key_env fix: _resolve_task_provider_model reads key_env from task config ─

def test_resolve_task_provider_model_reads_key_env(monkeypatch):
    """auxiliary.<task>.key_env must be resolved from the environment.

    Before the fix, ``key_env`` was silently dropped, leaving ``api_key``
    as "no-key-required" and causing 403 access_denied from the upstream
    provider even though the env var was set (#16254 sibling bug).
    """
    monkeypatch.setenv("TEST_KEY_ENV", "resolved-from-env-key")

    task_config = {
        "provider": "custom",
        "model": "some-model",
        "base_url": "https://example.com/v1",
        "key_env": "TEST_KEY_ENV",
        # no 'api_key' — only key_env
        "api_mode": "anthropic_messages",
    }

    from agent.auxiliary_client import _get_auxiliary_task_config, _resolve_task_provider_model

    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value=task_config,
    ):
        _, _, _, resolved_key, _ = _resolve_task_provider_model(task="vision")

    assert resolved_key == "resolved-from-env-key", (
        f"Expected 'resolved-from-env-key', got {resolved_key!r}.  "
        "key_env is not being read from the auxiliary task config."
    )


def test_resolve_task_provider_model_api_key_wins_over_key_env(monkeypatch):
    """Explicit api_key takes precedence over key_env when both are set."""
    monkeypatch.setenv("TEST_KEY_ENV", "env-key")

    task_config = {
        "provider": "custom",
        "base_url": "https://example.com",
        "api_key": "explicit-key",
        "key_env": "TEST_KEY_ENV",
        "api_mode": None,
    }

    from agent.auxiliary_client import _resolve_task_provider_model

    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value=task_config,
    ):
        _, _, _, resolved_key, _ = _resolve_task_provider_model(task="compression")

    assert resolved_key == "explicit-key"


def test_resolve_task_provider_model_no_key_env_set_returns_none(monkeypatch):
    """When key_env names an unset env var, api_key should remain None."""
    monkeypatch.delenv("MISSING_ENV_KEY", raising=False)

    task_config = {
        "provider": "custom",
        "base_url": "https://example.com",
        "key_env": "MISSING_ENV_KEY",
        "api_mode": None,
    }

    from agent.auxiliary_client import _resolve_task_provider_model

    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value=task_config,
    ):
        _, _, _, resolved_key, _ = _resolve_task_provider_model(task="compression")

    # Unset env var → None, not empty string or "no-key-required"
    assert resolved_key is None


def test_explicit_base_url_query_params_preserved_for_anthropic_client():
    """build_anthropic_client must receive the original URL including query params.

    Before the fix, _clean_base (query params stripped) was passed instead of
    custom_base, so Azure-style ?api-version= params were silently dropped
    before build_anthropic_client could route them via default_query.
    """
    from agent.auxiliary_client import resolve_provider_client

    received_urls: list[str] = []

    def _capture_build_anthropic_client(api_key, base_url=None, **kw):
        received_urls.append(base_url or "")
        return MagicMock(name="anthropic_client")

    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        side_effect=_capture_build_anthropic_client,
    ):
        resolve_provider_client(
            "custom",
            model="claude-3-haiku",
            explicit_base_url="https://my-azure-endpoint.openai.azure.com/v1?api-version=2025-04-15",
            explicit_api_key="sk-azure",
            api_mode="anthropic_messages",
        )

    assert received_urls, "build_anthropic_client was never called"
    assert "api-version=2025-04-15" in received_urls[0], (
        f"Query params were stripped before passing to build_anthropic_client: {received_urls[0]!r}.  "
        "Pass the original URL so the function can route api-version via default_query."
    )
