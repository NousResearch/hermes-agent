"""Tests for resolve_provider_client's ``custom`` + ``explicit_base_url`` branch
when the endpoint speaks Anthropic Messages.

When the main provider is ``custom`` and its ``base_url`` ends in ``/anthropic``
(a proxied Anthropic gateway — MiniMax, Zhipu GLM, LiteLLM, or a self-hosted
LLM proxy), auxiliary tasks reach ``resolve_provider_client("custom",
explicit_base_url=..., api_mode="anthropic_messages")`` — directly for a
per-task ``auxiliary.<task>`` override, or via ``_resolve_auto_route`` Step 1 which
forwards the main runtime's ``api_mode``.

The bug (issue #16254): this branch called ``_to_openai_base_url()``
unconditionally, stripping the ``/anthropic`` tail to ``/v1`` even for
``api_mode=anthropic_messages``.  The Anthropic wrapper then never saw the real
``/anthropic`` path, so every side task (title generation, compression, vision,
web_extract, session_search) hit ``.../v1/chat/completions`` on a Messages-only
endpoint and failed.  The sibling named-custom-provider branch already guarded
the rewrite on ``api_mode``; this makes the explicit-base branch consistent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


_ANTHROPIC_BASE = "https://gateway.example.com/proxy/anthropic"
# Known dual-surface (MiniMax) host: the only family still auto-rewritten to /v1
# after the host-anchored policy of #83782 / #83642.
_DUAL_SURFACE_BASE = "https://api.minimax.io/anthropic"
# The OpenAI-wire surface of the same family, as an explicit base_url or a
# credential-pool entry would spell it (#128830).
_DUAL_SURFACE_V1_BASE = "https://api.minimaxi.com/v1"
_DUAL_SURFACE_V1_RESTORED = "https://api.minimaxi.com/anthropic"


class _PlainOpenAIClient:
    """Stand-in with no wrapper opt-out declaration (a MagicMock attr is always truthy)."""


def _client_base_url(client) -> str:
    for chain in (("base_url",), ("_real_client", "base_url"), ("_client", "base_url")):
        obj = client
        try:
            for attr in chain:
                obj = getattr(obj, attr)
            return str(obj)
        except AttributeError:
            continue
    return ""


def test_explicit_base_anthropic_messages_keeps_anthropic_path():
    """api_mode=anthropic_messages must build the Anthropic wrapper on the raw
    ``/anthropic`` base — not the ``/v1``-rewritten one."""
    from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

    fake_anthropic = MagicMock(name="anthropic_sdk_client")
    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_anthropic,
    ) as mock_build:
        client, model = resolve_provider_client(
            "custom",
            model="claude-opus-4-8",
            explicit_base_url=_ANTHROPIC_BASE,
            explicit_api_key="k",
            api_mode="anthropic_messages",
        )

    assert isinstance(client, AnthropicAuxiliaryClient), (
        "custom endpoint with api_mode=anthropic_messages must return the native "
        f"Anthropic wrapper, got {type(client).__name__}"
    )
    # The wrapper — and the Anthropic SDK client it was built from — must keep
    # the /anthropic path, NOT the /v1-rewritten one.
    mock_build.assert_called_once_with("k", _ANTHROPIC_BASE)
    assert client.base_url == _ANTHROPIC_BASE
    assert model == "claude-opus-4-8"


def test_explicit_base_anthropic_messages_openai_fallback_uses_v1():
    """When the anthropic SDK is unavailable, _maybe_wrap_anthropic returns the
    plain OpenAI client — for a dual-surface host it must be on the /v1 base.
    (Anthropic-only gateways keep /anthropic since #83642 — there is no sibling
    /v1 to fall back to.)"""
    from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        side_effect=ImportError("anthropic package not installed"),
    ):
        client, model = resolve_provider_client(
            "custom",
            model="claude-opus-4-8",
            explicit_base_url=_DUAL_SURFACE_BASE,
            explicit_api_key="k",
            api_mode="anthropic_messages",
        )

    assert client is not None
    assert not isinstance(client, AnthropicAuxiliaryClient)
    # /anthropic → /v1 so the OpenAI SDK never hits /anthropic/chat/completions.
    assert _client_base_url(client).rstrip("/").endswith("/v1")


def test_explicit_base_without_anthropic_mode_preserves_v1_rewrite():
    """Regression: with no anthropic_messages api_mode, the /anthropic → /v1
    OpenAI-wire rewrite is preserved for known dual-surface hosts."""
    from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

    client, model = resolve_provider_client(
        "custom",
        model="my-model",
        explicit_base_url=_DUAL_SURFACE_BASE,
        explicit_api_key="k",
        api_mode="chat_completions",
    )

    assert client is not None
    assert not isinstance(client, AnthropicAuxiliaryClient)
    assert _client_base_url(client).rstrip("/").endswith("/v1")
    assert "/anthropic" not in _client_base_url(client)


def test_explicit_base_unknown_host_keeps_anthropic_path():
    """Anthropic-only custom gateways (unknown hosts) keep their /anthropic
    path even on the OpenAI wire — rewriting to /v1 404s (#83642)."""
    from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

    client, model = resolve_provider_client(
        "custom",
        model="my-model",
        explicit_base_url=_ANTHROPIC_BASE,
        explicit_api_key="k",
        api_mode="chat_completions",
    )

    assert client is not None
    assert not isinstance(client, AnthropicAuxiliaryClient)
    assert _client_base_url(client).rstrip("/").endswith("/proxy/anthropic")


def test_api_key_dual_surface_v1_restored_for_anthropic_messages():
    """A dual-surface host's OpenAI ``/v1`` base must never reach the Anthropic
    SDK: it appends its own ``/v1/messages``, so ``/v1/v1/messages`` 404s
    (#128830). The ``minimax-cn`` profile declares ``api_mode=anthropic_messages``,
    so an explicit ``/v1`` base_url (or one stored in the credential pool) is
    wrapped as Messages traffic exactly there."""
    from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

    fake_anthropic = MagicMock(name="anthropic_sdk_client")
    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_anthropic,
    ) as mock_build:
        client, model = resolve_provider_client(
            "minimax-cn",
            model="MiniMax-M2.7",
            explicit_base_url=_DUAL_SURFACE_V1_BASE,
            explicit_api_key="k",
        )

    assert isinstance(client, AnthropicAuxiliaryClient), (
        "minimax-cn declares anthropic_messages, so the wrap must happen even "
        f"without a task-level api_mode, got {type(client).__name__}"
    )
    mock_build.assert_called_once_with("k", _DUAL_SURFACE_V1_RESTORED)
    assert client.base_url == _DUAL_SURFACE_V1_RESTORED


def test_maybe_wrap_anthropic_restores_dual_surface_bare_host():
    """A bare dual-surface root has no Messages surface at ``/`` either — the
    SDK would request ``/v1/messages`` on the OpenAI host root."""
    from agent.auxiliary_client import _maybe_wrap_anthropic, AnthropicAuxiliaryClient

    fake_anthropic = MagicMock(name="anthropic_sdk_client")
    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_anthropic,
    ) as mock_build:
        wrapped = _maybe_wrap_anthropic(
            _PlainOpenAIClient(), "MiniMax-M2.7", "k", "https://api.minimax.io",
            "anthropic_messages",
        )

    assert isinstance(wrapped, AnthropicAuxiliaryClient)
    mock_build.assert_called_once_with("k", "https://api.minimax.io/anthropic")
    assert wrapped.base_url == "https://api.minimax.io/anthropic"


def test_maybe_wrap_anthropic_keeps_foreign_gateway_v1():
    """An unknown gateway's ``/v1`` stays verbatim — only the known dual-surface
    families have a Messages surface to restore; rewriting anything else would
    break proxies that really do serve Messages under /v1."""
    from agent.auxiliary_client import _maybe_wrap_anthropic, AnthropicAuxiliaryClient

    fake_anthropic = MagicMock(name="anthropic_sdk_client")
    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_anthropic,
    ) as mock_build:
        wrapped = _maybe_wrap_anthropic(
            _PlainOpenAIClient(), "my-model", "k", "https://gateway.example.com/v1",
            "anthropic_messages",
        )

    assert isinstance(wrapped, AnthropicAuxiliaryClient)
    mock_build.assert_called_once_with("k", "https://gateway.example.com/v1")
    assert wrapped.base_url == "https://gateway.example.com/v1"


def test_maybe_wrap_anthropic_auto_v1_stays_openai_wire():
    """Without an explicit api_mode, a ``/v1`` base does not speak Messages by
    URL heuristic, so no wrap happens and the OpenAI wire keeps using ``/v1``
    (the MiniMax OpenAI-compatible surface) — the restore must not change that."""
    from agent.auxiliary_client import _maybe_wrap_anthropic

    plain = _PlainOpenAIClient()
    wrapped = _maybe_wrap_anthropic(plain, "my-model", "k", _DUAL_SURFACE_V1_BASE, None)

    assert wrapped is plain
