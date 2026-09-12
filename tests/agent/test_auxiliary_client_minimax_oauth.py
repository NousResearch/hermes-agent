"""Regression: ``resolve_provider_client("minimax-oauth", ...)`` must build a
refresh-capable Anthropic auxiliary client, not silently return (None, None).

The auxiliary router dispatches on ``PROVIDER_REGISTRY[*].auth_type``.
``minimax-oauth`` registers ``auth_type == "oauth_minimax"``; without an
explicit arm the resolver falls through the ``oauth_device_code`` /
``oauth_external`` branch, logs a one-time warning, and returns (None, None).
Every aux task pinned to ``provider: minimax-oauth`` (compression,
title_generation, …) then silently re-routes to the Step-2 fallback chain —
the operator's explicit configuration never reaches the wire.

The fix routes through ``resolve_minimax_oauth_runtime_credentials(as_token_provider=True)``
and wraps the resulting Anthropic SDK client in
``AnthropicAuxiliaryClient(..., is_oauth=True)``. The callable bearer mints
a fresh access token per outbound request because MiniMax's tokens live
~15 minutes and a static string would 401 mid-session.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

INFERENCE_BASE_URL = "https://api.minimax.io/anthropic"


def _runtime_creds():
    """Return what resolve_minimax_oauth_runtime_credentials(as_token_provider=True) yields."""
    return {
        "provider": "minimax-oauth",
        "api_key": MagicMock(name="minimax_token_provider", return_value="fresh-bearer"),
        "base_url": INFERENCE_BASE_URL,
        "source": "oauth",
    }


def test_resolve_minimax_oauth_builds_anthropic_wrapper_with_oauth_semantics():
    """Happy path: token-provider + base_url → AnthropicAuxiliaryClient with is_oauth=True,
    Anthropic SDK built with the callable bearer (not a static string), and the
    resolved model passed through verbatim.
    """
    from agent.auxiliary_client import (
        AnthropicAuxiliaryClient,
        resolve_provider_client,
    )

    fake_anthropic_client = MagicMock(name="anthropic_sdk_client")
    creds = _runtime_creds()

    with patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        return_value=creds,
    ), patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_anthropic_client,
    ) as mock_build:
        client, model = resolve_provider_client("minimax-oauth", "MiniMax-M3")

    assert client is not None, (
        "minimax-oauth must produce a configured client when credentials are "
        "present, but the resolver returned (None, None). The oauth_minimax "
        "arm in _resolve_registry_branch is missing."
    )
    assert isinstance(client, AnthropicAuxiliaryClient), (
        f"minimax-oauth must build an AnthropicAuxiliaryClient (the inference "
        f"endpoint is /anthropic). Got {type(client).__name__}."
    )
    assert client.chat.completions._is_oauth is True
    # The callable token provider — not a string — must reach the Anthropic SDK so
    # the SDK mints a fresh access token per outbound request (MiniMax tokens
    # are short-lived; a static bearer 401s mid-session).
    positional, _kwargs = mock_build.call_args[0], mock_build.call_args[1]
    assert positional[0] is creds["api_key"], (
        "build_anthropic_client must receive the callable token provider, "
        "not a stringified snapshot of the current bearer."
    )
    assert positional[1] == INFERENCE_BASE_URL
    assert model == "MiniMax-M3"


def test_resolve_minimax_oauth_missing_credentials_returns_none_without_raising():
    """AuthError from the runtime resolver → (None, None), no exception.

    The resolver contract is "absent → call_llm's fallback chain", never
    "absent → exception"; the compression step must fall through to its
    Step-2 providers instead of crashing the turn.
    """
    from agent.auxiliary_client import resolve_provider_client

    with patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        side_effect=Exception("not logged in"),
    ):
        client, model = resolve_provider_client("minimax-oauth", "MiniMax-M3")

    assert client is None
    assert model is None
