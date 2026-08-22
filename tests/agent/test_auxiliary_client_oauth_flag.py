"""Auxiliary Anthropic clients must derive ``is_oauth`` from the token.

Three ``AnthropicAuxiliaryClient`` construction sites hardcoded
``is_oauth=False``.  On native ``api.anthropic.com`` with a Claude
Pro/Max subscription token that silently suppresses the Claude Code
identity transforms in ``build_anthropic_kwargs``, and Anthropic answers
HTTP 429 ``{"type":"rate_limit_error","message":"Error"}`` on Opus-class
models — an opaque failure that looks like a quota but is not one.

The flag must be True only when BOTH hold:
  1. the token is an Anthropic OAuth token, and
  2. the endpoint is native ``api.anthropic.com``.

Condition (2) preserves the #12846 invariant: no OAuth leak onto
third-party Anthropic-compatible gateways (MiniMax, GLM, Kimi, LiteLLM).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

OAUTH_TOKEN = "sk-ant-oat01-" + "x" * 95
API_KEY = "sk-ant-api03-" + "y" * 95


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


# ── _aux_is_oauth truth table ────────────────────────────────────────


@pytest.mark.parametrize(
    "token,base_url,expected",
    [
        # Native Anthropic + OAuth token → identity transforms must fire.
        (OAUTH_TOKEN, "https://api.anthropic.com", True),
        (OAUTH_TOKEN, "https://api.anthropic.com/v1", True),
        # Native Anthropic + plain API key → not OAuth.
        (API_KEY, "https://api.anthropic.com", False),
        # Third-party Anthropic-compatible gateways must NEVER get the
        # OAuth flag even when the token happens to look like one (#12846).
        (OAUTH_TOKEN, "https://api.minimaxi.com/anthropic", False),
        (OAUTH_TOKEN, "https://open.bigmodel.cn/api/anthropic", False),
        (OAUTH_TOKEN, "https://api.kimi.com/coding", False),
        (OAUTH_TOKEN, "http://127.0.0.1:8080/anthropic", False),
        # Degenerate inputs.
        ("", "https://api.anthropic.com", False),
        (None, "https://api.anthropic.com", False),
        (OAUTH_TOKEN, "", False),
    ],
)
def test_aux_is_oauth_truth_table(token, base_url, expected):
    from agent.auxiliary_client import _aux_is_oauth

    assert _aux_is_oauth(token, base_url) is expected


def test_aux_is_oauth_rejects_callable_token_provider():
    """Entra ID passes a callable bearer provider — never Anthropic OAuth."""
    from agent.auxiliary_client import _aux_is_oauth

    assert _aux_is_oauth(lambda: OAUTH_TOKEN, "https://api.anthropic.com") is False


# ── call site 1: _maybe_wrap_anthropic (the transport chokepoint) ────


def _fake_openai_client():
    client = MagicMock(name="openai_client")
    # Must not look like an already-wrapped adapter.
    return client


@pytest.mark.parametrize(
    "token,base_url,expected",
    [
        (OAUTH_TOKEN, "https://api.anthropic.com", True),
        (API_KEY, "https://api.anthropic.com", False),
        (OAUTH_TOKEN, "https://api.minimaxi.com/anthropic", False),
    ],
)
def test_maybe_wrap_anthropic_propagates_oauth_flag(token, base_url, expected):
    from agent import auxiliary_client as ac

    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=MagicMock(name="anthropic_sdk_client"),
    ):
        wrapped = ac._maybe_wrap_anthropic(
            _fake_openai_client(), "claude-opus-4-5", token, base_url,
        )

    assert isinstance(wrapped, ac.AnthropicAuxiliaryClient)
    assert wrapped.chat.completions._is_oauth is expected


# ── call site 2: custom endpoint with api_mode=anthropic_messages ────


@pytest.mark.parametrize(
    "token,base_url,expected",
    [
        (OAUTH_TOKEN, "https://api.anthropic.com", True),
        (OAUTH_TOKEN, "https://api.minimaxi.com/anthropic", False),
        (API_KEY, "https://api.anthropic.com", False),
    ],
)
def test_custom_anthropic_messages_endpoint_propagates_oauth_flag(
    token, base_url, expected,
):
    from agent import auxiliary_client as ac

    with patch(
        "agent.auxiliary_client._resolve_custom_runtime",
        return_value=(base_url, token, "anthropic_messages"),
    ), patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=MagicMock(name="anthropic_sdk_client"),
    ), patch(
        "agent.auxiliary_client._read_main_model_for_aux",
        return_value="claude-opus-4-5",
    ):
        client, _model = ac._try_custom_endpoint()

    assert isinstance(client, ac.AnthropicAuxiliaryClient)
    assert client.chat.completions._is_oauth is expected


# ── consistency with the already-correct branch ──────────────────────


def test_chokepoint_agrees_with_try_anthropic_on_native_oauth():
    """The two paths must not disagree for the same token + host.

    ``_try_anthropic`` already derives the flag correctly
    (``is_oauth=_is_oauth_token(token)``).  The chokepoint used to
    hardcode False, so identical inputs produced different wire
    behaviour depending on which branch happened to build the client.
    """
    from agent import auxiliary_client as ac

    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=MagicMock(name="anthropic_sdk_client"),
    ), patch(
        "agent.auxiliary_client._select_pool_entry",
        return_value=(False, None),
    ), patch(
        "agent.anthropic_adapter.resolve_anthropic_token",
        return_value=OAUTH_TOKEN,
    ), patch(
        "agent.auxiliary_client._get_aux_model_for_provider",
        return_value="claude-haiku-4-5",
    ):
        via_branch, _ = ac._try_anthropic()
        via_chokepoint = ac._maybe_wrap_anthropic(
            _fake_openai_client(), "claude-haiku-4-5",
            OAUTH_TOKEN, "https://api.anthropic.com",
        )

    assert (
        via_chokepoint.chat.completions._is_oauth
        == via_branch.chat.completions._is_oauth
        is True
    )
