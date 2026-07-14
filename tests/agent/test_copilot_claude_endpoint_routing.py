"""Tests for Copilot + Claude → Anthropic Messages (/v1/messages) endpoint routing.

The GitHub Copilot proxy serves Claude on two endpoints with very different limits:

  * POST /chat/completions  → CLAMPS Claude prompts (~168k, misleading
    "exceeds the limit of 168000" error). The OpenAI-compatible passthrough.
  * POST /v1/messages       → the genuine ~1,000,000-token input window for
    opus/sonnet 4.6 to 4.8, via the Anthropic Messages API.

Hermes' api_mode decision tree (agent_init) had no Copilot+Claude branch, so a
Claude model on provider=copilot fell through to ``chat_completions`` and hit the
168k clamp. These tests lock in the routing fix across all the surfaces:

  1. The agent_init override predicate routes copilot+claude → anthropic_messages
     (even over an explicit ``api_mode: chat_completions``), while leaving
     copilot+gpt, copilot-acp, and native-anthropic untouched.
  2. The debug request-dump URL label reflects the real endpoint per api_mode.
  3. The runtime wrong-route tripwire refuses to persist a sub-200k context
     limit for copilot+claude (the /chat/completions clamp signature).
"""

import pytest

from utils import base_url_host_matches


# ─────────────────────────────────────────────────────────────────────────────
# 1. agent_init api_mode routing override
#
# The override block lives inline in agent.agent_init.create_agent (right after
# the api_mode decision tree). We replicate its exact predicate here so the test
# is hermetic (no agent construction / network). If the predicate in agent_init
# changes, update this helper to match. The cases below encode the REQUIRED
# behavior.
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_api_mode(provider, model, base_url, api_mode_cfg=None):
    prov = (provider or "").lower()
    base_lower = (base_url or "").lower()
    if api_mode_cfg in {
        "chat_completions", "codex_responses", "anthropic_messages",
        "bedrock_converse", "codex_app_server",
    }:
        api_mode = api_mode_cfg
    elif prov == "anthropic":
        api_mode = "anthropic_messages"
    else:
        api_mode = "chat_completions"

    # Copilot+Claude override (mirror of agent_init.py).
    model_lower = (model or "").lower()
    is_copilot_native = (
        prov in {"copilot", "github-copilot"}
        or (
            prov not in {"copilot-acp"}
            and base_url_host_matches(base_lower, "api.githubcopilot.com")
        )
    )
    if (
        prov != "copilot-acp"
        and is_copilot_native
        and "claude" in model_lower
        and api_mode != "anthropic_messages"
    ):
        api_mode = "anthropic_messages"
    return api_mode


@pytest.mark.parametrize(
    "provider,model,base_url,cfg,expected",
    [
        # Copilot + Claude → anthropic_messages, even when config forces chat_completions.
        ("copilot", "claude-opus-4.8", "https://api.githubcopilot.com",
         "chat_completions", "anthropic_messages"),
        ("github-copilot", "claude-sonnet-4.8", "https://api.githubcopilot.com",
         None, "anthropic_messages"),
        # base-url-only detection (provider unset) still routes Claude.
        ("", "claude-opus-4.8", "https://api.githubcopilot.com",
         None, "anthropic_messages"),
        # Copilot + GPT stays on chat_completions (NOT forced).
        ("copilot", "gpt-5.5", "https://api.githubcopilot.com",
         None, "chat_completions"),
        # copilot-acp is excluded (ACP CLI does its own routing).
        ("copilot-acp", "claude-opus-4.8", "https://api.githubcopilot.com",
         None, "chat_completions"),
        # Native anthropic provider is untouched (already anthropic_messages).
        ("anthropic", "claude-opus-4.8", "https://api.anthropic.com",
         None, "anthropic_messages"),
    ],
)
def test_copilot_claude_routes_to_anthropic_messages(provider, model, base_url, cfg, expected):
    assert _resolve_api_mode(provider, model, base_url, cfg) == expected


# ─────────────────────────────────────────────────────────────────────────────
# 2. Debug request-dump URL label reflects the real endpoint (REAL function)
# ─────────────────────────────────────────────────────────────────────────────

def test_api_mode_endpoint_suffix_maps_each_mode():
    from agent.agent_runtime_helpers import _api_mode_endpoint_suffix

    assert _api_mode_endpoint_suffix("anthropic_messages") == "/v1/messages"
    assert _api_mode_endpoint_suffix("codex_responses") == "/responses"
    assert _api_mode_endpoint_suffix("chat_completions") == "/chat/completions"
    assert _api_mode_endpoint_suffix("bedrock_converse") == "/chat/completions"
    assert _api_mode_endpoint_suffix(None) == "/chat/completions"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Runtime wrong-route tripwire (REAL function)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "provider,base_url,model,new_ctx,expected",
    [
        # The misroute signature: copilot + claude + sub-200k clamp.
        ("copilot", "", "claude-opus-4.8", 168000, True),
        ("github-copilot", "", "claude-sonnet-4.8", 200000, True),
        # base-url-only copilot detection.
        ("", "https://api.githubcopilot.com", "claude-opus-4.8", 168000, True),
        # NOT a misroute: real /v1/messages 1M ceiling.
        ("copilot", "", "claude-opus-4.8", 1000000, False),
        # NOT a misroute: gpt on copilot (no claude).
        ("copilot", "", "gpt-5.5", 168000, False),
        # NOT a misroute: claude on a non-copilot provider.
        ("anthropic", "https://api.anthropic.com", "claude-opus-4.8", 168000, False),
    ],
)
def test_detect_copilot_claude_wrong_route(provider, base_url, model, new_ctx, expected):
    from agent.conversation_loop import _detect_copilot_claude_wrong_route

    assert _detect_copilot_claude_wrong_route(
        provider=provider, base_url=base_url, model=model, new_ctx=new_ctx
    ) is expected
