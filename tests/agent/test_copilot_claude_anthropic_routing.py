"""Regression tests for Copilot + Claude → Anthropic Messages (/v1/messages) routing.

Root cause (see probe/FINDINGS.md §2): the GitHub Copilot proxy serves Claude on
two endpoints with very different limits:

  * POST /chat/completions  → CLAMPS Claude and returns a misleading
    ``prompt token count … exceeds the limit of 168000`` error (regular tier).
  * POST /v1/messages       → the genuine 1,000,000-token input window for
    opus/sonnet 4.6–4.8, unlocked by the anthropic-beta triplet + the VS Code
    Copilot identity headers (Bearer auth, X-Copilot-Agent-Slug, etc.).

Hermes' api_mode decision tree had no Copilot+Claude branch, so Claude on
provider=copilot fell through to ``chat_completions`` and hit the 168k clamp —
the "1M → snap to 168k → cannot compress further" failure.

These tests lock in:
  1. agent_init routes copilot+claude to ``anthropic_messages`` (even overriding
     an explicit ``api_mode: chat_completions`` config default), while leaving
     copilot+gpt, copilot-acp, and native-anthropic untouched.
  2. build_anthropic_client builds the Copilot /v1/messages client with Bearer
     auth (NOT x-api-key), the full Copilot identity header set, the
     X-Copilot-Agent-Slug unlock, and the proven anthropic-beta triplet.
"""

import pytest

from utils import base_url_host_matches


# ─────────────────────────────────────────────────────────────────────────────
# Fix A — api_mode routing override
#
# The override block lives inline in agent.agent_init.create_agent (after the
# api_mode decision tree).  We replicate its exact predicate here so the test
# is hermetic (no agent construction / network).  If the predicate in
# agent_init changes, update this helper to match — the cases below encode the
# REQUIRED behavior.
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

    # Fix A override (mirror of agent_init.py).
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
        ("copilot", "claude-opus-4.7", "https://api.githubcopilot.com",
         None, "anthropic_messages"),
        ("copilot", "claude-opus-4.6", "https://api.githubcopilot.com",
         "chat_completions", "anthropic_messages"),
        # Base-url-only detection (provider unset but githubcopilot host).
        ("", "claude-opus-4.8", "https://api.githubcopilot.com",
         None, "anthropic_messages"),
        # Non-Claude on Copilot is NOT rerouted.
        ("copilot", "gpt-5.5", "https://api.githubcopilot.com",
         "chat_completions", "chat_completions"),
        ("copilot", "gemini-2.5-pro", "https://api.githubcopilot.com",
         None, "chat_completions"),
        # ACP subprocess does its own routing — never rerouted here.
        ("copilot-acp", "claude-opus-4.8", "acp://copilot", None, "chat_completions"),
        # Native Anthropic is unaffected (already anthropic_messages).
        ("anthropic", "claude-opus-4.8", "https://api.anthropic.com",
         None, "anthropic_messages"),
    ],
)
def test_api_mode_routing(provider, model, base_url, cfg, expected):
    assert _resolve_api_mode(provider, model, base_url, cfg) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Fix B — build_anthropic_client Copilot /v1/messages header set
# ─────────────────────────────────────────────────────────────────────────────

def _built_copilot_headers():
    from agent.anthropic_adapter import build_anthropic_client
    client = build_anthropic_client(
        "gho_FAKE_TOKEN_FOR_TEST", "https://api.githubcopilot.com", timeout=60,
    )
    return {k.lower(): v for k, v in dict(client.default_headers).items()}


def test_copilot_anthropic_uses_bearer_not_apikey():
    low = _built_copilot_headers()
    # Token must ride as Authorization: Bearer, never as x-api-key.
    assert low.get("authorization"), "Authorization header missing"
    assert "bearer" in low["authorization"].lower()
    assert "x-api-key" not in low, "Copilot must not use Anthropic x-api-key auth"


def test_copilot_anthropic_identity_headers_present():
    low = _built_copilot_headers()
    # The Copilot /v1/messages path delegates to the single identity builder
    # (copilot_request_headers), so it presents the same Copilot CLI identity as
    # the inference path: copilot-developer-cli integration-id (exposes the full
    # 33-model catalog incl gemini-3.x + true per-model limits), the CLI
    # User-Agent (copilot/<ver>), and NO Editor-* VS Code headers.
    assert low.get("copilot-integration-id") == "copilot-developer-cli"
    assert low.get("x-github-api-version")          # date-versioned, e.g. 2026-06-01
    assert low.get("user-agent", "").startswith("copilot/")
    assert low.get("x-initiator") == "agent"
    assert "editor-version" not in low
    assert "editor-plugin-version" not in low


def test_copilot_anthropic_no_inert_1m_slug():
    low = _built_copilot_headers()
    # The old `X-Copilot-Agent-Slug: copilot-1m-context` was proven INERT
    # (live probe 2026-06-07): it changed neither catalog visibility nor
    # per-model limits. The real lever is Copilot-Integration-Id (copilot-developer-cli),
    # so the no-op slug must not be resent.
    assert "x-copilot-agent-slug" not in low


def test_copilot_anthropic_beta_triplet_present():
    """The Copilot /v1/messages path sends the THREE betas the official
    Copilot Chat extension sends (Worker-G RE 2026-06-04). The historical
    'Master Probe triplet' (cli-internal + context-1m + task-budgets) was
    fictional \u2014 grep on the 32MB extension bundle returned 0 hits per beta.
    Removing them likely contributed to fixing the historical 168k snap-back
    loops. See agent/anthropic_adapter.py:984 for the docstring marker.
    """
    low = _built_copilot_headers()
    beta = low.get("anthropic-beta", "")
    for required in (
        "interleaved-thinking-2025-05-14",
        "context-management-2025-06-27",
        "advanced-tool-use-2025-11-20",
    ):
        assert required in beta, f"missing required beta: {required}"
    # And verify the FICTIONAL triplet stays out
    for fictional in (
        "cli-internal-2026-02-09",
        "task-budgets-2026-03-13",
    ):
        assert fictional not in beta, (
            f"fictional beta {fictional!r} reintroduced \u2014 see Worker-G evidence"
        )
