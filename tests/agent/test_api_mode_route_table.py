"""Behavior contract for ``_resolve_api_mode``'s rule-table refactor (Sep 2026).

The original 9-branch name-keyed ladder (root AGENTS.md forbids name/kind ladders
>= 4 branches) is now an ordered ``(predicate, applier)`` table. The order anchors
stay explicit in ``_resolve_api_mode``: an actual route pins ``chat_completions``,
an explicit ``api_mode`` wins next, and the host-mandate check is the LAST table
rule so provider-slug rewrites above it always win.

Asserted relationships (never snapshots):
- precedence: actual-route > explicit mode > table rules in order
- provider-slug rules rewrite provider; host auto-detections only fire when
  ``provider_name is None``; ``/anthropic`` suffix does NOT rewrite provider
- Meta wire is URL-driven BY DESIGN (#63425): a provider slug named "meta" with a
  non-api.meta.ai base_url must stay untouched, while api.meta.ai host still mandates
  codex_responses
- Nous dual-wire delegates to ``nous_api_mode`` (model-derived, ``nous.anthropic_wire``
  default "chat" — see providers.py measured note)
"""

import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, "/scratch/build/hermes-pr-apimode")

from agent.agent_init import _API_MODE_ROUTE_RULES, _resolve_api_mode  # noqa: E402


def _mk_agent(provider="", base_url="", model="test-model"):
    return SimpleNamespace(
        provider=provider,
        model=model,
        _base_url_hostname=(base_url.split("//")[-1].split("/")[0] if base_url else ""),
        _base_url_lower=(base_url or "").lower(),
    )


def _resolve(provider, api_mode, provider_name, base_url, model="test-model"):
    agent = _mk_agent(provider, base_url, model)
    _resolve_api_mode(agent, api_mode, provider_name, base_url)
    return agent


# (name, provider, api_mode, provider_name, base_url, want_mode, want_provider)
TABLE_CASES = [
    ("codex provider slug", "openai-codex", "", None,
     "https://chatgpt.com/backend-api/codex", "codex_responses", "openai-codex"),
    ("xai provider slug", "xai", "", None,
     "", "codex_responses", "xai"),
    ("xai-oauth slug", "xai-oauth", "", None,
     "", "codex_responses", "xai-oauth"),
    ("codex host autodetect", "", "", None,
     "https://chatgpt.com/backend-api/codex", "codex_responses", "openai-codex"),
    ("xai host autodetect", "", "", None,
     "https://api.x.ai/v1", "codex_responses", "xai"),
    ("anthropic provider", "anthropic", "", "anthropic",
     "", "anthropic_messages", "anthropic"),
    ("anthropic host autodetect", "", "", None,
     "https://api.anthropic.com/v1", "anthropic_messages", "anthropic"),
    ("anthropic suffix no rewrite", "custom", "", "custom",
     "https://api.minimax.io/v1/anthropic", "anthropic_messages", "custom"),
    ("bedrock provider", "bedrock", "", "bedrock",
     "", "bedrock_converse", "bedrock"),
    ("bedrock host", "", "", None,
     "https://bedrock-runtime.us-east-1.amazonaws.com", "bedrock_converse", ""),
    ("unknown host fallback", "custom", "", "custom",
     "https://api.example.com/v1", "chat_completions", "custom"),
    ("meta host mandate", "", "", None,
     "https://api.meta.ai/v1", "codex_responses", ""),
]

@pytest.mark.parametrize(
    "name,provider,api_mode,provider_name,base_url,want_mode,want_provider", TABLE_CASES,
    ids=[c[0] for c in TABLE_CASES],
)
def test_route_table_cases(name, provider, api_mode, provider_name, base_url, want_mode, want_provider):
    agent = _resolve(provider, api_mode, provider_name, base_url)
    assert agent.api_mode == want_mode
    assert agent.provider == want_provider


def test_actual_route_pins_chat_completions():
    agent = _resolve("custom", "", "custom", "https://openrouter.ai/api/v1")
    assert agent.api_mode == "chat_completions"


def test_explicit_mode_wins_over_host_detection():
    agent = _resolve("custom", "anthropic_messages", "custom", "https://api.anthropic.com/v1")
    assert agent.api_mode == "anthropic_messages"
    assert agent.provider == "custom"  # explicit mode must NOT rewrite the provider


def test_explicit_mode_wins_over_provider_slug():
    agent = _resolve("openai-codex", "chat_completions", "openai-codex", "")
    assert agent.api_mode == "chat_completions"


def test_meta_slug_with_foreign_base_url_is_untouched():
    # URL-driven by design (#63425): slug alone must never mandate codex_responses.
    agent = _resolve("meta", "", "meta", "https://api.example.com/v1")
    assert agent.api_mode == "chat_completions"
    assert agent.provider == "meta"


def test_nous_dual_wire_delegates_to_nous_api_mode():
    from unittest.mock import patch

    from agent.agent_init import _route_rule_nous

    agent = _mk_agent("nous", "")
    with patch("hermes_cli.providers.nous_api_mode", return_value="anthropic_messages") as m:
        _route_rule_nous(agent, "", None, "")
    m.assert_called_once_with(agent.model)
    assert agent.api_mode == "anthropic_messages"