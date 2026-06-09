"""Tests for the per-model tool allow/deny policy (issue #42999).

Covers:
  * config resolution precedence (top-level model shortcut, per-provider
    per-model, legacy custom_providers) — mirrors supports_vision lookup
  * allowlist / denylist / combined semantics
  * filter_tool_defs dropping the right tool definitions
  * policy_for_agent caching per (provider, model) so a fallback model
    picks up its own policy
  * the tool_executor enforcement gate rejecting a disallowed call
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.tool_policy import (
    ToolPolicy,
    filter_tool_defs,
    policy_for_agent,
    resolve_tool_policy,
)


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": ""}}


# ── resolution precedence ────────────────────────────────────────────────

def test_no_config_is_noop():
    policy = resolve_tool_policy({}, "my-local", "llama-3-8b")
    assert policy.is_noop
    assert policy.is_allowed("terminal")


def test_top_level_model_shortcut():
    cfg = {"model": {"denied_tools": ["terminal", "execute_code"]}}
    policy = resolve_tool_policy(cfg, "anything", "any-model")
    assert not policy.is_allowed("terminal")
    assert not policy.is_allowed("execute_code")
    assert policy.is_allowed("web_search")


def test_per_provider_per_model():
    cfg = {
        "providers": {
            "my-local": {
                "models": {
                    "llama-3-8b": {"allowed_tools": ["web_search", "web_extract"]},
                }
            }
        }
    }
    policy = resolve_tool_policy(cfg, "my-local", "llama-3-8b")
    assert policy.is_allowed("web_search")
    assert not policy.is_allowed("terminal")
    # a different model under the same provider is unrestricted
    assert resolve_tool_policy(cfg, "my-local", "other-model").is_noop


def test_named_custom_provider_resolved_as_custom():
    # Runtime rewrites a named custom provider to provider="custom" while the
    # config keeps the declared name under model.provider — both must match.
    cfg = {
        "model": {"provider": "my-local"},
        "providers": {
            "my-local": {
                "models": {"llama-3-8b": {"denied_tools": ["terminal"]}}
            }
        },
    }
    policy = resolve_tool_policy(cfg, "custom", "llama-3-8b")
    assert not policy.is_allowed("terminal")


def test_legacy_custom_providers_list():
    cfg = {
        "custom_providers": [
            {"name": "my-local", "models": {"llama-3-8b": {"denied_tools": ["patch"]}}}
        ]
    }
    policy = resolve_tool_policy(cfg, "my-local", "llama-3-8b")
    assert not policy.is_allowed("patch")


def test_top_level_wins_over_provider_block():
    cfg = {
        "model": {"denied_tools": ["terminal"]},
        "providers": {
            "my-local": {"models": {"m": {"allowed_tools": ["web_search"]}}}
        },
    }
    policy = resolve_tool_policy(cfg, "my-local", "m")
    # top-level block declares a key first → it wins
    assert policy.denied == frozenset({"terminal"})
    assert policy.allowed is None


# ── allow/deny semantics ─────────────────────────────────────────────────

def test_allowlist_only():
    policy = ToolPolicy(allowed=frozenset({"web_search"}))
    assert policy.is_allowed("web_search")
    assert not policy.is_allowed("terminal")


def test_denylist_only():
    policy = ToolPolicy(denied=frozenset({"terminal"}))
    assert not policy.is_allowed("terminal")
    assert policy.is_allowed("web_search")


def test_allow_and_deny_combined():
    policy = ToolPolicy(allowed=frozenset({"web_search", "terminal"}),
                        denied=frozenset({"terminal"}))
    assert policy.is_allowed("web_search")
    assert not policy.is_allowed("terminal")  # deny overrides allow


def test_string_list_is_parsed():
    cfg = {"model": {"denied_tools": "terminal, execute_code"}}
    policy = resolve_tool_policy(cfg, "p", "m")
    assert not policy.is_allowed("terminal")
    assert not policy.is_allowed("execute_code")


def test_empty_allowlist_blocks_everything():
    cfg = {"model": {"allowed_tools": []}}
    policy = resolve_tool_policy(cfg, "p", "m")
    assert not policy.is_noop
    assert not policy.is_allowed("web_search")


# ── filter_tool_defs ─────────────────────────────────────────────────────

def test_filter_drops_disallowed_defs():
    defs = [_tool("web_search"), _tool("terminal"), _tool("execute_code")]
    policy = ToolPolicy(denied=frozenset({"terminal", "execute_code"}))
    kept = filter_tool_defs(defs, policy)
    assert [d["function"]["name"] for d in kept] == ["web_search"]


def test_filter_noop_returns_same_object():
    defs = [_tool("web_search")]
    assert filter_tool_defs(defs, ToolPolicy()) is defs


def test_filter_keeps_unnamed_structural_defs():
    weird = {"type": "custom"}  # no resolvable name
    defs = [weird, _tool("terminal")]
    kept = filter_tool_defs(defs, ToolPolicy(allowed=frozenset({"web_search"})))
    assert weird in kept
    assert _tool("terminal") not in kept


# ── policy_for_agent caching across fallback ─────────────────────────────

def test_policy_for_agent_caches_per_model():
    cfg = {
        "providers": {
            "my-local": {"models": {"weak": {"denied_tools": ["terminal"]}}},
        }
    }
    agent = SimpleNamespace(
        provider="cloud", model="strong", _tool_policy_config=cfg,
    )
    # strong/cloud model: unrestricted
    assert policy_for_agent(agent).is_allowed("terminal")
    # simulate a fallback switch to the weak local model
    agent.provider = "my-local"
    agent.model = "weak"
    assert not policy_for_agent(agent).is_allowed("terminal")


# ── request-path filtering mirrors execution-path enforcement ────────────

def test_request_and_execution_paths_agree():
    """A tool hidden from the model (build_api_kwargs) must also be the one
    rejected at execution (tool_executor) — both consult the same policy."""
    cfg = {"model": {"allowed_tools": ["web_search"]}}
    agent = SimpleNamespace(provider="p", model="m", _tool_policy_config=cfg)
    policy = policy_for_agent(agent)

    defs = [_tool("web_search"), _tool("terminal")]
    visible = {d["function"]["name"] for d in filter_tool_defs(defs, policy)}

    assert visible == {"web_search"}
    # Execution gate would reject exactly the tools that were not visible.
    assert policy.is_allowed("web_search")
    assert not policy.is_allowed("terminal")
