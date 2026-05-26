"""Integration test for the policy + marker-placement join.

`agent/conversation_loop.py:898` calls `apply_anthropic_cache_control()` only
when `_anthropic_prompt_cache_policy()` returned `(True, native_layout)`.
The 24 + 14 existing unit tests in `tests/run_agent/test_anthropic_prompt_cache_policy.py`
and `tests/agent/test_prompt_caching.py` cover each side in isolation — but
not the join. That join is exactly what the PR-1 refactor replaces with
`profile.cache_strategy_for(model).apply(messages, intent)`.

This test pins the join: for each (provider × model) combo from the policy
matrix, given a known input message list, what does the FULL pipeline
produce? Same input/output contract must hold after the refactor.

When PR-1 lands, this file's `_apply_for_agent_combo()` helper changes one
function call — the asserts on the produced messages stay identical.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from agent.prompt_caching import apply_anthropic_cache_control
from agent.agent_runtime_helpers import anthropic_prompt_cache_policy


def _stub_agent(*, provider, base_url, api_mode, model):
    """Build a minimal AIAgent-shaped object for policy fn calls."""
    agent = MagicMock()
    agent.provider = provider
    agent.base_url = base_url
    agent.api_mode = api_mode
    agent.model = model
    return agent


def _apply_for_agent_combo(messages, *, provider, base_url, api_mode, model, ttl="5m"):
    """Run the policy + marker-placement combination once, mirroring
    `conversation_loop.py:898`. Returns the messages as they would be sent
    to the API (with cache_control markers if the policy said to cache).
    """
    agent = _stub_agent(provider=provider, base_url=base_url, api_mode=api_mode, model=model)
    should_cache, native_layout = anthropic_prompt_cache_policy(agent)
    if should_cache:
        return apply_anthropic_cache_control(
            messages,
            cache_ttl=ttl,
            native_anthropic=native_layout,
        )
    return messages


def _has_native_marker(msg) -> bool:
    """A 'native' cache_control marker lives on the inner content block,
    not on the message envelope."""
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and "cache_control" in item
        for item in content
    )


def _has_envelope_marker(msg) -> bool:
    """An 'envelope' cache_control marker is on the message-level dict.
    NOTE: When the marker is added to a string-content message, the
    string is first wrapped into a list, so the marker can land EITHER
    at the envelope level (tool messages on native layout, empty/None
    content) or on the inner block. For the envelope LAYOUT applied to
    a string-content user msg, the marker lands on the inner wrapped
    block — which collides with _has_native_marker.

    To distinguish layout reliably, check: does the message have
    `cache_control` at the top level? That's only true for envelope
    layout when content was already None / empty / a list."""
    return isinstance(msg.get("cache_control"), dict)


def _count_markers(messages):
    """Total cache_control markers across all messages, both layouts."""
    n = 0
    for m in messages:
        if _has_envelope_marker(m):
            n += 1
        content = m.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "cache_control" in item:
                    n += 1
    return n


# ── Fixtures: the message lists we feed every combo ─────────────────────

@pytest.fixture
def system_plus_three():
    """System prompt + 3 user/assistant exchanges (typical mid-session shape)."""
    return [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second question."},
        {"role": "assistant", "content": "Second answer."},
        {"role": "user", "content": "Third question."},
    ]


@pytest.fixture
def system_plus_five():
    """System + 5 user/asst (verifies 4-breakpoint cap is honored)."""
    return [
        {"role": "system", "content": "You are a helpful agent."},
        *[
            {"role": role, "content": f"Message {i}"}
            for i, role in enumerate(["user", "assistant"] * 5)
        ],
    ]


# ── Parameterized matrix ────────────────────────────────────────────────

# Each row: (provider, base_url, api_mode, model, expects_cache, layout_hint)
# layout_hint: "native" / "envelope" / None (when expects_cache is False)
MATRIX = [
    # Native Anthropic (branch 1)
    ("anthropic", "https://api.anthropic.com", "anthropic_messages", "claude-sonnet-4-6", True, "native"),
    # OpenRouter + Claude (branch 2)
    ("openrouter", "https://openrouter.ai/api/v1", "chat_completions", "anthropic/claude-sonnet-4.6", True, "envelope"),
    # Nous Portal + Claude (branch 3)
    ("nous", "https://inference.nousresearch.com/v1", "chat_completions", "anthropic/claude-sonnet-4.6", True, "envelope"),
    # Nous Portal + Qwen (branch 4)
    ("nous", "https://inference.nousresearch.com/v1", "chat_completions", "qwen3.6-plus", True, "envelope"),
    # Anthropic-wire third-party gateway + Claude (branch 5)
    ("litellm-proxy", "https://litellm.example.com/v1", "anthropic_messages", "claude-sonnet-4-6", True, "native"),
    # MiniMax + own model (branch 6)
    ("minimax", "https://api.minimax.io/anthropic", "anthropic_messages", "MiniMax-M2.7", True, "native"),
    # Alibaba family + Qwen (branch 7)
    ("opencode-go", "https://api.opencode.ai/v1", "chat_completions", "qwen3.6-plus", True, "envelope"),
    # Default-no-cache: OpenAI + GPT-4o (branch 8)
    ("openai", "https://api.openai.com/v1", "chat_completions", "gpt-4o", False, None),
    # Default-no-cache: OpenRouter + non-Claude
    ("openrouter", "https://openrouter.ai/api/v1", "chat_completions", "openai/gpt-4o", False, None),
    # Default-no-cache: OpenCode-Go + non-Qwen
    ("opencode-go", "https://api.opencode.ai/v1", "chat_completions", "kimi-k2", False, None),
]


@pytest.mark.parametrize("provider,base_url,api_mode,model,expects_cache,layout_hint", MATRIX)
def test_policy_plus_apply_matrix(
    system_plus_three, provider, base_url, api_mode, model, expects_cache, layout_hint
):
    """For each known matrix row, the join of policy + apply must produce
    cache markers (or not) consistent with the policy's decision."""
    result = _apply_for_agent_combo(
        system_plus_three,
        provider=provider,
        base_url=base_url,
        api_mode=api_mode,
        model=model,
    )

    marker_count = _count_markers(result)

    if not expects_cache:
        # No caching → no markers anywhere, list equals input length.
        assert marker_count == 0, f"{provider}/{model} should not cache but got {marker_count} markers"
        return

    # Cache path: 4 markers expected (system_and_3 strategy: system + last 3 non-system).
    # The 14 marker-placement tests already verify *which* messages get markers;
    # here we just verify the count survives the policy join.
    assert marker_count == 4, f"{provider}/{model} expected 4 markers, got {marker_count}"


def test_system_plus_five_caps_at_four_breakpoints(system_plus_five):
    """Even with 11 messages, max 4 cache_control markers (system + last 3)."""
    result = _apply_for_agent_combo(
        system_plus_five,
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        model="claude-sonnet-4-6",
    )
    assert _count_markers(result) == 4


def test_input_messages_not_mutated(system_plus_three):
    """The cache application must deep-copy — original messages must not
    grow cache_control fields. Regression guard against shared-state bugs."""
    snapshot = [dict(m) for m in system_plus_three]
    _apply_for_agent_combo(
        system_plus_three,
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        model="claude-sonnet-4-6",
    )
    for orig, snap in zip(system_plus_three, snapshot):
        assert orig == snap, "apply_anthropic_cache_control mutated the input message list"


def test_no_cache_path_passes_through_messages_unchanged(system_plus_three):
    """When the policy says no-cache, the helper returns the messages as-is."""
    result = _apply_for_agent_combo(
        system_plus_three,
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_mode="chat_completions",
        model="gpt-4o",
    )
    # Returns same list (or an equivalent one with no markers).
    assert _count_markers(result) == 0
    assert len(result) == len(system_plus_three)


def test_ttl_propagates_through_policy_apply_join(system_plus_three):
    """The cache_ttl parameter must survive the join — 1h tier must produce
    markers with the ttl field set."""
    result = _apply_for_agent_combo(
        system_plus_three,
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        model="claude-sonnet-4-6",
        ttl="1h",
    )
    # Find any cache_control marker and check its ttl.
    found_1h = False
    for m in result:
        if isinstance(m.get("content"), list):
            for item in m["content"]:
                if isinstance(item, dict) and item.get("cache_control", {}).get("ttl") == "1h":
                    found_1h = True
                    break
    assert found_1h, "1h TTL did not propagate through the policy+apply join"
