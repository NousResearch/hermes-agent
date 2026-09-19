"""Regression: fallback chain must skip a candidate whose declared context window is
smaller than the current session's estimated prompt tokens.

Real user report (Copilot GPT-4.1, 64k) — Anthropic and Codex both rate-limited, Hermes
fell through to Copilot with ~125k tokens of context and got

    400 prompt token count of 125516 exceeds the limit of 64000

which wedged the retry state. The fix is a pre-dispatch skip in
``_should_skip_fallback_candidate`` when the entry's declared context length is smaller
than the current prompt-token estimate (with a small safety margin).

These tests probe ONLY the pure decision function; the surrounding activation loop is
covered by the existing fallback tests. Behaviour contracts, not snapshots.
"""
from __future__ import annotations

import types

import pytest

from agent import chat_completion_helpers as cch


def _fb_entry(**kw):
    return {"provider": kw.pop("provider", "copilot"),
            "model": kw.pop("model", "gpt-4.1"),
            "base_url": kw.pop("base_url", "https://api.githubcopilot.com"),
            **kw}


def _make_agent(prompt_tokens: int):
    """Minimal duck-typed agent: our estimator reads ``last_prompt_tokens`` off the
    compressor when no usage anchor is present, which is the realistic pre-first-turn
    or provider-failed path.
    """
    agent = types.SimpleNamespace()
    agent.provider = "anthropic"
    agent.model = "claude-opus-4-7"
    agent.base_url = "https://api.anthropic.com"
    agent.messages = []
    agent._turn_base_usage_anchor = None
    agent._usage_anchor = None
    agent.context_compressor = types.SimpleNamespace(last_prompt_tokens=prompt_tokens)
    return agent


def test_declared_context_from_entry():
    assert cch._fallback_entry_declared_context_length({"context_length": 64000}) == 64000


def test_declared_context_missing_returns_zero():
    assert cch._fallback_entry_declared_context_length({}) == 0


@pytest.mark.parametrize("bad", [0, -1, "64000", True, False, None])
def test_declared_context_rejects_non_positive_int(bad):
    # bool is a subclass of int in Python — explicitly rejected so entries with a stray
    # ``context_length: true`` (yaml quirk) don't get treated as size 1.
    entry = {"context_length": bad} if bad is not None else {}
    got = cch._fallback_entry_declared_context_length(entry)
    assert isinstance(got, int) and got == 0


def test_current_prompt_token_estimate_reads_compressor_measurement():
    agent = _make_agent(prompt_tokens=125_000)
    assert cch._current_prompt_token_estimate(agent) == 125_000


def test_current_prompt_token_estimate_never_raises_on_garbage_agent():
    # Estimation must be side-effect-free and never block a real fallback path.
    garbage = types.SimpleNamespace()
    assert cch._current_prompt_token_estimate(garbage) == 0


def test_skip_when_declared_context_smaller_than_estimated_prompt(monkeypatch):
    # 125k prompt vs a 64k-context fallback (the real Copilot GPT-4.1 case).
    agent = _make_agent(prompt_tokens=125_000)
    fb = _fb_entry(context_length=64_000)
    unavailable: set = set()
    assert cch._should_skip_fallback_candidate(
        agent, fb, ("copilot", "gpt-4.1"), "copilot", "gpt-4.1", unavailable,
    ) is True
    # A too-small context is a per-turn skip, NOT permanent. If a later turn is smaller
    # (post-compression) the same entry must be reachable again.
    assert ("copilot", "gpt-4.1") not in unavailable


def test_do_not_skip_when_declared_context_comfortably_larger():
    agent = _make_agent(prompt_tokens=125_000)
    fb = _fb_entry(model="gpt-5.5", base_url="https://chatgpt.com/backend-api/codex",
                   provider="openai-codex", context_length=272_000)
    assert cch._should_skip_fallback_candidate(
        agent, fb, ("openai-codex", "gpt-5.5"), "openai-codex", "gpt-5.5", set(),
    ) is False


def test_do_not_skip_when_declared_context_unknown():
    # Entry didn't declare context_length AND cache miss → the caller must NOT gate on
    # this signal (undecidable). Leaves the eventual 400 as the fallback for edge cases.
    agent = _make_agent(prompt_tokens=125_000)
    fb = {"provider": "openrouter", "model": "some-model",
          "base_url": "https://openrouter.ai/api/v1"}
    assert cch._should_skip_fallback_candidate(
        agent, fb, ("openrouter", "some-model"), "openrouter", "some-model", set(),
    ) is False


def test_safety_margin_skips_just_over_threshold():
    # 61k prompt vs 64k context — 61000 * 1.05 = 64050 > 64000, so we skip.
    agent = _make_agent(prompt_tokens=61_000)
    fb = _fb_entry(context_length=64_000)
    assert cch._should_skip_fallback_candidate(
        agent, fb, ("copilot", "gpt-4.1"), "copilot", "gpt-4.1", set(),
    ) is True


def test_safety_margin_allows_comfortable_fit():
    # 50k prompt vs 64k context — well below the 5% margin, entry should be usable.
    agent = _make_agent(prompt_tokens=50_000)
    fb = _fb_entry(context_length=64_000)
    assert cch._should_skip_fallback_candidate(
        agent, fb, ("copilot", "gpt-4.1"), "copilot", "gpt-4.1", set(),
    ) is False
