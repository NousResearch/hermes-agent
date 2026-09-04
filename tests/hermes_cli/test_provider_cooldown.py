"""The shared chain walk: which fallback entry takes over a benched primary.

Four callers (gateway, interactive CLI, cron, one-shot) route around a
rate-limited credential pool. The walk lives here so they cannot drift apart
on the same decision, which makes this module worth exercising directly
rather than only through whichever caller happens to be under test.
"""

from __future__ import annotations

import time

from hermes_cli.provider_cooldown import (
    Demotion,
    demote_if_rate_limited,
    resolve_non_cooling_fallback_runtime,
)
from hermes_cli.runtime_provider import CREDENTIALS_COOLING_DOWN_KEY


def _runtime(provider, **extra):
    return {"provider": provider, "api_key": "sk-test-000-not-a-real-key", **extra}


def test_an_entry_with_no_model_is_skipped_not_adopted():
    """Swapping the provider while keeping the primary's model name would send
    e.g. a Gemini model id to OpenRouter -- a 400, not a fallback."""
    chain = [
        {"provider": "openai-codex"},  # no model: unusable as a route
        {"provider": "openrouter", "model": "z-ai/glm-5.2"},
    ]
    seen = []

    def _resolve(entry):
        seen.append(entry.get("provider"))
        return _runtime(entry["provider"])

    runtime, model, entry = resolve_non_cooling_fallback_runtime(
        chain, is_rate_limited=lambda _rt: False, resolve_entry=_resolve
    )

    assert model == "z-ai/glm-5.2"
    assert runtime["provider"] == "openrouter"
    assert entry is chain[1]
    # The model-less entry is refused before it is even resolved.
    assert seen == ["openrouter"]


def test_a_cooling_fallback_is_passed_over_for_a_healthy_one_further_down():
    """Picking a fallback whose own pool is cooling just moves the doomed
    request one hop down the chain."""
    chain = [
        {"provider": "zai", "model": "glm-5.2"},
        {"provider": "openrouter", "model": "z-ai/glm-5.2"},
    ]

    runtime, model, entry = resolve_non_cooling_fallback_runtime(
        chain,
        is_rate_limited=lambda rt: rt["provider"] == "zai",
        resolve_entry=lambda e: _runtime(e["provider"]),
    )

    assert runtime["provider"] == "openrouter"
    assert entry is chain[1]


def test_every_fallback_cooling_still_beats_the_benched_primary():
    """A different quota bucket has a chance the primary provably does not."""
    chain = [{"provider": "zai", "model": "glm-5.2"}]

    runtime, model, _entry = resolve_non_cooling_fallback_runtime(
        chain,
        is_rate_limited=lambda _rt: True,
        resolve_entry=lambda e: _runtime(e["provider"]),
    )

    assert runtime["provider"] == "zai"
    assert model == "glm-5.2"


def test_a_healthy_primary_is_never_demoted():
    result = demote_if_rate_limited(_runtime("gemini"), [{"provider": "x", "model": "y"}])

    assert isinstance(result, Demotion)
    assert result.switched is False
    assert result.model is None
    assert result.cooling_until is None


def test_a_benched_primary_with_an_empty_chain_reports_its_window_and_stays():
    """A cooldown DEMOTES a provider; it does not disqualify it.

    The caller still needs the reset time on the turn it had to stay put --
    that is what a long-lived session's return is scheduled against.
    """
    until = time.time() + 1800
    primary = _runtime("gemini", **{CREDENTIALS_COOLING_DOWN_KEY: until})

    result = demote_if_rate_limited(primary, [])

    assert result.switched is False
    assert result.runtime is primary
    assert result.cooling_until == until
