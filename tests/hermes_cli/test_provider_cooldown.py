"""The shared chain walk: which fallback entry takes over a benched primary.

Three callers (gateway, cron, one-shot) route around a rate-limited
credential pool. The walk lives here so they cannot drift apart on the same
decision, which makes this module worth exercising directly rather than only
through whichever caller happens to be under test.
"""

from __future__ import annotations

import logging
import time

from hermes_cli.provider_cooldown import (
    Demotion,
    demote_if_rate_limited,
    resolve_non_cooling_fallback_runtime,
    runtime_is_rate_limited,
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

    runtime, model = resolve_non_cooling_fallback_runtime(
        chain, is_rate_limited=lambda _rt: False, resolve_entry=_resolve
    )

    assert model == "z-ai/glm-5.2"
    assert runtime["provider"] == "openrouter"
    # The model-less entry is refused before it is even resolved.
    assert seen == ["openrouter"]


def test_a_cooling_fallback_is_passed_over_for_a_healthy_one_further_down():
    """Picking a fallback whose own pool is cooling just moves the doomed
    request one hop down the chain."""
    chain = [
        {"provider": "zai", "model": "glm-5.2"},
        {"provider": "openrouter", "model": "z-ai/glm-5.2"},
    ]

    runtime, model = resolve_non_cooling_fallback_runtime(
        chain,
        is_rate_limited=lambda rt: rt["provider"] == "zai",
        resolve_entry=lambda e: _runtime(e["provider"]),
    )

    assert runtime["provider"] == "openrouter"
    assert model == "z-ai/glm-5.2"


def test_every_fallback_cooling_still_beats_the_benched_primary():
    """A different quota bucket has a chance the primary provably does not."""
    chain = [{"provider": "zai", "model": "glm-5.2"}]

    runtime, model = resolve_non_cooling_fallback_runtime(
        chain,
        is_rate_limited=lambda _rt: True,
        resolve_entry=lambda e: _runtime(e["provider"]),
    )

    assert runtime["provider"] == "zai"
    assert model == "glm-5.2"


def test_a_healthy_primary_is_never_demoted():
    chain_reads = []

    def _chain():
        chain_reads.append(1)
        return [{"provider": "x", "model": "y"}]

    result = demote_if_rate_limited(_runtime("gemini"), _chain)

    assert isinstance(result, Demotion)
    assert result.switched is False
    assert result.model is None
    # The healthy path must not pay for a config read it cannot use.
    assert chain_reads == []


def test_a_benched_primary_with_an_empty_chain_keeps_running_on_it():
    """A cooldown DEMOTES a provider; it does not disqualify it.

    With nothing to hand over to, the real upstream 429 beats refusing to run.
    """
    until = time.time() + 1800
    primary = _runtime("gemini", **{CREDENTIALS_COOLING_DOWN_KEY: until})

    result = demote_if_rate_limited(primary, lambda: [])

    assert result.switched is False
    assert result.runtime is primary
    assert result.model is None


def test_a_broken_probe_says_so_instead_of_reporting_health(monkeypatch, caplog):
    """The failure this module actually shipped: a name the probe called was no
    longer imported, the blanket handler swallowed the NameError, and every
    benched pool read as healthy -- invisible until the tests ran, because the
    only trace was a debug-level line.

    Failing open stays right (a cooldown must never take a turn down), so the
    answer is still False. What is pinned here is that it is no longer silent:
    a defect in this code reports itself as one.
    """
    import hermes_cli.runtime_provider as rp

    def _broken(_runtime):
        raise NameError("name 'get_custom_provider_pool_key' is not defined")

    monkeypatch.setattr(rp, "runtime_credentials_cooling_down_until", _broken)

    with caplog.at_level(logging.ERROR, logger="hermes_cli.provider_cooldown"):
        assert runtime_is_rate_limited(_runtime("gemini")) is False

    assert [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "a probe broken by a code defect must not be swallowed at debug level"
    )
