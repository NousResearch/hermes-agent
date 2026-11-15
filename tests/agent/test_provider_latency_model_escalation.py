"""Unit tests for the latency → model-fallback escalation (DQ #218).

When every provider of the CURRENT model is breaker-demoted for latency, the turn
loop must escalate to the next fallback MODEL (``_try_activate_fallback``) instead of
hammering degraded providers — the breaker alone only reorders providers, so an
all-demoted model keeps stalling (15/10: 5 949 breaker events, turns interrupted
mid-flight).

Coverage: ``all_providers_demoted`` (pinned order + seen-population), the escalation
throttle, and the turn-level guard verdicts (escalate / healthy provider / exhausted
chain).
"""
import time
from types import SimpleNamespace
from unittest import mock

from agent.error_classifier import FailoverReason
from agent.provider_latency_circuit_breaker import (
    all_providers_demoted,
    latency_escalation_due,
    mark_latency_escalation,
    record_provider_failure,
    record_provider_latency,
    reset_circuit_breaker,
)


def _fake_agent(model="z-ai/glm-5.3", order=("Phala", "Baidu", "Novita")):
    return SimpleNamespace(
        model=model,
        provider="openrouter",
        base_url="https://openrouter.example/api/v1",
        providers_order=list(order) if order else None,
        providers_allowed=None,
        providers_ignored=None,
        provider_sort=None,
        provider_require_parameters=None,
        provider_data_collection=None,
        _try_activate_fallback=mock.Mock(return_value=True),
        _buffer_vprint=mock.Mock(),
    )


# ── all_providers_demoted ────────────────────────────────────────────────


def test_all_demoted_with_pinned_order():
    reset_circuit_breaker()
    for p in ("Phala", "Baidu", "Novita"):
        record_provider_failure("z-ai/glm-5.3", p)
    assert all_providers_demoted("z-ai/glm-5.3", ["Phala", "Baidu", "Novita"])


def test_all_demoted_with_tag_pinned_order_matches_served_names():
    reset_circuit_breaker()
    # DQ #500: argon's pinned order uses endpoint tags (google-ai-studio/flex, ...),
    # while the API reports served names ("Google AI Studio", "Google") — both must
    # count as the same population for the escalation verdict.
    order = ["google-ai-studio/flex", "google-ai-studio", "google-vertex"]
    record_provider_failure("google/gemini-4-argon", "Google AI Studio")
    assert not all_providers_demoted("google/gemini-4-argon", order)
    record_provider_failure("google/gemini-4-argon", "Google")
    assert all_providers_demoted("google/gemini-4-argon", order)


def test_not_all_demoted_with_pinned_order_unseen_healthy():
    reset_circuit_breaker()
    # Only two of the three pinned providers have ever served; the third is healthy
    # on paper — a pinned order is a full population, so no escalation.
    record_provider_failure("z-ai/glm-5.3", "Phala")
    record_provider_failure("z-ai/glm-5.3", "Novita")
    assert not all_providers_demoted("z-ai/glm-5.3", ["Phala", "Baidu", "Novita"])


def test_seen_population_requires_two_and_all():
    reset_circuit_breaker()
    record_provider_failure("z-ai/glm-5.3-flash", "Novita")
    assert not all_providers_demoted("z-ai/glm-5.3-flash", None)  # single seen provider
    record_provider_failure("z-ai/glm-5.3-flash", "Relace")
    assert all_providers_demoted("z-ai/glm-5.3-flash", None)


def test_seen_population_with_healthy_history_not_all_demoted():
    reset_circuit_breaker()
    # A provider seen healthy earlier in the session counts towards the population.
    record_provider_latency("z-ai/glm-5.3-flash", "Baidu", 3.0)
    record_provider_failure("z-ai/glm-5.3-flash", "Novita")
    record_provider_failure("z-ai/glm-5.3-flash", "Relace")
    assert not all_providers_demoted("z-ai/glm-5.3-flash", None)


def test_expired_cooldown_is_not_all_demoted():
    reset_circuit_breaker()
    for p in ("Phala", "Baidu"):
        record_provider_failure("z-ai/glm-5.3", p)
    assert all_providers_demoted("z-ai/glm-5.3", ["Phala", "Baidu"])
    with mock.patch("agent.provider_latency_circuit_breaker.time.monotonic",
                    return_value=time.monotonic() + 601.0):
        assert not all_providers_demoted("z-ai/glm-5.3", ["Phala", "Baidu"])


def test_empty_inputs_never_escalate():
    reset_circuit_breaker()
    assert not all_providers_demoted("z-ai/glm-5.3", [])
    assert not all_providers_demoted("z-ai/glm-5.3", None)
    assert not all_providers_demoted("", ["Phala"])


# ── escalation throttle ──────────────────────────────────────────────────


def test_escalation_throttle():
    reset_circuit_breaker()
    assert latency_escalation_due("z-ai/glm-5.3")
    mark_latency_escalation("z-ai/glm-5.3")
    assert not latency_escalation_due("z-ai/glm-5.3")
    with mock.patch("agent.provider_latency_circuit_breaker.time.monotonic",
                    return_value=time.monotonic() + 61.0):
        assert latency_escalation_due("z-ai/glm-5.3")


# ── turn-level guard ─────────────────────────────────────────────────────


def _run_guard(agent, retry=None):
    from agent.turn_api_call import provider_latency_escalation_guard

    _retry = SimpleNamespace(primary_recovery_attempted=False, restart_with_rebuilt_messages=False)
    if retry is not None:
        _retry = retry
    return provider_latency_escalation_guard(
        agent,
        _retry=_retry,
        api_messages=[{"role": "user", "content": "x"}],
        active_system_prompt="sys",
        retry_count=3,
        compression_attempts=2,
    )


def test_guard_escalates_when_all_demoted():
    reset_circuit_breaker()
    for p in ("Phala", "Baidu", "Novita"):
        record_provider_failure("z-ai/glm-5.3", p)
    agent = _fake_agent()
    verdict = _run_guard(agent)
    assert verdict.action == "break"
    agent._try_activate_fallback.assert_called_once_with(reason=FailoverReason.provider_latency)
    assert verdict.retry_count == 0
    assert verdict.compression_attempts == 0
    assert agent._buffer_vprint.called


def test_guard_fallthrough_when_healthy_provider_remains():
    reset_circuit_breaker()
    record_provider_failure("z-ai/glm-5.3", "Phala")
    agent = _fake_agent()
    verdict = _run_guard(agent)
    assert verdict.action == "fallthrough"
    assert verdict.retry_count == 3  # untouched
    agent._try_activate_fallback.assert_not_called()


def test_guard_fallthrough_when_chain_exhausted():
    reset_circuit_breaker()
    for p in ("Phala", "Baidu", "Novita"):
        record_provider_failure("z-ai/glm-5.3", p)
    agent = _fake_agent()
    agent._try_activate_fallback = mock.Mock(return_value=False)
    verdict = _run_guard(agent)
    assert verdict.action == "fallthrough"
    assert verdict.retry_count == 3


def test_guard_throttles_repeated_attempts():
    reset_circuit_breaker()
    for p in ("Phala", "Baidu", "Novita"):
        record_provider_failure("z-ai/glm-5.3", p)
    agent = _fake_agent()
    assert _run_guard(agent).action == "break"
    # Chain exhausted on the follow-up model: next attempt within the window is throttled.
    agent2 = _fake_agent(model="z-ai/glm-5.3")  # same model → throttle window applies
    agent2._try_activate_fallback = mock.Mock(return_value=False)
    verdict = _run_guard(agent2)
    assert verdict.action == "fallthrough"
    agent2._try_activate_fallback.assert_not_called()


def test_guard_never_raises_on_guard_internals_failure():
    reset_circuit_breaker()
    agent = _fake_agent()
    agent.providers_order = object()  # unhashable-ish garbage that breaks order handling
    verdict = _run_guard(agent)
    assert verdict.action == "fallthrough"
