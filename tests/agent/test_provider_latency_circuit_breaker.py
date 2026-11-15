"""Unit tests for agent/provider_latency_circuit_breaker.py."""
from agent.provider_latency_circuit_breaker import (
    adjust_provider_order,
    is_provider_demoted,
    record_provider_failure,
    record_provider_latency,
    reset_circuit_breaker,
)


def test_provider_latency_circuit_breaker_healthy():
    reset_circuit_breaker()
    # Fast requests do not trigger demotion
    record_provider_latency("z-ai/glm-5.3-flash", "Novita", 3.5)
    record_provider_latency("z-ai/glm-5.3-flash", "Novita", 4.0)
    assert not is_provider_demoted("z-ai/glm-5.3-flash", "Novita")
    assert adjust_provider_order("z-ai/glm-5.3-flash", ["Novita", "Relace"]) == ["Novita", "Relace"]


def test_provider_latency_circuit_breaker_demotes_degraded():
    reset_circuit_breaker()
    # Slow requests trigger demotion
    record_provider_latency("z-ai/glm-5.3-flash", "Novita", 35.0)
    record_provider_latency("z-ai/glm-5.3-flash", "Novita", 45.0)
    assert is_provider_demoted("z-ai/glm-5.3-flash", "Novita")

    # Demoted provider gets moved to the end of the order
    order = ["Novita", "Relace", "Phala"]
    assert adjust_provider_order("z-ai/glm-5.3-flash", order) == ["Relace", "Phala", "Novita"]


def test_provider_latency_circuit_breaker_record_failure():
    reset_circuit_breaker()
    record_provider_failure("z-ai/glm-5.3-flash", "Novita")
    assert is_provider_demoted("z-ai/glm-5.3-flash", "Novita")
    order = ["Novita", "Relace"]
    assert adjust_provider_order("z-ai/glm-5.3-flash", order) == ["Relace", "Novita"]


def test_provider_latency_circuit_breaker_case_insensitive():
    reset_circuit_breaker()
    record_provider_latency("Z-AI/GLM-5.3-FLASH", "novita", 30.0)
    record_provider_latency("z-ai/glm-5.3-flash", "NOVITA", 40.0)
    assert is_provider_demoted("z-ai/glm-5.3-flash", "Novita")
    assert adjust_provider_order("z-ai/glm-5.3-flash", ["Novita", "Relace"]) == ["Relace", "Novita"]


def test_provider_latency_circuit_breaker_suppresses_repeat_warnings_while_demoted(caplog):
    """Once a (model, provider) pair enters active cooldown, subsequent slow calls
    refresh the cooldown without re-emitting WARNING on every turn (prevents 70+
    warnings/10m on single-provider reasoning models like google/gemini-4-argon @ Google
    from tripping provider-latency-storm-watchdog).

    DQ #500: argon's base threshold is 25.0s (20.0s × 1.25 model multiplier, no
    context tokens) — the flapping boundary below is scaled accordingly.
    """
    import logging

    reset_circuit_breaker()
    with caplog.at_level(logging.WARNING, logger="agent.provider_latency_circuit_breaker"):
        record_provider_latency("google/gemini-4-argon", "Google", 31.0)
        record_provider_latency("google/gemini-4-argon", "Google", 35.0)  # transitions to demoted -> 1st WARNING
        record_provider_latency("google/gemini-4-argon", "Google", 39.0)  # already demoted -> must NOT warn again
        record_provider_latency("google/gemini-4-argon", "Google", 30.0)  # already demoted -> must NOT warn again
        # Flapping across the 25.0s boundary within the same 600s window must also NOT re-warn:
        record_provider_latency("google/gemini-4-argon", "Google", 12.0)
        record_provider_latency("google/gemini-4-argon", "Google", 12.0)
        record_provider_latency("google/gemini-4-argon", "Google", 12.0)  # avg drops below 25s (early recovery)
        record_provider_latency("google/gemini-4-argon", "Google", 46.0)
        record_provider_latency("google/gemini-4-argon", "Google", 46.0)  # avg back to >= 25s again within 600s

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"Expected exactly 1 transition warning while in 600s window, got {len(warnings)}: {warnings}"
    assert is_provider_demoted("google/gemini-4-argon", "Google")


def test_context_aware_threshold_keeps_healthy_long_context_calls():
    reset_circuit_breaker()
    # DQ #500 (census 21/10): argon at 100-200k input tokens has healthy p90 41.8s —
    # the flat 20s base demoted that physics. Threshold at 150k ctx =
    # (20 + 0.25*150) * 1.25 = 71.9s.
    record_provider_latency("google/gemini-4-argon", "Google AI Studio", 55.0, context_tokens=150_000)
    record_provider_latency("google/gemini-4-argon", "Google AI Studio", 60.0, context_tokens=150_000)
    assert not is_provider_demoted("google/gemini-4-argon", "Google AI Studio")
    # A genuinely stalled provider still demotes at the scaled threshold.
    record_provider_latency("google/gemini-4-argon", "Google AI Studio", 85.0, context_tokens=150_000)
    record_provider_latency("google/gemini-4-argon", "Google AI Studio", 95.0, context_tokens=150_000)
    assert is_provider_demoted("google/gemini-4-argon", "Google AI Studio")


def test_served_name_and_order_tag_share_breaker_key():
    reset_circuit_breaker()
    # DQ #500 del 2b: OpenRouter serves both google-ai-studio tags under the name
    # "Google AI Studio" and Vertex under "Google"; the pinned order uses tags.
    # Demotion recorded under a served name must demote the tag entries and vice versa.
    record_provider_latency("google/gemini-4-argon", "Google AI Studio", 35.0)
    record_provider_latency("google/gemini-4-argon", "Google AI Studio", 40.0)
    assert is_provider_demoted("google/gemini-4-argon", "google-ai-studio/flex")
    assert is_provider_demoted("google/gemini-4-argon", "google-ai-studio")
    assert not is_provider_demoted("google/gemini-4-argon", "google-vertex")
    order = ["google-ai-studio/flex", "google-ai-studio", "google-vertex"]
    assert adjust_provider_order("google/gemini-4-argon", order) == [
        "google-vertex", "google-ai-studio/flex", "google-ai-studio"]

    # The same holds for Vertex's served name "Google".
    record_provider_failure("google/gemini-4-argon", "Google")
    assert is_provider_demoted("google/gemini-4-argon", "google-vertex")


def test_explicit_failure_demotion_is_not_cleared_early_by_inflight_latency_sample():
    reset_circuit_breaker()
    try:
        record_provider_failure("google/gemini-4-argon", "Google AI Studio")
        assert is_provider_demoted("google/gemini-4-argon", "google-ai-studio/flex")
        # An already-in-flight call completing fast right after a 429 must NOT wipe the failure cooldown.
        record_provider_latency("google/gemini-4-argon", "Google AI Studio", 12.0, context_tokens=150_000)
        assert is_provider_demoted("google/gemini-4-argon", "google-ai-studio/flex")
        assert is_provider_demoted("google/gemini-4-argon", "google-ai-studio")
    finally:
        reset_circuit_breaker()


def test_mixed_context_window_uses_rolling_threshold_mean():
    """CF-73 / DQ #1991: When large-context calls (165k-311k tokens) are followed by a
    small-context call (36.8k tokens, 9.4s), the rolling average duration (37.02s) must
    be compared against the rolling average threshold of the 5-call window (92.50s),
    not the final small-context call's threshold (36.5s).
    """
    reset_circuit_breaker()
    try:
        samples = [
            (12.8, 277_928),
            (22.5, 165_259),
            (116.4, 311_630),
            (24.0, 288_398),
            (9.4, 36_769),
        ]
        for dur, ctx in samples:
            record_provider_latency("google/gemini-4-argon", "Google AI Studio", dur, context_tokens=ctx)
        assert not is_provider_demoted("google/gemini-4-argon", "Google AI Studio")
    finally:
        reset_circuit_breaker()



