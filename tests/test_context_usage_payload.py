"""Coverage for the native Hermes context_usage telemetry payload.

The helper in run_agent.py turns a CanonicalUsage from a successful API call
into the `hermes_context_usage` block mc-api consumes for MC Live. The bar
must reflect the full provider-reported prefill (input + cache_read +
cache_creation), not just `input_tokens`, or long cached sessions show as
falsely small.
"""

from dataclasses import dataclass


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


def test_payload_sums_full_prefill_including_cache(monkeypatch):
    monkeypatch.setenv("HERMES_CONTEXT_LIMIT", "400000")
    monkeypatch.setenv("HERMES_CONTEXT_FLOOR", "320000")
    from run_agent import _build_hermes_context_usage_payload

    usage = _Usage(
        input_tokens=1000,
        output_tokens=5000,
        cache_read_tokens=120000,
        cache_write_tokens=2000,
    )
    out = _build_hermes_context_usage_payload(canonical_usage=usage, model="claude-x")

    assert out["tokens"] == 1000 + 120000 + 2000
    assert out["estimated"] is False
    assert out["found"] is True
    assert out["source"] == "provider_usage"
    assert out["model"] == "claude-x"
    assert out["limit"] == 400_000
    assert out["floor"] == 320_000
    assert out["status"] == "green"
    assert out["pct"] > 0
    comp = out["components"]
    assert comp["input_tokens"] == 1000
    assert comp["cache_read_input_tokens"] == 120000
    assert comp["cache_creation_input_tokens"] == 2000
    assert comp["output_tokens"] == 5000
    assert comp["total_context_tokens"] == out["tokens"]


def test_payload_excludes_output_tokens_from_total(monkeypatch):
    monkeypatch.setenv("HERMES_CONTEXT_LIMIT", "400000")
    monkeypatch.setenv("HERMES_CONTEXT_FLOOR", "320000")
    from run_agent import _build_hermes_context_usage_payload

    usage = _Usage(input_tokens=10, output_tokens=999_999, cache_read_tokens=0, cache_write_tokens=0)
    out = _build_hermes_context_usage_payload(canonical_usage=usage, model="m")

    assert out["tokens"] == 10
    assert out["components"]["output_tokens"] == 999_999


def test_payload_status_thresholds(monkeypatch):
    monkeypatch.setenv("HERMES_CONTEXT_LIMIT", "400000")
    monkeypatch.setenv("HERMES_CONTEXT_FLOOR", "320000")
    from run_agent import _build_hermes_context_usage_payload

    # Green: under floor*0.5 = 160_000
    out = _build_hermes_context_usage_payload(
        canonical_usage=_Usage(input_tokens=100_000), model="m")
    assert out["status"] == "green"

    # Yellow: between 160_000 and 320_000
    out = _build_hermes_context_usage_payload(
        canonical_usage=_Usage(input_tokens=200_000), model="m")
    assert out["status"] == "yellow"

    # Red: at or above floor
    out = _build_hermes_context_usage_payload(
        canonical_usage=_Usage(input_tokens=320_000), model="m")
    assert out["status"] == "red"


def test_payload_response_shape_matches_frontend_contract(monkeypatch):
    """Top-level keys must satisfy MC's hermes_context_usage TS interface."""
    from run_agent import _build_hermes_context_usage_payload

    out = _build_hermes_context_usage_payload(
        canonical_usage=_Usage(input_tokens=1, cache_read_tokens=2, cache_write_tokens=3),
        model="claude-opus-4-7",
    )
    required = {"tokens", "limit", "floor", "remaining", "pct",
                "floor_pct", "status", "found", "estimated"}
    assert required.issubset(out.keys())
    assert out["status"] in ("green", "yellow", "red")


def test_payload_handles_zero_cache_fields(monkeypatch):
    """Models that don't report cache (e.g. local Ollama) still produce a valid payload."""
    monkeypatch.setenv("HERMES_CONTEXT_LIMIT", "200000")
    monkeypatch.setenv("HERMES_CONTEXT_FLOOR", "100000")
    from run_agent import _build_hermes_context_usage_payload

    usage = _Usage(input_tokens=42, output_tokens=10)  # cache_read/write default to 0
    out = _build_hermes_context_usage_payload(canonical_usage=usage, model="gemma:26b")

    assert out["tokens"] == 42
    assert out["components"]["cache_read_input_tokens"] == 0
    assert out["components"]["cache_creation_input_tokens"] == 0
    assert out["limit"] == 200_000
    assert out["floor"] == 100_000
