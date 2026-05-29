from __future__ import annotations

from types import SimpleNamespace

from provider_gateway.config import GatewayConfig
from provider_gateway.policy import ProviderRouteCandidate
from provider_gateway.status import build_gateway_status, format_gateway_status_lines


class FakeTracker:
    def summarize_by_provider(self):
        return [
            {
                "provider": "openrouter",
                "request_count": 3,
                "success_count": 2,
                "error_count": 1,
                "total_tokens": 1200,
                "estimated_cost_usd": 0.012345,
                "avg_latency_ms": 640.5,
            }
        ]


class ErrorTracker:
    """Tracker that raises on summarize — must not break status."""

    def summarize_by_provider(self):
        raise RuntimeError("database locked")


def test_status_disabled_without_usage_is_quiet() -> None:
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(enabled=False),
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[],
        _provider_usage_tracker=None,
    )

    status = build_gateway_status(agent)

    assert status["enabled"] is False
    assert status["usage_summary"] == []
    assert format_gateway_status_lines(status) == []


def test_status_reports_policy_and_last_observed_candidate() -> None:
    last_candidate = ProviderRouteCandidate(
        provider="openrouter",
        model="openai/gpt-5.4",
        source="provider_gateway.routing.fallback_models",
        base_url="https://openrouter.ai/api/v1",
    )
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(
            enabled=True,
            backend="native",
            track_usage=True,
            track_cost=False,
            routing_strategy="lowest-cost",
            fallback_models=["openai/gpt-5.4"],
        ),
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[{"provider": "openai", "model": "gpt-4o"}],
        _provider_gateway_last_route_candidate=last_candidate,
        _provider_usage_tracker=FakeTracker(),
    )

    status = build_gateway_status(agent)

    assert status["enabled"] is True
    assert status["backend"] == "native"
    assert status["routing_strategy"] == "lowest-cost"
    assert status["candidate_count"] == 3
    assert status["last_observed_candidate"] == {
        "provider": "openrouter",
        "model": "openai/gpt-5.4",
        "source": "provider_gateway.routing.fallback_models",
        "base_url": "https://openrouter.ai/api/v1",
    }
    assert status["usage_summary"] == [
        {
            "provider": "openrouter",
            "request_count": 3,
            "success_count": 2,
            "error_count": 1,
            "total_tokens": 1200,
            "estimated_cost_usd": 0.012345,
            "avg_latency_ms": 640.5,
        }
    ]


def test_format_status_lines_are_compact_for_cli_usage() -> None:
    status = {
        "enabled": True,
        "backend": "native",
        "routing_strategy": "round-robin",
        "track_usage": True,
        "track_cost": True,
        "candidate_count": 2,
        "last_observed_candidate": {
            "provider": "openrouter",
            "model": "openai/gpt-5.4",
            "source": "provider_gateway.routing.fallback_models",
            "base_url": "https://openrouter.ai/api/v1",
        },
        "usage_summary": [
            {
                "provider": "openrouter",
                "request_count": 3,
                "success_count": 2,
                "error_count": 1,
                "total_tokens": 1200,
                "estimated_cost_usd": 0.012345,
                "avg_latency_ms": 640.5,
            }
        ],
    }

    assert format_gateway_status_lines(status) == [
        "Provider Gateway: enabled (backend=native, strategy=round-robin, candidates=2)",
        "Tracking: usage=on, cost=on",
        "Next observed: openrouter/openai/gpt-5.4 via provider_gateway.routing.fallback_models",
        "Usage openrouter: 3 req, 2 ok, 1 error, 1,200 tokens, $0.0123, avg 640.5ms",
    ]


# --- New status tests below ---


def test_status_disabled_with_usage_still_shows_output() -> None:
    """Even when disabled, if there's usage data, show it."""
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(enabled=False),
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[],
        _provider_usage_tracker=FakeTracker(),
    )

    status = build_gateway_status(agent)
    lines = format_gateway_status_lines(status)

    assert status["enabled"] is False
    assert len(status["usage_summary"]) == 1
    # Should produce output because usage_summary is non-empty
    assert len(lines) > 0
    assert "disabled" in lines[0]
    assert "Usage openrouter:" in lines[-1]


def test_status_without_last_candidate_omits_next_line() -> None:
    """If no route observation has been made, omit the 'Next observed' line."""
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(enabled=True),
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[],
        _provider_usage_tracker=None,
    )

    status = build_gateway_status(agent)
    lines = format_gateway_status_lines(status)

    assert status["last_observed_candidate"] is None
    # No "Next observed" line should be present
    for line in lines:
        assert "Next observed" not in line


def test_status_handles_tracker_error_gracefully() -> None:
    """If tracker.summarize_by_provider() raises, status should still work."""
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(enabled=True),
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[],
        _provider_usage_tracker=ErrorTracker(),
    )

    status = build_gateway_status(agent)

    assert status["enabled"] is True
    assert status["usage_summary"] == []


def test_format_handles_empty_usage_summary_gracefully() -> None:
    """Format with empty usage summary should not include any Usage lines."""
    status = {
        "enabled": True,
        "backend": "native",
        "routing_strategy": "round-robin",
        "track_usage": True,
        "track_cost": True,
        "candidate_count": 1,
        "last_observed_candidate": None,
        "usage_summary": [],
    }

    lines = format_gateway_status_lines(status)

    assert len(lines) == 2  # Just header + tracking line
    assert "Provider Gateway:" in lines[0]
    assert "Tracking:" in lines[1]


def test_format_handles_malformed_usage_row() -> None:
    """Non-dict entries in usage_summary should be skipped safely."""
    status = {
        "enabled": True,
        "backend": "native",
        "routing_strategy": "round-robin",
        "track_usage": True,
        "track_cost": True,
        "candidate_count": 1,
        "last_observed_candidate": None,
        "usage_summary": ["not_a_dict", 42, None],
    }

    lines = format_gateway_status_lines(status)

    # Should only have header + tracking, no crash
    assert len(lines) == 2
