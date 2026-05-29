import pytest
from provider_gateway.circuit_breaker import CircuitBreaker
from provider_gateway.policy import ProviderRouteCandidate
from provider_gateway.router import ProviderRouter


def test_router_select_round_robin_basic() -> None:
    """Test basic round-robin route selection."""
    breaker = CircuitBreaker()
    router = ProviderRouter(breaker)

    candidates = [
        ProviderRouteCandidate(provider="anthropic", model="claude-sonnet", source="test"),
        ProviderRouteCandidate(provider="openrouter", model="gpt-4o", source="test"),
        ProviderRouteCandidate(provider="ollama", model="llama3.2", source="test"),
    ]

    # No current route -> returns first candidate
    selected = router.select_route(candidates, strategy="round-robin")
    assert selected.provider == "anthropic"

    # Current route is anthropic -> returns openrouter
    selected = router.select_route(
        candidates,
        strategy="round-robin",
        current_provider="anthropic",
        current_model="claude-sonnet",
    )
    assert selected.provider == "openrouter"

    # Current route is ollama (last) -> wraps around to anthropic
    selected = router.select_route(
        candidates,
        strategy="round-robin",
        current_provider="ollama",
        current_model="llama3.2",
    )
    assert selected.provider == "anthropic"


def test_router_skips_unhealthy_providers() -> None:
    """Test that router skips unhealthy (circuit-open) providers."""
    breaker = CircuitBreaker()
    router = ProviderRouter(breaker)

    # Trip the "openrouter" provider circuit
    for _ in range(5):
        breaker.record_failure("openrouter")
    assert breaker.is_available("openrouter") is False

    candidates = [
        ProviderRouteCandidate(provider="anthropic", model="claude-sonnet", source="test"),
        ProviderRouteCandidate(provider="openrouter", model="gpt-4o", source="test"),
        ProviderRouteCandidate(provider="ollama", model="llama3.2", source="test"),
    ]

    # Current is anthropic -> next should be openrouter, but it's unhealthy, so it skips to ollama!
    selected = router.select_route(
        candidates,
        strategy="round-robin",
        current_provider="anthropic",
        current_model="claude-sonnet",
    )
    assert selected.provider == "ollama"


def test_router_failsafe_when_all_unhealthy() -> None:
    """Test that router falls back to all candidates if everything is circuit-open."""
    breaker = CircuitBreaker()
    router = ProviderRouter(breaker)

    for _ in range(5):
        breaker.record_failure("anthropic")
        breaker.record_failure("openrouter")

    candidates = [
        ProviderRouteCandidate(provider="anthropic", model="claude-sonnet", source="test"),
        ProviderRouteCandidate(provider="openrouter", model="gpt-4o", source="test"),
    ]

    # Even though both are unhealthy, returns first (failsafe) instead of None
    selected = router.select_route(candidates, strategy="round-robin")
    assert selected is not None
    assert selected.provider in {"anthropic", "openrouter"}


def test_router_select_lowest_cost() -> None:
    """Test that lowest-cost strategy selects the cheapest option."""
    breaker = CircuitBreaker()
    router = ProviderRouter(breaker)

    candidates = [
        # Anthropic claude-sonnet-4-6 has a pricing profile in agent.usage_pricing
        ProviderRouteCandidate(provider="anthropic", model="claude-sonnet-4-6", source="test"),
        # Ollama local model is generally free (cost = 0.0)
        ProviderRouteCandidate(provider="ollama", model="llama3.2", source="test"),
    ]

    selected = router.select_route(candidates, strategy="lowest-cost")
    assert selected.provider == "ollama"  # Ollama is free (0.0 cost)


def test_router_select_lowest_latency() -> None:
    """Test lowest-latency strategy selects the proven fastest provider."""
    breaker = CircuitBreaker()
    router = ProviderRouter(breaker)

    # Record latencies: anthropic is fast (120ms), openrouter is slower (450ms)
    breaker.record_success("anthropic", latency_ms=120.0)
    breaker.record_success("openrouter", latency_ms=450.0)

    candidates = [
        ProviderRouteCandidate(provider="openrouter", model="gpt-4o", source="test"),
        ProviderRouteCandidate(provider="anthropic", model="claude-sonnet", source="test"),
        ProviderRouteCandidate(provider="ollama", model="llama3.2", source="test"),  # untested (0.0)
    ]

    # Untested models are placed last. proven fastest wins.
    selected = router.select_route(candidates, strategy="lowest-latency")
    assert selected.provider == "anthropic"
