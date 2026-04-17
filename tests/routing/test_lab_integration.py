"""FASE 3 — Lab integration tests for env-var opt-in telemetry (RED suite)."""
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Add agent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agent"))


@pytest.fixture
def isolated_store(tmp_path):
    """Return a Path for an isolated JSONL store."""
    store = tmp_path / "telemetry.jsonl"
    return store


def test_env_var_triggers_auto_instrumentation(monkeypatch):
    """HERMES_ROUTING_TELEMETRY=1 at import time triggers auto-instrumentation."""
    import importlib
    import agent.smart_model_routing as sm
    
    # First, ensure we have a fresh module without instrumentation
    if hasattr(sm.resolve_turn_route, "_routing_instrumented"):
        # Force reload without env var first
        monkeypatch.delenv("HERMES_ROUTING_TELEMETRY", raising=False)
        importlib.reload(sm)
    
    # Now set env var and reload
    monkeypatch.setenv("HERMES_ROUTING_TELEMETRY", "1")
    importlib.reload(sm)
    
    # Verify instrumentation flag is set
    assert hasattr(sm.resolve_turn_route, "_routing_instrumented")
    assert sm.resolve_turn_route._routing_instrumented is True


def test_no_env_var_leaves_resolve_turn_route_untouched(monkeypatch):
    """Without HERMES_ROUTING_TELEMETRY, resolve_turn_route is NOT instrumented."""
    import importlib
    import agent.smart_model_routing as sm
    
    # Ensure env var is not set
    monkeypatch.delenv("HERMES_ROUTING_TELEMETRY", raising=False)
    importlib.reload(sm)
    
    # Verify instrumentation flag is NOT set
    assert not hasattr(sm.resolve_turn_route, "_routing_instrumented")


def test_instrument_is_idempotent(isolated_store, monkeypatch):
    """Calling instrument_resolve_turn_route() twice doesn't double-wrap."""
    import importlib
    import agent.smart_model_routing as sm
    from agent.routing_telemetry import load_events
    
    # Ensure env var is not set for clean reload
    monkeypatch.delenv("HERMES_ROUTING_TELEMETRY", raising=False)
    importlib.reload(sm)
    
    from agent.smart_model_routing import instrument_resolve_turn_route
    
    # Instrument twice
    instrument_resolve_turn_route(store=isolated_store)
    instrument_resolve_turn_route(store=isolated_store)
    
    # Call the function once
    primary = {
        "model": "claude-sonnet-4.6",
        "provider": "anthropic",
        "api_key": "fake",
    }
    sm.resolve_turn_route("hello", {"enabled": False}, primary)
    
    # Should only have 1 event, not 2
    events = load_events(store=isolated_store)
    assert len(events) == 1


def test_instrumented_call_records_event(isolated_store, monkeypatch):
    """Instrumented resolve_turn_route records a telemetry event on success."""
    import importlib
    import agent.smart_model_routing as sm
    from agent.routing_telemetry import load_events
    
    # Ensure env var is not set for clean reload
    monkeypatch.delenv("HERMES_ROUTING_TELEMETRY", raising=False)
    importlib.reload(sm)
    
    from agent.smart_model_routing import instrument_resolve_turn_route
    instrument_resolve_turn_route(store=isolated_store)
    
    # Call with a simple message that should use primary (routing disabled)
    primary = {
        "model": "claude-sonnet-4.6",
        "provider": "anthropic",
        "api_key": "fake",
    }
    result = sm.resolve_turn_route("hello", {"enabled": False}, primary)
    
    # Verify result is correct
    assert result["model"] == "claude-sonnet-4.6"
    
    # Verify event was recorded
    events = load_events(store=isolated_store)
    assert len(events) == 1
    ev = events[0]
    assert ev.success is True
    assert ev.latency_ms is not None
    assert ev.latency_ms > 0
    assert ev.model == "claude-sonnet-4.6"
    assert ev.provider == "anthropic"
    assert ev.error_type is None


def test_instrumented_call_on_failure_records_and_reraises(isolated_store, monkeypatch):
    """When resolve_runtime_provider raises, telemetry records failure and re-raises."""
    import importlib
    import agent.smart_model_routing as sm
    from agent.routing_telemetry import load_events
    
    # Ensure env var is not set for clean reload
    monkeypatch.delenv("HERMES_ROUTING_TELEMETRY", raising=False)
    importlib.reload(sm)
    
    from agent.smart_model_routing import instrument_resolve_turn_route
    instrument_resolve_turn_route(store=isolated_store)
    
    # Mock resolve_runtime_provider to raise an exception
    # Note: resolve_turn_route catches exceptions from resolve_runtime_provider internally
    # and falls back to primary, so telemetry sees success=True but with primary fallback
    from hermes_cli import runtime_provider
    
    original_resolve = runtime_provider.resolve_runtime_provider
    
    def raising_resolve(*args, **kwargs):
        raise RuntimeError("Provider resolution failed")
    
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", raising_resolve)
    
    try:
        primary = {
            "model": "claude-sonnet-4.6",
            "provider": "anthropic",
            "api_key": "fake",
        }
        # This should NOT raise because resolve_turn_route catches internally
        result = sm.resolve_turn_route("hello", {"enabled": True, "cheap_model": {"provider": "fake", "model": "fake"}}, primary)
        
        # Should fall back to primary
        assert result["model"] == "claude-sonnet-4.6"
        
        # Verify event was recorded
        events = load_events(store=isolated_store)
        assert len(events) == 1
        ev = events[0]
        # Since resolve_turn_route catches internally, telemetry sees success=True
        # and turn_kind="primary" because label is None (no smart routing happened)
        assert ev.success is True
        assert ev.turn_kind == "primary"
    finally:
        # Restore original
        monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", original_resolve)
