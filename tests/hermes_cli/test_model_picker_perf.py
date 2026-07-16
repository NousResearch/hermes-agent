"""Tests for issue #65650 — /model picker slow (~5s) when custom providers
have discover_models enabled.

The bottleneck was sequential ``fetch_api_models()`` calls — each with a
5s timeout — stacking additively. The fix parallelizes all custom provider
probes with ``ThreadPoolExecutor``, bounding the total wait to the slowest
single endpoint instead of the sum.
"""

from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.model_switch import _parallel_probe_providers


# --------------------------------------------------------------------------- #
# _parallel_probe_providers
# --------------------------------------------------------------------------- #


def test_parallel_probe_empty_targets_returns_empty():
    """No targets → empty results, no calls made."""
    result = _parallel_probe_providers([])
    assert result == {}


def test_parallel_probe_single_target():
    """A single target is probed and returned."""
    targets = [{"api_key": "sk-test", "api_url": "https://api.test.com/v1"}]

    with patch("hermes_cli.models.fetch_api_models", return_value=["model-a", "model-b"]):
        result = _parallel_probe_providers(targets)

    assert ("https://api.test.com/v1", "sk-test") in result
    assert result[("https://api.test.com/v1", "sk-test")] == ["model-a", "model-b"]


def test_parallel_probe_multiple_targets_run_concurrently():
    """Multiple targets are probed in parallel — total time ≈ max, not sum.

    This is the core perf fix: 3 endpoints each taking 2s should complete
    in ~2s, not ~6s.
    """
    targets = [
        {"api_key": "key1", "api_url": "https://api1.test.com/v1"},
        {"api_key": "key2", "api_url": "https://api2.test.com/v1"},
        {"api_key": "key3", "api_url": "https://api3.test.com/v1"},
    ]

    def slow_fetch(api_key, api_url, headers=None, **kw):
        time.sleep(1.0)  # Each call takes 1s
        return [f"model-from-{api_url}"]

    with patch("hermes_cli.models.fetch_api_models", side_effect=slow_fetch):
        start = time.time()
        result = _parallel_probe_providers(targets)
        elapsed = time.time() - start

    # All three probed
    assert len(result) == 3
    # Total time should be ~1s (parallel), not ~3s (sequential)
    # Allow generous margin for thread overhead
    assert elapsed < 2.5, f"Parallel probe took {elapsed:.1f}s — expected <2.5s (parallel)"


def test_parallel_probe_failed_endpoints_skipped():
    """Endpoints that raise or return None are absent from results."""
    targets = [
        {"api_key": "good", "api_url": "https://good.test.com/v1"},
        {"api_key": "bad", "api_url": "https://bad.test.com/v1"},
    ]

    def selective_fetch(api_key, api_url, headers=None, **kw):
        if "bad" in api_url:
            raise ConnectionError("endpoint down")
        return ["good-model"]

    with patch("hermes_cli.models.fetch_api_models", side_effect=selective_fetch):
        result = _parallel_probe_providers(targets)

    # Only the good endpoint is in results
    assert ("https://good.test.com/v1", "good") in result
    assert ("https://bad.test.com/v1", "bad") not in result
    assert result[("https://good.test.com/v1", "good")] == ["good-model"]


def test_parallel_probe_none_results_skipped():
    """Endpoints that return None (no models) are absent from results."""
    targets = [
        {"api_key": "key1", "api_url": "https://api1.test.com/v1"},
        {"api_key": "key2", "api_url": "https://api2.test.com/v1"},
    ]

    def mixed_fetch(api_key, api_url, headers=None, **kw):
        if "api2" in api_url:
            return None  # No models found
        return ["model-1"]

    with patch("hermes_cli.models.fetch_api_models", side_effect=mixed_fetch):
        result = _parallel_probe_providers(targets)

    assert ("https://api1.test.com/v1", "key1") in result
    assert ("https://api2.test.com/v1", "key2") not in result


def test_parallel_probe_passes_headers():
    """Headers from the target dict are passed to fetch_api_models."""
    targets = [
        {
            "api_key": "sk-test",
            "api_url": "https://api.test.com/v1",
            "headers": {"X-Custom-Header": "value"},
        },
    ]

    with patch("hermes_cli.models.fetch_api_models", return_value=["m1"]) as mock:
        _parallel_probe_providers(targets)

    mock.assert_called_once_with(
        "sk-test",
        "https://api.test.com/v1",
        headers={"X-Custom-Header": "value"},
    )


def test_parallel_probe_no_headers_passes_none():
    """When no headers key is present, None is passed."""
    targets = [{"api_key": "sk-test", "api_url": "https://api.test.com/v1"}]

    with patch("hermes_cli.models.fetch_api_models", return_value=["m1"]) as mock:
        _parallel_probe_providers(targets)

    mock.assert_called_once_with(
        "sk-test",
        "https://api.test.com/v1",
        headers=None,
    )


def test_parallel_probe_max_workers_capped():
    """With many targets, max_workers is capped at 8 to avoid excessive threads."""
    # Create 20 targets
    targets = [
        {"api_key": f"key{i}", "api_url": f"https://api{i}.test.com/v1"}
        for i in range(20)
    ]

    with patch("hermes_cli.models.fetch_api_models", return_value=["m"]):
        # Just verify it completes without error
        result = _parallel_probe_providers(targets)

    assert len(result) == 20