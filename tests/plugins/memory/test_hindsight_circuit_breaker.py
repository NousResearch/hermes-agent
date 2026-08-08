"""Tests for Hindsight circuit breaker behaviour.

Adapted from PR #17416 (kagura-agent) with local_external mode support.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    _CIRCUIT_BREAKER_COOLDOWN,
    _CIRCUIT_BREAKER_THRESHOLD,
)


def _make_provider(tmp_path, monkeypatch, mode="local_embedded"):
    """Create a provider with a mock client for the given mode."""
    config = {
        "mode": mode,
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("plugins.memory.hindsight._check_local_runtime", lambda: (True, ""))

    p = HindsightMemoryProvider()
    p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")

    client = MagicMock()
    client.aretain = AsyncMock(return_value=SimpleNamespace(ok=True))
    client.arecall = AsyncMock(
        return_value=SimpleNamespace(results=[SimpleNamespace(text="m1")])
    )
    client.aretain_batch = AsyncMock()
    client.aclose = AsyncMock()
    p._client = client
    # Stub _get_client so it never tries to create a real connection
    p._get_client = lambda: client
    return p


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    return _make_provider(tmp_path, monkeypatch, mode="local_embedded")


@pytest.fixture()
def provider_external(tmp_path, monkeypatch):
    return _make_provider(tmp_path, monkeypatch, mode="local_external")


class TestCircuitBreaker:

    def test_opens_after_threshold_connection_failures(self, provider):
        """Only backend connection failures open the circuit."""
        provider._client.arecall = AsyncMock(side_effect=RuntimeError("server disconnected"))

        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            with pytest.raises(RuntimeError, match="server disconnected"):
                provider._run_hindsight_operation(lambda c: c.arecall(bank_id="b", query="q"))

        with pytest.raises(RuntimeError, match="circuit breaker open"):
            provider._run_hindsight_operation(lambda c: c.arecall(bank_id="b", query="q"))

    @pytest.mark.parametrize("message", [
        "authentication failed", "invalid request", "unsupported operation",
        "database migration failed",
    ])
    def test_non_backend_errors_do_not_open_circuit(self, provider, message):
        provider._client.arecall = AsyncMock(side_effect=RuntimeError(message))

        for _ in range(_CIRCUIT_BREAKER_THRESHOLD + 1):
            with pytest.raises(RuntimeError, match=message):
                provider._run_hindsight_operation(lambda c: c.arecall(bank_id="b", query="q"))

        assert provider._circuit_breaker_failures == 0

    def test_half_open_probe_is_single_flight(self, provider):
        """After cooldown, one caller probes and concurrent callers fail fast."""
        import threading

        provider._circuit_breaker_failures = _CIRCUIT_BREAKER_THRESHOLD
        provider._circuit_breaker_open_until = time.monotonic() - 1
        started = threading.Event()
        release = threading.Event()
        results = []

        def run_sync(_operation):
            started.set()
            assert release.wait(timeout=2)
            return SimpleNamespace(results=[])

        provider._run_sync = run_sync

        def caller():
            try:
                provider._run_hindsight_operation(lambda c: "probe")
                results.append("success")
            except RuntimeError as exc:
                results.append(str(exc))

        first = threading.Thread(target=caller)
        first.start()
        assert started.wait(timeout=2)
        second = threading.Thread(target=caller)
        second.start()
        second.join(timeout=2)
        release.set()
        first.join(timeout=2)

        assert any("half-open probe in flight" in result for result in results)
        assert results.count("success") == 1
        assert provider._circuit_breaker_failures == 0

    def test_resets_after_cooldown(self, provider):
        """After cooldown, circuit allows a retry (half-open)."""
        provider._circuit_breaker_failures = _CIRCUIT_BREAKER_THRESHOLD
        provider._circuit_breaker_open_until = time.monotonic() - 1  # expired

        result = provider._run_hindsight_operation(lambda c: c.arecall(bank_id="b", query="q"))
        assert result.results
        assert provider._circuit_breaker_failures == 0

    def test_success_resets_counter(self, provider):
        """A successful operation resets the failure counter to 0."""
        provider._circuit_breaker_failures = 2

        provider._run_hindsight_operation(lambda c: c.arecall(bank_id="b", query="q"))
        assert provider._circuit_breaker_failures == 0

    def test_sync_turn_skips_when_circuit_open(self, provider):
        """sync_turn returns early without calling aretain_batch when circuit is open."""
        provider._circuit_breaker_failures = _CIRCUIT_BREAKER_THRESHOLD
        provider._circuit_breaker_open_until = time.monotonic() + 300
        provider._auto_retain = True
        provider._retain_every_n_turns = 1

        provider.sync_turn("hello", "world")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)

        provider._client.aretain_batch.assert_not_called()


class TestRetriableErrorDetection:

    def test_embedded_mode_markers_still_work(self, provider):
        """Original local_embedded markers are still detected."""
        assert provider._is_retriable_connection_error(
            RuntimeError("Failed to start daemon on port 9999")
        )
        assert provider._is_retriable_connection_error(
            RuntimeError("Cannot use HindsightEmbedded after it has been closed")
        )

    def test_external_mode_markers(self, provider_external):
        """local_external mode connection errors are detected as retriable."""
        assert provider_external._is_retriable_connection_error(
            RuntimeError("Server disconnected")
        )
        assert provider_external._is_retriable_connection_error(
            RuntimeError("Connection reset by peer")
        )
        assert provider_external._is_retriable_connection_error(
            RuntimeError("Operation timed out")
        )
        assert provider_external._is_retriable_connection_error(
            RuntimeError("Connection refused")
        )
        assert provider_external._is_retriable_connection_error(
            RuntimeError("Cannot connect to host 192.168.10.252:8888")
        )
        assert provider_external._is_retriable_connection_error(
            RuntimeError("ClientConnectorError")
        )

    def test_non_connection_errors_not_retriable(self, provider_external):
        """Genuine non-connection errors are NOT retriable."""
        assert not provider_external._is_retriable_connection_error(
            RuntimeError("Invalid query syntax")
        )
        assert not provider_external._is_retriable_connection_error(
            ValueError("bad argument")
        )


class TestLocalExternalCircuitBreaker:
    """Verify the circuit breaker works end-to-end in local_external mode
    (our production configuration)."""

    def test_circuit_opens_on_server_disconnected(self, provider_external):
        """The exact error we see in production ('Server disconnected') trips the circuit.

        'Server disconnected' is retriable, so each call attempts a retry.
        The circuit opens once failures reach the threshold.
        """
        provider_external._client.arecall = AsyncMock(
            side_effect=RuntimeError("Server disconnected")
        )

        # Keep calling until the circuit opens
        circuit_opened = False
        for i in range(10):
            try:
                provider_external._run_hindsight_operation(
                    lambda c: c.arecall(bank_id="b", query="q")
                )
            except RuntimeError as e:
                if "circuit breaker open" in str(e).lower():
                    circuit_opened = True
                    break
            except Exception:
                pass

        assert circuit_opened, "Circuit breaker should have opened after repeated Server disconnected errors"
