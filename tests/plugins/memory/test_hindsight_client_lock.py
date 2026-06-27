"""Tests for HindsightMemoryProvider._get_client() thread-safety.

The _get_client() method uses a double-checked locking pattern so two
concurrent callers don't race to create duplicate clients.
"""
import threading
from unittest.mock import MagicMock, patch

import pytest


def _make_provider():
    """Create a HindsightMemoryProvider with minimal init."""
    from plugins.memory.hindsight import HindsightMemoryProvider
    provider = HindsightMemoryProvider.__new__(HindsightMemoryProvider)
    provider._config = {"mode": "cloud"}
    provider._mode = "cloud"
    provider._client = None
    provider._client_lock = threading.Lock()
    provider._api_url = "https://api.hindsight.vectorize.io"
    provider._api_key = "test-key"
    provider._timeout = 10
    provider._idle_timeout = 300
    provider._llm_base_url = ""
    return provider


class TestGetClientLock:
    def test_concurrent_clients_created_once(self):
        """Two threads racing _get_client should create only one client."""
        provider = _make_provider()
        creation_count = 0
        mock_client = MagicMock()

        original_init = type(mock_client).__init__

        def counting_init(self_inner, *args, **kwargs):
            nonlocal creation_count
            creation_count += 1

        mock_client.__class__.__init__ = counting_init

        barrier = threading.Barrier(2)
        results = []

        def get_client():
            barrier.wait()
            with patch(
                "plugins.memory.hindsight._ensure_cloud_client_dependency"
            ), patch(
                "plugins.memory.hindsight.Hindsight", return_value=mock_client
            ):
                results.append(provider._get_client())

        t1 = threading.Thread(target=get_client)
        t2 = threading.Thread(target=get_client)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert len(results) == 2
        assert results[0] is results[1]

    def test_second_call_returns_cached(self):
        """After first creation, subsequent calls return cached client."""
        provider = _make_provider()
        mock_client = MagicMock()
        provider._client = mock_client

        result = provider._get_client()
        assert result is mock_client

    def test_lock_acquired_during_creation(self):
        """Lock is held during client creation."""
        provider = _make_provider()
        lock_acquire_count = 0
        original_acquire = threading.Lock.acquire

        def counting_acquire(self_inner, *args, **kwargs):
            nonlocal lock_acquire_count
            lock_acquire_count += 1
            return original_acquire(self_inner, *args, **kwargs)

        mock_client = MagicMock()
        with patch.object(
            threading.Lock, "acquire", counting_acquire
        ), patch(
            "plugins.memory.hindsight._ensure_cloud_client_dependency"
        ), patch(
            "plugins.memory.hindsight.Hindsight", return_value=mock_client
        ):
            result = provider._get_client()

        assert result is mock_client
        assert lock_acquire_count >= 1
