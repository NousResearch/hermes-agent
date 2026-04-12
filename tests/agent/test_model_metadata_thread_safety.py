"""Tests for thread safety of model_metadata.py caches.

Verifies that concurrent access to the module-level mutable caches
(_model_metadata_cache, _endpoint_model_metadata_cache, etc.) does
not raise RuntimeError or corrupt data, thanks to _metadata_lock.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import agent.model_metadata as mm
from agent.model_metadata import (
    fetch_model_metadata,
    fetch_endpoint_model_metadata,
    _metadata_lock,
)


def _reset_caches():
    """Reset all module-level caches to a clean state."""
    with _metadata_lock:
        mm._model_metadata_cache = {}
        mm._model_metadata_cache_time = 0
        mm._endpoint_model_metadata_cache = {}
        mm._endpoint_model_metadata_cache_time = {}


class TestFetchModelMetadataThreadSafety:
    """Concurrent calls to fetch_model_metadata must not corrupt state."""

    def test_concurrent_fetches_no_exception(self):
        """Many threads calling fetch_model_metadata concurrently must not raise."""
        _reset_caches()
        errors: list[Exception] = []
        barrier = threading.Barrier(10)

        def _make_response():
            resp = MagicMock()
            resp.json.return_value = {
                "data": [
                    {"id": f"model/{i}", "context_length": 128000, "name": f"Model {i}"}
                    for i in range(50)
                ]
            }
            resp.raise_for_status = MagicMock()
            return resp

        def worker():
            try:
                barrier.wait(timeout=5)
                fetch_model_metadata(force_refresh=True)
            except Exception as exc:
                errors.append(exc)

        with patch("agent.model_metadata.requests.get", side_effect=lambda *a, **k: _make_response()):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert not errors, f"Threads raised exceptions: {errors}"

    def test_concurrent_fetch_and_read_no_exception(self):
        """One thread refreshes while others read the cache."""
        _reset_caches()
        # Seed the cache so readers have something to read
        mm._model_metadata_cache = {"seed/model": {"context_length": 1000}}
        mm._model_metadata_cache_time = time.time()

        errors: list[Exception] = []
        stop = threading.Event()

        def _make_response():
            resp = MagicMock()
            resp.json.return_value = {
                "data": [
                    {"id": f"model/{i}", "context_length": 64000, "name": f"M{i}"}
                    for i in range(100)
                ]
            }
            resp.raise_for_status = MagicMock()
            return resp

        def reader():
            try:
                while not stop.is_set():
                    result = fetch_model_metadata()
                    # Result must be a dict (never a corrupted partial state)
                    assert isinstance(result, dict)
            except Exception as exc:
                errors.append(exc)

        def writer():
            try:
                for _ in range(20):
                    fetch_model_metadata(force_refresh=True)
            except Exception as exc:
                errors.append(exc)

        with patch("agent.model_metadata.requests.get", side_effect=lambda *a, **k: _make_response()):
            readers = [threading.Thread(target=reader) for _ in range(5)]
            writers = [threading.Thread(target=writer) for _ in range(3)]
            for t in readers + writers:
                t.start()
            for t in writers:
                t.join(timeout=10)
            stop.set()
            for t in readers:
                t.join(timeout=10)

        assert not errors, f"Threads raised exceptions: {errors}"

    def test_concurrent_fetch_failure_returns_consistent_cache(self):
        """Concurrent failures must return consistent (possibly stale) data."""
        _reset_caches()
        mm._model_metadata_cache = {"stale/model": {"context_length": 42000}}
        mm._model_metadata_cache_time = 0  # expired

        errors: list[Exception] = []
        results: list[dict] = []
        barrier = threading.Barrier(5)

        def worker():
            try:
                barrier.wait(timeout=5)
                r = fetch_model_metadata(force_refresh=True)
                results.append(r)
            except Exception as exc:
                errors.append(exc)

        with patch("agent.model_metadata.requests.get", side_effect=Exception("network down")):
            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert not errors, f"Threads raised exceptions: {errors}"
        # All results should contain the stale cache data
        for r in results:
            assert "stale/model" in r


class TestFetchEndpointModelMetadataThreadSafety:
    """Concurrent calls to fetch_endpoint_model_metadata must not corrupt state."""

    def test_concurrent_endpoint_fetches_no_exception(self):
        """Many threads fetching endpoint metadata concurrently must not raise."""
        _reset_caches()
        errors: list[Exception] = []
        barrier = threading.Barrier(10)

        def _make_response():
            resp = MagicMock()
            resp.json.return_value = {
                "data": [
                    {"id": f"local-model-{i}", "context_length": 32768}
                    for i in range(20)
                ]
            }
            resp.raise_for_status = MagicMock()
            return resp

        def worker(url_index):
            try:
                barrier.wait(timeout=5)
                # All threads hit the same endpoint URL to maximise contention
                fetch_endpoint_model_metadata(
                    f"http://localhost:808{url_index % 2}/v1",
                    api_key="test",
                    force_refresh=True,
                )
            except Exception as exc:
                errors.append(exc)

        with patch("agent.model_metadata.requests.get", side_effect=lambda *a, **k: _make_response()):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert not errors, f"Threads raised exceptions: {errors}"

    def test_concurrent_endpoint_same_url_no_corruption(self):
        """Multiple threads fetching the same endpoint URL produce valid caches."""
        _reset_caches()
        errors: list[Exception] = []
        results: list[dict] = []
        barrier = threading.Barrier(8)
        base_url = "http://localhost:9999/v1"

        def _make_response():
            resp = MagicMock()
            resp.json.return_value = {
                "data": [
                    {"id": "the-model", "context_length": 65536}
                ]
            }
            resp.raise_for_status = MagicMock()
            return resp

        def worker():
            try:
                barrier.wait(timeout=5)
                r = fetch_endpoint_model_metadata(base_url, api_key="k", force_refresh=True)
                results.append(r)
            except Exception as exc:
                errors.append(exc)

        with patch("agent.model_metadata.requests.get", side_effect=lambda *a, **k: _make_response()):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert not errors, f"Threads raised exceptions: {errors}"
        for r in results:
            assert isinstance(r, dict)
            assert "the-model" in r
            assert r["the-model"]["context_length"] == 65536


class TestMetadataLockExists:
    """Verify the lock is a proper threading.Lock instance at module level."""

    def test_lock_is_threading_lock(self):
        assert isinstance(_metadata_lock, type(threading.Lock()))
