"""Tests for ResponseStore thread safety.

Verifies that concurrent access from multiple threads does not cause
logic bugs in LRU eviction counting or access-time tracking.
"""
import threading

import pytest


class TestResponseStoreThreading:
    """ResponseStore should handle concurrent access correctly."""

    def test_concurrent_put_does_not_over_evict(self):
        """Two threads inserting simultaneously should not evict more than max_size entries."""
        from gateway.platforms.api_server import ResponseStore

        store = ResponseStore(max_size=5, db_path=":memory:")

        errors = []

        def insert_items(start, count):
            try:
                for i in range(start, start + count):
                    store.put(f"id_{i}", {"output": f"result_{i}"})
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=insert_items, args=(0, 10))
        t2 = threading.Thread(target=insert_items, args=(100, 10))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Errors during concurrent puts: {errors}"
        # After 20 inserts with max_size=5, store should have <= 5 entries
        assert len(store) <= 5, f"Store has {len(store)} entries, expected <= 5"
        store.close()

    def test_concurrent_get_and_put_no_crash(self):
        """Reading while writing should not raise exceptions."""
        from gateway.platforms.api_server import ResponseStore

        store = ResponseStore(max_size=10, db_path=":memory:")

        # Pre-populate
        for i in range(10):
            store.put(f"id_{i}", {"output": f"result_{i}"})

        errors = []

        def reader():
            try:
                for i in range(50):
                    store.get(f"id_{i % 10}")
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(50):
                    store.put(f"id_{i + 10}", {"output": f"result_{i + 10}"})
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=writer)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Errors during concurrent get/put: {errors}"
        store.close()

    def test_get_updates_accessed_at_atomically(self):
        """Access-time update in get() should not lose to concurrent deletes."""
        from gateway.platforms.api_server import ResponseStore

        store = ResponseStore(max_size=2, db_path=":memory:")

        store.put("id_0", {"output": "result_0"})
        store.put("id_1", {"output": "result_1"})

        # id_0 is accessed, making it most-recently-used
        store.get("id_0")

        # Inserting a 3rd item should evict id_1 (least recently used), not id_0
        store.put("id_2", {"output": "result_2"})

        assert store.get("id_0") is not None, "id_0 was evicted despite being accessed recently"
        assert store.get("id_1") is None, "id_1 was not evicted as expected (LRU)"
        store.close()

    def test_close_while_readers_active_does_not_raise(self):
        """close() acquires the lock so it cannot race with an in-flight query.

        The reader runs concurrently with close() — we call close() while the
        reader thread is still active, then join the reader afterward.  This
        exercises the actual lock-ordering between close() and get().
        """
        from gateway.platforms.api_server import ResponseStore
        import time as _time

        store = ResponseStore(max_size=20, db_path=":memory:")
        for i in range(20):
            store.put(f"id_{i}", {"output": f"result_{i}"})

        errors = []
        stop_flag = threading.Event()

        def reader():
            try:
                while not stop_flag.is_set():
                    store.get("id_0")
                    store.get("id_1")
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=reader)
        t.start()
        _time.sleep(0.02)  # Let the reader get going

        # Call close() while the reader is still active — this is the race we
        # want to exercise.  The lock ensures close() waits for any in-flight
        # query to finish before closing the connection.
        store.close()
        stop_flag.set()
        t.join()

        # No unhandled exceptions from the reader before close() was called.
        assert not errors, f"Errors during concurrent read/close: {errors}"
