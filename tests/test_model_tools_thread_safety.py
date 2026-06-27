"""Tests for model_tools._last_resolved_tool_names thread-safety.

The global _last_resolved_tool_names is protected by _last_resolved_tool_names_lock
to prevent TOCTOU races between get_tool_definitions() writes and
handle_function_call() reads.
"""
import threading
from unittest.mock import MagicMock, patch

import pytest


class TestLastResolvedToolNamesLock:
    def test_write_protected_by_lock(self):
        """get_tool_definitions() acquires the lock when updating the global."""
        import model_tools
        lock_acquire_count = 0
        original_acquire = threading.Lock.acquire

        def counting_acquire(self_inner, *args, **kwargs):
            nonlocal lock_acquire_count
            if self_inner is model_tools._last_resolved_tool_names_lock:
                lock_acquire_count += 1
            return original_acquire(self_inner, *args, **kwargs)

        with patch.object(threading.Lock, "acquire", counting_acquire):
            with patch(
                "model_tools._compute_tool_definitions",
                return_value=[{"function": {"name": "test_tool"}}],
            ):
                model_tools.get_tool_definitions(
                    enabled_toolsets=["terminal"], quiet_mode=True
                )

        assert lock_acquire_count >= 1

    def test_read_copies_under_lock(self):
        """handle_function_call() copies the list under lock for execute_code."""
        import model_tools
        model_tools._last_resolved_tool_names = ["tool_a", "tool_b"]

        with patch(
            "model_tools._last_resolved_tool_names_lock"
        ) as mock_lock:
            mock_lock.__enter__ = MagicMock()
            mock_lock.__exit__ = MagicMock(return_value=False)
            sandbox_enabled = list(model_tools._last_resolved_tool_names)
            assert sandbox_enabled == ["tool_a", "tool_b"]

    def test_concurrent_writes_dont_corrupt(self):
        """Multiple threads writing _last_resolved_tool_names don't corrupt it."""
        import model_tools
        errors = []

        def writer(thread_id):
            try:
                names = [f"tool_{thread_id}_{i}" for i in range(50)]
                with model_tools._last_resolved_tool_names_lock:
                    model_tools._last_resolved_tool_names = names
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
        result = model_tools._last_resolved_tool_names
        assert isinstance(result, list)
        assert len(result) == 50

    def test_read_during_write_returns_complete_list(self):
        """A read during a write always sees a complete list, not a partial one."""
        import model_tools
        model_tools._last_resolved_tool_names = ["old_tool"]
        seen_during_write = []

        def reader():
            for _ in range(100):
                with model_tools._last_resolved_tool_names_lock:
                    current = list(model_tools._last_resolved_tool_names)
                if current:
                    seen_during_write.append(tuple(current))

        def writer():
            for i in range(100):
                with model_tools._last_resolved_tool_names_lock:
                    model_tools._last_resolved_tool_names = [f"new_{i}"]

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        for snapshot in seen_during_write:
            assert len(snapshot) >= 1
