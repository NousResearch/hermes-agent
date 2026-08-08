# tests/agent/test_registry_v2.py
import pytest
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tools.registry_v2 import ToolRegistryV2, tool_registry_v2


class TestToolRegistryV2:
    """Tests for ToolRegistryV2 lifecycle hook system."""

    def test_singleton_module_level_instance_exists(self):
        """Module-level tool_registry_v2 instance exists."""
        assert tool_registry_v2 is not None
        assert isinstance(tool_registry_v2, ToolRegistryV2)

    def test_register_lifecycle_hook_adds_hook(self):
        """register_lifecycle_hook() adds a callback for the specified event."""
        registry = ToolRegistryV2()
        calls = []

        def hook(tool_name, **kwargs):
            calls.append((tool_name, kwargs))

        registry.register_lifecycle_hook("on_tool_loaded", hook)
        registry.on_tool_registered("my_tool", toolset="my_toolset")

        assert len(calls) == 1
        assert calls[0][0] == "my_tool"
        assert calls[0][1]["toolset"] == "my_toolset"

    def test_on_tool_registered_fires_on_tool_loaded_hooks(self):
        """on_tool_registered() fires all registered on_tool_loaded hooks."""
        registry = ToolRegistryV2()
        calls = []

        def hook(tool_name, **kwargs):
            calls.append(tool_name)

        registry.register_lifecycle_hook("on_tool_loaded", hook)
        registry.on_tool_registered("tool_one", toolset="ts1")
        registry.on_tool_registered("tool_two", toolset="ts2")

        assert calls == ["tool_one", "tool_two"]

    def test_on_tool_deregistered_fires_on_tool_unloaded_hooks(self):
        """on_tool_deregistered() fires all registered on_tool_unloaded hooks."""
        registry = ToolRegistryV2()
        calls = []

        def hook(tool_name, **kwargs):
            calls.append(tool_name)

        registry.register_lifecycle_hook("on_tool_unloaded", hook)
        registry.on_tool_deregistered("old_tool")

        assert calls == ["old_tool"]

    def test_hook_exceptions_are_fire_and_forget(self):
        """Hook exceptions don't propagate - they're logged but not raised."""
        registry = ToolRegistryV2()

        def bad_hook(tool_name, **kwargs):
            raise RuntimeError("hook error")

        registry.register_lifecycle_hook("on_tool_loaded", bad_hook)

        # Should not raise
        registry.on_tool_registered("some_tool", toolset="ts")

        # Should still have called the hook (even though it errored)
        # We can't easily verify the error was logged, but we verified it didn't raise

    def test_multiple_hooks_all_fire(self):
        """Multiple hooks for the same event all get called."""
        registry = ToolRegistryV2()
        calls = []

        def hook1(tool_name, **kwargs):
            calls.append(("hook1", tool_name))

        def hook2(tool_name, **kwargs):
            calls.append(("hook2", tool_name))

        registry.register_lifecycle_hook("on_tool_loaded", hook1)
        registry.register_lifecycle_hook("on_tool_loaded", hook2)
        registry.on_tool_registered("tool_x")

        assert ("hook1", "tool_x") in calls
        assert ("hook2", "tool_x") in calls

    def test_register_lifecycle_hook_unknown_event_raises(self):
        """register_lifecycle_hook() raises ValueError for unknown events."""
        registry = ToolRegistryV2()

        with pytest.raises(ValueError) as excinfo:
            registry.register_lifecycle_hook("on_unknown_event", lambda *args: None)

        assert "Unknown lifecycle event" in str(excinfo.value)

    def test_unregister_lifecycle_hook(self):
        """unregister_lifecycle_hook() removes a previously registered hook."""
        registry = ToolRegistryV2()
        calls = []

        def hook(tool_name, **kwargs):
            calls.append(tool_name)

        registry.register_lifecycle_hook("on_tool_loaded", hook)
        registry.on_tool_registered("before_unregister")

        assert len(calls) == 1

        removed = registry.unregister_lifecycle_hook("on_tool_loaded", hook)
        assert removed is True

        registry.on_tool_registered("after_unregister")
        assert len(calls) == 1  # Should not have increased

    def test_unregister_lifecycle_hook_returns_false_if_not_found(self):
        """unregister_lifecycle_hook() returns False if hook wasn't registered."""
        registry = ToolRegistryV2()

        def hook(tool_name, **kwargs):
            pass

        result = registry.unregister_lifecycle_hook("on_tool_loaded", hook)
        assert result is False

    def test_thread_safety_register_and_fire(self):
        """Thread-safe: concurrent register/unregister and fire operations work."""
        registry = ToolRegistryV2()
        num_hooks = 20
        hooks = [lambda *args, idx=idx: None for idx in range(num_hooks)]

        def register_and_fire():
            for i, h in enumerate(hooks):
                registry.register_lifecycle_hook("on_tool_loaded", h)
            registry.on_tool_registered("concurrent_tool")

        threads = [threading.Thread(target=register_and_fire) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not have raised any exceptions

    def test_thread_safety_concurrent_fire(self):
        """Thread-safe: concurrent on_tool_registered() calls don't cause issues."""
        registry = ToolRegistryV2()
        call_count = 0
        lock = threading.Lock()

        def counting_hook(tool_name, **kwargs):
            nonlocal call_count
            with lock:
                call_count += 1

        # Register hook
        registry.register_lifecycle_hook("on_tool_loaded", counting_hook)

        def fire():
            for _ in range(50):
                registry.on_tool_registered("tool")

        threads = [threading.Thread(target=fire) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All calls should have been made
        assert call_count == 500

    def test_set_base_registry(self):
        """set_base_registry() stores the reference."""
        registry = ToolRegistryV2()
        base_reg = object()

        registry.set_base_registry(base_reg)

        # Read back via property (it's wrapped in a property for thread safety)
        assert registry.base_registry is base_reg

    def test_base_registry_initially_none(self):
        """base_registry property returns None when no base registry is set."""
        registry = ToolRegistryV2()
        assert registry.base_registry is None

    def test_hook_order_preserved(self):
        """Hooks fire in the order they were registered (FIFO)."""
        registry = ToolRegistryV2()
        call_order = []

        def hook1(tool_name, **kwargs):
            call_order.append(1)

        def hook2(tool_name, **kwargs):
            call_order.append(2)

        def hook3(tool_name, **kwargs):
            call_order.append(3)

        registry.register_lifecycle_hook("on_tool_loaded", hook1)
        registry.register_lifecycle_hook("on_tool_loaded", hook2)
        registry.register_lifecycle_hook("on_tool_loaded", hook3)
        registry.on_tool_registered("tool")

        assert call_order == [1, 2, 3]
