"""Tests for the thread tool whitelist used by background review forks and /btw side
questions (hermes_cli.plugins_thread_whitelist, re-exported by hermes_cli.plugins)."""

import threading

from hermes_cli.plugins import get_pre_tool_call_block_message
from hermes_cli.plugins_thread_whitelist import (
    clear_thread_tool_whitelist,
    set_thread_tool_whitelist,
)
from tools.thread_context import propagate_context_to_thread


class TestThreadToolWhitelist:
    """Tests for the thread-local tool whitelist used by background review forks."""

    def test_allowed_tool_passes_through_to_hooks(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [],
        )
        set_thread_tool_whitelist({"memory", "skill_manage"})
        try:
            assert get_pre_tool_call_block_message("memory", {}) is None
        finally:
            clear_thread_tool_whitelist()

    def test_clear_restores_unrestricted_behavior(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [],
        )
        set_thread_tool_whitelist({"memory"})
        clear_thread_tool_whitelist()
        # After clearing, any tool should pass through to plugin hooks (which
        # return [] here, so result is None).
        assert get_pre_tool_call_block_message("terminal", {}) is None

    def test_whitelist_is_thread_local(self, monkeypatch):
        """Setting a whitelist in one thread must NOT leak into another."""
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [],
        )

        # Main thread: install a restrictive whitelist.
        set_thread_tool_whitelist({"memory"})
        try:
            assert get_pre_tool_call_block_message("terminal", {}) is not None

            # Worker thread: should NOT inherit main thread's whitelist.
            result = {}

            def worker():
                result["msg"] = get_pre_tool_call_block_message("terminal", {})

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            assert result["msg"] is None, "thread-local whitelist leaked across threads"
        finally:
            clear_thread_tool_whitelist()

    def test_whitelist_enforced_on_propagate_context_worker(self, monkeypatch):
        """The whitelist must reach tool-dispatch workers: every tool call runs on a
        DaemonThreadPoolExecutor worker wrapped in propagate_context_to_thread, which
        copies ContextVars but not thread-locals (#136623)."""
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [],
        )

        set_thread_tool_whitelist({"skill_view"}, "denied {tool_name}")
        try:
            assert get_pre_tool_call_block_message("skill_view", {}) is None

            result = {}

            def worker():
                result["allowed"] = get_pre_tool_call_block_message("skill_view", {})
                result["denied"] = get_pre_tool_call_block_message("terminal", {})

            t = threading.Thread(target=propagate_context_to_thread(worker))
            t.start()
            t.join()
            assert result["allowed"] is None, (
                "whitelisted tool blocked on propagate_context_to_thread worker"
            )
            assert result["denied"] == "denied terminal", (
                "whitelist not enforced on propagate_context_to_thread worker"
            )
        finally:
            clear_thread_tool_whitelist()

    def test_cleared_whitelist_not_enforced_on_worker(self, monkeypatch):
        """After clear_thread_tool_whitelist the ContextVar mirror must not keep
        denying tools on workers (#136623)."""
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [],
        )

        set_thread_tool_whitelist(set())
        clear_thread_tool_whitelist()
        result = {}

        def worker():
            result["msg"] = get_pre_tool_call_block_message("terminal", {})

        t = threading.Thread(target=propagate_context_to_thread(worker))
        t.start()
        t.join()
        assert result["msg"] is None, (
            "cleared whitelist still enforced on propagate_context_to_thread worker"
        )
