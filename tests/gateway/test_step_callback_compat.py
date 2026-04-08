"""Tests for step_callback backward compatibility.

Verifies that the gateway's step_callback normalization keeps
``tool_names`` as a list of strings for backward-compatible hooks,
while also providing the enriched ``tools`` list with results.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestStepCallbackNormalization:
    """The gateway's _step_callback_sync normalizes prev_tools from run_agent."""

    def _extract_step_callback(self):
        """Build a minimal _step_callback_sync using the same logic as gateway/run.py.

        We replicate the closure so we can test normalisation in isolation
        without spinning up the full gateway.
        """
        captured_events = []

        class FakeHooks:
            async def emit(self, event_type, data):
                captured_events.append((event_type, data))

        hooks_ref = FakeHooks()
        loop = asyncio.new_event_loop()

        def _step_callback_sync(iteration: int, prev_tools: list) -> None:
            _names: list[str] = []
            for _t in (prev_tools or []):
                if isinstance(_t, dict):
                    _names.append(_t.get("name") or "")
                else:
                    _names.append(str(_t))
            asyncio.run_coroutine_threadsafe(
                hooks_ref.emit("agent:step", {
                    "iteration": iteration,
                    "tool_names": _names,
                    "tools": prev_tools,
                }),
                loop,
            )

        return _step_callback_sync, captured_events, loop

    def test_dict_prev_tools_produce_string_tool_names(self):
        """When prev_tools is list[dict], tool_names should be list[str]."""
        cb, events, loop = self._extract_step_callback()

        # Simulate the enriched format from run_agent.py
        prev_tools = [
            {"name": "terminal", "result": '{"output": "hello"}'},
            {"name": "read_file", "result": '{"content": "..."}'},
        ]

        try:
            loop.run_until_complete(asyncio.sleep(0))  # prime the loop
            import threading
            t = threading.Thread(target=cb, args=(1, prev_tools))
            t.start()
            t.join(timeout=2)
            loop.run_until_complete(asyncio.sleep(0.1))
        finally:
            loop.close()

        assert len(events) == 1
        _, data = events[0]
        # tool_names must be strings for backward compat
        assert data["tool_names"] == ["terminal", "read_file"]
        assert all(isinstance(n, str) for n in data["tool_names"])
        # tools should be the enriched dicts
        assert data["tools"] == prev_tools

    def test_string_prev_tools_still_work(self):
        """When prev_tools is list[str] (legacy), tool_names should pass through."""
        cb, events, loop = self._extract_step_callback()

        prev_tools = ["terminal", "read_file"]

        try:
            loop.run_until_complete(asyncio.sleep(0))
            import threading
            t = threading.Thread(target=cb, args=(2, prev_tools))
            t.start()
            t.join(timeout=2)
            loop.run_until_complete(asyncio.sleep(0.1))
        finally:
            loop.close()

        assert len(events) == 1
        _, data = events[0]
        assert data["tool_names"] == ["terminal", "read_file"]

    def test_empty_prev_tools(self):
        """Empty or None prev_tools should produce empty tool_names."""
        cb, events, loop = self._extract_step_callback()

        try:
            loop.run_until_complete(asyncio.sleep(0))
            import threading
            t = threading.Thread(target=cb, args=(1, []))
            t.start()
            t.join(timeout=2)
            loop.run_until_complete(asyncio.sleep(0.1))
        finally:
            loop.close()

        assert len(events) == 1
        _, data = events[0]
        assert data["tool_names"] == []

    def test_joinable_for_hook_example(self):
        """The documented hook example: ', '.join(tool_names) should work."""
        # This is the exact pattern from the docs
        prev_tools = [
            {"name": "terminal", "result": "ok"},
            {"name": "web_search", "result": None},
        ]

        _names = []
        for _t in prev_tools:
            if isinstance(_t, dict):
                _names.append(_t.get("name") or "")
            else:
                _names.append(str(_t))

        # This must not raise — documented hook pattern
        result = ", ".join(_names)
        assert result == "terminal, web_search"
