"""Tests for gateway/hooks.py — event hook system."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.hooks import HookRegistry


def _create_hook(hooks_dir, hook_name, events, handler_code):
    """Helper to create a hook directory with HOOK.yaml and handler.py."""
    hook_dir = hooks_dir / hook_name
    hook_dir.mkdir(parents=True)
    (hook_dir / "HOOK.yaml").write_text(
        f"name: {hook_name}\n"
        f"description: Test hook\n"
        f"events: {events}\n"
    )
    (hook_dir / "handler.py").write_text(handler_code)
    return hook_dir


class TestHookRegistryInit:
    def test_empty_registry(self):
        reg = HookRegistry()
        assert reg.loaded_hooks == []
        assert reg._handlers == {}


class TestDiscoverAndLoad:
    def test_loads_valid_hook(self, tmp_path):
        _create_hook(tmp_path, "my-hook", '["agent:start"]',
                      "def handle(event_type, context):\n    pass\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 1
        assert reg.loaded_hooks[0]["name"] == "my-hook"
        assert "agent:start" in reg.loaded_hooks[0]["events"]

    def test_skips_missing_hook_yaml(self, tmp_path):
        hook_dir = tmp_path / "bad-hook"
        hook_dir.mkdir()
        (hook_dir / "handler.py").write_text("def handle(e, c): pass\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 0

    def test_skips_missing_handler_py(self, tmp_path):
        hook_dir = tmp_path / "bad-hook"
        hook_dir.mkdir()
        (hook_dir / "HOOK.yaml").write_text("name: bad\nevents: ['agent:start']\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 0

    def test_skips_no_events(self, tmp_path):
        hook_dir = tmp_path / "empty-hook"
        hook_dir.mkdir()
        (hook_dir / "HOOK.yaml").write_text("name: empty\nevents: []\n")
        (hook_dir / "handler.py").write_text("def handle(e, c): pass\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 0

    def test_skips_no_handle_function(self, tmp_path):
        hook_dir = tmp_path / "no-handle"
        hook_dir.mkdir()
        (hook_dir / "HOOK.yaml").write_text("name: no-handle\nevents: ['agent:start']\n")
        (hook_dir / "handler.py").write_text("def something_else(): pass\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 0

    def test_nonexistent_hooks_dir(self, tmp_path):
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path / "nonexistent"):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 0

    def test_multiple_hooks(self, tmp_path):
        _create_hook(tmp_path, "hook-a", '["agent:start"]',
                      "def handle(e, c): pass\n")
        _create_hook(tmp_path, "hook-b", '["session:start", "session:reset"]',
                      "def handle(e, c): pass\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 2


class TestEmit:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_emit_calls_sync_handler(self, tmp_path):
        results = []

        _create_hook(tmp_path, "sync-hook", '["agent:start"]',
                      "results = []\n"
                      "def handle(event_type, context):\n"
                      "    results.append(event_type)\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        # Inject our results list into the handler's module globals
        handler_fn = reg._handlers["agent:start"][0]
        handler_fn.__globals__["results"] = results

        await reg.emit("agent:start", {"test": True})
        assert "agent:start" in results

    @pytest.mark.asyncio(loop_scope="function")
    async def test_emit_calls_async_handler(self, tmp_path):
        results = []

        hook_dir = tmp_path / "async-hook"
        hook_dir.mkdir()
        (hook_dir / "HOOK.yaml").write_text(
            "name: async-hook\nevents: ['agent:end']\n"
        )
        (hook_dir / "handler.py").write_text(
            "import asyncio\n"
            "results = []\n"
            "async def handle(event_type, context):\n"
            "    results.append(event_type)\n"
        )

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        handler_fn = reg._handlers["agent:end"][0]
        handler_fn.__globals__["results"] = results

        await reg.emit("agent:end", {})
        assert "agent:end" in results

    @pytest.mark.asyncio(loop_scope="function")
    async def test_wildcard_matching(self, tmp_path):
        results = []

        _create_hook(tmp_path, "wildcard-hook", '["command:*"]',
                      "results = []\n"
                      "def handle(event_type, context):\n"
                      "    results.append(event_type)\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        handler_fn = reg._handlers["command:*"][0]
        handler_fn.__globals__["results"] = results

        await reg.emit("command:reset", {})
        assert "command:reset" in results

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_handlers_for_event(self, tmp_path):
        reg = HookRegistry()
        # Should not raise
        await reg.emit("unknown:event", {})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_handler_error_does_not_propagate(self, tmp_path):
        _create_hook(tmp_path, "bad-hook", '["agent:start"]',
                      "def handle(event_type, context):\n"
                      "    raise ValueError('boom')\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        # Should not raise even though handler throws
        await reg.emit("agent:start", {})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_emit_default_context(self, tmp_path):
        captured = []

        _create_hook(tmp_path, "ctx-hook", '["agent:start"]',
                      "captured = []\n"
                      "def handle(event_type, context):\n"
                      "    captured.append(context)\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        handler_fn = reg._handlers["agent:start"][0]
        handler_fn.__globals__["captured"] = captured

        await reg.emit("agent:start")  # no context arg
        assert captured[0] == {}
