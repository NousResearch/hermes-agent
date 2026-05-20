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


def _patch_no_builtins(reg):
    """Suppress built-in hook registration so tests only exercise user-hook discovery."""
    return patch.object(reg, "_register_builtin_hooks")


class TestDiscoverAndLoad:
    def test_loads_valid_hook(self, tmp_path):
        _create_hook(tmp_path, "my-hook", '["agent:start"]',
                      "def handle(event_type, context):\n    pass\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path), _patch_no_builtins(reg):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 1
        assert reg.loaded_hooks[0]["name"] == "my-hook"
        assert "agent:start" in reg.loaded_hooks[0]["events"]

    def test_skips_missing_hook_yaml(self, tmp_path):
        hook_dir = tmp_path / "bad-hook"
        hook_dir.mkdir()
        (hook_dir / "handler.py").write_text("def handle(e, c): pass\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path), _patch_no_builtins(reg):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 0

    def test_skips_missing_handler_py(self, tmp_path):
        hook_dir = tmp_path / "bad-hook"
        hook_dir.mkdir()
        (hook_dir / "HOOK.yaml").write_text("name: bad\nevents: ['agent:start']\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path), _patch_no_builtins(reg):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 0

    def test_skips_no_events(self, tmp_path):
        hook_dir = tmp_path / "empty-hook"
        hook_dir.mkdir()
        (hook_dir / "HOOK.yaml").write_text("name: empty\nevents: []\n")
        (hook_dir / "handler.py").write_text("def handle(e, c): pass\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path), _patch_no_builtins(reg):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 0

    def test_skips_no_handle_function(self, tmp_path):
        hook_dir = tmp_path / "no-handle"
        hook_dir.mkdir()
        (hook_dir / "HOOK.yaml").write_text("name: no-handle\nevents: ['agent:start']\n")
        (hook_dir / "handler.py").write_text("def something_else(): pass\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path), _patch_no_builtins(reg):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 0

    def test_nonexistent_hooks_dir(self, tmp_path):
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path / "nonexistent"), _patch_no_builtins(reg):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 0

    def test_multiple_hooks(self, tmp_path):
        _create_hook(tmp_path, "hook-a", '["agent:start"]',
                      "def handle(e, c): pass\n")
        _create_hook(tmp_path, "hook-b", '["session:start", "session:reset"]',
                      "def handle(e, c): pass\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path), _patch_no_builtins(reg):
            reg.discover_and_load()

        assert len(reg.loaded_hooks) == 2


class TestEmit:
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_no_handlers_for_event(self, tmp_path):
        reg = HookRegistry()
        # Should not raise and should have no handlers registered
        result = await reg.emit("unknown:event", {})
        assert result is None
        assert not reg._handlers.get("unknown:event")

    @pytest.mark.asyncio
    async def test_handler_error_does_not_propagate(self, tmp_path):
        _create_hook(tmp_path, "bad-hook", '["agent:start"]',
                      "def handle(event_type, context):\n"
                      "    raise ValueError('boom')\n")

        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        assert len(reg._handlers.get("agent:start", [])) == 1
        # Should not raise even though handler throws
        result = await reg.emit("agent:start", {})
        assert result is None

    @pytest.mark.asyncio
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


class TestEmitCollect:
    """Tests for emit_collect() — returns handler return values for decision-style hooks."""

    @pytest.mark.asyncio
    async def test_collects_sync_return_values(self):
        reg = HookRegistry()
        reg._handlers["command:status"] = [
            lambda _e, _c: {"decision": "allow"},
            lambda _e, _c: {"decision": "deny", "message": "nope"},
        ]

        results = await reg.emit_collect("command:status", {})

        assert results == [
            {"decision": "allow"},
            {"decision": "deny", "message": "nope"},
        ]

    @pytest.mark.asyncio
    async def test_collects_async_return_values(self):
        reg = HookRegistry()

        async def _async_handler(_event_type, _ctx):
            return {"decision": "handled", "message": "done"}

        reg._handlers["command:ping"] = [_async_handler]

        results = await reg.emit_collect("command:ping", {})

        assert results == [{"decision": "handled", "message": "done"}]

    @pytest.mark.asyncio
    async def test_drops_none_return_values(self):
        reg = HookRegistry()
        reg._handlers["command:x"] = [
            lambda _e, _c: None,  # fire-and-forget, returns nothing
            lambda _e, _c: {"decision": "deny"},
            lambda _e, _c: None,
        ]

        results = await reg.emit_collect("command:x", {})

        assert results == [{"decision": "deny"}]

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_abort_chain(self):
        reg = HookRegistry()

        def _raises(_e, _c):
            raise ValueError("boom")

        reg._handlers["command:x"] = [
            _raises,
            lambda _e, _c: {"decision": "allow"},
        ]

        results = await reg.emit_collect("command:x", {})

        # First handler's exception is swallowed; second handler's value still collected.
        assert results == [{"decision": "allow"}]

    @pytest.mark.asyncio
    async def test_wildcard_match_also_collected(self):
        reg = HookRegistry()
        reg._handlers["command:*"] = [lambda _e, _c: {"decision": "allow"}]
        reg._handlers["command:reset"] = [lambda _e, _c: {"decision": "deny"}]

        results = await reg.emit_collect("command:reset", {})

        # Exact match fires first, then wildcard.
        assert results == [{"decision": "deny"}, {"decision": "allow"}]

    @pytest.mark.asyncio
    async def test_no_handlers_returns_empty_list(self):
        reg = HookRegistry()

        results = await reg.emit_collect("unknown:event", {})

        assert results == []

    @pytest.mark.asyncio
    async def test_default_context(self):
        reg = HookRegistry()
        captured = []

        def _handler(event_type, context):
            captured.append((event_type, context))
            return None

        reg._handlers["agent:start"] = [_handler]

        await reg.emit_collect("agent:start")  # no context arg

        assert captured == [("agent:start", {})]
