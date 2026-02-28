"""Tests for the gateway HookRegistry — discover, load, and emit.

Covers:
- discover_and_load: skips invalid dirs, loads valid hooks
- emit: exact match, wildcard (command:*), combined
- session:start / agent:step / command:* events fired correctly
- async and sync handlers both supported
- errors in handlers are swallowed (never block pipeline)
- step_callback integration with AIAgent
"""

import asyncio
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.hooks import HookRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_hook(hooks_dir: Path, name: str, events: list, handler_code: str) -> Path:
    """Create a minimal hook directory with HOOK.yaml + handler.py."""
    hook_dir = hooks_dir / name
    hook_dir.mkdir(parents=True, exist_ok=True)
    (hook_dir / "HOOK.yaml").write_text(
        f"name: {name}\ndescription: test\nevents:\n"
        + "".join(f"  - {e}\n" for e in events),
        encoding="utf-8",
    )
    (hook_dir / "handler.py").write_text(
        textwrap.dedent(handler_code), encoding="utf-8"
    )
    return hook_dir


# ---------------------------------------------------------------------------
# discover_and_load
# ---------------------------------------------------------------------------

class TestHookRegistryDiscovery:
    def test_empty_hooks_dir_loads_zero_hooks(self, tmp_path):
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()
        assert reg.loaded_hooks == []

    def test_nonexistent_hooks_dir_does_not_raise(self, tmp_path):
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path / "does_not_exist"):
            reg.discover_and_load()  # should not raise
        assert reg.loaded_hooks == []

    def test_valid_sync_hook_is_loaded(self, tmp_path):
        _write_hook(
            tmp_path, "my_hook", ["agent:start"],
            "def handle(event_type, ctx): pass\n",
        )
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()
        assert len(reg.loaded_hooks) == 1
        assert reg.loaded_hooks[0]["name"] == "my_hook"
        assert "agent:start" in reg.loaded_hooks[0]["events"]

    def test_hook_missing_yaml_is_skipped(self, tmp_path):
        hook_dir = tmp_path / "incomplete_hook"
        hook_dir.mkdir()
        (hook_dir / "handler.py").write_text("def handle(e, c): pass\n")
        # No HOOK.yaml
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()
        assert reg.loaded_hooks == []

    def test_hook_missing_handler_is_skipped(self, tmp_path):
        hook_dir = tmp_path / "no_handler"
        hook_dir.mkdir()
        (hook_dir / "HOOK.yaml").write_text(
            "name: no_handler\nevents:\n  - agent:start\n"
        )
        # No handler.py
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()
        assert reg.loaded_hooks == []

    def test_hook_with_no_handle_function_is_skipped(self, tmp_path):
        _write_hook(
            tmp_path, "bad_handler", ["agent:start"],
            "# no handle() function here\nfoo = 42\n",
        )
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()
        assert reg.loaded_hooks == []

    def test_hook_with_invalid_yaml_is_skipped(self, tmp_path):
        hook_dir = tmp_path / "bad_yaml"
        hook_dir.mkdir()
        (hook_dir / "HOOK.yaml").write_text(":\n::invalid yaml::\n")
        (hook_dir / "handler.py").write_text("def handle(e, c): pass\n")
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()
        assert reg.loaded_hooks == []

    def test_multiple_hooks_all_loaded(self, tmp_path):
        for i in range(3):
            _write_hook(tmp_path, f"hook_{i}", [f"event:{i}"], "def handle(e,c): pass\n")
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()
        assert len(reg.loaded_hooks) == 3


# ---------------------------------------------------------------------------
# emit — exact match
# ---------------------------------------------------------------------------

class TestHookRegistryEmitExact:
    @pytest.mark.asyncio
    async def test_sync_handler_called_for_matching_event(self, tmp_path):
        calls = []
        _write_hook(
            tmp_path, "spy", ["agent:start"],
            "calls = []\ndef handle(event_type, ctx):\n    calls.append((event_type, ctx))\n",
        )
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path):
            reg.discover_and_load()

        # Inject a direct Python callable instead of relying on module-level list
        received = []
        reg._handlers["agent:start"] = [lambda et, ctx: received.append((et, ctx))]

        await reg.emit("agent:start", {"foo": "bar"})
        assert len(received) == 1
        assert received[0][0] == "agent:start"
        assert received[0][1]["foo"] == "bar"

    @pytest.mark.asyncio
    async def test_async_handler_awaited(self):
        reg = HookRegistry()
        received = []

        async def async_handle(event_type, ctx):
            received.append(event_type)

        reg._handlers["agent:end"] = [async_handle]
        await reg.emit("agent:end", {})
        assert received == ["agent:end"]

    @pytest.mark.asyncio
    async def test_non_matching_event_not_fired(self):
        reg = HookRegistry()
        called = []
        reg._handlers["session:start"] = [lambda e, c: called.append(e)]
        await reg.emit("agent:start", {})
        assert called == []

    @pytest.mark.asyncio
    async def test_emit_with_no_handlers_does_not_raise(self):
        reg = HookRegistry()
        await reg.emit("some:unknown:event", {"x": 1})  # should not raise

    @pytest.mark.asyncio
    async def test_emit_none_context_defaults_to_empty_dict(self):
        reg = HookRegistry()
        received_ctx = []
        reg._handlers["test:event"] = [lambda e, ctx: received_ctx.append(ctx)]
        await reg.emit("test:event", None)
        assert received_ctx == [{}]


# ---------------------------------------------------------------------------
# emit — wildcard (command:*)
# ---------------------------------------------------------------------------

class TestHookRegistryWildcard:
    @pytest.mark.asyncio
    async def test_wildcard_fires_for_any_command_subtype(self):
        reg = HookRegistry()
        received = []
        reg._handlers["command:*"] = [lambda e, c: received.append(e)]

        await reg.emit("command:reset", {})
        await reg.emit("command:model", {})
        await reg.emit("command:help", {})

        assert received == ["command:reset", "command:model", "command:help"]

    @pytest.mark.asyncio
    async def test_wildcard_does_not_fire_for_non_command_events(self):
        reg = HookRegistry()
        received = []
        reg._handlers["command:*"] = [lambda e, c: received.append(e)]

        await reg.emit("agent:start", {})
        await reg.emit("session:start", {})

        assert received == []

    @pytest.mark.asyncio
    async def test_exact_and_wildcard_both_fire(self):
        reg = HookRegistry()
        exact_calls = []
        wildcard_calls = []
        reg._handlers["command:model"] = [lambda e, c: exact_calls.append(e)]
        reg._handlers["command:*"] = [lambda e, c: wildcard_calls.append(e)]

        await reg.emit("command:model", {"args": "gpt-4"})

        assert exact_calls == ["command:model"]
        assert wildcard_calls == ["command:model"]


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------

class TestHookRegistryErrorIsolation:
    @pytest.mark.asyncio
    async def test_exception_in_handler_does_not_propagate(self):
        reg = HookRegistry()

        def bad_handler(event_type, ctx):
            raise RuntimeError("kaboom")

        reg._handlers["agent:start"] = [bad_handler]
        # Should not raise
        await reg.emit("agent:start", {})

    @pytest.mark.asyncio
    async def test_subsequent_handlers_still_called_after_exception(self):
        reg = HookRegistry()
        called = []

        def bad_handler(event_type, ctx):
            raise ValueError("oops")

        def good_handler(event_type, ctx):
            called.append("good")

        reg._handlers["agent:start"] = [bad_handler, good_handler]
        await reg.emit("agent:start", {})
        assert called == ["good"]


# ---------------------------------------------------------------------------
# session:start / agent:step / command:* integration shape
# ---------------------------------------------------------------------------

class TestHookEventContextShapes:
    """Verify the context dicts emitted have the expected keys.

    These tests don't run the full gateway — they just inspect the shape
    of contexts emitted via the hook system.
    """

    @pytest.mark.asyncio
    async def test_session_start_context_keys(self):
        reg = HookRegistry()
        captured = []
        reg._handlers["session:start"] = [lambda e, ctx: captured.append(ctx)]

        ctx = {
            "platform": "telegram",
            "user_id": "123",
            "session_id": "sess_abc",
            "session_key": "agent:main:telegram:dm",
        }
        await reg.emit("session:start", ctx)
        assert captured[0]["platform"] == "telegram"
        assert "session_id" in captured[0]
        assert "session_key" in captured[0]

    @pytest.mark.asyncio
    async def test_agent_step_context_keys(self):
        reg = HookRegistry()
        captured = []
        reg._handlers["agent:step"] = [lambda e, ctx: captured.append(ctx)]

        ctx = {
            "platform": "discord",
            "user_id": "456",
            "session_id": "sess_xyz",
            "iteration": 3,
            "tool_names": ["terminal", "web_search"],
        }
        await reg.emit("agent:step", ctx)
        assert captured[0]["iteration"] == 3
        assert "tool_names" in captured[0]

    @pytest.mark.asyncio
    async def test_command_context_keys(self):
        reg = HookRegistry()
        captured = []
        reg._handlers["command:*"] = [lambda e, ctx: captured.append(ctx)]

        ctx = {
            "platform": "slack",
            "user_id": "789",
            "command": "model",
            "args": "openai/gpt-4o",
        }
        await reg.emit("command:model", ctx)
        assert captured[0]["command"] == "model"
        assert captured[0]["args"] == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# step_callback integration with AIAgent
# ---------------------------------------------------------------------------

class TestStepCallbackIntegration:
    """Verify AIAgent.__init__ accepts step_callback and fires it in the loop."""

    def test_step_callback_parameter_accepted(self):
        """AIAgent should not raise when step_callback is provided."""
        from run_agent import AIAgent

        cb = MagicMock()
        # We don't run a full conversation — just verify __init__ accepts the param
        with patch("run_agent.OpenAI"):
            agent = AIAgent(
                model="test/model",
                api_key="sk-test",
                step_callback=cb,
                quiet_mode=True,
            )
        assert agent.step_callback is cb

    def test_step_callback_none_by_default(self):
        """step_callback should default to None for backwards compatibility."""
        from run_agent import AIAgent

        with patch("run_agent.OpenAI"):
            agent = AIAgent(
                model="test/model",
                api_key="sk-test",
                quiet_mode=True,
            )
        assert agent.step_callback is None

    def test_step_callback_called_each_iteration(self):
        """step_callback(iteration, tool_names) must be called once per loop turn."""
        from run_agent import AIAgent

        calls = []

        def cb(iteration, tool_names):
            calls.append({"iteration": iteration, "tool_names": tool_names})

        # Build a minimal mock response that ends after one tool-less turn
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "done"
        mock_choice.message.tool_calls = None
        mock_choice.message.reasoning_content = None
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with patch("run_agent.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            agent = AIAgent(
                model="test/model",
                api_key="sk-test",
                step_callback=cb,
                quiet_mode=True,
                enabled_toolsets=[],  # no tools to keep it simple
            )
            agent.run_conversation("hello")

        assert len(calls) >= 1
        assert calls[0]["iteration"] == 1
        assert isinstance(calls[0]["tool_names"], list)

    def test_step_callback_error_does_not_crash_agent(self):
        """A buggy step_callback must not propagate and crash the agent loop."""
        from run_agent import AIAgent

        def bad_cb(iteration, tool_names):
            raise RuntimeError("step callback exploded")

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_choice.message.reasoning_content = None
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=3)

        with patch("run_agent.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            agent = AIAgent(
                model="test/model",
                api_key="sk-test",
                step_callback=bad_cb,
                quiet_mode=True,
                enabled_toolsets=[],
            )
            # Should not raise
            result = agent.run_conversation("hello")

        assert result is not None
