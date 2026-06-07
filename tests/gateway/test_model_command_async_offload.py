"""Regression tests for gateway /model command not blocking the event loop.

Bug: #41289 — _handle_model_command calls list_picker_providers() and
list_authenticated_providers() synchronously on the async event loop.
When the provider model cache is stale, these functions perform blocking
network I/O (urllib.request.urlopen), freezing the event loop for 120-150s
and causing Discord "application did not respond" timeouts.

The fix wraps both calls in asyncio.to_thread() so they run in a worker
thread while the event loop stays responsive.
"""

import asyncio
import functools
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(session_db=None, current_session_id="sess_001"):
    """Create a minimal GatewayRunner for testing slash commands."""
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._session_db = session_db
    runner._pending_model_notes = {}
    runner._session_key_for_source = lambda source: (
        f"{source.platform.value}:{source.chat_id}"
    )
    runner._evict_cached_agent = MagicMock()
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._thread_metadata_for_source = MagicMock(return_value={})
    return runner


def _make_event(text="/model", platform=Platform.TELEGRAM):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=platform,
            user_id="12345",
            chat_id="67890",
            chat_type="dm",
        ),
    )


class TestModelCommandAsyncOffload:
    """Tests that list_picker_providers and list_authenticated_providers
    are called via asyncio.to_thread, not directly on the event loop."""

    @pytest.mark.asyncio
    async def test_picker_path_uses_to_thread(self, tmp_path, monkeypatch):
        """list_picker_providers should be offloaded to a thread.

        Verify that _handle_model_command does NOT call list_picker_providers
        directly on the event loop when a picker-capable adapter is present.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        import yaml

        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({
                "model": {"default": "test-model", "provider": "openrouter"},
            }),
            encoding="utf-8",
        )

        import gateway.run as gateway_run

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

        runner = _make_runner()

        # Create a mock adapter with send_model_picker defined on the TYPE
        # (not just the instance) so the has_picker check passes
        class PickerAdapter:
            async def send_model_picker(self, **kwargs):
                return SimpleNamespace(success=False)

        mock_adapter = PickerAdapter()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        # Mock list_picker_providers to track whether it was called from
        # a worker thread (asyncio.to_thread) or directly on the event loop
        call_thread_ids = []

        def slow_list_picker(**kwargs):
            try:
                call_thread_ids.append(asyncio.get_running_loop())
            except RuntimeError as e:
                call_thread_ids.append(e)
            return [{"name": "Test", "slug": "test", "models": ["m1"], "total_models": 1, "is_current": True}]

        with patch(
            "hermes_cli.model_switch.list_picker_providers",
            side_effect=slow_list_picker,
        ):
            result = await runner._handle_model_command(_make_event())

        # If offloaded to thread, asyncio.get_running_loop() should have raised
        # RuntimeError (no running loop in worker thread), so the stored value
        # should be a RuntimeError instance
        assert len(call_thread_ids) == 1
        thread_id_result = call_thread_ids[0]
        assert isinstance(thread_id_result, RuntimeError), (
            "list_picker_providers was called directly on the event loop "
            "(not via to_thread) — this blocks the loop. Fix: wrap in "
            "asyncio.to_thread()."
        )

    @pytest.mark.asyncio
    async def test_text_fallback_uses_to_thread(self, tmp_path, monkeypatch):
        """list_authenticated_providers should also be offloaded to a thread.

        This tests the fallback path (no picker adapter) where
        list_authenticated_providers is called to build the text model list.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        import yaml

        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({
                "model": {"default": "test-model", "provider": "openrouter"},
            }),
            encoding="utf-8",
        )

        import gateway.run as gateway_run

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

        runner = _make_runner()
        # No adapter with picker → falls through to text list path
        runner.adapters = {}

        call_thread_ids = []

        def slow_list_authenticated(**kwargs):
            try:
                call_thread_ids.append(asyncio.get_running_loop())
            except RuntimeError as e:
                call_thread_ids.append(e)
            return []

        with patch(
            "hermes_cli.model_switch.list_authenticated_providers",
            side_effect=slow_list_authenticated,
        ):
            result = await runner._handle_model_command(_make_event())

        assert result is not None
        assert len(call_thread_ids) == 1
        thread_id_result = call_thread_ids[0]
        assert isinstance(thread_id_result, RuntimeError), (
            "list_authenticated_providers was called directly on the event loop "
            "(not via to_thread) — this blocks the loop. Fix: wrap in "
            "asyncio.to_thread()."
        )

    @pytest.mark.asyncio
    async def test_event_loop_remains_responsive_during_model_command(
        self, tmp_path, monkeypatch
    ):
        """While _handle_model_command is running, other async tasks should
        still be able to execute (event loop not blocked).

        This is the user-visible symptom: Discord heartbeats and incoming
        messages are delayed because the loop is frozen.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        import yaml

        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({
                "model": {"default": "test-model", "provider": "openrouter"},
            }),
            encoding="utf-8",
        )

        import gateway.run as gateway_run

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

        runner = _make_runner()
        runner.adapters = {}  # text fallback path

        import time

        def blocking_list_authenticated(**kwargs):
            """Simulates 200ms of blocking network I/O."""
            time.sleep(0.2)
            return []

        other_task_ran = asyncio.Event()

        async def concurrent_task():
            """This should be able to run while model command is executing."""
            other_task_ran.set()

        with patch(
            "hermes_cli.model_switch.list_authenticated_providers",
            side_effect=blocking_list_authenticated,
        ):
            # Start the model command
            model_task = asyncio.ensure_future(
                runner._handle_model_command(_make_event())
            )
            # Give the event loop a chance to schedule
            await asyncio.sleep(0.01)
            # Start the concurrent task
            concurrent = asyncio.ensure_future(concurrent_task())
            # Wait a short time for the concurrent task to complete
            # If the event loop is blocked, this will timeout
            try:
                await asyncio.wait_for(other_task_ran.wait(), timeout=0.15)
            except asyncio.TimeoutError:
                pytest.fail(
                    "Event loop was blocked during _handle_model_command — "
                    "concurrent tasks could not run. The synchronous call to "
                    "list_authenticated_providers must be wrapped in "
                    "asyncio.to_thread()."
                )

            # Clean up
            await model_task
