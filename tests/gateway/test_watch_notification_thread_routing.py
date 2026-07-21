"""Regression tests for watch_match notification thread routing (issue #10411).

When a background process is started from a Discord thread, the watch_match
notification should be routed back to that thread.  The session store origin
may lack thread_id (if the session was first created from the parent channel),
but the event dict carries thread_id from the watcher metadata.

``_build_process_event_source`` must use the event's thread_id when the
origin lacks one.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_runner(monkeypatch, tmp_path) -> GatewayRunner:
    """Create a GatewayRunner with a fake config."""
    import gateway.run as gateway_run

    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: all\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter
    return runner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWatchNotificationThreadRouting:
    """Verify watch notifications route to the correct thread when the
    session store origin lacks thread_id."""

    @pytest.mark.asyncio
    async def test_event_thread_id_used_when_origin_lacks_it(self, monkeypatch, tmp_path):
        """When origin has no thread_id but the event does, use the event's."""
        runner = _build_runner(monkeypatch, tmp_path)
        adapter = runner.adapters[Platform.DISCORD]

        # Session was first created from the parent channel (no thread_id)
        runner.session_store._entries["agent:main:discord:group:123:456"] = SimpleNamespace(
            origin=SessionSource(
                platform=Platform.DISCORD,
                chat_id="123",
                chat_type="group",
                thread_id=None,  # <-- Missing thread_id
                user_id="789",
                user_name="Emiliyan",
            )
        )

        # Background process was started from thread 456
        evt = {
            "session_id": "proc_abc",
            "session_key": "agent:main:discord:group:123:456",
            "platform": "discord",
            "chat_id": "123",
            "thread_id": "456",  # <-- Correct thread from watcher metadata
            "user_id": "789",
            "user_name": "Emiliyan",
        }

        await runner._inject_watch_notification("[SYSTEM: Background process matched]", evt)

        adapter.handle_message.assert_awaited_once()
        synth_event = adapter.handle_message.await_args.args[0]
        assert synth_event.source.thread_id == "456"

    @pytest.mark.asyncio
    async def test_origin_thread_id_preserved_when_present(self, monkeypatch, tmp_path):
        """When origin already has thread_id, it should be used (not overridden)."""
        runner = _build_runner(monkeypatch, tmp_path)
        adapter = runner.adapters[Platform.DISCORD]

        runner.session_store._entries["agent:main:discord:group:123:100"] = SimpleNamespace(
            origin=SessionSource(
                platform=Platform.DISCORD,
                chat_id="123",
                chat_type="group",
                thread_id="100",  # <-- Already has thread_id
                user_id="789",
                user_name="Emiliyan",
            )
        )

        evt = {
            "session_id": "proc_abc",
            "session_key": "agent:main:discord:group:123:100",
            "platform": "discord",
            "chat_id": "123",
            "thread_id": "999",  # <-- Different thread in event
            "user_id": "789",
            "user_name": "Emiliyan",
        }

        await runner._inject_watch_notification("[SYSTEM: Background process matched]", evt)

        adapter.handle_message.assert_awaited_once()
        synth_event = adapter.handle_message.await_args.args[0]
        # Origin's thread_id takes precedence
        assert synth_event.source.thread_id == "100"

    def test_build_source_patches_thread_id_from_event(self, monkeypatch, tmp_path):
        """_build_process_event_source patches thread_id from event when origin lacks it."""
        runner = _build_runner(monkeypatch, tmp_path)

        runner.session_store._entries["agent:main:discord:group:123:456"] = SimpleNamespace(
            origin=SessionSource(
                platform=Platform.DISCORD,
                chat_id="123",
                chat_type="group",
                thread_id=None,
                user_id="789",
                user_name="Emiliyan",
            )
        )

        evt = {
            "session_id": "proc_abc",
            "session_key": "agent:main:discord:group:123:456",
            "platform": "discord",
            "chat_id": "123",
            "thread_id": "456",
            "user_id": "789",
            "user_name": "Emiliyan",
        }

        source = runner._build_process_event_source(evt)

        assert source is not None
        assert source.thread_id == "456"
        assert source.chat_id == "123"
        assert source.platform == Platform.DISCORD

    def test_build_source_no_event_thread_returns_origin_as_is(self, monkeypatch, tmp_path):
        """When event also has no thread_id, origin is returned unchanged."""
        runner = _build_runner(monkeypatch, tmp_path)

        runner.session_store._entries["agent:main:discord:group:123"] = SimpleNamespace(
            origin=SessionSource(
                platform=Platform.DISCORD,
                chat_id="123",
                chat_type="group",
                thread_id=None,
                user_id="789",
                user_name="Emiliyan",
            )
        )

        evt = {
            "session_id": "proc_abc",
            "session_key": "agent:main:discord:group:123",
        }

        source = runner._build_process_event_source(evt)

        assert source is not None
        assert source.thread_id is None

    def test_build_source_no_origin_uses_event_thread_id(self, monkeypatch, tmp_path):
        """When no session store entry exists, event's thread_id is used directly."""
        runner = _build_runner(monkeypatch, tmp_path)

        evt = {
            "session_id": "proc_abc",
            "session_key": "agent:main:discord:group:123:456",
            "platform": "discord",
            "chat_id": "123",
            "thread_id": "456",
            "user_id": "789",
            "user_name": "Emiliyan",
        }

        source = runner._build_process_event_source(evt)

        assert source is not None
        assert source.thread_id == "456"

    def test_all_origin_fields_preserved_when_thread_patched(self, monkeypatch, tmp_path):
        """All 11 SessionSource fields must survive when thread_id is patched.

        Regression test: a manual SessionSource() constructor previously
        dropped chat_name, chat_topic, user_id_alt, chat_id_alt, and is_bot.
        Using dataclasses.replace() ensures all fields carry over.
        """
        runner = _build_runner(monkeypatch, tmp_path)

        origin = SessionSource(
            platform=Platform.DISCORD,
            chat_id="123",
            chat_name="general",
            chat_type="group",
            thread_id=None,
            user_id="789",
            user_name="Emiliyan",
            chat_topic="Project discussion",
            user_id_alt="alt-uuid-abc",
            chat_id_alt="alt-group-xyz",
            is_bot=False,
        )
        runner.session_store._entries["agent:main:discord:group:123:456"] = SimpleNamespace(
            origin=origin,
        )

        evt = {
            "session_id": "proc_abc",
            "session_key": "agent:main:discord:group:123:456",
            "thread_id": "456",
        }

        source = runner._build_process_event_source(evt)

        assert source is not None
        assert source.thread_id == "456"
        assert source.platform == Platform.DISCORD
        assert source.chat_id == "123"
        assert source.chat_name == "general"
        assert source.chat_type == "group"
        assert source.user_id == "789"
        assert source.user_name == "Emiliyan"
        assert source.chat_topic == "Project discussion"
        assert source.user_id_alt == "alt-uuid-abc"
        assert source.chat_id_alt == "alt-group-xyz"
        assert source.is_bot is False
