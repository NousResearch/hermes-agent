"""Tests for the /fresh gateway command: reset session + first prompt in one turn.

Uses stdlib unittest so it runs in installs without pytest.
"""

from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import EphemeralReply
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="group",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner._session_key_for_source = lambda source: build_session_key(source)
    # Called through asyncio.to_thread, so this must be a sync callable.
    runner._is_telegram_topic_root_lobby = MagicMock(return_value=False)
    runner._handle_reset_command = AsyncMock(return_value=EphemeralReply("new session"))
    runner._send_command_ack = AsyncMock()
    return runner


class FreshCommandTest(unittest.IsolatedAsyncioTestCase):

    async def test_fresh_resets_then_falls_through_with_query(self):
        seen_args = []
        seen_flags = []

        async def _record_reset(ev):
            seen_args.append(ev.get_command_args())
            seen_flags.append(getattr(ev, "_fresh_reset", False))
            return EphemeralReply("new session")

        runner = _make_runner()
        runner._handle_reset_command = AsyncMock(side_effect=_record_reset)
        event = _make_event("/fresh What is the capital of France?")

        handled, reply = await runner._hm_cmd_fresh(
            event, event.source, build_session_key(event.source)
        )

        self.assertEqual((handled, reply), (False, None))
        self.assertEqual(event.text, "What is the capital of France?")
        self.assertTrue(getattr(event, "_fresh_reset", False))
        runner._handle_reset_command.assert_awaited_once()
        # The title-consuming /new path must see no arguments at reset time, and the
        # reset must see the /fresh flag so it can pin the epoch it produces.
        self.assertEqual(seen_args, [""])
        self.assertEqual(seen_flags, [True])

    async def test_fresh_without_query_shows_usage(self):
        runner = _make_runner()
        event = _make_event("/fresh")

        handled, reply = await runner._hm_cmd_fresh(
            event, event.source, build_session_key(event.source)
        )

        self.assertTrue(handled)
        self.assertIn("/fresh", reply)
        runner._handle_reset_command.assert_not_awaited()

    async def test_fresh_turn_keeps_epoch_pinned_by_its_reset(self):
        runner = _make_runner()
        event = _make_event("/fresh hello")
        key = build_session_key(event.source)

        async def _canonical(_event, _source, _key, canonical):
            self.assertEqual(canonical, "fresh")
            event.text = "hello"
            event._fresh_reset = True
            # The reset pinned epoch 2; a later /stop bumps the route to 3.
            event._fresh_stop_epoch = 2
            return False, None

        async def _run_in_executor(fn, *args):
            return fn(*args)

        runner._hm_resolve_command = AsyncMock(return_value=(False, None, "fresh", "fresh"))
        runner._hm_dispatch_canonical_command = _canonical
        runner._hm_dispatch_quick_and_plugin_commands = AsyncMock(return_value=(False, None, "fresh"))
        runner._run_in_executor_with_context = _run_in_executor
        runner._hm_skill_slash_rewrite = MagicMock(return_value=None)
        runner._route_stop_epoch = MagicMock(return_value=3)
        runner._admit_command_turn = AsyncMock(return_value=(True, "ok"))

        handled, reply = await runner._hm_dispatch_idle_commands(event, event.source, key)

        self.assertEqual((handled, reply), (True, "ok"))
        # Only the entry-time read happens; the later epoch must not be adopted, so a
        # /stop after the reset still fences this turn out.
        self.assertEqual(runner._route_stop_epoch.call_count, 1)
        self.assertEqual(runner._admit_command_turn.await_args.args[3], 2)

    async def test_fresh_without_stamped_epoch_is_rejected(self):
        runner = _make_runner()
        event = _make_event("/fresh hello")
        key = build_session_key(event.source)

        async def _canonical(_event, _source, _key, canonical):
            event.text = "hello"
            event._fresh_reset = True  # reset happened, but no epoch was stamped
            return False, None

        async def _run_in_executor(fn, *args):
            return fn(*args)

        runner._hm_resolve_command = AsyncMock(return_value=(False, None, "fresh", "fresh"))
        runner._hm_dispatch_canonical_command = _canonical
        runner._hm_dispatch_quick_and_plugin_commands = AsyncMock(return_value=(False, None, "fresh"))
        runner._run_in_executor_with_context = _run_in_executor
        runner._hm_skill_slash_rewrite = MagicMock(return_value=None)
        runner._admit_command_turn = AsyncMock()

        handled, reply = await runner._hm_dispatch_idle_commands(event, event.source, key)

        self.assertTrue(handled)
        self.assertIn("resend", reply)
        runner._admit_command_turn.assert_not_awaited()

    def test_fresh_registered_as_gateway_command(self):
        from hermes_cli.commands import COMMAND_REGISTRY, is_gateway_known_command

        fresh = next(c for c in COMMAND_REGISTRY if c.name == "fresh")
        self.assertTrue(fresh.gateway_only)
        self.assertEqual(fresh.args_hint, "<prompt>")
        self.assertTrue(is_gateway_known_command("fresh"))


if __name__ == "__main__":
    unittest.main()
