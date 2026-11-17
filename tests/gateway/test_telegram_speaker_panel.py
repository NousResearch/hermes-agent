"""Contract tests for the Telegram-native Kitchen Speaker panel."""

import sys
import asyncio
import hashlib
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from hermes_constants import get_hermes_home
from plugins.platforms.telegram import adapter as telegram_adapter_module
from plugins.platforms.telegram.adapter import (
    InlineKeyboardMarkup,
    TELEGRAM_AVAILABLE,
    TelegramAdapter,
    _podcast_reader_command_argv,
    _speaker_button_rows,
    _speaker_command_argv,
)
from integrations import sonos_fm_scheduler
from integrations.sonos_fm_scheduler import _wrapper_text, parse_schedule_spec




class _FakeInlineButton:
    def __init__(self, text, callback_data=None, **kwargs):
        self.text = text
        self.callback_data = callback_data


class _FakeInlineMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard


class TelegramSpeakerPanelTests(unittest.TestCase):
    def test_fm_schedule_form_parses_time_duration_and_days(self):
        self.assertEqual(
            parse_schedule_spec("08:05 1.5h weekdays"),
            {
                "time": "08:05",
                "duration": "90m",
                "days": "weekdays",
                "cron": "5 8 * * 1-5",
            },
        )
        with self.assertRaises(ValueError):
            parse_schedule_spec("25:00 1h daily")

    def test_fm_schedule_list_does_not_claim_success_when_cron_is_unavailable(self):
        record = {"id": "deadbeef", "job_id": "job-42", "enabled": True}
        with patch.object(sonos_fm_scheduler, "_read_manifest", return_value=[record]), patch.object(
            sonos_fm_scheduler, "_cron_call", return_value={"success": False, "error": "scheduler unavailable"}
        ):
            result = sonos_fm_scheduler.list_fm_schedules()
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "scheduler unavailable")
        self.assertNotIn("schedules", result)

    def test_fm_schedule_list_rejects_malformed_job_list(self):
        with patch.object(sonos_fm_scheduler, "_read_manifest", return_value=[]), patch.object(
            sonos_fm_scheduler, "_cron_call", return_value={"success": True}
        ):
            result = sonos_fm_scheduler.list_fm_schedules()
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "scheduler returned an invalid job list")

    def test_fm_schedule_list_does_not_claim_manifest_job_when_scheduler_omits_it(self):
        record = {"id": "deadbeef", "job_id": "job-42", "enabled": True}
        with patch.object(sonos_fm_scheduler, "_read_manifest", return_value=[record]), patch.object(
            sonos_fm_scheduler, "_cron_call", return_value={"success": True, "jobs": []}
        ):
            result = sonos_fm_scheduler.list_fm_schedules()
        self.assertFalse(result["success"])
        self.assertIn("missing", result["error"].lower())
        self.assertNotIn("schedules", result)

    def test_fm_schedule_wrapper_does_not_echo_command_output(self):
        generated = _wrapper_text("60m")
        self.assertIn("result.returncode", generated)
        self.assertNotIn("result.stderr", generated)
        self.assertNotIn("result.stdout", generated)

    def test_fm_schedule_detail_does_not_fall_back_to_stale_manifest_state(self):
        record = {"id": "deadbeef", "job_id": "job-42", "enabled": True}
        with patch.object(sonos_fm_scheduler, "_read_manifest", return_value=[record]), patch.object(
            sonos_fm_scheduler, "_cron_call", return_value={"success": False, "error": "scheduler unavailable"}
        ):
            result = sonos_fm_scheduler.get_fm_schedule("deadbeef")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "scheduler unavailable")
        self.assertNotIn("schedule", result)

    def test_manual_fm_schedule_run_validates_id_and_uses_scheduler_run_action(self):
        with patch.object(sonos_fm_scheduler, "_find", return_value=([], {"job_id": "job-42"})) as find, patch.object(
            sonos_fm_scheduler, "_cron_call", return_value={"success": True}
        ) as cron_call:
            result = sonos_fm_scheduler.run_fm_schedule("deadbeef")
            self.assertTrue(result["success"])
            cron_call.assert_called_once_with("run", job_id="job-42")
            find.assert_called_once_with("deadbeef")

            find.reset_mock()
            cron_call.reset_mock()
            result = sonos_fm_scheduler.run_fm_schedule("not-an-id")

        self.assertFalse(result["success"])
        find.assert_not_called()
        cron_call.assert_not_called()

    def test_fm_schedule_wrapper_path_cannot_escape_schedule_directory(self):
        record = {
            "id": "deadbeef",
            "script": "sonos-fm-schedules/../../../../.ssh/id_rsa",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            schedule_dir = root / "sonos-fm-schedules"
            with patch.object(sonos_fm_scheduler, "SCRIPTS_DIR", root), patch.object(
                sonos_fm_scheduler, "SCHEDULE_DIR", schedule_dir
            ):
                resolved = sonos_fm_scheduler._wrapper_path(record).resolve()
                self.assertTrue(resolved.is_relative_to(schedule_dir.resolve()))

    def test_fm_schedule_keyboards_are_allowlisted_and_compact(self):
        adapter = object.__new__(TelegramAdapter)
        schedule_markup = adapter._speaker_schedule_keyboard()
        detail_markup = adapter._speaker_schedule_detail_keyboard({
            "id": "deadbeef",
            "enabled": True,
        })
        callbacks = [
            button.callback_data
            for markup in (schedule_markup, detail_markup)
            for row in getattr(markup, "inline_keyboard", [])
            for button in row
        ]
        if not callbacks:
            self.skipTest("Telegram SDK is mocked in this test environment")
        self.assertIn("sp:schedule:list", callbacks)
        self.assertIn("sp:schedule:add", callbacks)
        self.assertIn("sp:schedule:job:deadbeef:run", callbacks)
        self.assertIn("sp:schedule:job:deadbeef:edit", callbacks)
        self.assertIn("sp:schedule:job:deadbeef:delete", callbacks)
        self.assertTrue(all(len(callback.encode("utf-8")) <= 64 for callback in callbacks))

    def test_schedule_add_callback_opens_bound_reply_form(self):
        events = []

        class QueryMessage:
            async def reply_text(self, text):
                events.append(text)
                return type("Prompt", (), {"message_id": 123})()

        class Query:
            from_user = type("User", (), {"id": 1})()
            message = QueryMessage()

            async def answer(self):
                events.append("ack")

            async def edit_message_text(self, text, **kwargs):
                events.append(text)

        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                Query(), "sp:schedule:add", chat_id="7", chat_type="private", thread_id=None, user_name="Oleg"
            ))

        self.assertEqual(adapter._speaker_form_state["7:1"]["action"], "schedule-add")
        self.assertEqual(adapter._speaker_form_state["7:1"]["prompt_message_id"], 123)
        self.assertIn("HH:MM duration days", events[1])

    def test_schedule_list_callback_uses_read_only_manager_and_renders_allowlisted_rows(self):
        events = []

        class Query:
            from_user = type("User", (), {"id": 1})()
            message = object()

            async def answer(self):
                events.append("answer")

            async def edit_message_text(self, text, **kwargs):
                events.append((text, kwargs.get("reply_markup")))

        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        manager = AsyncMock(return_value={"success": True, "schedules": [
            {"id": "deadbeef", "time": "08:05", "duration": "90m", "days": "weekdays", "enabled": True},
        ]})
        adapter._run_fm_schedule_manager = manager
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                Query(), "sp:schedule:list", chat_id="1", chat_type="private", thread_id=None, user_name="Oleg"
            ))

        manager.assert_awaited_once_with("list")
        self.assertIn("08:05", events[-1][0])
        callbacks = [button.callback_data for row in events[-1][1].inline_keyboard for button in row]
        self.assertIn("sp:schedule:job:deadbeef:detail", callbacks)

    def test_schedule_delete_callback_requires_a_second_confirmation(self):
        events = []

        class Query:
            from_user = type("User", (), {"id": 1})()
            message = object()

            async def answer(self):
                return None

            async def edit_message_text(self, text, **kwargs):
                events.append((text, kwargs.get("reply_markup")))

        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        manager = AsyncMock()
        adapter._run_fm_schedule_manager = manager
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                Query(), "sp:schedule:job:deadbeef:delete", chat_id="1", chat_type="private", thread_id=None, user_name="Oleg"
            ))

        manager.assert_not_awaited()
        callbacks = [button.callback_data for row in events[-1][1].inline_keyboard for button in row]
        self.assertIn("sp:schedule:job:deadbeef:confirm-delete", callbacks)
        self.assertIn("sp:schedule:job:deadbeef:cancel", callbacks)

    def test_fm_center_callbacks_do_not_dispatch_unverified_speaker_actions(self):
        rendered = []

        class Query:
            from_user = type("User", (), {"id": 1})()
            message = object()

            async def answer(self):
                return None

            async def edit_message_text(self, text, **kwargs):
                rendered.append(text)

        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        run_speaker = AsyncMock()
        adapter._run_speaker_subprocess = run_speaker
        asyncio.run(adapter._handle_speaker_callback(
            Query(), "sp:fmcenter:mic", chat_id="1", chat_type="private", thread_id=None, user_name="Oleg"
        ))

        self.assertIn("unavailable", rendered[0].lower())
        self.assertIn("no speaker action was sent", rendered[0].lower())
        run_speaker.assert_not_awaited()

    def test_speaker_button_rows_are_mobile_friendly_and_safe(self):
        rows = _speaker_button_rows()
        callbacks = [callback for row in rows for _, callback in row]

        self.assertIn("sp:status", callbacks)
        self.assertIn("sp:stop", callbacks)
        self.assertIn("sp:fmcenter", callbacks)
        self.assertIn("sp:podcast", callbacks)
        self.assertIn("sp:schedule", callbacks)
        self.assertIn("sp:news", callbacks)
        self.assertIn("sp:volume", callbacks)
        self.assertIn("sp:npc", callbacks)
        self.assertIn("sp:haunted", callbacks)
        self.assertIn("sp:roulette", callbacks)
        self.assertIn("sp:sleep", callbacks)
        self.assertTrue(all(len(callback.encode("utf-8")) <= 64 for callback in callbacks))
        self.assertTrue(all(1 <= len(row) <= 2 for row in rows))
        self.assertIn("sp:sources", callbacks)

    def test_hourly_news_settings_card_exposes_interrupt_and_idle_modes(self):
        adapter = object.__new__(TelegramAdapter)
        text, markup = adapter._format_hourly_news_settings(json.dumps({
            "success": True,
            "settings": {
                "enabled": True,
                "station": "france24",
                "mode": "interrupt",
                "duration_minutes": 10,
            },
            "stations": {
                "france24": "France 24 English",
                "dw": "DW News",
                "bbc": "BBC News (when live)",
            },
        }))
        self.assertIn("interrupt current audio", text)
        callbacks = [
            button.callback_data
            for row in getattr(markup, "inline_keyboard", [])
            for button in row
        ]
        if not callbacks:
            self.skipTest("Telegram SDK is mocked in this test environment")
        self.assertIn("sp:news:mode:interrupt", callbacks)
        self.assertIn("sp:news:mode:idle", callbacks)
        self.assertIn("sp:news:station:bbc", callbacks)

    def test_hourly_news_settings_card_exposes_kill_switch_duration_and_post_news_radio(self):
        adapter = object.__new__(TelegramAdapter)
        payload = {
            "success": True,
            "settings": {
                "enabled": True,
                "station": "bbc",
                "mode": "interrupt",
                "duration_minutes": 15,
                "after_news_action": "radio",
                "after_news_station_id": "w_9jYX5B",
            },
            "stations": {"bbc": "BBC News (when live)"},
            "radio_stations": [{"id": "w_9jYX5B", "name": "WFMU"}],
        }
        with patch.object(telegram_adapter_module, "InlineKeyboardButton", _FakeInlineButton), patch.object(
            telegram_adapter_module, "InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            text, markup = adapter._format_hourly_news_settings(json.dumps(payload))
        callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
        self.assertIn("News: <b>ON</b>", text)
        self.assertIn("15 minutes", text)
        self.assertIn("WFMU", text)
        self.assertIn("sp:news:enabled:off", callbacks)
        self.assertIn("sp:news:kill", callbacks)
        self.assertIn("sp:news:play", callbacks)
        self.assertIn("sp:news:duration:5", callbacks)
        self.assertIn("sp:news:duration:15", callbacks)
        self.assertIn("sp:news:duration:30", callbacks)
        self.assertIn("sp:news:after:radio", callbacks)
        self.assertIn("sp:news:after:resume", callbacks)
        self.assertIn("sp:news:after:idle", callbacks)
        self.assertIn("sp:news:after-station:w_9jYX5B", callbacks)

    def test_hourly_news_settings_card_exposes_enable_when_disabled(self):
        adapter = object.__new__(TelegramAdapter)
        with patch.object(telegram_adapter_module, "InlineKeyboardButton", _FakeInlineButton), patch.object(
            telegram_adapter_module, "InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            _, markup = adapter._format_hourly_news_settings(json.dumps({
                "success": True,
                "settings": {"enabled": False, "duration_minutes": 15, "after_news_action": "idle"},
                "stations": {},
                "radio_stations": [],
            }))
        callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
        self.assertIn("sp:news:enabled:on", callbacks)

    def test_direct_speaker_parser_accepts_only_allowlisted_operations(self):
        parse = telegram_adapter_module._parse_speaker_command
        self.assertEqual(parse(["volume"]), ("volume", {}))
        self.assertEqual(parse(["volume", "up"]), ("volume", {"volume_action": "up"}))
        self.assertEqual(parse(["volume", "75"]), ("volume", {"volume_level": "75"}))
        self.assertEqual(parse(["fm", "1h"]), ("fm", {"duration": "1h", "switch": False}))
        with self.assertRaises(ValueError):
            parse(["volume", "101"])
        with self.assertRaises(ValueError):
            parse(["volume", "up", "now"])
        with self.assertRaises(ValueError):
            parse(["start", "--shell"])

    def test_direct_speaker_parser_supports_advertised_arguments(self):
        parse = telegram_adapter_module._parse_speaker_command
        self.assertEqual(parse(["schedule"]), ("panel", {}))
        self.assertEqual(parse(["npc", "Use", "Normal", "Case"]), ("npc", {"text": "Use Normal Case"}))
        self.assertEqual(parse(["roulette", "longest-title"]), ("roulette", {"rule": "longest-title"}))
        self.assertEqual(parse(["oracle", "Launch", "details here"]),
                         ("oracle", {"oracle_event": "Launch", "oracle_detail": "details here"}))

    def test_direct_speaker_message_does_not_claim_success_for_failed_backend(self):
        adapter = object.__new__(TelegramAdapter)
        message = SimpleNamespace(text="/speaker start", reply_text=AsyncMock())
        update = SimpleNamespace(message=message)
        adapter._should_process_message = lambda *args, **kwargs: True
        adapter._ensure_forum_commands = AsyncMock()
        adapter._run_speaker_subprocess = AsyncMock(return_value='{"success":false,"error":"secret token"}')
        with patch.object(TelegramAdapter, "_effective_update_message", return_value=message), patch.object(
            TelegramAdapter, "_is_user_authorized_from_message", return_value=True
        ):
            asyncio.run(adapter._handle_command(update, object()))
        response = message.reply_text.await_args.args[0]
        self.assertIn("failed", response.lower())
        self.assertNotIn("secret token", response)

    def test_fm_custom_form_reply_is_consumed_without_agent_dispatch(self):
        from types import SimpleNamespace

        adapter = object.__new__(TelegramAdapter)
        message = SimpleNamespace(
            text="1h", chat_id="42", from_user=SimpleNamespace(id=7),
            reply_to_message=SimpleNamespace(message_id=99), reply_text=AsyncMock(),
        )
        update = SimpleNamespace(message=message)
        run_speaker = AsyncMock(return_value="FM started")
        adapter._run_speaker_subprocess = run_speaker
        adapter._speaker_form_state = {
            "42:7": {"action": "fm", "prompt_message_id": 99, "expires_at": 9999999999},
        }
        adapter._ensure_forum_commands = AsyncMock()
        enqueue = AsyncMock()
        adapter._enqueue_text_event = enqueue
        with patch.object(TelegramAdapter, "_effective_update_message", return_value=message), patch.object(
            TelegramAdapter, "_is_user_authorized_from_message", return_value=True
        ), patch.object(TelegramAdapter, "_gate_or_observe", return_value=True):
            asyncio.run(adapter._handle_text_message(update, object()))

        run_speaker.assert_awaited_once_with("fm", duration="1h", switch=False)
        message.reply_text.assert_awaited_once()
        enqueue.assert_not_awaited()

    def test_expired_speaker_form_does_not_consume_an_unrelated_message(self):
        from types import SimpleNamespace

        adapter = object.__new__(TelegramAdapter)
        reply_text = AsyncMock()
        message = SimpleNamespace(
            text="Hi there", chat_id="42", from_user=SimpleNamespace(id=7),
            reply_to_message=SimpleNamespace(message_id=98), reply_text=reply_text,
        )
        adapter._speaker_form_state = {
            "42:7": {"action": "fm", "prompt_message_id": 99, "expires_at": 999},
        }

        with patch("plugins.platforms.telegram.adapter.time.monotonic", return_value=1000):
            consumed = asyncio.run(adapter._consume_speaker_form_reply(message))

        self.assertFalse(consumed)
        reply_text.assert_not_awaited()

    def test_direct_speaker_message_routes_to_allowlisted_subprocess_not_agent(self):
        adapter = object.__new__(TelegramAdapter)
        message = SimpleNamespace(text="/speaker volume up", reply_text=AsyncMock())
        update = SimpleNamespace(message=message)
        adapter._should_process_message = lambda *args, **kwargs: True
        adapter._ensure_forum_commands = AsyncMock()
        run_speaker = AsyncMock(return_value=json.dumps({"volume": 64, "muted": False}))
        adapter._run_speaker_subprocess = run_speaker
        with patch.object(TelegramAdapter, "_effective_update_message", return_value=message), patch.object(
            TelegramAdapter, "_is_user_authorized_from_message", return_value=True
        ):
            asyncio.run(adapter._handle_command(update, object()))

        run_speaker.assert_awaited_once_with("volume", volume_action="up")
        message.reply_text.assert_awaited_once()

    def test_speaker_command_builder_uses_exec_arguments_and_overlay(self):
        command = _speaker_command_argv("haunted", text="booo killing you")

        self.assertEqual(command[:2], [sys.executable, str(get_hermes_home() / "sonos-brain" / "sonos_brain.py")])
        self.assertEqual(command[2:], [
            "haunted",
            "--overlay",
            "--duck-by",
            "18",
            "--no-wait",
            "--text",
            "booo killing you",
        ])
        self.assertNotIn("&&", command)
        self.assertNotIn(";", command)

        npc_command = _speaker_command_argv("npc", text="radio voice")
        self.assertIn("--overlay", npc_command)
        self.assertIn("--duck-by", npc_command)

    def test_fm_command_requires_and_preserves_airtime(self):
        with self.assertRaises(ValueError):
            _speaker_command_argv("fm")
        command = _speaker_command_argv("fm", duration="1.5h")
        self.assertEqual(command[2:5], ["fm", "--duration", "1.5h"])

    def test_fm_center_keyboard_exposes_dj_controls(self):
        adapter = object.__new__(TelegramAdapter)
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            markup = adapter._speaker_fm_center_keyboard()
        callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
        self.assertEqual(
            set(callbacks),
            {
                "sp:fmcenter:onair",
                "sp:fmcenter:mic",
                "sp:fmcenter:playlist",
                "sp:fmcenter:jingles",
                "sp:fmcenter:next",
                "sp:fmcenter:pause",
                "sp:fmcenter:end",
                "sp:menu",
            },
        )

    def test_next_and_clip_commands_are_argv_only(self):
        self.assertEqual(_speaker_command_argv("next")[2:], ["next"])
        with tempfile.NamedTemporaryFile(suffix=".ogg") as clip:
            self.assertEqual(
                _speaker_command_argv("clip", clip_path=clip.name)[2:],
                ["clip", clip.name, "--duck-by", "18"],
            )

        adapter = object.__new__(TelegramAdapter)
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            markup = adapter._speaker_fm_duration_keyboard()
        rows = markup.inline_keyboard
        buttons = [button for row in rows for button in row]
        callbacks = [button.callback_data for button in buttons]

        self.assertEqual(
            set(callbacks),
            {"sp:fm:30m", "sp:fm:1h", "sp:fm:1.5h", "sp:fm:2h", "sp:fm:custom", "sp:menu"},
        )
        self.assertTrue(all(len(callback.encode("utf-8")) <= 64 for callback in callbacks))
        self.assertTrue(all(1 <= len(row) <= 2 for row in rows))

    def test_sleep_timer_command_builder_and_keyboard_are_allowlisted(self):
        command = _speaker_command_argv("sleep_timer", sleep_timer_hours="2h")
        self.assertEqual(command[2:], ["sleep-timer", "--hours", "2", "--no-wait"])
        self.assertEqual(
            _speaker_command_argv("sleep_timer", sleep_timer_hours="off")[2:],
            ["sleep-timer", "--off"],
        )
        command = _speaker_command_argv("sleep_timer", sleep_timer_hours="3h")
        self.assertEqual(command[2:], ["sleep-timer", "--hours", "3", "--no-wait"])
        with self.assertRaisesRegex(ValueError, "Sleep timer"):
            _speaker_command_argv("sleep_timer", sleep_timer_hours="5h")

        adapter = object.__new__(TelegramAdapter)
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            markup = adapter._speaker_sleep_timer_keyboard()
        callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
        self.assertEqual(
            set(callbacks),
            {
                "sp:sleep:off", "sp:sleep:1h", "sp:sleep:2h", "sp:sleep:3h",
                "sp:sleep:4h", "sp:sleep:6h", "sp:sleep:8h", "sp:sleep:12h", "sp:menu",
            },
        )
        self.assertTrue(all(len(callback.encode("utf-8")) <= 64 for callback in callbacks))
        self.assertTrue(all(1 <= len(row) <= 2 for row in markup.inline_keyboard))

    def test_fm_callback_opens_duration_buttons(self):
        events = []

        class Query:
            from_user = type("User", (), {"id": 1})()
            message = object()

            async def answer(self):
                events.append("answer")

            async def edit_message_text(self, text, **kwargs):
                events.append((text, kwargs.get("reply_markup")))

        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                Query(), "sp:fm", chat_id="1", chat_type="private", thread_id=None, user_name="Oleg"
            ))

        self.assertEqual(events[0], "answer")
        self.assertEqual(events[1][1].inline_keyboard[0][0].callback_data, "sp:fm:30m")

    def test_fm_duration_callback_dispatches_only_selected_allowlisted_duration(self):
        events = []

        class Query:
            from_user = type("User", (), {"id": 1})()
            message = object()

            async def answer(self):
                events.append("answer")

            async def edit_message_text(self, text, **kwargs):
                events.append((text, kwargs.get("reply_markup")))

        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True

        async def run_speaker(action, **kwargs):
            events.append((action, kwargs))
            return "✅ <b>FM</b> accepted."

        adapter._run_speaker_subprocess = run_speaker
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                Query(), "sp:fm:1.5h", chat_id="1", chat_type="private", thread_id=None, user_name="Oleg"
            ))

        self.assertIn(("fm", {"duration": "1.5h", "switch": False}), events)
        self.assertIn("FM · 1.5h", events[-1][0])

    def test_fm_custom_button_retains_reply_form_flow(self):
        events = []

        class Message:
            chat = type("Chat", (), {"id": 42})()

            async def reply_text(self, text, **kwargs):
                events.append((text, kwargs))
                return type("Prompt", (), {"message_id": 99})()

        class Query:
            from_user = type("User", (), {"id": 7})()
            message = Message()

            async def answer(self):
                events.append("answer")

            async def edit_message_text(self, text, **kwargs):
                events.append((text, kwargs))

        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        adapter._speaker_form_state = {}
        asyncio.run(adapter._handle_speaker_callback(
            Query(), "sp:fm:custom", chat_id="42", chat_type="private", thread_id=None, user_name="Oleg"
        ))

        self.assertIn("42:7", adapter._speaker_form_state)
        self.assertEqual(adapter._speaker_form_state["42:7"]["action"], "fm")
        self.assertEqual(adapter._speaker_form_state["42:7"]["prompt_message_id"], 99)

    def test_podcast_command_maps_to_brief_and_idle_guard(self):
        command = _speaker_command_argv("podcast")

        self.assertEqual(command[2:], [
            "podcast-brief",
            "--only-if-idle",
            "--no-wait",
        ])

    def test_podcast_reader_commands_use_short_allowlisted_ids(self):
        list_command = _podcast_reader_command_argv("list")
        self.assertEqual(list_command[2:], ["podcast-list", "--limit", "20"])
        play_command = _podcast_reader_command_argv("play", summary_id="deadbeef", regenerate=True)
        self.assertEqual(play_command[2:], [
            "podcast-readout", "deadbeef", "--only-if-idle", "--no-wait", "--regenerate",
        ])
        switch_command = _podcast_reader_command_argv("switch", summary_id="deadbeef")
        self.assertEqual(switch_command[2:], [
            "podcast-readout", "deadbeef", "--switch", "--no-wait",
        ])
        with self.assertRaises(ValueError):
            _podcast_reader_command_argv("play", summary_id="../../secret")

    def test_podcast_reader_picker_is_compact_and_paginated(self):
        adapter = object.__new__(TelegramAdapter)
        records = [
            {"id": f"{index:08x}", "podcast": "Show", "title": f"Episode {index}"}
            for index in range(11)
        ]
        text, parsed = adapter._format_podcast_picker(json.dumps(records), page=1)
        self.assertIn("11–11 of 11", text)
        self.assertEqual(len(parsed), 11)
        markup = adapter._podcast_picker_keyboard(parsed, page=1)
        callbacks = [
            button.callback_data
            for row in getattr(markup, "inline_keyboard", [])
            for button in row
        ]
        if not callbacks:
            self.skipTest("Telegram SDK is mocked in this test environment")
        self.assertIn("sp:pr:select:0000000a", callbacks)
        self.assertIn("sp:pr:list:0", callbacks)
        self.assertTrue(all(len(callback.encode("utf-8")) <= 64 for callback in callbacks))

    def test_podcast_detail_offers_explicit_switch_without_changing_safe_play(self):
        adapter = object.__new__(TelegramAdapter)
        markup = adapter._podcast_detail_keyboard("deadbeef")
        callbacks = [
            button.callback_data
            for row in getattr(markup, "inline_keyboard", [])
            for button in row
        ]
        if not callbacks:
            self.skipTest("Telegram SDK is mocked in this test environment")
        self.assertIn("sp:pr:play:deadbeef", callbacks)
        self.assertIn("sp:pr:switch:deadbeef", callbacks)
        self.assertIn("sp:pr:regen:deadbeef", callbacks)
        self.assertTrue(all(len(callback.encode("utf-8")) <= 64 for callback in callbacks))

        switched = adapter._format_podcast_reader_result(
            '{"success":true,"started":true,"switched":true,"provider":"edge","voice":"en-US-AriaNeural"}'
        )
        self.assertIn("Switched to the readout", switched)

    def test_podcast_reader_result_does_not_claim_success_for_failures(self):
        adapter = object.__new__(TelegramAdapter)
        failed = adapter._format_podcast_reader_result(
            '{"success":false,"error":"secret-token must not be shown"}'
        )
        malformed = adapter._format_podcast_reader_result("not json")
        self.assertEqual(failed, "Podcast reader request failed.")
        self.assertEqual(malformed, "Podcast reader response unavailable.")
        self.assertNotIn("secret-token", failed)

    def test_podcast_callback_acknowledges_before_reader_listing(self):
        events = []

        class Query:
            from_user = type("User", (), {"id": 1})()
            message = type("Message", (), {})()

            async def answer(self):
                events.append("answer")

            async def edit_message_text(self, *args, **kwargs):
                events.append("edit")

        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True

        async def run_reader(*args, **kwargs):
            events.append("reader")
            return "[]"

        adapter._run_podcast_reader_subprocess = run_reader
        asyncio.run(adapter._handle_speaker_callback(
            Query(),
            "sp:podcast",
            chat_id="1",
            chat_type="private",
            thread_id=None,
            user_name="Oleg",
        ))

        self.assertEqual(events, ["answer", "reader", "edit"])

    def test_stale_podcast_callback_aborts_without_subprocess(self):
        events = []

        class Query:
            from_user = type("User", (), {"id": 1})()
            message = None

            async def answer(self):
                events.append("answer")
                raise RuntimeError("Query is too old and response timeout expired")

        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True

        async def run_command(*args, **kwargs):
            events.append("subprocess")
            return "Podcast briefing started."

        adapter._run_speaker_subprocess = run_command
        asyncio.run(adapter._handle_speaker_callback(
            Query(),
            "sp:podcast",
            chat_id="1",
            chat_type="private",
            thread_id=None,
            user_name="Oleg",
        ))

        self.assertEqual(events, ["answer"])

    def test_speaker_command_builder_validates_roulette_rules(self):
        command = _speaker_command_argv("roulette", rule="longest-title")

        self.assertEqual(command[2:], [
            "roulette",
            "--rule",
            "longest-title",
            "--only-if-idle",
            "--no-wait",
        ])

        with self.assertRaisesRegex(ValueError, "roulette"):
            _speaker_command_argv("roulette", rule="run-shell")

    def test_speaker_command_builder_supports_safe_plex_search_and_play(self):
        search = _speaker_command_argv("plex-search", query="Björk / Vespertine", limit=10)
        self.assertEqual(search[2:], ["plex-search", "--query", "Björk / Vespertine", "--limit", "10"])
        play = _speaker_command_argv("plex-play", track_id="12345")
        self.assertEqual(play[2:], ["plex-play", "12345", "--no-wait"])
        with self.assertRaises(ValueError):
            _speaker_command_argv("plex-search", query="invalid\x01query")
        with self.assertRaises(ValueError):
            _speaker_command_argv("plex-play", track_id="12345; stop")

    def test_plex_search_results_are_escaped_and_only_expose_safe_metadata(self):
        adapter = object.__new__(TelegramAdapter)
        raw = json.dumps({"success": True, "tracks": [{
            "id": "12345", "title": "A < B", "artist": "Artist & Co", "album": "Record", "duration_ms": 123000,
        }]})
        text, records = adapter._format_plex_search_results(raw)
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            markup = adapter._speaker_plex_results_keyboard(records)
        callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
        self.assertIn("A &lt; B", text)
        self.assertIn("Artist &amp; Co", text)
        self.assertIn("sp:plex:play:12345", callbacks)
        self.assertNotIn("http", text.lower())
        self.assertNotIn("token", text.lower())

    def test_volume_command_builder_and_status_format(self):
        self.assertEqual(_speaker_command_argv("volume")[2:], ["volume", "status"])
        self.assertEqual(
            _speaker_command_argv("volume", volume_action="down")[2:],
            ["volume", "down"],
        )
        self.assertEqual(
            _speaker_command_argv("volume", volume_level="55")[2:],
            ["volume", "55"],
        )
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            _speaker_command_argv("volume", volume_level="101")

        adapter = object.__new__(TelegramAdapter)
        rendered = adapter._format_speaker_volume('{"volume":42,"muted":false}')
        self.assertIn("42/100", rendered)
        self.assertIn("Muted: <b>no</b>", rendered)

    def test_volume_formatter_reports_unknown_state_instead_of_assuming_unmuted(self):
        adapter = object.__new__(TelegramAdapter)
        rendered = adapter._format_speaker_volume("not-json")
        self.assertIn("Volume: <b>unknown</b>", rendered)
        self.assertIn("Muted: <b>unknown</b>", rendered)

    def test_volume_submenu_reads_status_and_dispatches_allowlisted_controls(self):
        events = []

        class Query:
            from_user = type("User", (), {"id": 1})()
            message = object()

            async def answer(self):
                events.append("answer")

            async def edit_message_text(self, text, **kwargs):
                events.append(("edit", text, kwargs.get("reply_markup")))

        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True

        async def run_speaker(action, **kwargs):
            events.append(("speaker", action, kwargs))
            return '{"volume":42,"muted":false}'

        adapter._run_speaker_subprocess = run_speaker
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                Query(), "sp:volume", chat_id="1", chat_type="private", thread_id=None, user_name="Oleg"
            ))
            asyncio.run(adapter._handle_speaker_callback(
                Query(), "sp:vol:up", chat_id="1", chat_type="private", thread_id=None, user_name="Oleg"
            ))

        self.assertIn(("speaker", "volume", {}), events)
        self.assertIn(("speaker", "volume", {"volume_action": "up"}), events)

    def test_hourly_news_callbacks_update_settings_without_speaker_playback(self):
        events = []

        class Query:
            from_user = type("User", (), {"id": 1})()
            message = object()

            async def answer(self):
                events.append("answer")

            async def edit_message_text(self, text, **kwargs):
                events.append(("edit", text))

        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True

        async def run_news(action, **kwargs):
            events.append((action, kwargs))
            return json.dumps({
                "success": True,
                "settings": {"enabled": True, "station": "france24", "mode": "idle", "duration_minutes": 10},
                "stations": {"france24": "France 24 English", "dw": "DW News", "bbc": "BBC News"},
            })

        async def forbidden_speaker(*_args, **_kwargs):
            raise AssertionError("news settings must not call the playback controller")

        adapter._run_hourly_news_manager = run_news
        adapter._run_speaker_subprocess = forbidden_speaker
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                Query(), "sp:news:mode:idle", chat_id="1", chat_type="private", thread_id=None, user_name="Oleg"
            ))

        self.assertIn(("set_mode", {"mode": "idle"}), events)
        self.assertGreaterEqual(sum(isinstance(item, tuple) and item[0] == "status" for item in events), 1)

    def test_markup_builds_and_status_is_compact_and_redacted(self):
        adapter = object.__new__(TelegramAdapter)
        rows = _speaker_button_rows()
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(rows[0]), 2)
        self.assertEqual(rows[0][0][1], "sp:status")

        if TELEGRAM_AVAILABLE and isinstance(InlineKeyboardMarkup, type):
            markup = adapter._speaker_panel_keyboard()
            self.assertEqual(len(markup.inline_keyboard), 7)
            self.assertEqual(len(markup.inline_keyboard[0]), 2)
            self.assertEqual(markup.inline_keyboard[0][0].callback_data, "sp:status")

        status = adapter._format_speaker_status(
            '{"transport":"STOPPED","media":{'
            '"current_uri":"http://plex/library/file.flac?X-Plex-Token=secret",'
            '"track_count":"1","play_medium":"NETWORK"}}'
        )
        self.assertIn("State: <code>STOPPED</code>", status)
        self.assertIn("Track loaded: <b>yes</b>", status)
        self.assertNotIn("X-Plex-Token", status)
        self.assertNotIn("plex/library", status)

    def test_fm_schedule_wrapper_is_idle_safe_and_argument_based(self):
        wrapper = _wrapper_text("90m")
        self.assertIn('"fm", "--duration", DURATION', wrapper)
        self.assertIn('"--only-if-idle", "--no-wait"', wrapper)
        self.assertNotIn("shell=True", wrapper)

    def test_fm_schedule_creation_without_job_id_removes_the_created_job(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            scripts_dir = Path(temporary_directory) / "scripts"
            schedule_dir = scripts_dir / "sonos-fm-schedules"
            manifest_path = schedule_dir / "schedules.json"
            calls = []

            def fake_cron_call(action, **kwargs):
                calls.append((action, kwargs))
                if action == "create":
                    return {"success": True}
                if action == "list":
                    return {"success": True, "jobs": [{"job_id": "job-42", "script": calls[0][1]["script"]}]}
                return {"success": True}

            with patch.object(sonos_fm_scheduler, "SCRIPTS_DIR", scripts_dir), patch.object(
                sonos_fm_scheduler, "SCHEDULE_DIR", schedule_dir
            ), patch.object(sonos_fm_scheduler, "MANIFEST_PATH", manifest_path), patch.object(
                sonos_fm_scheduler, "_cron_call", side_effect=fake_cron_call
            ):
                result = sonos_fm_scheduler.create_fm_schedule("08:00 1h weekdays")

            self.assertFalse(result["success"])
            self.assertEqual([action for action, _ in calls], ["create", "list", "remove"])
            self.assertEqual(calls[-1][1]["job_id"], "job-42")
            self.assertFalse((scripts_dir / calls[0][1]["script"]).exists())
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), [])

    def test_fm_schedule_keeps_wrapper_when_orphan_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            scripts_dir = Path(temporary_directory) / "scripts"
            schedule_dir = scripts_dir / "sonos-fm-schedules"
            manifest_path = schedule_dir / "schedules.json"
            calls = []

            def fake_cron_call(action, **kwargs):
                calls.append((action, kwargs))
                if action == "create":
                    return {"success": True}
                if action == "list":
                    return {"success": True, "jobs": [{"job_id": "job-42", "script": calls[0][1]["script"]}]}
                return {"success": False, "error": "remove failed"}

            with patch.object(sonos_fm_scheduler, "SCRIPTS_DIR", scripts_dir), patch.object(
                sonos_fm_scheduler, "SCHEDULE_DIR", schedule_dir
            ), patch.object(sonos_fm_scheduler, "MANIFEST_PATH", manifest_path), patch.object(
                sonos_fm_scheduler, "_cron_call", side_effect=fake_cron_call
            ):
                result = sonos_fm_scheduler.create_fm_schedule("08:00 1h weekdays")
                wrapper = scripts_dir / calls[0][1]["script"]

            self.assertFalse(result["success"])
            self.assertIn("could not be confirmed", result["error"])
            self.assertTrue(wrapper.exists())

    def test_fm_schedule_create_uses_a_cron_allowed_relative_script(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            scripts_dir = Path(temporary_directory) / "scripts"
            schedule_dir = scripts_dir / "sonos-fm-schedules"
            manifest_path = schedule_dir / "schedules.json"
            calls = []

            def fake_cron_call(action, **kwargs):
                calls.append((action, kwargs))
                return {"success": True, "job_id": "job-1"}

            with patch.object(sonos_fm_scheduler, "SCRIPTS_DIR", scripts_dir), patch.object(
                sonos_fm_scheduler, "SCHEDULE_DIR", schedule_dir
            ), patch.object(sonos_fm_scheduler, "MANIFEST_PATH", manifest_path), patch.object(
                sonos_fm_scheduler, "_cron_call", side_effect=fake_cron_call
            ):
                result = sonos_fm_scheduler.create_fm_schedule("08:00 1h weekdays")

            script = calls[0][1]["script"]
            self.assertTrue((scripts_dir / script).exists())

        self.assertTrue(result["success"])
        self.assertEqual(calls[0][0], "create")
        self.assertTrue(script.startswith("sonos-fm-schedules/sonos-fm-"))



    def test_source_menu_separates_plex_sonos_and_radio(self):
        adapter = object.__new__(TelegramAdapter)
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            markup = adapter._speaker_source_keyboard()
        callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
        self.assertEqual(callbacks[:3], ["sp:source:plex", "sp:source:sonos", "sp:source:radio"])
        self.assertIn("sp:menu", callbacks)

    def test_radio_station_picker_has_safe_dynamic_callbacks_and_add_button(self):
        adapter = object.__new__(TelegramAdapter)
        stations = [{"id": "Abc_1-xyz", "name": "WFMU", "url": "https://stream.example/live.mp3?token=secret"}]
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            markup = adapter._speaker_radio_station_keyboard(stations)
        rows = markup.inline_keyboard
        callbacks = [button.callback_data for row in rows for button in row]
        labels = [button.text for row in rows for button in row]
        self.assertIn("sp:radio:play:Abc_1-xyz", callbacks)
        self.assertIn("sp:news:after-station:Abc_1-xyz", callbacks)
        self.assertIn("sp:radio:add", callbacks)
        self.assertIn("sp:radio:remove:Abc_1-xyz", callbacks)
        self.assertIn("WFMU", labels)
        self.assertNotIn("token=secret", " ".join(labels))
        self.assertTrue(all(len(callback.encode("utf-8")) <= 64 for callback in callbacks))

    def test_sonos_favorite_picker_only_exposes_title_and_position(self):
        adapter = object.__new__(TelegramAdapter)
        favorites = [{"position": 2, "title": "Rare Grooves", "uri": "x-sonos-http:secret"}]
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            markup = adapter._speaker_sonos_favorites_keyboard(favorites)
        rows = markup.inline_keyboard
        callbacks = [button.callback_data for row in rows for button in row]
        labels = [button.text for row in rows for button in row]
        self.assertTrue(any(callback.startswith("sp:source:sonos:2:") for callback in callbacks))
        self.assertIn("Rare Grooves", labels)
        self.assertNotIn("x-sonos-http", " ".join(labels))

    def test_sonos_favorite_revalidates_choice_and_honors_quiet_hours(self):
        favorite = {"position": 2, "title": "Rare Grooves", "uri": "x-sonos-http:opaque"}
        token = hashlib.sha256(b"2|Rare Grooves|x-sonos-http:opaque").hexdigest()[:8]
        calls = []
        brain = SimpleNamespace(
            SPEAKER_OPERATION_LOCK=Path("/unused/test-lock"),
            _exclusive_operation_lock=lambda _path: nullcontext(),
            quiet_hours_active=lambda: True,
            _set_manual_stop=Mock(),
        )
        original_path = sys.path.copy()
        try:
            with patch.object(TelegramAdapter, "_read_sonos_favorites", return_value=[favorite]), patch(
                "hermes_constants.get_hermes_home", return_value=Path("/unused/hermes")
            ), patch.object(
                telegram_adapter_module.subprocess,
                "run",
                side_effect=lambda *args, **kwargs: calls.append(args) or SimpleNamespace(returncode=0),
            ), patch.object(telegram_adapter_module.importlib, "import_module", return_value=brain):
                stale = TelegramAdapter._play_sonos_favorite(2, "00000000")
                blocked = TelegramAdapter._play_sonos_favorite(2, token)
                brain.quiet_hours_active = lambda: False
                started = TelegramAdapter._play_sonos_favorite(2, token)
        finally:
            sys.path[:] = original_path
        self.assertFalse(stale["success"])
        self.assertFalse(blocked["success"])
        self.assertTrue(started["success"])
        self.assertEqual(len(calls), 1)
        self.assertIn("--index", calls[0][0])
        self.assertEqual(calls[0][0][-1], "2")
        brain._set_manual_stop.assert_called_once_with(False)

    def test_radio_station_removal_requires_confirmation_then_refreshes_list(self):
        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        station = {"id": "Abc_1-xyz", "name": "WFMU", "url": "https://stream.example/live.mp3"}
        adapter._load_radio_stations = Mock(side_effect=[[station], []])
        remove = Mock(return_value=True)
        queries = [
            SimpleNamespace(from_user=SimpleNamespace(id=7), answer=AsyncMock(), edit_message_text=AsyncMock()),
            SimpleNamespace(from_user=SimpleNamespace(id=7), answer=AsyncMock(), edit_message_text=AsyncMock()),
        ]
        with patch("integrations.sonos_radio_stations.remove_station", remove), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton
        ), patch("plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup):
            asyncio.run(adapter._handle_speaker_callback(
                queries[0], "sp:radio:remove:Abc_1-xyz", chat_id="42", chat_type="private", thread_id=None, user_name="Oleg"
            ))
            remove.assert_not_called()
            asyncio.run(adapter._handle_speaker_callback(
                queries[1], "sp:radio:confirm-remove:Abc_1-xyz", chat_id="42", chat_type="private", thread_id=None, user_name="Oleg"
            ))
        remove.assert_called_once_with("Abc_1-xyz")
        self.assertIn("WFMU", queries[0].edit_message_text.await_args.args[0])
        self.assertIn("removed", queries[1].edit_message_text.await_args.args[0].lower())

    def test_radio_add_form_reply_saves_station_without_echoing_url(self):
        adapter = object.__new__(TelegramAdapter)
        message = SimpleNamespace(
            text="WFMU | https://stream.example/live.mp3",
            chat_id="42",
            from_user=SimpleNamespace(id=7),
            reply_to_message=SimpleNamespace(message_id=99),
            reply_text=AsyncMock(),
        )
        adapter._speaker_form_state = {
            "42:7": {"action": "radio-add", "prompt_message_id": 99, "expires_at": 9999999999},
        }
        adapter._load_radio_stations = lambda: [
            {"id": "abc123def456", "name": "WFMU", "url": "https://stream.example/live.mp3"}
        ]
        with patch(
            "integrations.sonos_radio_stations.add_station",
            return_value={"id": "abc123def456", "name": "WFMU"},
        ) as add:
            consumed = asyncio.run(adapter._consume_speaker_form_reply(message))
        self.assertTrue(consumed)
        add.assert_called_once_with("WFMU", "https://stream.example/live.mp3")
        response = message.reply_text.await_args.args[0]
        self.assertIn("WFMU", response)
        self.assertNotIn("https://", response)

    def test_plex_source_requires_duration_choice_then_explicit_switch(self):
        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        adapter._run_speaker_subprocess = AsyncMock(return_value='{"success":true}')
        query = SimpleNamespace(
            from_user=SimpleNamespace(id=7),
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
        )
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                query, "sp:source:plex:1h", chat_id="42", chat_type="private", thread_id=None, user_name="Oleg"
            ))
        adapter._run_speaker_subprocess.assert_awaited_once_with("fm", duration="1h", switch=True)
        self.assertIn("Plex Music", query.edit_message_text.await_args.args[0])
        self.assertIn("not been independently verified", query.edit_message_text.await_args.args[0])

    def test_news_duration_callback_saves_choice_and_reads_settings_back(self):
        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        status = json.dumps({
            "success": True,
            "settings": {"enabled": True, "station": "bbc", "mode": "idle", "duration_minutes": 15,
                          "after_news_action": "idle", "after_news_station_id": ""},
            "stations": {"bbc": "BBC News"}, "radio_stations": [],
        })
        manager = AsyncMock(side_effect=[json.dumps({"success": True}), status])
        adapter._run_hourly_news_manager = manager
        query = SimpleNamespace(from_user=SimpleNamespace(id=7), answer=AsyncMock(), edit_message_text=AsyncMock())
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                query, "sp:news:duration:15", chat_id="42", chat_type="private", thread_id=None, user_name="Oleg"
            ))
        self.assertEqual(manager.await_args_list[0].args, ("set_duration",))
        self.assertEqual(manager.await_args_list[0].kwargs, {"duration_minutes": 15})
        self.assertEqual(manager.await_args_list[1].args, ("status",))
        self.assertIn("15 minutes", query.edit_message_text.await_args.args[0])

    def test_news_post_station_callback_saves_station_and_enables_radio_action(self):
        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        status = json.dumps({
            "success": True,
            "settings": {"enabled": True, "station": "bbc", "mode": "interrupt", "duration_minutes": 15,
                          "after_news_action": "radio", "after_news_station_id": "w_9jYX5B"},
            "stations": {"bbc": "BBC News"}, "managed_radio_stations": [{"id": "w_9jYX5B", "name": "WFMU"}],
        })
        manager = AsyncMock(side_effect=[json.dumps({"success": True}), json.dumps({"success": True}), status])
        adapter._run_hourly_news_manager = manager
        query = SimpleNamespace(from_user=SimpleNamespace(id=7), answer=AsyncMock(), edit_message_text=AsyncMock())
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                query, "sp:news:after-station:w_9jYX5B", chat_id="42", chat_type="private", thread_id=None,
                user_name="Oleg",
            ))
        self.assertEqual(manager.await_args_list[0].args, ("set_after_news_station",))
        self.assertEqual(manager.await_args_list[0].kwargs, {"station_id": "w_9jYX5B"})
        self.assertEqual(manager.await_args_list[1].args, ("set_after_news_action",))
        self.assertEqual(manager.await_args_list[1].kwargs, {"after_news_action": "radio"})
        self.assertEqual(manager.await_args_list[2].args, ("status",))
        self.assertIn("WFMU", query.edit_message_text.await_args.args[0])

    def test_news_kill_callback_disables_automation_and_reads_status_back(self):
        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        status = json.dumps({
            "success": True,
            "settings": {"enabled": False, "station": "bbc", "mode": "idle", "duration_minutes": 15,
                          "after_news_action": "idle", "after_news_station_id": ""},
            "stations": {"bbc": "BBC News"}, "managed_radio_stations": [],
        })
        manager = AsyncMock(side_effect=[json.dumps({"success": True}), status])
        adapter._run_hourly_news_manager = manager
        query = SimpleNamespace(from_user=SimpleNamespace(id=7), answer=AsyncMock(), edit_message_text=AsyncMock())
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                query, "sp:news:kill", chat_id="42", chat_type="private", thread_id=None, user_name="Oleg"
            ))
        self.assertEqual(manager.await_args_list[0].args, ("kill_automations",))
        self.assertEqual(manager.await_args_list[1].args, ("status",))
        self.assertIn("Automatic news and post-news radio paused", query.edit_message_text.await_args.args[0])

    def test_news_play_now_uses_idle_only_manager_and_reports_busy_speaker(self):
        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        manager = AsyncMock(return_value=json.dumps({"success": True, "started": False, "skipped": "busy"}))
        adapter._run_hourly_news_manager = manager
        query = SimpleNamespace(from_user=SimpleNamespace(id=7), answer=AsyncMock(), edit_message_text=AsyncMock())
        asyncio.run(adapter._handle_speaker_callback(
            query, "sp:news:play", chat_id="42", chat_type="private", thread_id=None, user_name="Oleg"
        ))
        manager.assert_awaited_once_with("trigger_if_idle")
        self.assertIn("speaker is not idle", query.edit_message_text.await_args.args[0])
        self.assertIn("left alone", query.edit_message_text.await_args.args[0])

    def test_hourly_news_manager_routes_global_kill_to_atomic_backend_operation(self):
        adapter = object.__new__(TelegramAdapter)
        with patch("integrations.sonos_hourly_news.settings_operation", return_value={"success": False}) as operation:
            raw = asyncio.run(adapter._run_hourly_news_manager("kill_automations"))
        operation.assert_called_once_with("kill_automations")
        self.assertFalse(json.loads(raw)["success"])

    def test_plex_search_form_runs_bounded_query_and_renders_results(self):
        adapter = object.__new__(TelegramAdapter)
        adapter._speaker_form_state = {"42:7": {
            "action": "plex-search", "prompt_message_id": 19, "expires_at": float("inf"),
        }}
        adapter._run_speaker_subprocess = AsyncMock(return_value=json.dumps({"success": True, "tracks": [
            {"id": "12345", "title": "Song", "artist": "Artist", "album": "Album", "duration_ms": 120000},
        ]}))
        message = SimpleNamespace(
            from_user=SimpleNamespace(id=7), chat_id=42, reply_to_message=SimpleNamespace(message_id=19),
            text="Artist", reply_text=AsyncMock(),
        )
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            consumed = asyncio.run(adapter._consume_speaker_form_reply(message))
        self.assertTrue(consumed)
        adapter._run_speaker_subprocess.assert_awaited_once_with("plex-search", query="Artist", limit=10)
        self.assertIn("Artist — Song", message.reply_text.await_args.args[0])
        self.assertIn("sp:plex:play:12345", [
            button.callback_data
            for row in message.reply_text.await_args.kwargs["reply_markup"].inline_keyboard
            for button in row
        ])

    def test_plex_play_callback_uses_allowlisted_track_id_and_reads_back_status(self):
        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        adapter._run_speaker_subprocess = AsyncMock(side_effect=[
            json.dumps({"success": True, "started": True, "track": {"title": "Song", "artist": "Artist", "album": "Album"}}),
            json.dumps({"state": "PLAYING", "track": "Song", "source": "Plex"}),
        ])
        query = SimpleNamespace(from_user=SimpleNamespace(id=7), answer=AsyncMock(), edit_message_text=AsyncMock())
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                query, "sp:plex:play:12345", chat_id="42", chat_type="private", thread_id=None, user_name="Oleg"
            ))
        self.assertEqual(adapter._run_speaker_subprocess.await_args_list[0].args, ("plex-play",))
        self.assertEqual(adapter._run_speaker_subprocess.await_args_list[0].kwargs, {"track_id": "12345"})
        self.assertEqual(adapter._run_speaker_subprocess.await_args_list[1].args, ("status",))
        self.assertIn("Artist — Song", query.edit_message_text.await_args.args[0])

    def test_radio_picker_callback_runs_only_the_selected_station(self):
        adapter = object.__new__(TelegramAdapter)
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        adapter._play_radio_station = lambda station_id: {
            "success": True,
            "message": f"Started station {station_id}.",
        }
        query = SimpleNamespace(
            from_user=SimpleNamespace(id=7),
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
        )
        with patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", _FakeInlineButton), patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", _FakeInlineMarkup
        ):
            asyncio.run(adapter._handle_speaker_callback(
                query, "sp:radio:play:aBcd_123", chat_id="42", chat_type="private", thread_id=None, user_name="Oleg"
            ))
        self.assertIn("Started station aBcd_123", query.edit_message_text.await_args.args[0])


if __name__ == "__main__":
    unittest.main()
