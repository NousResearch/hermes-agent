"""A Discord join must not report listening after its receiver fails to start.

These adapter-level tests use fake Discord transport and transcription only. They do not
qualify native codec reception, playback, provider latency or full-duplex conversation.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from plugins.platforms.discord import adapter as discord_adapter


def make_adapter():
    adapter = object.__new__(discord_adapter.DiscordAdapter)
    adapter._client = MagicMock()
    adapter._allowed_user_ids = {"42"}
    for name in ("clients", "locks", "receivers", "listen_tasks", "timeout_tasks",
                 "text_channels", "sources", "mixers"):
        setattr(adapter, f"_voice_{name}", {})
    adapter._voice_timeout_seconds = 0
    adapter._voice_fx_cfg = {"enabled": True}
    adapter._install_voice_mixer = AsyncMock()
    adapter._voice_input_callback = AsyncMock()
    return adapter


def make_channel():
    channel = SimpleNamespace(id=900, guild=SimpleNamespace(id=111))
    client = MagicMock()
    client.channel = channel
    client.is_connected.return_value = True
    client.disconnect = AsyncMock()
    client.move_to = AsyncMock()
    channel.connect = AsyncMock(return_value=client)
    return channel, client


class DiscordVoiceReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_receiver_partial_start_removes_its_socket_listener(self):
        from discord.utils import MISSING

        adapter = make_adapter()
        channel, client = make_channel()
        connection = SimpleNamespace(secret_key=bytes(32), dave_session=None, ssrc=1, hook=None, ws=MISSING,
                                     add_socket_listener=MagicMock(side_effect=RuntimeError("fake socket failure")),
                                     remove_socket_listener=MagicMock())
        client._connection = connection
        with patch.object(discord_adapter, "DISCORD_AVAILABLE", True):
            self.assertFalse(await adapter.join_voice_channel(channel))
        connection.add_socket_listener.assert_called_once()
        connection.remove_socket_listener.assert_called_once_with(connection.add_socket_listener.call_args.args[0])
        client.disconnect.assert_awaited_once()
        self.assertFalse(adapter.is_listening_in_voice_channel(111))

    async def test_failed_receiver_cleans_only_owned_join_and_never_reports_listening(self):
        adapter = make_adapter()
        channel, client = make_channel()
        receiver = MagicMock()
        receiver.start.side_effect = RuntimeError("private receiver exception")
        timer = MagicMock()
        adapter._reset_voice_timeout = lambda guild: adapter._voice_timeout_tasks.update({guild: timer})
        other_client, other_receiver, other_task = MagicMock(), MagicMock(), MagicMock()
        adapter._voice_clients[222] = other_client
        adapter._voice_receivers[222] = other_receiver
        adapter._voice_timeout_tasks[222] = other_task
        with patch.object(discord_adapter, "DISCORD_AVAILABLE", True), \
             patch.object(discord_adapter, "VoiceReceiver", return_value=receiver), \
             self.assertLogs(discord_adapter.logger, logging.DEBUG) as logs:
            joined = await adapter.join_voice_channel(channel, text_channel_id=700, source={"chat_id": "700"})
        self.assertFalse(joined)
        self.assertFalse(adapter.is_in_voice_channel(111))
        self.assertFalse(adapter.is_listening_in_voice_channel(111))
        receiver.stop.assert_called_once()
        receiver.flush_pending.assert_not_called()
        client.disconnect.assert_awaited_once()
        timer.cancel.assert_called_once()
        adapter._install_voice_mixer.assert_not_awaited()
        adapter._voice_input_callback.assert_not_awaited()
        self.assertNotIn(111, adapter._voice_receivers)
        self.assertNotIn(111, adapter._voice_listen_tasks)
        self.assertNotIn(111, adapter._voice_timeout_tasks)
        self.assertNotIn(111, adapter._voice_text_channels)
        self.assertNotIn(111, adapter._voice_sources)
        self.assertIs(adapter._voice_clients[222], other_client)
        self.assertIs(adapter._voice_receivers[222], other_receiver)
        other_task.cancel.assert_not_called()
        self.assertIn("error_type=RuntimeError", "\n".join(logs.output))
        self.assertNotIn("private receiver exception", "\n".join(logs.output))

    async def test_failed_receiver_does_not_disconnect_superseding_client_or_clear_its_binding(self):
        adapter = make_adapter()
        channel, client = make_channel()
        receiver, replacement = MagicMock(), MagicMock()
        replacement_receiver, replacement_timer = MagicMock(), MagicMock()
        replacement_mixer = object()

        def replace_then_fail():
            adapter._voice_clients[111] = replacement
            adapter._voice_receivers[111] = replacement_receiver
            adapter._voice_timeout_tasks[111] = replacement_timer
            adapter._voice_text_channels[111] = 800
            adapter._voice_sources[111] = {"chat_id": "800"}
            adapter._voice_mixers[111] = replacement_mixer
            raise RuntimeError("failed old receiver")

        receiver.start.side_effect = replace_then_fail
        with patch.object(discord_adapter, "DISCORD_AVAILABLE", True), \
             patch.object(discord_adapter, "VoiceReceiver", return_value=receiver):
            self.assertFalse(await adapter.join_voice_channel(channel, text_channel_id=700))
        receiver.stop.assert_called_once()
        client.disconnect.assert_not_awaited()
        replacement.disconnect.assert_not_called()
        replacement.stop.assert_not_called()
        replacement_timer.cancel.assert_not_called()
        self.assertIs(adapter._voice_clients[111], replacement)
        self.assertIs(adapter._voice_receivers[111], replacement_receiver)
        self.assertEqual(adapter._voice_text_channels[111], 800)
        self.assertEqual(adapter._voice_sources[111], {"chat_id": "800"})
        self.assertIs(adapter._voice_mixers[111], replacement_mixer)

    async def test_successful_join_and_rejoin_require_running_receiver_and_listener(self):
        adapter = make_adapter()
        channel, client = make_channel()
        receiver = MagicMock()
        receiver._running = True
        release = asyncio.Event()

        async def listen(_guild):
            await release.wait()

        adapter._voice_listen_loop = listen
        with patch.object(discord_adapter, "DISCORD_AVAILABLE", True), \
             patch.object(discord_adapter, "VoiceReceiver", return_value=receiver):
            self.assertTrue(await adapter.join_voice_channel(channel, text_channel_id=700))
            self.assertTrue(adapter.is_in_voice_channel(111))
            self.assertTrue(adapter.is_listening_in_voice_channel(111))
            self.assertTrue(await adapter.join_voice_channel(channel, text_channel_id=800))
            channel.connect.assert_awaited_once()
            self.assertEqual(adapter._voice_text_channels[111], 800)
            task = adapter._voice_listen_tasks[111]
            release.set()
            await task
            self.assertTrue(adapter.is_in_voice_channel(111))
            self.assertFalse(adapter.is_listening_in_voice_channel(111))
            self.assertFalse(await adapter.join_voice_channel(channel))
        await adapter.leave_voice_channel(111)
        client.disconnect.assert_awaited_once()

    async def test_listener_that_finishes_during_mixer_setup_is_not_ready(self):
        adapter = make_adapter()
        channel, client = make_channel()
        receiver = MagicMock()
        receiver._running = True
        adapter._voice_listen_loop = AsyncMock()

        async def install_mixer(_guild, _client):
            adapter._voice_mixers[111] = object()
            client.is_playing.return_value = True
            await adapter._voice_listen_tasks[111]

        adapter._install_voice_mixer.side_effect = install_mixer
        with patch.object(discord_adapter, "DISCORD_AVAILABLE", True), \
             patch.object(discord_adapter, "VoiceReceiver", return_value=receiver):
            self.assertFalse(await adapter.join_voice_channel(channel))
        self.assertFalse(adapter.is_listening_in_voice_channel(111))
        receiver.stop.assert_called_once()
        self.assertNotIn(111, adapter._voice_mixers)
        self.assertFalse(adapter.voice_mixer_active(111))
        client.stop.assert_called_once()
        client.disconnect.assert_awaited_once()

    async def test_transcript_reaches_callback_but_ingress_logs_are_metadata_only(self):
        adapter = make_adapter()
        adapter._voice_text_channels[111] = 700
        transcript = "Private speech marker 9ae32f and its sensitive contents"
        error = subprocess.CalledProcessError(1, ["fake", transcript], stderr=transcript.encode())
        with patch.object(discord_adapter.VoiceReceiver, "pcm_to_wav"), \
             patch("tools.transcription_tools.transcribe_audio", return_value={"success": True, "transcript": transcript}), \
             patch("tools.voice_mode_transcript.is_whisper_hallucination", return_value=False), \
             self.assertLogs(discord_adapter.logger, logging.DEBUG) as logs:
            await adapter._process_voice_input(111, 42, b"fake PCM", 700)
            adapter._voice_input_callback.assert_awaited_once_with(guild_id=111, user_id=42, transcript=transcript)
            adapter._voice_input_callback.side_effect = error
            await adapter._process_voice_input(111, 42, b"fake PCM", 700)
        rendered = "\n".join(logs.output)
        self.assertNotIn(transcript, rendered)
        self.assertNotIn("9ae32f", rendered)
        self.assertIn("guild=111, user=42, characters=", rendered)
        self.assertIn("error_type=CalledProcessError", rendered)
        self.assertTrue(all(record.exc_info is None for record in logs.records))

    async def test_listener_error_logs_are_metadata_only(self):
        adapter = make_adapter()
        receiver = MagicMock()
        receiver._running = True
        receiver.check_silence.side_effect = RuntimeError("sensitive exception marker 79be2c")
        adapter._voice_receivers[111] = receiver
        with patch.object(discord_adapter.asyncio, "sleep", AsyncMock()), \
             self.assertLogs(discord_adapter.logger, logging.DEBUG) as logs:
            await adapter._voice_listen_loop(111)
        rendered = "\n".join(logs.output)
        self.assertIn("guild=111, error_type=RuntimeError", rendered)
        self.assertNotIn("79be2c", rendered)
        self.assertTrue(all(record.exc_info is None for record in logs.records))


if __name__ == "__main__":
    unittest.main()
