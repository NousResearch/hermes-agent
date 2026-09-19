import asyncio
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.discord.adapter import (
    DiscordAdapter,
    VoiceReceiver,
    _read_discord_stt_settings,
)


class FakeLiveController:
    def __init__(self, result):
        self.result = result
        self.appended = []
        self.finished = []
        self.aborted = []
        self.closed = 0

    async def append_pcm48(self, user_id, pcm, *, admit=None):
        assert admit is None or admit()
        self.appended.append((user_id, pcm))

    async def finish_utterance(self, user_id, *, expected_source_bytes, admit=None):
        assert admit is None or admit()
        self.finished.append((user_id, expected_source_bytes))
        return dict(self.result)

    async def _prepare_utterance(self, user_id, *, expected_source_bytes, admit=None):
        return await self.finish_utterance(user_id, expected_source_bytes=expected_source_bytes, admit=admit)

    async def _wait_for_utterance(self, user_id, prepared):
        return prepared

    async def _retire_prepared_utterance(self, user_id, prepared):
        pass

    async def abort_user(self, user_id):
        self.aborted.append(user_id)

    async def close(self):
        self.closed += 1


def _receiver(*, allowed_user_ids=None, members=None):
    vc = MagicMock()
    vc.channel = SimpleNamespace(members=members or [])
    vc.user = SimpleNamespace(id=999)
    vc._connection = MagicMock()
    receiver = VoiceReceiver(vc, allowed_user_ids=allowed_user_ids)
    receiver._running = True
    receiver.require_explicit_mapping = True
    return receiver


def _adapter():
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_input_callback = AsyncMock()
    adapter._voice_live_transcribers = {}
    receiver = _receiver()
    receiver.map_ssrc(100, 42)
    receiver.map_ssrc(101, 43)
    adapter._voice_receivers = {7: receiver}
    adapter._client = None
    adapter._is_allowed_user = lambda user_id, **kwargs: user_id in {"42", "43"}
    return adapter


def test_voice_receiver_exposes_mapped_pcm_chunks_once():
    receiver = _receiver(allowed_user_ids={"42"})
    receiver.map_ssrc(100, 42)
    pcm = b"\x00\x00\x00\x00" * 10
    receiver._buffer_decoded_pcm(100, pcm)
    assert receiver.drain_stream_chunks() == [(0, 100, 1, 42, pcm)]
    assert receiver.drain_stream_chunks() == []
    assert bytes(receiver._buffers[100]) == pcm


def test_voice_receiver_stop_restores_installed_speaking_hooks():
    original_connection_hook = MagicMock()
    original_ws_hook = MagicMock()
    connection = SimpleNamespace(
        hook=original_connection_hook,
        ws=SimpleNamespace(_hook=original_ws_hook),
    )
    voice_client = MagicMock()
    voice_client._connection = connection
    receiver = VoiceReceiver(voice_client, allowed_user_ids={"42"})
    receiver._running = True
    with patch.dict(
        "sys.modules",
        {"discord.utils": SimpleNamespace(MISSING=object())},
    ):
        receiver._install_speaking_hook(connection)

    assert connection.hook is not original_connection_hook
    assert connection.ws._hook is not original_ws_hook

    receiver.stop()

    assert connection.hook is original_connection_hook
    assert connection.ws._hook is original_ws_hook


def test_voice_receiver_waits_for_explicit_mapping_then_streams_local_preroll():
    members = [
        SimpleNamespace(id=999),
        SimpleNamespace(id=42),
        SimpleNamespace(id=43),
    ]
    receiver = _receiver(allowed_user_ids={"42"}, members=members)
    pcm = b"\x01\x00\x02\x00" * 10
    receiver._buffer_decoded_pcm(100, pcm)
    assert receiver.drain_stream_chunks() == []
    assert 100 not in receiver._ssrc_to_user

    receiver.map_ssrc(100, 42)
    assert receiver.drain_stream_chunks() == [(0, 100, 1, 42, pcm)]
    assert receiver._ssrc_to_user[100] == 42


def test_voice_receiver_never_streams_unmapped_ambiguous_audio():
    members = [
        SimpleNamespace(id=999),
        SimpleNamespace(id=42),
        SimpleNamespace(id=43),
    ]
    receiver = _receiver(allowed_user_ids={"42", "43"}, members=members)
    pcm = b"\x01\x00\x02\x00" * 10
    receiver._buffer_decoded_pcm(100, pcm)
    assert receiver.drain_stream_chunks() == []
    assert bytes(receiver._buffers[100]) == pcm


def test_voice_receiver_remap_invalidates_old_buffer_decoder_and_stream_chunks():
    receiver = _receiver(allowed_user_ids={"42", "43"})
    old_pcm = b"\x01\x00\x02\x00" * 10
    new_pcm = b"\x03\x00\x04\x00" * 10
    receiver.map_ssrc(100, 42)
    receiver._decoders[100] = object()
    receiver._buffer_decoded_pcm(100, old_pcm)

    receiver.map_ssrc(100, 43)
    receiver._buffer_decoded_pcm(100, new_pcm)

    assert receiver.drain_stream_chunks() == [(0, 100, 2, 43, new_pcm)]
    assert bytes(receiver._buffers[100]) == new_pcm
    assert 100 not in receiver._decoders


def test_voice_receiver_rechecks_running_state_before_buffer_publication():
    receiver = _receiver(allowed_user_ids={"42"})
    receiver.map_ssrc(100, 42)
    receiver._running = False
    receiver._buffer_decoded_pcm(100, b"\x01\x00\x02\x00" * 10)

    assert receiver.drain_stream_chunks() == []
    assert bytes(receiver._buffers[100]) == b""


def test_voice_receiver_rejects_frame_decoded_across_ssrc_remap():
    receiver = _receiver()
    receiver.map_ssrc(100, 42)
    snapshot = receiver._snapshot_packet_generation(100)

    receiver.map_ssrc(100, 43)
    receiver._buffer_decoded_pcm(100, b"old", *snapshot)

    assert receiver.drain_stream_chunks() == []
    assert bytes(receiver._buffers[100]) == b""


def test_voice_receiver_rejects_frame_decoded_across_pause_resume():
    receiver = _receiver()
    receiver.map_ssrc(100, 42)
    snapshot = receiver._snapshot_packet_generation(100)

    receiver.pause()
    receiver.resume()
    receiver._buffer_decoded_pcm(100, b"old", *snapshot)

    assert receiver.drain_stream_chunks() == []


def test_drained_chunk_retains_generation_for_egress_recheck():
    receiver = _receiver()
    receiver.map_ssrc(100, 42)
    receiver._buffer_decoded_pcm(100, b"frame")
    chunk = receiver.drain_stream_chunks()[0]

    receiver.map_ssrc(100, 43)

    assert receiver.stream_chunk_is_current(*chunk[:4]) is False


def test_unmap_user_revokes_mapping_buffers_decoder_and_queued_audio():
    receiver = _receiver()
    receiver.map_ssrc(100, 42)
    receiver._decoders[100] = object()
    receiver._buffer_decoded_pcm(100, b"frame")

    receiver.unmap_user(42)

    assert 100 not in receiver._ssrc_to_user
    assert 100 not in receiver._buffers
    assert 100 not in receiver._decoders
    assert receiver.drain_stream_chunks() == []


@pytest.mark.asyncio
async def test_revoke_voice_user_unmaps_ssrc_and_aborts_live_session():
    adapter = _adapter()
    receiver = MagicMock()
    controller = MagicMock()
    controller.abort_user = AsyncMock()
    adapter._voice_receivers = {7: receiver}
    adapter._voice_live_transcribers = {7: controller}

    await adapter._revoke_voice_stt_user(7, 42)

    receiver.unmap_user.assert_called_once_with(42)
    controller.abort_user.assert_awaited_once_with(42)


def test_voice_receiver_bounds_stream_queue_and_preserves_full_local_buffer():
    receiver = _receiver(allowed_user_ids={"42"})
    receiver.map_ssrc(100, 42)
    pcm = b"\x01\x00\x02\x00" * 10
    receiver.MAX_STREAM_QUEUE_BYTES = len(pcm)

    receiver._buffer_decoded_pcm(100, pcm)
    receiver._buffer_decoded_pcm(100, pcm)

    assert receiver.drain_stream_chunks() == [(0, 100, 1, 42, pcm)]
    assert bytes(receiver._buffers[100]) == pcm + pcm


def test_voice_receiver_turn_cap_bounds_buffer_and_streamed_bytes_together():
    receiver = _receiver(allowed_user_ids={"42"})
    receiver.map_ssrc(100, 42)
    pcm = b"\x01\x00\x02\x00" * 10
    receiver.MAX_UTTERANCE_BYTES = len(pcm)
    receiver.MIN_SPEECH_DURATION = 0.0

    receiver._buffer_decoded_pcm(100, pcm)
    receiver._buffer_decoded_pcm(100, pcm)

    assert receiver.drain_stream_chunks() == [(0, 100, 1, 42, pcm)]
    assert receiver.check_silence() == [(42, pcm)]


@pytest.mark.asyncio
async def test_contextual_mode_uses_explicit_openai_gpt_transcribe(tmp_path):
    adapter = _adapter()
    pcm = b"\x00\x00\x00\x00" * 100

    def fake_wav(_pcm, path):
        with open(path, "wb") as handle:
            handle.write(b"wav")

    with patch.object(VoiceReceiver, "pcm_to_wav", side_effect=fake_wav), \
         patch(
             "tools.transcription_tools._transcribe_audio_with_provider",
             return_value={
                 "success": True,
                 "transcript": "Kontext gewinnt",
                 "provider": "openai",
             },
         ) as transcribe:
        await adapter._process_voice_input(
            guild_id=7,
            user_id=42,
            pcm_data=pcm,
            stt_mode="openai_contextual",
        )

    assert transcribe.call_args.kwargs == {
        "model": "gpt-transcribe",
        "provider": "openai",
        "source": "discord",
    }
    adapter._voice_input_callback.assert_awaited_once_with(
        guild_id=7,
        user_id=42,
        transcript="Kontext gewinnt",
    )


@pytest.mark.asyncio
async def test_configured_mode_preserves_existing_transcription_dispatch():
    adapter = _adapter()
    pcm = b"\x00\x00\x00\x00" * 100

    def fake_wav(_pcm, path):
        with open(path, "wb") as handle:
            handle.write(b"wav")

    with patch.object(VoiceReceiver, "pcm_to_wav", side_effect=fake_wav), \
         patch(
             "tools.transcription_tools.transcribe_audio",
             return_value={"success": True, "transcript": "Default"},
         ) as transcribe:
        await adapter._process_voice_input(
            guild_id=7,
            user_id=42,
            pcm_data=pcm,
            stt_mode="configured",
        )

    assert transcribe.call_args.args[0].endswith(".wav")
    assert transcribe.call_args.kwargs == {}


@pytest.mark.asyncio
async def test_live_mode_finishes_streamed_turn_and_dispatches_transcript():
    adapter = _adapter()
    controller = FakeLiveController(
        {
            "success": True,
            "transcript": "Live gewinnt",
            "provider": "openai_realtime",
            "item_id": "item_42",
        }
    )
    adapter._voice_live_transcribers[7] = controller
    pcm = b"\x00\x00\x00\x00" * 100

    await adapter._process_completed_voice_utterance(
        guild_id=7,
        user_id=42,
        pcm_data=pcm,
        stt_mode="openai_live_high",
    )

    assert controller.finished == [(42, len(pcm))]
    adapter._voice_input_callback.assert_awaited_once_with(
        guild_id=7,
        user_id=42,
        transcript="Live gewinnt",
    )


@pytest.mark.asyncio
async def test_live_failure_does_not_secretly_call_paid_contextual_fallback():
    adapter = _adapter()
    controller = FakeLiveController(
        {
            "success": False,
            "transcript": "",
            "error": "incomplete_live_stream",
            "provider": "openai_realtime",
        }
    )
    adapter._voice_live_transcribers[7] = controller
    adapter._process_voice_input = AsyncMock()

    await adapter._process_completed_voice_utterance(
        guild_id=7,
        user_id=42,
        pcm_data=b"\x00\x00\x00\x00" * 100,
        stt_mode="openai_live_high",
    )

    adapter._process_voice_input.assert_not_awaited()
    adapter._voice_input_callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_live_mode_delegates_to_file_pipeline():
    adapter = _adapter()
    adapter._process_voice_input = AsyncMock()
    pcm = b"\x00\x00\x00\x00" * 100

    await adapter._process_completed_voice_utterance(
        guild_id=7,
        user_id=42,
        pcm_data=pcm,
        stt_mode="openai_contextual",
    )

    adapter._process_voice_input.assert_awaited_once_with(
        7,
        42,
        pcm,
        stt_mode="openai_contextual",
    )


def test_discord_stt_mode_is_read_from_discord_config(monkeypatch):
    config = {
        "stt": {"openai": {"languages": ["de", "en"]}},
        "discord": {"voice_stt": {"mode": "live-high"}},
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    mode, loaded = _read_discord_stt_settings()
    assert mode == "openai_live_high"
    assert loaded is config


def test_default_config_keeps_dual_discord_stt_opt_in():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    voice_stt = DEFAULT_CONFIG["discord"]["voice_stt"]
    assert voice_stt["mode"] == "configured"
    assert (
        voice_stt["openai_live"]["delay"]
        == "high"
    )
    assert (
        voice_stt["openai_live"]["max_session_seconds"]
        == 3300.0
    )


@pytest.mark.asyncio
async def test_stt_disabled_never_creates_live_controller():
    adapter = _adapter()
    config = {
        "stt": {"enabled": False},
        "discord": {"voice_stt": {"mode": "openai_live_high"}},
    }
    with patch(
        "plugins.platforms.discord.live_transcription.DiscordLiveTranscriptionController"
    ) as controller_type:
        with pytest.raises(ValueError, match="stt.enabled"):
            await adapter._activate_voice_stt_mode(7, "openai_live_high", config)

    controller_type.assert_not_called()
    assert 7 not in adapter._voice_live_transcribers


@pytest.mark.asyncio
async def test_runtime_stt_kill_switch_closes_live_controller_and_discards_audio():
    adapter = _adapter()
    controller = FakeLiveController({"success": True, "transcript": "unused"})
    adapter._voice_live_transcribers[7] = controller
    receiver = MagicMock()
    adapter._voice_receivers[7] = receiver

    with patch(
        "plugins.platforms.discord.adapter._read_runtime_stt_enabled",
        return_value=False,
    ):
        await adapter._process_voice_listener_tick(
            guild_id=7,
            receiver=receiver,
            stt_mode="openai_live_high",
            guild=MagicMock(),
        )

    receiver.discard_pending.assert_called_once_with()
    assert controller.closed == 1
    assert 7 not in adapter._voice_live_transcribers


@pytest.mark.asyncio
async def test_listener_tick_streams_authorized_chunks_before_live_commit():
    adapter = _adapter()
    order = []

    class OrderedController(FakeLiveController):
        async def append_pcm48(self, user_id, pcm, *, admit=None):
            order.append(("append", user_id, pcm))
            await super().append_pcm48(user_id, pcm, admit=admit)

        async def _prepare_utterance(self, user_id, *, expected_source_bytes, admit=None):
            order.append(("finish", user_id, expected_source_bytes))
            return await super()._prepare_utterance(user_id, expected_source_bytes=expected_source_bytes, admit=admit)

    controller = OrderedController({"success": True, "transcript": "Live"})
    adapter._voice_live_transcribers[7] = controller
    adapter._is_allowed_user = lambda user_id, **kwargs: user_id == "42"
    receiver = MagicMock()
    adapter._voice_receivers[7] = receiver
    receiver.drain_stream_chunks.return_value = [
        (0, 100, 1, 42, b"authorized-1"),
        (0, 100, 1, 42, b"-authorized-2"),
        (0, 101, 1, 99, b"blocked"),
    ]
    receiver.stream_chunk_is_current.return_value = True
    receiver.snapshot_sources.return_value = {42: ((0, 100, 1, 42),)}
    receiver.check_silence.return_value = [(42, b"full utterance")]
    guild = MagicMock()

    await adapter._process_voice_listener_tick(
        guild_id=7,
        receiver=receiver,
        stt_mode="openai_live_high",
        guild=guild,
    )

    assert controller.appended == [(42, b"authorized-1-authorized-2")]
    assert order[0] == ("append", 42, b"authorized-1-authorized-2")
    assert order[1] == ("finish", 42, len(b"full utterance"))
    assert controller.finished == [(42, len(b"full utterance"))]
    adapter._voice_input_callback.assert_awaited_once_with(guild_id=7, user_id=42, transcript="Live")


@pytest.mark.asyncio
async def test_listener_drops_drained_chunk_after_ssrc_generation_changes():
    adapter = _adapter()
    receiver = MagicMock()
    adapter._voice_receivers[7] = receiver
    receiver.drain_stream_chunks.return_value = [(0, 100, 1, 42, b"stale")]
    receiver.stream_chunk_is_current.return_value = False
    receiver.check_silence.return_value = []
    controller = FakeLiveController({"success": True, "transcript": "unused"})
    adapter._voice_live_transcribers[7] = controller
    adapter._process_completed_voice_utterance = AsyncMock()

    await adapter._process_voice_listener_tick(
        guild_id=7,
        receiver=receiver,
        stt_mode="openai_live_high",
        guild=MagicMock(),
    )

    receiver.stream_chunk_is_current.assert_called_once_with(0, 100, 1, 42)
    assert controller.appended == []
    adapter._process_completed_voice_utterance.assert_not_awaited()


@pytest.mark.asyncio
async def test_listener_drops_later_drained_chunks_after_pause_resume():
    adapter = _adapter()
    receiver = _receiver(allowed_user_ids={"42", "43"})
    adapter._voice_receivers[7] = receiver
    receiver.map_ssrc(100, 42)
    receiver.map_ssrc(101, 43)
    receiver._buffer_decoded_pcm(100, b"first")
    receiver._buffer_decoded_pcm(101, b"second")
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class BlockingController(FakeLiveController):
        async def append_pcm48(self, user_id, pcm, *, admit=None):
            assert admit is None or admit()
            self.appended.append((user_id, pcm))
            if user_id == 42:
                first_started.set()
                await release_first.wait()

    controller = BlockingController({"success": True, "transcript": "unused"})
    adapter._voice_live_transcribers[7] = controller
    adapter._is_allowed_user = lambda user_id, **kwargs: True
    adapter._process_completed_voice_utterance = AsyncMock()

    tick = asyncio.create_task(
        adapter._process_voice_listener_tick(
            guild_id=7,
            receiver=receiver,
            stt_mode="openai_live_high",
            guild=MagicMock(),
        )
    )
    await first_started.wait()
    receiver.pause()
    receiver.resume()
    release_first.set()
    await tick

    assert controller.appended == [(42, b"first")]


@pytest.mark.asyncio
async def test_listener_tick_rechecks_kill_switch_before_each_live_append():
    adapter = _adapter()
    controller = FakeLiveController({"success": True, "transcript": "unused"})
    adapter._voice_live_transcribers[7] = controller
    adapter._is_allowed_user = lambda user_id, **kwargs: True
    receiver = MagicMock()
    adapter._voice_receivers[7] = receiver
    receiver.drain_stream_chunks.return_value = [
        (0, 100, 1, 42, b"first"),
        (0, 101, 1, 43, b"second"),
    ]
    receiver.stream_chunk_is_current.return_value = True
    receiver.check_silence.return_value = []

    enabled = [True]
    original_append = controller.append_pcm48
    async def append_then_disable(user_id, pcm, *, admit=None):
        await original_append(user_id, pcm, admit=admit)
        enabled[0] = False
    controller.append_pcm48 = append_then_disable
    with patch(
        "plugins.platforms.discord.adapter._read_runtime_stt_enabled",
        side_effect=lambda: enabled[0],
    ):
        await adapter._process_voice_listener_tick(
            guild_id=7,
            receiver=receiver,
            stt_mode="openai_live_high",
            guild=MagicMock(),
        )

    assert controller.appended == [(42, b"first")]
    assert controller.closed == 1
    assert 7 not in adapter._voice_live_transcribers
    receiver.discard_pending.assert_called_once_with()


@pytest.mark.asyncio
async def test_listener_tick_discards_stream_chunks_for_contextual_mode():
    adapter = _adapter()
    adapter._is_allowed_user = lambda user_id, **kwargs: True
    adapter._process_completed_voice_utterance = AsyncMock()
    receiver = MagicMock()
    adapter._voice_receivers[7] = receiver
    receiver.drain_stream_chunks.return_value = [
        (0, 100, 1, 42, b"unused live chunk")
    ]
    receiver.check_silence.return_value = [(42, b"full utterance")]
    receiver.snapshot_sources.return_value = {42: ((0, 100, 1, 42),)}
    receiver.stream_chunk_is_current.return_value = True

    await adapter._process_voice_listener_tick(
        guild_id=7,
        receiver=receiver,
        stt_mode="openai_contextual",
        guild=MagicMock(),
    )

    adapter._process_completed_voice_utterance.assert_awaited_once_with(
        7,
        42,
        b"full utterance",
        stt_mode="openai_contextual",
        admit=ANY,
    )


@pytest.mark.asyncio
async def test_activating_new_mode_closes_existing_live_controller():
    adapter = _adapter()
    adapter._voice_stt_modes = {7: "openai_live_high"}
    old = FakeLiveController({"success": True, "transcript": "old"})
    adapter._voice_live_transcribers[7] = old

    await adapter._activate_voice_stt_mode(
        7,
        "openai_contextual",
        {},
    )

    assert old.closed == 1
    assert adapter._voice_stt_modes[7] == "openai_contextual"
    assert 7 not in adapter._voice_live_transcribers


@pytest.mark.asyncio
async def test_reactivating_same_live_mode_replaces_controller_to_apply_new_config():
    adapter = object.__new__(DiscordAdapter)
    old_controller = FakeLiveController({"success": True, "transcript": "old"})
    new_controller = MagicMock()
    adapter._voice_live_transcribers = {7: old_controller}
    adapter._voice_stt_modes = {7: "openai_live_high"}
    config = {
        "discord": {
            "voice_stt": {
                "openai_live": {
                    "delay": "high",
                    "max_session_seconds": 3200,
                }
            }
        }
    }

    with (
        patch(
            "plugins.platforms.discord.live_transcription.resolve_openai_realtime_api_key",
            return_value="test-key",
        ),
        patch(
            "plugins.platforms.discord.live_transcription.DiscordLiveTranscriptionController",
            return_value=new_controller,
        ),
    ):
        await adapter._activate_voice_stt_mode(
            7,
            "openai_live_high",
            config,
        )

    assert old_controller.closed == 1
    assert adapter._voice_live_transcribers[7] is new_controller


@pytest.mark.asyncio
async def test_join_does_not_replace_latched_mode_while_already_connected():
    adapter = _adapter()
    adapter._client = MagicMock()
    adapter._voice_locks = {}
    adapter._voice_clients = {}
    adapter._voice_stt_modes = {}
    adapter._activate_voice_stt_mode = AsyncMock()
    adapter._reset_voice_timeout = MagicMock()
    existing = MagicMock()
    existing.is_connected.return_value = True
    existing.channel.id = 70
    adapter._voice_clients[7] = existing
    channel = MagicMock()
    channel.id = 70
    channel.guild.id = 7
    config = {"discord": {"voice_stt": {"mode": "openai_contextual"}}}

    with patch(
        "plugins.platforms.discord.adapter.DISCORD_AVAILABLE", True
    ), patch(
        "plugins.platforms.discord.adapter._read_discord_stt_settings",
        return_value=("openai_contextual", config),
    ):
        result = await adapter.join_voice_channel(channel)

    assert result is True
    adapter._activate_voice_stt_mode.assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_join_latches_selected_mode_before_receiver_start():
    adapter = _adapter()
    adapter._client = MagicMock()
    adapter._voice_locks = {}
    adapter._voice_clients = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._voice_fx_cfg = {"enabled": False}
    adapter._allowed_user_ids = {"42"}
    adapter._activate_voice_stt_mode = AsyncMock()
    adapter._reset_voice_timeout = MagicMock()
    voice_client = MagicMock()
    channel = MagicMock()
    channel.guild.id = 7
    channel.connect = AsyncMock(return_value=voice_client)
    config = {"discord": {"voice_stt": {"mode": "openai_contextual"}}}

    def consume_coroutine(coroutine):
        coroutine.close()
        return MagicMock()

    with (
        patch("plugins.platforms.discord.adapter.DISCORD_AVAILABLE", True),
        patch(
            "plugins.platforms.discord.adapter._read_discord_stt_settings",
            return_value=("openai_contextual", config),
        ),
        patch("plugins.platforms.discord.adapter.VoiceReceiver") as receiver_cls,
        patch(
            "plugins.platforms.discord.adapter.asyncio.ensure_future",
            side_effect=consume_coroutine,
        ),
    ):
        assert await adapter.join_voice_channel(channel) is True

    adapter._activate_voice_stt_mode.assert_awaited_once_with(
        7,
        "openai_contextual",
        config,
    )
    receiver_cls.assert_called_once_with(
        voice_client,
        allowed_user_ids={"42"},
    )


@pytest.mark.asyncio
async def test_disconnected_rejoin_discards_old_receiver_listener_and_mode():
    adapter = _adapter()
    adapter._client = MagicMock()
    adapter._voice_locks = {}
    old_client = MagicMock()
    old_client.is_connected.return_value = False
    adapter._voice_clients = {7: old_client}
    old_receiver = MagicMock()
    adapter._voice_receivers = {7: old_receiver}
    old_task = asyncio.create_task(asyncio.sleep(60))
    adapter._voice_listen_tasks = {7: old_task}
    adapter._voice_mixers = {7: MagicMock()}
    timeout_task = MagicMock()
    adapter._voice_timeout_tasks = {7: timeout_task}
    adapter._voice_text_channels = {7: 70}
    adapter._voice_sources = {7: {"channel_id": "70"}}
    adapter._voice_stt_modes = {7: "openai_live_high"}
    adapter._voice_live_transcribers = {7: MagicMock()}
    adapter._close_voice_stt_mode = AsyncMock()
    adapter._activate_voice_stt_mode = AsyncMock()
    adapter._reset_voice_timeout = MagicMock()
    adapter._voice_fx_cfg = {"enabled": False}
    adapter._allowed_user_ids = {"42"}
    new_client = MagicMock()
    channel = MagicMock()
    channel.guild.id = 7
    channel.connect = AsyncMock(return_value=new_client)

    def consume_coroutine(coroutine):
        coroutine.close()
        return MagicMock()

    with (
        patch("plugins.platforms.discord.adapter.DISCORD_AVAILABLE", True),
        patch(
            "plugins.platforms.discord.adapter._read_discord_stt_settings",
            return_value=("openai_contextual", {}),
        ),
        patch("plugins.platforms.discord.adapter.VoiceReceiver"),
        patch(
            "plugins.platforms.discord.adapter.asyncio.ensure_future",
            side_effect=consume_coroutine,
        ),
    ):
        assert await adapter.join_voice_channel(channel) is True

    assert old_task.cancelled()
    old_receiver.stop.assert_called_once_with()
    adapter._close_voice_stt_mode.assert_awaited_once_with(7)
    timeout_task.cancel.assert_called_once_with()
    assert adapter._voice_clients[7] is new_client
    adapter._activate_voice_stt_mode.assert_awaited_once_with(
        7,
        "openai_contextual",
        {},
    )


@pytest.mark.asyncio
async def test_real_listen_loop_calls_mode_aware_tick(monkeypatch):
    adapter = _adapter()
    receiver = MagicMock()
    receiver._running = True
    adapter._voice_receivers = {7: receiver}
    adapter._voice_stt_modes = {7: "openai_live_high"}
    adapter._voice_clients = {}
    adapter._client = None

    async def one_tick(**kwargs):
        receiver._running = False

    adapter._process_voice_listener_tick = AsyncMock(side_effect=one_tick)
    with patch("asyncio.sleep", new=AsyncMock()):
        await adapter._voice_listen_loop(7)

    adapter._process_voice_listener_tick.assert_awaited_once_with(
        guild_id=7,
        receiver=receiver,
        stt_mode="openai_live_high",
        guild=None,
        live_batch=ANY,
    )

    from plugins.platforms.discord import adapter as discord_adapter
    from plugins.platforms.discord.live_transcription import (
        DiscordLiveTranscriptionController, LiveTranscriptionConfig,
        OpenAIRealtimeTranscriptionSession,
    )
    from tests.plugins.platforms.test_discord_live_transcription import FakeWebSocket

    monkeypatch.setattr(discord_adapter, "_read_runtime_stt_enabled", lambda: True)
    violations = []
    for trigger in ("cancel", "failure", "timeout"):
        baseline_tasks = asyncio.all_tasks()
        committed, retiring, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        reader_settled = asyncio.Event()
        sockets, sessions, children = [], [], []

        class Socket(FakeWebSocket):
            async def send(self, payload):
                await super().send(payload)
                if self.sent[-1]["type"] == "input_audio_buffer.commit":
                    await self.incoming.put({"type": "input_audio_buffer.committed", "item_id": "turn"})
                    if self is not sockets[0]:
                        await self.incoming.put({
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": "turn", "transcript": "replacement speech",
                        })
                    elif trigger == "failure":
                        await self.incoming.put({
                            "type": "conversation.item.input_audio_transcription.failed", "item_id": "turn",
                        })
                    committed.set()

            async def recv(self):
                try:
                    return await super().recv()
                except asyncio.CancelledError:
                    if self is sockets[0]:
                        self.retiring_reader = asyncio.current_task()
                        retiring.set()
                        try:
                            # The native reader is being gathered before socket close.
                            # A second cancellation must interrupt this unshielded barrier.
                            await release.wait()
                        finally:
                            reader_settled.set()
                    raise

        async def connect(*args, **kwargs):
            ws = Socket()
            sockets.append(ws)
            await ws.incoming.put({"type": "session.updated"})
            return ws

        config = LiveTranscriptionConfig(completion_timeout_seconds=0.05 if trigger == "timeout" else 2)

        def session_factory():
            session = OpenAIRealtimeTranscriptionSession(
                api_key="synthetic", config=config, websocket_connect=connect)
            sessions.append(session)
            return session

        # Observe the native listener's children without replacing their bodies.
        def create_task(coro):
            task = asyncio.create_task(coro)
            children.append(task)
            return task

        monkeypatch.setattr(discord_adapter, "asyncio", SimpleNamespace(
            **{name: getattr(asyncio, name) for name in dir(asyncio) if name != "create_task"},
            create_task=create_task,
        ))
        controller = DiscordLiveTranscriptionController(
            api_key="synthetic", config=config, session_factory=session_factory)
        adapter = _adapter()
        receiver = adapter._voice_receivers[7]
        receiver.MIN_SPEECH_DURATION = 0
        pcm = b"\x01\x00\x02\x00" * 24000
        receiver._buffer_decoded_pcm(100, pcm)
        receiver._last_packet_time[100] = 0
        adapter._voice_stt_modes = {7: "openai_live_high"}
        adapter._voice_live_transcribers = {7: controller}
        adapter._voice_clients = {}
        listener = asyncio.create_task(adapter._voice_listen_loop(7))
        replacement_append = None
        reader = None
        try:
            await asyncio.wait_for(committed.wait(), 2)
            if trigger == "cancel":
                listener.cancel()  # Ordinary cancellation while WAIT is still pending.
            await asyncio.wait_for(retiring.wait(), 2)
            reader = sockets[0].retiring_reader
            assert reader is not None
            assert 42 not in controller._sessions
            assert controller._operation_lock.locked()
            assert not sockets[0].closed
            assert not reader.done()
            waiter = next(task for task in children
                          if task.get_coro().__qualname__.endswith("._wait_for_utterance"))
            replacement_append = asyncio.create_task(controller.append_pcm48(42, pcm))
            if trigger != "cancel":
                listener.cancel()  # Failure/timeout already owns the popped transport.

            async def cancellation_reached_waiter():
                while not waiter.cancelling():
                    await asyncio.sleep(0)

            await asyncio.wait_for(cancellation_reached_waiter(), 2)
            if waiter.done() or reader_settled.is_set():
                violations.append((trigger, "retirement escaped before barrier release"))
            # A stale final must never reach the old listener or the replacement turn.
            await sockets[0].incoming.put({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "turn", "transcript": "stale speech",
            })
            release.set()
            await asyncio.wait_for(listener, 2)
            await asyncio.wait_for(replacement_append, 2)
            if not sockets[0].closed:
                violations.append((trigger, "captured socket leaked"))
            assert reader_settled.is_set() and reader.done()
            assert waiter.cancelled()
            assert all(task.done() for task in children)
            assert not sessions[0]._pending_unassigned and not sessions[0]._pending_by_item
            assert len(sessions) == 2
            assert controller._sessions[42] is sessions[1]
            assert not sessions[1]._closed and not sockets[1].closed
            assert controller._source_bytes[42] == len(pcm)
            result = await controller.finish_utterance(42, expected_source_bytes=len(pcm))
            assert result["success"] and result["transcript"] == "replacement speech"
            adapter._voice_input_callback.assert_not_awaited()
            await controller.close()
            assert sockets[1].closed
            assert not (asyncio.all_tasks() - baseline_tasks), "retirement left an orphan task"
        finally:
            release.set()
            tasks = [listener, *children]
            if replacement_append is not None:
                tasks.append(replacement_append)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # RED cleanup cannot change the already-recorded socket-leak evidence.
            for session in sessions:
                await session.close()
            await controller.close()
            receiver.stop()

    # Keep failed-close swallowing and cancellation priority at the same local owner.
    for outcome in ("exception", "cancel_then_exception", "child_cancel"):
        entered, release = asyncio.Event(), asyncio.Event()
        settled = []

        class ClosingSession:
            async def close(self):
                entered.set()
                try:
                    await release.wait()
                    if outcome == "child_cancel":
                        raise asyncio.CancelledError()
                    raise RuntimeError("synthetic close failure")
                finally:
                    settled.append(True)

        controller = DiscordLiveTranscriptionController(
            api_key="synthetic", config=LiveTranscriptionConfig(), session_factory=ClosingSession)
        controller._get_session(42)
        owner = asyncio.create_task(controller.abort_user(42))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            if outcome == "cancel_then_exception":
                for _ in range(2):
                    owner.cancel()
                    await asyncio.sleep(0)
                if owner.done() or settled:
                    violations.append((outcome, "close did not retain cancellation ownership"))
            release.set()
            result = (await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), 2))[0]
            assert settled == [True]
            assert not controller._sessions and not controller._operation_lock.locked()
            if outcome == "exception":
                assert result is None
            else:
                assert isinstance(result, asyncio.CancelledError)
        finally:
            release.set()
            await asyncio.gather(owner, return_exceptions=True)
            await controller.close()
    assert violations == [], violations


@pytest.mark.asyncio
async def test_leave_closes_latched_live_mode_without_receiver():
    adapter = _adapter()
    adapter._voice_locks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {}
    adapter._voice_clients = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._client = None
    adapter._close_voice_stt_mode = AsyncMock()

    await adapter.leave_voice_channel(7)

    adapter._close_voice_stt_mode.assert_awaited_once_with(7)


@pytest.mark.parametrize("flush", [False, True])
@pytest.mark.parametrize("change", ["revoke_other", "disable", "replace_receiver", "replace_controller", "remap"])
@pytest.mark.asyncio
async def test_operation_barriers_revalidate_egress(monkeypatch, flush, change):
    from plugins.platforms.discord.live_transcription import (
        DiscordLiveTranscriptionController, LiveTranscriptionConfig,
        OpenAIRealtimeTranscriptionSession,
    )
    from tests.plugins.platforms.test_discord_live_transcription import FakeWebSocket
    import json

    adapter = _adapter()
    receiver = _receiver()
    adapter._voice_receivers[7] = receiver
    receiver.MIN_SPEECH_DURATION = 0
    receiver.map_ssrc(100, 42)
    receiver.map_ssrc(101, 43)
    pcm = b"\x01\x00\x02\x00" * 8
    receiver._buffer_decoded_pcm(100, pcm)
    receiver._buffer_decoded_pcm(101, pcm)
    receiver._last_packet_time = {100: 0, 101: 0}
    if change == "disable":
        receiver.unmap_user(43)  # Block the FINAL append, not an earlier source.
    adapter._voice_receivers = {7: receiver}
    adapter._voice_stt_modes = {7: "openai_live_high"}
    adapter._client = MagicMock()
    adapter._voice_locks = {}
    adapter._voice_clients = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._reset_voice_timeout = MagicMock()
    allowed = {"42", "43"}
    enabled = [True]
    adapter._is_allowed_user = lambda user_id, **kwargs: user_id in allowed
    monkeypatch.setattr("plugins.platforms.discord.adapter._read_runtime_stt_enabled", lambda: enabled[0])
    started, release = asyncio.Event(), asyncio.Event()
    sockets = []

    class Socket(FakeWebSocket):
        async def send(self, payload):
            event = json.loads(payload)
            if event["type"] == "input_audio_buffer.append" and self is sockets[0]:
                started.set()
                await release.wait()
            await super().send(payload)
            if event["type"] == "input_audio_buffer.commit":
                await self.incoming.put({"type": "input_audio_buffer.committed", "item_id": "one"})
                await self.incoming.put({"type": "conversation.item.input_audio_transcription.completed",
                                         "item_id": "one", "transcript": "synthetic result"})

    async def connect(*args, **kwargs):
        ws = Socket()
        sockets.append(ws)
        await ws.incoming.put({"type": "session.updated"})
        return ws

    controller = DiscordLiveTranscriptionController(
        api_key="synthetic", config=LiveTranscriptionConfig(),
        session_factory=lambda: OpenAIRealtimeTranscriptionSession(
            api_key="synthetic", config=LiveTranscriptionConfig(), websocket_connect=connect),
    )
    adapter._voice_live_transcribers[7] = controller
    replacement = FakeLiveController({"success": False})
    task = asyncio.create_task(adapter.leave_voice_channel(7) if flush else adapter._process_voice_listener_tick(
        guild_id=7, receiver=receiver, stt_mode="openai_live_high", guild=MagicMock()))
    try:
        await asyncio.wait_for(started.wait(), 2)
        if change == "revoke_other":
            allowed.remove("43")  # No capture-generation change: authorization is independent.
        elif change == "disable":
            enabled[0] = False
        elif change == "replace_receiver":
            adapter._voice_receivers[7] = _receiver()
        elif change == "replace_controller":
            adapter._voice_live_transcribers[7] = replacement
        else:
            receiver.map_ssrc(100, 44)
        release.set()
        await asyncio.wait_for(task, 3)
        if change in {"revoke_other", "disable", "replace_receiver", "replace_controller"}:
            assert len(sockets) == 1, "a later source must never begin its append/connect"
        if change != "revoke_other":
            assert not any(e["type"] == "input_audio_buffer.commit" for e in sockets[0].sent)
            assert all(call.kwargs["user_id"] != 42 for call in adapter._voice_input_callback.await_args_list)
        if change == "disable":
            assert sockets[0].closed
            assert controller._closed
        assert replacement.appended == []
        assert replacement.finished == []
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await controller.close()


def test_configured_receiver_preserves_native_single_member_inference():
    receiver = _receiver(allowed_user_ids={"42"}, members=[SimpleNamespace(id=999), SimpleNamespace(id=42)])
    receiver.require_explicit_mapping = False
    receiver.MIN_SPEECH_DURATION = 0
    receiver._buffer_decoded_pcm(100, b"\x00" * 8)
    receiver._last_packet_time[100] = 0
    assert receiver.check_silence() == [(42, b"\x00" * 8)]


@pytest.fixture
def decoded_packet_receiver(monkeypatch):
    import struct
    import nacl.secret

    receiver = _receiver(
        allowed_user_ids={"42", "43"},
        members=[SimpleNamespace(id=999), SimpleNamespace(id=42), SimpleNamespace(id=43)],
    )
    receiver.require_explicit_mapping = False
    receiver._secret_key = b"\x01" * 32
    box = nacl.secret.Aead(receiver._secret_key)

    class Decoder:
        def decode(self, payload):
            return payload

    monkeypatch.setattr("plugins.platforms.discord.adapter.discord.opus.Decoder", Decoder)
    sequence = 0

    def receive(ssrc, pcm):
        nonlocal sequence
        sequence += 1
        header = struct.pack(">BBHII", 0x80, 0x78, sequence, sequence * 960, ssrc)
        nonce_suffix = struct.pack(">I", sequence)
        encrypted = box.encrypt(pcm, header, nonce_suffix + b"\x00" * 20)
        receiver._on_packet(header + encrypted.ciphertext + nonce_suffix)

    return receiver, receive


@pytest.mark.parametrize("retirement", ["silence_noise", "silence_speech", "flush", "pause", "discard"])
def test_abandoned_unmapped_decoders_allow_next_authorized_source(decoded_packet_receiver, retirement):
    receiver, receive = decoded_packet_receiver
    pcm = b"\x01\x00\x02\x00" * (24000 if retirement == "silence_speech" else 960)
    for ssrc in range(100, 104):
        receive(ssrc, pcm)
    assert len(receiver._decoders) == receiver.MAX_ACTIVE_SSRC_BUFFERS
    assert len(receiver._buffers) == receiver.MAX_ACTIVE_SSRC_BUFFERS
    assert receiver.drain_stream_chunks() == []

    receiver.map_ssrc(500, 42)
    receive(500, pcm)
    assert 500 not in receiver._buffers, "the four-source cap must still reject a fifth source"

    if retirement.startswith("silence"):
        receiver._last_packet_time = dict.fromkeys(range(100, 104), 0)
        assert receiver.check_silence() == []
    elif retirement == "flush":
        assert receiver.flush_pending() == []
    elif retirement == "pause":
        receiver.pause()
        receiver.resume()
    else:
        receiver.discard_pending()
    assert not receiver._buffers

    speech = b"\x03\x00\x04\x00" * 24000
    receive(500, speech)
    assert receiver.flush_pending() == [(42, speech)]
    assert b"".join(chunk[4] for chunk in receiver.drain_stream_chunks()) == speech


@pytest.mark.parametrize("retirement", ["check_silence", "flush_pending"])
@pytest.mark.parametrize("mapping", ["explicit", "inferred"])
def test_buffer_retirement_preserves_active_mapped_decoder(decoded_packet_receiver, retirement, mapping):
    receiver, receive = decoded_packet_receiver
    if mapping == "explicit":
        receiver.map_ssrc(100, 42)
    else:
        receiver._vc.channel.members = [SimpleNamespace(id=999), SimpleNamespace(id=42)]
    first = b"\x01\x00\x02\x00" * 24000
    receive(100, first)
    decoder = receiver._decoders[100]
    receiver._last_packet_time[100] = 0
    assert getattr(receiver, retirement)() == [(42, first)]
    assert receiver._ssrc_to_user[100] == 42
    assert receiver._decoders[100] is decoder

    second = b"\x03\x00\x04\x00" * 24000
    receive(100, second)
    assert receiver.flush_pending() == [(42, second)]
    assert receiver._decoders[100] is decoder


def test_capture_caps_hold_without_any_polling():
    receiver = _receiver()
    for ssrc in range(5):
        receiver.map_ssrc(ssrc, 42 + ssrc)
        receiver._buffer_decoded_pcm(ssrc, b"\x00" * (receiver.MAX_UTTERANCE_BYTES + 8))
    assert len(receiver._buffers) == 4
    assert all(len(buf) == 60 * 48000 * 2 * 2 for buf in receiver._buffers.values())
    assert receiver._stream_chunk_bytes <= 5 * 48000 * 2 * 2


@pytest.mark.parametrize("disabled", [False, "false", "0", "off"])
def test_runtime_enablement_uses_native_boolean_semantics(monkeypatch, disabled):
    from plugins.platforms.discord.adapter import _read_runtime_stt_enabled
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"stt": {"enabled": disabled}})
    assert _read_runtime_stt_enabled() is False


@pytest.fixture(params=["openai_contextual", "openai_live_high"])
def timeout_voice_runtime(request, monkeypatch):
    import json
    import subprocess
    import wave

    import httpx

    from hermes_constants import get_hermes_home
    from plugins.platforms.discord.live_transcription import (
        DiscordLiveTranscriptionController,
        LiveTranscriptionConfig,
        OpenAIRealtimeTranscriptionSession,
    )
    from tests.plugins.platforms.test_discord_live_transcription import FakeWebSocket

    # Use the real config and authorization readers in the isolated test home.
    (get_hermes_home() / "config.yaml").write_text(json.dumps({
        "stt": {"enabled": True, "provider": "openai", "cloud_trim_silence": False,
                "openai": {"api_key": "synthetic-timeout-test"}},
    }))
    adapter = object.__new__(DiscordAdapter)
    adapter._client = None
    adapter._allowed_user_ids = {"42"}
    adapter._allowed_role_ids = set()
    adapter._voice_input_callback = AsyncMock()
    adapter._on_voice_disconnect = MagicMock()
    adapter._voice_locks = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_timeout_seconds = 1
    adapter._voice_text_channels = {7: 70}
    adapter._voice_sources = {7: {"channel_id": "70"}}
    adapter._voice_mixers = {}
    adapter._voice_stt_modes = {7: request.param}
    adapter._voice_live_transcribers = {}
    receiver = _receiver(allowed_user_ids={"42"})
    receiver.map_ssrc(100, 42)
    receiver.SILENCE_THRESHOLD = 60  # Keep the final utterance pending until leave.
    adapter._voice_receivers = {7: receiver}
    vc = receiver._vc
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False

    async def disconnect():
        await asyncio.sleep(0)  # A real Discord disconnect suspends.
        vc.is_connected.return_value = False

    vc.disconnect = AsyncMock(side_effect=disconnect)
    adapter._voice_clients = {7: vc}
    pcm = b"\x01\x00\x02\x00" * receiver.SAMPLE_RATE
    assert len(pcm) / receiver.PCM_BYTES_PER_SECOND > receiver.MIN_SPEECH_DURATION
    receiver._buffer_decoded_pcm(100, pcm)
    requests = []

    def ffmpeg_run(command, *, input, **kwargs):
        assert "pipe:0" in command
        assert input == pcm
        with wave.open(command[-1], "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x01\x00" * 16000)

    def http_send(client, request, **kwargs):
        requests.append(request)
        assert str(request.url) == "https://api.openai.com/v1/audio/transcriptions"
        request.read()
        return httpx.Response(200, json={"text": "Pending speech"}, request=request)

    # Replace external I/O, retaining the native WAV worker, provider routing and SDK.
    monkeypatch.setenv("FFMPEG_PATH", "ffmpeg")
    monkeypatch.setattr("plugins.platforms.discord.adapter.subprocess",
                        SimpleNamespace(run=ffmpeg_run, PIPE=subprocess.PIPE))
    monkeypatch.setattr(httpx.Client, "send", http_send)

    class Socket(FakeWebSocket):
        async def send(self, payload):
            await super().send(payload)
            if json.loads(payload)["type"] == "input_audio_buffer.commit":
                await asyncio.sleep(0)  # The transport can suspend before acknowledging.
                await self.incoming.put({"type": "input_audio_buffer.committed", "item_id": "pending"})
                await self.incoming.put({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "pending", "transcript": "Pending speech",
                })

    ws = Socket()

    async def connect(*args, **kwargs):
        await ws.incoming.put({"type": "session.updated"})
        return ws

    controller = None
    if request.param == "openai_live_high":
        config = LiveTranscriptionConfig()
        controller = DiscordLiveTranscriptionController(
            api_key="synthetic-timeout-test", config=config,
            session_factory=lambda: OpenAIRealtimeTranscriptionSession(
                api_key="synthetic-timeout-test", config=config, websocket_connect=connect),
        )
        adapter._voice_live_transcribers[7] = controller
    return SimpleNamespace(adapter=adapter, receiver=receiver, vc=vc, mode=request.param,
                           controller=controller, ws=ws, requests=requests)


@pytest.mark.asyncio
async def test_inactivity_timeout_flush_disconnects_and_clears_runtime(timeout_voice_runtime):
    runtime = timeout_voice_runtime
    adapter, receiver = runtime.adapter, runtime.receiver
    listener = asyncio.create_task(adapter._voice_listen_loop(7))
    adapter._voice_listen_tasks[7] = listener
    adapter._reset_voice_timeout(7)
    timeout_task = adapter._voice_timeout_tasks[7]
    try:
        # Await the actual registered timer; cancellation is part of native teardown.
        await asyncio.wait_for(asyncio.gather(timeout_task, return_exceptions=True), 5)

        runtime.vc.disconnect.assert_awaited_once_with()
        assert not runtime.vc.is_connected()
        adapter._voice_input_callback.assert_awaited_once_with(
            guild_id=7, user_id=42, transcript="Pending speech")
        adapter._on_voice_disconnect.assert_called_once_with("70")
        assert listener.done()
        assert timeout_task.done()
        assert not receiver._running
        assert not receiver._buffers
        assert not receiver._decoders
        assert not receiver._ssrc_to_user
        assert not receiver._last_packet_time
        assert receiver.drain_stream_chunks() == []
        for name in ("_voice_receivers", "_voice_live_transcribers", "_voice_stt_modes",
                     "_voice_listen_tasks", "_voice_timeout_tasks", "_voice_clients",
                     "_voice_text_channels", "_voice_sources"):
            assert 7 not in getattr(adapter, name), name
        if runtime.controller is not None:
            assert runtime.controller._closed
            assert not runtime.controller._sessions
            assert not runtime.controller._source_bytes
            assert runtime.ws.closed
            assert sum(e["type"] == "input_audio_buffer.commit" for e in runtime.ws.sent) == 1
        else:
            assert len(runtime.requests) == 1
    finally:
        # Retire any leaked replacement timer on RED without hiding the assertions.
        tasks = {listener, timeout_task, *adapter._voice_timeout_tasks.values()}
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if runtime.controller is not None:
            await runtime.controller.close()


@pytest.mark.parametrize("authorized", [True, False])
@pytest.mark.asyncio
async def test_normal_voice_activity_renews_timeout_only_when_authorized(timeout_voice_runtime, authorized):
    runtime = timeout_voice_runtime
    adapter, receiver = runtime.adapter, runtime.receiver
    adapter._voice_timeout_seconds = 60
    if not authorized:
        adapter._allowed_user_ids = {"43"}
    receiver._last_packet_time[100] = 0  # A completed ordinary listening utterance.
    adapter._reset_voice_timeout(7)
    original_timeout = adapter._voice_timeout_tasks[7]
    await asyncio.sleep(0)
    try:
        await adapter._process_voice_listener_tick(
            guild_id=7, receiver=receiver, stt_mode=runtime.mode, guild=None)
        current_timeout = adapter._voice_timeout_tasks[7]
        assert (current_timeout is not original_timeout) is authorized
        assert not current_timeout.done()
        runtime.vc.disconnect.assert_not_awaited()
        if authorized:
            adapter._voice_input_callback.assert_awaited_once_with(
                guild_id=7, user_id=42, transcript="Pending speech")
        else:
            adapter._voice_input_callback.assert_not_awaited()
            assert runtime.requests == []
            assert runtime.ws.sent == []
    finally:
        tasks = {original_timeout, *adapter._voice_timeout_tasks.values()}
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        receiver.stop()
        if runtime.controller is not None:
            await runtime.controller.close()
