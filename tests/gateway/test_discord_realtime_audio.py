"""Tests for Discord realtime voice audio conversion and playback source."""

from gateway.discord_realtime_audio import (
    DISCORD_PCM_FRAME_BYTES,
    RealtimeDiscordAudioSource,
    discord_pcm_to_realtime_pcm,
    realtime_pcm_to_discord_pcm,
)


def test_read_returns_silence_frame_when_queue_empty():
    source = RealtimeDiscordAudioSource()

    frame = source.read()

    assert len(frame) == DISCORD_PCM_FRAME_BYTES
    assert frame == b"\x00" * DISCORD_PCM_FRAME_BYTES


def test_push_pcm_outputs_discord_sized_frames():
    source = RealtimeDiscordAudioSource()
    pcm_100ms_24k_mono = b"\x00" * 4800

    source.push_pcm_24k_mono(pcm_100ms_24k_mono)

    for _ in range(5):
        assert len(source.read()) == DISCORD_PCM_FRAME_BYTES


def test_clear_drops_buffered_audio():
    source = RealtimeDiscordAudioSource()
    pcm_20ms_24k_mono = b"\x01\x02" * 480

    source.push_pcm_24k_mono(pcm_20ms_24k_mono)
    source.clear()

    assert source.read() == b"\x00" * DISCORD_PCM_FRAME_BYTES


def test_resampler_roundtrip_lengths_are_reasonable():
    discord_frame = b"\x00" * DISCORD_PCM_FRAME_BYTES

    realtime_frame = discord_pcm_to_realtime_pcm(discord_frame)
    converted_back = realtime_pcm_to_discord_pcm(realtime_frame)

    assert len(realtime_frame) == 960
    assert len(converted_back) == DISCORD_PCM_FRAME_BYTES
