"""Real-codec receive proof and privacy-safe Discord voice transport metrics."""

import ctypes.util
import logging
import os
import struct
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.integration
pytest.importorskip("nacl.secret", reason="PyNaCl required for voice transport tests")
pytest.importorskip("discord", reason="discord.py required for voice transport tests")


def _load_real_opus(discord):
    if discord.opus.is_loaded():
        return
    candidates = [ctypes.util.find_library("opus")]
    if sys.platform == "darwin":
        candidates += [
            "/opt/homebrew/lib/libopus.dylib",
            "/usr/local/lib/libopus.dylib",
        ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.sep in candidate and not os.path.isfile(candidate):
            continue
        try:
            discord.opus.load_opus(candidate)
        except Exception:
            continue
        if discord.opus.is_loaded():
            return
    pytest.skip("a real libopus shared library is unavailable")


def _receiver(secret_key, *, bot_ssrc=9999, **receiver_kwargs):
    from plugins.platforms.discord.adapter import VoiceReceiver

    voice_client = MagicMock()
    voice_client._connection.secret_key = list(secret_key)
    voice_client._connection.dave_session = None
    voice_client._connection.ssrc = bot_ssrc
    voice_client._connection.add_socket_listener = MagicMock()
    voice_client._connection.remove_socket_listener = MagicMock()
    voice_client._connection.hook = None
    voice_client.user = SimpleNamespace(id=bot_ssrc)
    voice_client.channel = MagicMock()
    voice_client.channel.members = []
    receiver = VoiceReceiver(
        voice_client,
        allowed_user_ids={"42"},
        **receiver_kwargs,
    )
    receiver.start()
    receiver.map_ssrc(100, 42)
    return receiver


def _encrypted_opus_rtp_packets(secret_key, *, count=30, ssrc=100):
    import discord
    import nacl.secret

    _load_real_opus(discord)
    encoder = discord.opus.Encoder()
    aead = nacl.secret.Aead(secret_key)
    # 20 ms, 48 kHz, stereo, signed 16-bit PCM. A non-zero deterministic
    # waveform ensures the real encoder/decoder path is exercised.
    frame = b"\x10\x00\xf0\xff" * 960
    for index in range(count):
        sequence = index + 1
        timestamp = sequence * 960
        header = struct.pack(">BBHII", 0x80, 0x78, sequence, timestamp, ssrc)
        opus = encoder.encode(frame, 960)
        nonce_suffix = sequence.to_bytes(4, "big")
        nonce = nonce_suffix + (b"\x00" * 20)
        encrypted = bytes(aead.encrypt(opus, header, nonce).ciphertext)
        yield header + encrypted + nonce_suffix


def test_real_nacl_opus_packets_survive_while_playback_capture_is_armed(caplog):
    """Prove UDP callback -> NaCl -> Opus -> PCM -> token -> endpoint."""
    caplog.set_level(logging.INFO)
    secret_key = bytes(range(32))
    receiver = _receiver(secret_key)
    receiver.begin_playback_capture(77)

    for packet in _encrypted_opus_rtp_packets(secret_key):
        receiver._on_packet(packet)

    receiver._last_packet_time[100] = time.monotonic() - 2.0
    completed = receiver.check_silence(with_context=True)
    receiver.end_playback_capture(77)

    assert len(completed) == 1
    user_id, pcm, playback_token = completed[0]
    assert user_id == 42
    assert playback_token == 77
    assert len(pcm) >= 30 * 3840

    stats = receiver.snapshot_transport_stats()
    assert stats["udp_callbacks"] == 30
    assert stats["voice_rtp_packets"] == 30
    assert stats["non_bot_rtp_packets"] == 30
    assert stats["decoded_packets"] == 30
    assert stats["playback_tagged_packets"] == 30
    assert stats["decoded_pcm_bytes"] == len(pcm)
    assert stats.get("nacl_decrypt_failures", 0) == 0
    assert stats.get("opus_decode_failures", 0) == 0
    assert "Discord voice endpoint" in caplog.text
    assert "Discord voice playback capture summary" in caplog.text
    assert "ssrcs=1" in caplog.text
    assert "tagged=30" in caplog.text


def test_paused_boundary_counts_callback_but_drops_before_rtp_decode():
    secret_key = bytes(range(32))
    receiver = _receiver(secret_key)
    receiver.pause()

    packet = next(_encrypted_opus_rtp_packets(secret_key, count=1))
    receiver._on_packet(packet)

    stats = receiver.snapshot_transport_stats()
    assert stats["udp_callbacks"] == 1
    assert stats["paused_drops"] == 1
    assert stats.get("voice_rtp_packets", 0) == 0
    assert stats.get("decoded_packets", 0) == 0
    assert not receiver._buffers


def test_playback_end_defers_summary_until_token_pinned_decode_finishes(
    monkeypatch,
    caplog,
):
    import discord

    caplog.set_level(logging.INFO)
    secret_key = bytes(range(32))
    packet = next(_encrypted_opus_rtp_packets(secret_key, count=1))
    callback_order = []
    pcm_events = []
    drained = []
    receiver = _receiver(
        secret_key,
        playback_pcm_callback=lambda *event: (
            callback_order.append("pcm"),
            pcm_events.append(event),
        ),
        playback_drained_callback=lambda token: (
            callback_order.append("drained"),
            drained.append(token),
        ),
    )
    decode_started = threading.Event()
    release_decode = threading.Event()

    class BlockingDecoder:
        def decode(self, _payload):
            decode_started.set()
            assert release_decode.wait(timeout=2)
            return b"\x00" * 3840

    monkeypatch.setattr(discord.opus, "Decoder", lambda: BlockingDecoder())
    receiver.begin_playback_capture(91)
    worker = threading.Thread(target=receiver._on_packet, args=(packet,))
    worker.start()
    assert decode_started.wait(timeout=1)

    receiver.end_playback_capture(91)
    assert "capture draining token=91 inflight=1" in caplog.text
    assert receiver._playback_inflight[91] == 1
    assert pcm_events == []
    assert drained == []

    release_decode.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert receiver._buffer_playback_tokens[100] == 91
    assert receiver.snapshot_transport_stats()["decoded_packets"] == 1
    assert "capture summary token=91" in caplog.text
    assert "decoded=1 tagged=1" in caplog.text
    assert len(pcm_events) == 1
    assert pcm_events[0][0] == 91
    assert pcm_events[0][1] == 42
    assert pcm_events[0][2] == b"\x00" * 3840
    assert drained == [91]
    assert callback_order == ["pcm", "drained"]
    assert 91 not in receiver._playback_inflight
    assert 91 not in receiver._playback_transport_stats


def test_shadow_streaming_callback_does_not_retain_playback_pcm():
    secret_key = bytes(range(32))
    packet = next(_encrypted_opus_rtp_packets(secret_key, count=1))
    events = []
    receiver = _receiver(
        secret_key,
        playback_pcm_callback=lambda *event: events.append(event),
        retain_playback_pcm=False,
    )
    receiver.begin_playback_capture(92)

    receiver._on_packet(packet)

    assert len(events) == 1
    assert events[0][0] == 92
    assert events[0][1] == 42
    assert receiver._buffers.get(100, bytearray()) == bytearray()
    assert 100 not in receiver._last_packet_time
    assert 100 not in receiver._buffer_playback_tokens


def test_nacl_and_opus_failure_counters_are_recorded(monkeypatch):
    import discord

    secret_key = bytes(range(32))
    wrong_key = bytes(reversed(range(32)))
    receiver = _receiver(secret_key)
    receiver.begin_playback_capture(33)
    bad_packet = next(_encrypted_opus_rtp_packets(wrong_key, count=1))
    receiver._on_packet(bad_packet)
    assert receiver.snapshot_transport_stats()["nacl_decrypt_failures"] == 1

    good_packet = next(_encrypted_opus_rtp_packets(secret_key, count=1))

    class BrokenDecoder:
        def decode(self, _payload):
            raise RuntimeError("decode failed")

    monkeypatch.setattr(discord.opus, "Decoder", lambda: BrokenDecoder())
    receiver._on_packet(good_packet)
    assert receiver.snapshot_transport_stats()["opus_decode_failures"] == 1


def test_flush_pending_logs_only_bounded_metadata(caplog):
    caplog.set_level(logging.INFO)
    receiver = _receiver(bytes(range(32)))
    pcm = b"\x00" * 96000  # 0.5 seconds at 48 kHz, stereo, 16-bit
    receiver._buffers[100].extend(pcm)
    receiver._buffer_playback_tokens[100] = 88

    completed = receiver.flush_pending(with_context=True)

    assert completed == [(42, pcm, 88)]
    assert "Discord voice flush playback=88" in caplog.text
    assert "pcm_bytes=96000" in caplog.text
    assert "duration_ms=500" in caplog.text
    assert "user=42" not in caplog.text


def test_normal_flush_does_not_emit_monitor_metadata(caplog):
    caplog.set_level(logging.INFO)
    receiver = _receiver(bytes(range(32)))
    pcm = b"\x00" * 96000
    receiver._buffers[100].extend(pcm)

    completed = receiver.flush_pending(with_context=True)

    assert completed == [(42, pcm, None)]
    assert "Discord voice flush" not in caplog.text
