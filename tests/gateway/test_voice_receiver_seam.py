"""Seam-identity + aggressive tests for the VoiceReceiver extraction (R1-S1).

``plugins/platforms/discord/voice_receiver.py`` holds the VoiceReceiver
class, moved verbatim out of ``plugins/platforms/discord/adapter.py``
(god-file slice R1-S1, epic #78647) and re-exported through
``adapter.VoiceReceiver`` (identity-preserving seam — three existing test
files bind the adapter module global: test_voice_command.py,
test_voice_channel_flow.py, test_discord_race_polish.py).

The seam-identity tests pin the regression this extraction is meant to
prevent: adapter must resolve every moved name to the *same object* the
voice_receiver module defines.  The aggressive tests exercise the voice
path failure modes: RTP/NaCl decrypt, DAVE E2EE passthrough and failure,
RTP padding-strip edges, SPEAKING-hook chaining, sole-member SSRC
inference, and the pcm_to_wav ffmpeg argv contract.
"""

import asyncio
import struct
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.discord import adapter
from plugins.platforms.discord import voice_receiver

# Every member of the moved cluster: class + 12 methods. All must resolve
# through the adapter re-export to the identical object in voice_receiver.
# (Class identity itself is pinned by test_class_is_seam_identical.)
MOVED_MEMBERS = (
    "__init__",
    "start",
    "stop",
    "pause",
    "resume",
    "map_ssrc",
    "_install_speaking_hook",
    "_on_packet",
    "_infer_user_for_ssrc",
    "check_silence",
    "flush_pending",
    "pcm_to_wav",
)


# ---------------------------------------------------------------------------
# Seam identity
# ---------------------------------------------------------------------------

def test_class_is_seam_identical():
    assert adapter.VoiceReceiver is voice_receiver.VoiceReceiver
    assert voice_receiver.VoiceReceiver.__module__ == (
        "plugins.platforms.discord.voice_receiver"
    )


def test_all_moved_members_are_seam_identical():
    for name in MOVED_MEMBERS:
        assert getattr(adapter.VoiceReceiver, name) is getattr(
            voice_receiver.VoiceReceiver, name
        ), name


def test_class_consts_survive_move():
    for const in ("SILENCE_THRESHOLD", "MIN_SPEECH_DURATION", "SAMPLE_RATE", "CHANNELS"):
        assert getattr(adapter.VoiceReceiver, const) == getattr(
            voice_receiver.VoiceReceiver, const
        )
        assert getattr(adapter.VoiceReceiver, const) is getattr(
            voice_receiver.VoiceReceiver, const
        )


def test_no_back_import_of_adapter():
    """voice_receiver must import standalone (module-level back-import of
    adapter would be a cycle: adapter imports voice_receiver).

    The parent package __init__ imports adapter, so we first import the
    package normally (caching adapter + voice_receiver), then None-out the
    adapter module and re-execute voice_receiver from scratch: if it (or
    anything in its import chain) back-imported adapter, the None entry in
    sys.modules makes that import raise ImportError.
    """
    code = (
        "import sys\n"
        "import plugins.platforms.discord\n"
        "import plugins.platforms.discord.voice_receiver\n"
        "sys.modules['plugins.platforms.discord.adapter'] = None\n"
        "del sys.modules['plugins.platforms.discord.voice_receiver']\n"
        "import plugins.platforms.discord.voice_receiver as v\n"
        "print(v.VoiceReceiver.__name__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "VoiceReceiver"


# ---------------------------------------------------------------------------
# Receiver lifecycle
# ---------------------------------------------------------------------------

def _make_receiver():
    mock_vc = MagicMock()
    mock_vc._connection.secret_key = [0] * 32
    mock_vc._connection.dave_session = None
    mock_vc._connection.ssrc = 9999
    mock_vc._connection.hook = None
    mock_vc._connection.add_socket_listener = MagicMock()
    mock_vc._connection.remove_socket_listener = MagicMock()
    return voice_receiver.VoiceReceiver(mock_vc)


def test_start_installs_listener_and_hook():
    receiver = _make_receiver()
    receiver.start()
    conn = receiver._vc._connection
    assert receiver._running is True
    assert receiver._bot_ssrc == 9999
    conn.add_socket_listener.assert_called_once_with(receiver._on_packet)
    # Speaking hook chained onto the connection state
    assert conn.hook is not None
    assert conn.hook.__name__ == "wrapped_hook"


def test_stop_cleans_up():
    receiver = _make_receiver()
    receiver.start()
    receiver._buffers[1111] = bytearray(b"\x01" * 64)
    receiver._last_packet_time[1111] = 1.0
    receiver._decoders[1111] = object()
    receiver._ssrc_to_user[1111] = 42
    receiver.stop()
    assert receiver._running is False
    receiver._vc._connection.remove_socket_listener.assert_called_once_with(
        receiver._on_packet
    )
    assert receiver._buffers == {}
    assert receiver._last_packet_time == {}
    assert receiver._decoders == {}
    assert receiver._ssrc_to_user == {}


def test_pause_resume_gate_packet_processing():
    receiver = _make_receiver()
    receiver.start()
    receiver.pause()
    assert receiver._paused is True
    receiver._on_packet(b"\x80\x78" + b"\x00" * 20)
    assert len(receiver._buffers) == 0
    receiver.resume()
    assert receiver._paused is False


# ---------------------------------------------------------------------------
# Speaking hook (op-5 SPEAKING) chaining
# ---------------------------------------------------------------------------

def test_speaking_hook_maps_ssrc_and_chains_original():
    receiver = _make_receiver()
    original = AsyncMock()
    receiver._vc._connection.hook = original
    receiver.start()
    hooked = receiver._vc._connection.hook
    msg = {"op": 5, "d": {"ssrc": 1234, "user_id": 5678}}
    asyncio.run(hooked(None, msg))
    assert receiver._ssrc_to_user[1234] == 5678
    original.assert_awaited_once_with(None, msg)


def test_speaking_hook_passthrough_non_speaking_op():
    receiver = _make_receiver()
    original = AsyncMock()
    receiver._vc._connection.hook = original
    receiver.start()
    hooked = receiver._vc._connection.hook
    msg = {"op": 3, "d": {"ssrc": 999, "user_id": 1}}
    asyncio.run(hooked(None, msg))
    assert 999 not in receiver._ssrc_to_user
    original.assert_awaited_once_with(None, msg)


# ---------------------------------------------------------------------------
# Packet path: NaCl decrypt, DAVE E2EE, RTP padding edges
# ---------------------------------------------------------------------------

class _FakeAead:
    """nacl.secret.Aead stand-in. decrypt() returns ``plaintext`` verbatim."""

    def __init__(self, key):
        self.key = key

    def decrypt(self, encrypted, header, nonce):
        return _FakeAead._plaintext


def _rtp_packet(ssrc, seq=42, payload=b"\xaa" * 40, flags=0x80):
    header = struct.pack(">BBHII", flags, 0x78, seq, 1000, ssrc)
    return header + payload + b"\xbb\xbb\xbb\xbb"


def _patched_receiver(decrypted_payload, **kwargs):
    receiver = _make_receiver()
    receiver.start()
    _FakeAead._plaintext = decrypted_payload
    fake_decoder = MagicMock()
    fake_decoder.decode.return_value = b"\x00\x01" * 48  # 96 bytes PCM
    fake_discord = MagicMock()
    fake_discord.opus.Decoder.return_value = fake_decoder
    fake_nacl_secret = SimpleNamespace(Aead=_FakeAead)
    patches = [
        patch.dict(
            sys.modules,
            {
                "nacl": SimpleNamespace(secret=fake_nacl_secret),
                "nacl.secret": fake_nacl_secret,
            },
        ),
        patch.object(voice_receiver, "discord", fake_discord),
    ]
    if "dave_session" in kwargs:
        patches.append(
            patch.object(receiver, "_dave_session", kwargs["dave_session"])
        )
    for p in patches:
        p.start()
    receiver._patches = patches
    receiver._fake_decoder = fake_decoder
    return receiver


def test_on_packet_decrypts_buffers_pcm():
    ssrc = 1111
    receiver = _patched_receiver(b"\x01" * 40)
    receiver.map_ssrc(ssrc, 42)
    receiver._on_packet(_rtp_packet(ssrc))
    assert receiver._buffers[ssrc] == bytearray(b"\x00\x01" * 48)
    assert ssrc in receiver._last_packet_time
    for p in receiver._patches:
        p.stop()


def test_on_packet_skips_bot_own_ssrc():
    receiver = _patched_receiver(b"\x01" * 40)
    receiver._on_packet(_rtp_packet(9999))  # bot ssrc
    assert receiver._buffers == {}
    for p in receiver._patches:
        p.stop()


def test_on_packet_skips_non_rtp_and_short_packets():
    receiver = _patched_receiver(b"\x01" * 40)
    receiver._on_packet(b"\x00\x78" + b"\x00" * 20)  # version != 2
    receiver._on_packet(b"\x80\x78" + b"\x00" * 10)  # too short
    assert receiver._buffers == {}
    for p in receiver._patches:
        p.stop()


def test_on_packet_drops_bad_rtp_padding():
    ssrc = 2222
    # Padding bit set; decrypted payload's last byte is the pad count.
    # pad_len == 0 -> invalid -> drop.
    receiver = _patched_receiver(b"\x01" * 40 + b"\x00")
    receiver._on_packet(_rtp_packet(ssrc, flags=0xA0))
    assert receiver._buffers == {}
    for p in receiver._patches:
        p.stop()


def test_on_packet_strips_valid_rtp_padding():
    ssrc = 3333
    # pad_len == 4 (valid): 4 trailing bytes stripped before Opus decode.
    receiver = _patched_receiver(b"\x01" * 40 + b"\x04")
    receiver.map_ssrc(ssrc, 7)
    receiver._on_packet(_rtp_packet(ssrc, flags=0xA0))
    assert receiver._buffers[ssrc] == bytearray(b"\x00\x01" * 48)
    for p in receiver._patches:
        p.stop()


def test_on_packet_dave_passthrough_on_unencrypted():
    ssrc = 4444
    dave_session = MagicMock()
    dave_session.decrypt.side_effect = Exception("Unencrypted media")
    receiver = _patched_receiver(
        b"\x01" * 40, dave_session=dave_session
    )
    receiver.map_ssrc(ssrc, 9)
    with patch.dict(sys.modules, {"davey": SimpleNamespace(
        MediaType=SimpleNamespace(audio="audio"))}):
        receiver._on_packet(_rtp_packet(ssrc))
    assert receiver._buffers[ssrc] == bytearray(b"\x00\x01" * 48)
    for p in receiver._patches:
        p.stop()


def test_on_packet_drops_dave_hard_failure():
    ssrc = 5555
    dave_session = MagicMock()
    dave_session.decrypt.side_effect = RuntimeError("E2EE exploded")
    receiver = _patched_receiver(
        b"\x01" * 40, dave_session=dave_session
    )
    receiver.map_ssrc(ssrc, 9)
    with patch.dict(sys.modules, {"davey": SimpleNamespace(
        MediaType=SimpleNamespace(audio="audio"))}):
        receiver._on_packet(_rtp_packet(ssrc))
    assert receiver._buffers == {}
    for p in receiver._patches:
        p.stop()


# ---------------------------------------------------------------------------
# Silence detection / SSRC inference
# ---------------------------------------------------------------------------

def test_infer_user_for_ssrc_sole_allowed_member():
    receiver = _make_receiver()
    receiver._allowed_user_ids = {"42"}
    channel = MagicMock()
    bot = SimpleNamespace(id=999)
    member = SimpleNamespace(id=42)
    receiver._vc.channel = channel
    receiver._vc.user = bot
    channel.members = [bot, member]
    assert receiver._infer_user_for_ssrc(77) == 42
    assert receiver._ssrc_to_user[77] == 42


def test_infer_user_for_ssrc_ambiguous_returns_zero():
    receiver = _make_receiver()
    channel = MagicMock()
    receiver._vc.channel = channel
    receiver._vc.user = SimpleNamespace(id=999)
    channel.members = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    assert receiver._infer_user_for_ssrc(77) == 0
    assert 77 not in receiver._ssrc_to_user


# ---------------------------------------------------------------------------
# pcm_to_wav ffmpeg argv contract
# ---------------------------------------------------------------------------

def test_pcm_to_wav_ffmpeg_argv_contract():
    run = MagicMock()
    with patch.object(voice_receiver, "resolve_ffmpeg_executable",
                      return_value="/fake/ffmpeg"), \
         patch.object(voice_receiver.subprocess, "run", run), \
         patch("hermes_cli._subprocess_compat.windows_hide_flags",
               return_value=0):
        voice_receiver.VoiceReceiver.pcm_to_wav(
            b"\x00\x01" * 100, "/tmp/out.wav"
        )
    run.assert_called_once()
    args, kwargs = run.call_args
    assert args[0] == [
        "/fake/ffmpeg", "-y", "-loglevel", "error",
        "-f", "s16le", "-ar", "48000", "-ac", "2",
        "-i", "pipe:0", "-ar", "16000", "-ac", "1",
        "/tmp/out.wav",
    ]
    assert kwargs["input"] == b"\x00\x01" * 100
    assert kwargs["check"] is True
    assert kwargs["timeout"] == 10
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["creationflags"] == 0


def test_pcm_to_wav_propagates_ffmpeg_failure():
    run = MagicMock(
        side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr=b"boom")
    )
    with patch.object(voice_receiver, "resolve_ffmpeg_executable",
                      return_value="/fake/ffmpeg"), \
         patch.object(voice_receiver.subprocess, "run", run), \
         patch("hermes_cli._subprocess_compat.windows_hide_flags",
               return_value=0):
        with pytest.raises(subprocess.CalledProcessError):
            voice_receiver.VoiceReceiver.pcm_to_wav(b"\x00", "/tmp/out.wav")
