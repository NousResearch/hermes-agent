"""Integration tests for Discord voice channel audio flow.

Uses real NaCl encryption and Opus codec (no mocks for crypto/codec).
Does NOT require a Discord connection — tests the VoiceReceiver
packet processing pipeline end-to-end.

Requires: PyNaCl>=1.5.0, discord.py[voice] (opus codec)
"""

import struct
import time
import pytest

pytestmark = pytest.mark.integration

# Skip entire module if voice deps are missing
pytest.importorskip("nacl.secret", reason="PyNaCl required for voice integration tests")
discord = pytest.importorskip("discord", reason="discord.py required for voice integration tests")

import nacl.secret

try:
    if not discord.opus.is_loaded():
        import ctypes.util
        opus_path = ctypes.util.find_library("opus")
        if not opus_path:
            import sys
            for p in ("/opt/homebrew/lib/libopus.dylib", "/usr/local/lib/libopus.dylib"):
                import os
                if os.path.isfile(p):
                    opus_path = p
                    break
        if opus_path:
            discord.opus.load_opus(opus_path)
    OPUS_AVAILABLE = discord.opus.is_loaded()
except Exception:
    OPUS_AVAILABLE = False

from types import SimpleNamespace
from unittest.mock import MagicMock
from gateway.platforms.discord import VoiceReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_secret_key():
    """Generate a random 32-byte key."""
    import os
    return os.urandom(32)


def _build_encrypted_rtp_packet(secret_key, opus_payload, ssrc=100, seq=1, timestamp=960):
    """Build a real NaCl-encrypted RTP packet matching Discord's format.

    Format: RTP header (12 bytes) + encrypted(opus) + 4-byte nonce
    Encryption: aead_xchacha20_poly1305 with RTP header as AAD.
    """
    # RTP header: version=2, payload_type=0x78, no extension, no CSRC
    header = struct.pack(">BBHII", 0x80, 0x78, seq, timestamp, ssrc)

    # Encrypt with NaCl AEAD
    box = nacl.secret.Aead(secret_key)
    nonce_counter = struct.pack(">I", seq)  # 4-byte counter as nonce seed
    # Full 24-byte nonce: counter in first 4 bytes, rest zeros
    full_nonce = nonce_counter + b'\x00' * 20

    enc_msg = box.encrypt(opus_payload, header, full_nonce)
    ciphertext = enc_msg.ciphertext  # without nonce prefix

    # Discord format: header + ciphertext + 4-byte nonce
    return header + ciphertext + nonce_counter


def _build_padded_rtp_packet(
    secret_key, opus_payload, pad_len, ssrc=100, seq=1, timestamp=960,
    declared_pad_len=None, ext_words=0,
):
    """Build a NaCl-encrypted RTP packet with the P bit set and padding appended.

    Per RFC 3550 §5.1, the last padding byte declares how many trailing bytes
    (including itself) to discard. ``pad_len`` is the actual padding appended;
    ``declared_pad_len`` lets a test forge a mismatched declared length to
    exercise the validation path. ``ext_words`` > 0 also sets the X bit and
    prepends a synthetic extension block (4-byte preamble in cleartext header,
    ext_words*4 bytes of encrypted extension data prepended to the payload).
    """
    if pad_len < 1:
        raise ValueError("pad_len must be >= 1 (last byte includes itself)")
    declared = pad_len if declared_pad_len is None else declared_pad_len
    if declared < 0 or declared > 255:
        raise ValueError("declared_pad_len must fit in one byte")

    has_extension = ext_words > 0
    first_byte = 0xA0 | (0x10 if has_extension else 0)  # V=2, P=1, [X=?], CC=0
    fixed_header = struct.pack(">BBHII", first_byte, 0x78, seq, timestamp, ssrc)
    if has_extension:
        # 4-byte extension preamble: 2 bytes "defined by profile" + 2 bytes length-in-words
        ext_preamble = struct.pack(">HH", 0xBEDE, ext_words)
        header = fixed_header + ext_preamble
        ext_data = b"\xab" * (ext_words * 4)
    else:
        header = fixed_header
        ext_data = b""

    padding = b"\x00" * (pad_len - 1) + bytes([declared])
    plaintext = ext_data + opus_payload + padding

    box = nacl.secret.Aead(secret_key)
    nonce_counter = struct.pack(">I", seq)
    full_nonce = nonce_counter + b"\x00" * 20

    enc_msg = box.encrypt(plaintext, header, full_nonce)
    ciphertext = enc_msg.ciphertext

    return header + ciphertext + nonce_counter


def _make_voice_receiver(secret_key, dave_session=None, bot_ssrc=9999,
                         allowed_user_ids=None, members=None):
    """Create a VoiceReceiver with real secret key."""
    vc = MagicMock()
    vc._connection.secret_key = list(secret_key)
    vc._connection.dave_session = dave_session
    vc._connection.ssrc = bot_ssrc
    vc._connection.add_socket_listener = MagicMock()
    vc._connection.remove_socket_listener = MagicMock()
    vc._connection.hook = None
    vc.user = SimpleNamespace(id=bot_ssrc)
    vc.channel = MagicMock()
    vc.channel.members = members or []
    receiver = VoiceReceiver(vc, allowed_user_ids=allowed_user_ids)
    receiver.start()
    return receiver


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRealNaClDecrypt:
    """End-to-end: real NaCl encrypt → _on_packet decrypt → buffer."""

    def test_valid_encrypted_packet_buffered(self):
        """Real NaCl encrypted packet → decrypted → buffered."""
        key = _make_secret_key()
        opus_silence = b'\xf8\xff\xfe'
        receiver = _make_voice_receiver(key)

        packet = _build_encrypted_rtp_packet(key, opus_silence, ssrc=100)
        receiver._on_packet(packet)

        assert 100 in receiver._buffers
        assert len(receiver._buffers[100]) > 0

    def test_wrong_key_packet_dropped(self):
        """Packet encrypted with wrong key → NaCl fails → not buffered."""
        real_key = _make_secret_key()
        wrong_key = _make_secret_key()
        opus_silence = b'\xf8\xff\xfe'
        receiver = _make_voice_receiver(real_key)

        packet = _build_encrypted_rtp_packet(wrong_key, opus_silence, ssrc=100)
        receiver._on_packet(packet)

        assert len(receiver._buffers.get(100, b"")) == 0

    def test_bot_ssrc_ignored(self):
        """Packet from bot's own SSRC → ignored."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key, bot_ssrc=9999)

        packet = _build_encrypted_rtp_packet(key, b'\xf8\xff\xfe', ssrc=9999)
        receiver._on_packet(packet)

        assert len(receiver._buffers) == 0

    def test_multiple_packets_accumulate(self):
        """Multiple valid packets → buffer grows."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)

        for seq in range(1, 6):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver._on_packet(packet)

        assert 100 in receiver._buffers
        buf_size = len(receiver._buffers[100])
        assert buf_size > 0, "Multiple packets should accumulate in buffer"

    def test_different_ssrcs_separate_buffers(self):
        """Packets from different SSRCs → separate buffers."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)

        for ssrc in [100, 200, 300]:
            packet = _build_encrypted_rtp_packet(key, b'\xf8\xff\xfe', ssrc=ssrc)
            receiver._on_packet(packet)

        assert len(receiver._buffers) == 3
        for ssrc in [100, 200, 300]:
            assert ssrc in receiver._buffers


class TestRealNaClWithDAVE:
    """NaCl decrypt + DAVE passthrough scenarios with real crypto."""

    def test_dave_unknown_ssrc_passthrough(self):
        """DAVE enabled but SSRC unknown → skip DAVE, buffer audio."""
        key = _make_secret_key()
        dave = MagicMock()  # DAVE session present but SSRC not mapped
        receiver = _make_voice_receiver(key, dave_session=dave)

        packet = _build_encrypted_rtp_packet(key, b'\xf8\xff\xfe', ssrc=100)
        receiver._on_packet(packet)

        # DAVE decrypt not called (SSRC unknown)
        dave.decrypt.assert_not_called()
        # Audio still buffered via passthrough
        assert 100 in receiver._buffers
        assert len(receiver._buffers[100]) > 0

    def test_dave_unencrypted_error_passthrough(self):
        """DAVE raises 'Unencrypted' → use NaCl-decrypted data as-is."""
        key = _make_secret_key()
        dave = MagicMock()
        dave.decrypt.side_effect = Exception(
            "DecryptionFailed(UnencryptedWhenPassthroughDisabled)"
        )
        receiver = _make_voice_receiver(key, dave_session=dave)
        receiver.map_ssrc(100, 42)

        packet = _build_encrypted_rtp_packet(key, b'\xf8\xff\xfe', ssrc=100)
        receiver._on_packet(packet)

        # DAVE was called but failed → passthrough
        dave.decrypt.assert_called_once()
        assert 100 in receiver._buffers
        assert len(receiver._buffers[100]) > 0

    def test_dave_real_error_drops(self):
        """DAVE raises non-Unencrypted error → packet dropped."""
        key = _make_secret_key()
        dave = MagicMock()
        dave.decrypt.side_effect = Exception("KeyRotationFailed")
        receiver = _make_voice_receiver(key, dave_session=dave)
        receiver.map_ssrc(100, 42)

        packet = _build_encrypted_rtp_packet(key, b'\xf8\xff\xfe', ssrc=100)
        receiver._on_packet(packet)

        assert len(receiver._buffers.get(100, b"")) == 0


class TestRTPPaddingStrip:
    """RFC 3550 §5.1 — strip RTP padding before DAVE/Opus decode."""

    def test_padded_packet_stripped_and_buffered(self):
        """P bit set → trailing padding stripped → opus payload decoded."""
        key = _make_secret_key()
        opus_silence = b"\xf8\xff\xfe"
        receiver = _make_voice_receiver(key)

        # 5 bytes of padding (4 zeros + count byte = 5)
        packet = _build_padded_rtp_packet(key, opus_silence, pad_len=5, ssrc=100)
        receiver._on_packet(packet)

        assert 100 in receiver._buffers
        assert len(receiver._buffers[100]) > 0

    def test_padded_packet_matches_unpadded_output(self):
        """Same opus payload with/without padding → same decoded PCM."""
        key = _make_secret_key()
        opus_silence = b"\xf8\xff\xfe"

        recv_plain = _make_voice_receiver(key)
        recv_plain._on_packet(
            _build_encrypted_rtp_packet(key, opus_silence, ssrc=100)
        )

        recv_padded = _make_voice_receiver(key)
        recv_padded._on_packet(
            _build_padded_rtp_packet(key, opus_silence, pad_len=7, ssrc=100)
        )

        assert bytes(recv_plain._buffers[100]) == bytes(recv_padded._buffers[100])

    def test_padding_with_dave_passthrough(self):
        """Padding stripped before DAVE → passthrough buffers cleanly."""
        key = _make_secret_key()
        opus_silence = b"\xf8\xff\xfe"
        dave = MagicMock()  # SSRC unmapped → DAVE skipped, passthrough used
        receiver = _make_voice_receiver(key, dave_session=dave)

        packet = _build_padded_rtp_packet(key, opus_silence, pad_len=4, ssrc=100)
        receiver._on_packet(packet)

        dave.decrypt.assert_not_called()
        assert 100 in receiver._buffers
        assert len(receiver._buffers[100]) > 0

    def test_invalid_padding_length_zero_dropped(self):
        """Declared pad_len=0 is invalid (RFC requires count includes itself)."""
        key = _make_secret_key()
        opus_silence = b"\xf8\xff\xfe"
        receiver = _make_voice_receiver(key)

        packet = _build_padded_rtp_packet(
            key, opus_silence, pad_len=4, declared_pad_len=0, ssrc=100
        )
        receiver._on_packet(packet)

        assert len(receiver._buffers.get(100, b"")) == 0

    def test_invalid_padding_length_overflow_dropped(self):
        """Declared pad_len > payload size → packet dropped."""
        key = _make_secret_key()
        opus_silence = b"\xf8\xff\xfe"
        receiver = _make_voice_receiver(key)

        packet = _build_padded_rtp_packet(
            key, opus_silence, pad_len=4, declared_pad_len=255, ssrc=100
        )
        receiver._on_packet(packet)

        assert len(receiver._buffers.get(100, b"")) == 0

    def test_padding_consuming_entire_payload_dropped(self):
        """Padding consumes entire payload → no opus data → dropped."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)

        # Empty opus payload, 6 bytes of padding (count byte declares 6)
        packet = _build_padded_rtp_packet(key, b"", pad_len=6, ssrc=100)
        receiver._on_packet(packet)

        assert len(receiver._buffers.get(100, b"")) == 0

    def test_padding_with_extension_stripped_correctly(self):
        """X+P bits both set → strip extension from start, padding from end."""
        key = _make_secret_key()
        opus_silence = b"\xf8\xff\xfe"

        # Same opus payload sent two ways: plain, and with both ext+padding
        recv_plain = _make_voice_receiver(key)
        recv_plain._on_packet(
            _build_encrypted_rtp_packet(key, opus_silence, ssrc=100)
        )

        recv_ext_pad = _make_voice_receiver(key)
        recv_ext_pad._on_packet(
            _build_padded_rtp_packet(
                key, opus_silence, pad_len=5, ext_words=2, ssrc=100
            )
        )

        # Both must yield identical decoded PCM — ext data and padding both
        # stripped before opus decode.
        assert bytes(recv_plain._buffers[100]) == bytes(recv_ext_pad._buffers[100])


class TestFullVoiceFlow:
    """End-to-end: encrypt → receive → buffer → silence detect → complete."""

    def test_single_utterance_flow(self):
        """Encrypt packets → buffer → silence → check_silence returns utterance."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)
        receiver.map_ssrc(100, 42)

        # Send enough packets to exceed MIN_SPEECH_DURATION (0.5s)
        # At 48kHz stereo 16-bit, each Opus silence frame decodes to ~3840 bytes
        # Need 96000 bytes = ~25 frames
        for seq in range(1, 30):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver._on_packet(packet)

        # Simulate silence by setting last_packet_time in the past
        receiver._last_packet_time[100] = time.monotonic() - 3.0

        completed = receiver.check_silence()
        assert len(completed) == 1
        user_id, pcm_data = completed[0]
        assert user_id == 42
        assert len(pcm_data) > 0

    def test_utterance_with_ssrc_automap(self):
        """No SPEAKING event → auto-map sole allowed user → utterance processed."""
        key = _make_secret_key()
        members = [
            SimpleNamespace(id=9999, name="Bot"),
            SimpleNamespace(id=42, name="Alice"),
        ]
        receiver = _make_voice_receiver(
            key, allowed_user_ids={"42"}, members=members
        )
        # No map_ssrc call — simulating missing SPEAKING event

        for seq in range(1, 30):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver._on_packet(packet)

        receiver._last_packet_time[100] = time.monotonic() - 3.0

        completed = receiver.check_silence()
        assert len(completed) == 1
        assert completed[0][0] == 42  # auto-mapped to sole allowed user

    def test_pause_blocks_during_playback(self):
        """Pause receiver → packets ignored → resume → packets accepted."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)

        # Pause (echo prevention during TTS playback)
        receiver.pause()
        packet = _build_encrypted_rtp_packet(key, b'\xf8\xff\xfe', ssrc=100)
        receiver._on_packet(packet)
        assert len(receiver._buffers.get(100, b"")) == 0

        # Resume
        receiver.resume()
        receiver._on_packet(packet)
        assert 100 in receiver._buffers
        assert len(receiver._buffers[100]) > 0

    def test_corrupted_packet_ignored(self):
        """Corrupted/truncated packet → silently ignored."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)

        # Too short
        receiver._on_packet(b"\x00" * 5)
        assert len(receiver._buffers) == 0

        # Wrong RTP version
        bad_header = struct.pack(">BBHII", 0x00, 0x78, 1, 960, 100)
        receiver._on_packet(bad_header + b"\x00" * 20)
        assert len(receiver._buffers) == 0

        # Wrong payload type
        bad_pt = struct.pack(">BBHII", 0x80, 0x00, 1, 960, 100)
        receiver._on_packet(bad_pt + b"\x00" * 20)
        assert len(receiver._buffers) == 0

    def test_stop_cleans_everything(self):
        """stop() clears all state cleanly."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)
        receiver.map_ssrc(100, 42)

        for seq in range(1, 10):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver._on_packet(packet)

        assert len(receiver._buffers[100]) > 0

        receiver.stop()
        assert receiver._running is False
        assert len(receiver._buffers) == 0
        assert len(receiver._ssrc_to_user) == 0
        assert len(receiver._decoders) == 0


class TestSPEAKINGHook:
    """SPEAKING event hook correctly maps SSRC to user_id."""

    def test_speaking_hook_installed(self):
        """start() installs speaking hook on connection."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)
        conn = receiver._vc._connection
        # hook should be set (wrapped)
        assert conn.hook is not None

    def test_map_ssrc_via_speaking(self):
        """SPEAKING op 5 event maps SSRC to user_id."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)
        receiver.map_ssrc(500, 12345)
        assert receiver._ssrc_to_user[500] == 12345

    def test_map_ssrc_overwrites(self):
        """New SPEAKING event for same SSRC overwrites old mapping."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)
        receiver.map_ssrc(500, 111)
        receiver.map_ssrc(500, 222)
        assert receiver._ssrc_to_user[500] == 222

    def test_speaking_mapped_audio_processed(self):
        """After SSRC is mapped, audio from that SSRC gets correct user_id."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)
        receiver.map_ssrc(100, 42)

        for seq in range(1, 30):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver._on_packet(packet)

        receiver._last_packet_time[100] = time.monotonic() - 3.0
        completed = receiver.check_silence()
        assert len(completed) == 1
        assert completed[0][0] == 42


class TestAuthFiltering:
    """Only allowed users' audio should be processed."""

    def test_allowed_user_audio_processed(self):
        """Allowed user's utterance is returned by check_silence."""
        key = _make_secret_key()
        members = [
            SimpleNamespace(id=9999, name="Bot"),
            SimpleNamespace(id=42, name="Alice"),
        ]
        receiver = _make_voice_receiver(
            key, allowed_user_ids={"42"}, members=members,
        )
        receiver.map_ssrc(100, 42)

        for seq in range(1, 30):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver._on_packet(packet)

        receiver._last_packet_time[100] = time.monotonic() - 3.0
        completed = receiver.check_silence()
        assert len(completed) == 1
        assert completed[0][0] == 42

    def test_automap_rejects_unallowed_user(self):
        """Auto-map refuses to map SSRC to user not in allowed list."""
        key = _make_secret_key()
        members = [
            SimpleNamespace(id=9999, name="Bot"),
            SimpleNamespace(id=42, name="Alice"),
        ]
        receiver = _make_voice_receiver(
            key, allowed_user_ids={"99"},  # Alice not allowed
            members=members,
        )
        # No map_ssrc — SSRC unknown, auto-map should reject

        for seq in range(1, 30):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver._on_packet(packet)

        receiver._last_packet_time[100] = time.monotonic() - 3.0
        completed = receiver.check_silence()
        assert len(completed) == 0

    def test_empty_allowlist_allows_all(self):
        """Empty allowed_user_ids means no restriction."""
        key = _make_secret_key()
        members = [
            SimpleNamespace(id=9999, name="Bot"),
            SimpleNamespace(id=42, name="Alice"),
        ]
        receiver = _make_voice_receiver(
            key, allowed_user_ids=None, members=members,
        )

        for seq in range(1, 30):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver._on_packet(packet)

        receiver._last_packet_time[100] = time.monotonic() - 3.0
        completed = receiver.check_silence()
        # Auto-mapped to sole non-bot member
        assert len(completed) == 1
        assert completed[0][0] == 42


class TestRejoinFlow:
    """Leave and rejoin: state cleanup and fresh receiver."""

    def test_stop_then_new_receiver_clean_state(self):
        """After stop(), a new receiver starts with empty state."""
        key = _make_secret_key()
        receiver1 = _make_voice_receiver(key)
        receiver1.map_ssrc(100, 42)

        for seq in range(1, 10):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver1._on_packet(packet)

        assert len(receiver1._buffers[100]) > 0
        receiver1.stop()

        # New receiver (simulates rejoin)
        receiver2 = _make_voice_receiver(key)
        assert len(receiver2._buffers) == 0
        assert len(receiver2._ssrc_to_user) == 0
        assert len(receiver2._decoders) == 0

    def test_rejoin_new_ssrc_works(self):
        """After rejoin, user may get new SSRC — still works."""
        key = _make_secret_key()
        receiver1 = _make_voice_receiver(key)
        receiver1.map_ssrc(100, 42)  # old SSRC
        receiver1.stop()

        receiver2 = _make_voice_receiver(key)
        receiver2.map_ssrc(200, 42)  # new SSRC after rejoin

        for seq in range(1, 30):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=200, seq=seq, timestamp=960 * seq
            )
            receiver2._on_packet(packet)

        receiver2._last_packet_time[200] = time.monotonic() - 3.0
        completed = receiver2.check_silence()
        assert len(completed) == 1
        assert completed[0][0] == 42

    def test_rejoin_without_speaking_event_automap(self):
        """Rejoin without SPEAKING event — auto-map sole allowed user."""
        key = _make_secret_key()
        members = [
            SimpleNamespace(id=9999, name="Bot"),
            SimpleNamespace(id=42, name="Alice"),
        ]

        # First session
        receiver1 = _make_voice_receiver(
            key, allowed_user_ids={"42"}, members=members,
        )
        receiver1.stop()

        # Rejoin — new key (Discord may assign new secret_key)
        new_key = _make_secret_key()
        receiver2 = _make_voice_receiver(
            new_key, allowed_user_ids={"42"}, members=members,
        )
        # No map_ssrc — simulating missing SPEAKING event

        for seq in range(1, 30):
            packet = _build_encrypted_rtp_packet(
                new_key, b'\xf8\xff\xfe', ssrc=300, seq=seq, timestamp=960 * seq
            )
            receiver2._on_packet(packet)

        receiver2._last_packet_time[300] = time.monotonic() - 3.0
        completed = receiver2.check_silence()
        assert len(completed) == 1
        assert completed[0][0] == 42


class TestMultiGuildIsolation:
    """Each guild has independent voice state."""

    def test_separate_receivers_independent(self):
        """Two receivers (different guilds) don't interfere."""
        key1 = _make_secret_key()
        key2 = _make_secret_key()

        receiver1 = _make_voice_receiver(key1, bot_ssrc=1111)
        receiver2 = _make_voice_receiver(key2, bot_ssrc=2222)

        receiver1.map_ssrc(100, 42)
        receiver2.map_ssrc(200, 99)

        # Send to receiver1
        for seq in range(1, 10):
            packet = _build_encrypted_rtp_packet(
                key1, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver1._on_packet(packet)

        # receiver2 should be empty
        assert len(receiver2._buffers) == 0
        assert 100 in receiver1._buffers

    def test_stop_one_doesnt_affect_other(self):
        """Stopping one receiver doesn't affect another."""
        key1 = _make_secret_key()
        key2 = _make_secret_key()

        receiver1 = _make_voice_receiver(key1)
        receiver2 = _make_voice_receiver(key2)

        receiver1.map_ssrc(100, 42)
        receiver2.map_ssrc(200, 99)

        for seq in range(1, 10):
            packet = _build_encrypted_rtp_packet(
                key2, b'\xf8\xff\xfe', ssrc=200, seq=seq, timestamp=960 * seq
            )
            receiver2._on_packet(packet)

        receiver1.stop()

        # receiver2 still has data
        assert receiver2._running is True
        assert len(receiver2._buffers[200]) > 0


class TestEchoPreventionFlow:
    """Receiver pause/resume during TTS playback prevents echo."""

    def test_audio_during_pause_ignored(self):
        """Audio arriving while paused is completely ignored."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)
        receiver.map_ssrc(100, 42)
        receiver.pause()

        for seq in range(1, 30):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver._on_packet(packet)

        assert len(receiver._buffers.get(100, b"")) == 0

    def test_audio_after_resume_processed(self):
        """Audio arriving after resume is processed normally."""
        key = _make_secret_key()
        receiver = _make_voice_receiver(key)
        receiver.map_ssrc(100, 42)

        # Pause → send packets → resume → send more packets
        receiver.pause()
        for seq in range(1, 5):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver._on_packet(packet)
        assert len(receiver._buffers.get(100, b"")) == 0

        receiver.resume()
        for seq in range(5, 35):
            packet = _build_encrypted_rtp_packet(
                key, b'\xf8\xff\xfe', ssrc=100, seq=seq, timestamp=960 * seq
            )
            receiver._on_packet(packet)

        assert len(receiver._buffers[100]) > 0
        receiver._last_packet_time[100] = time.monotonic() - 3.0
        completed = receiver.check_silence()
        assert len(completed) == 1
        assert completed[0][0] == 42
