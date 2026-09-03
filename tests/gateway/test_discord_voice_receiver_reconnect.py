"""Regression coverage for Discord voice reconnect and DAVE activation.

Discord.py retains one VoiceConnectionState for a VoiceClient, but mutates its
key, SSRC and DAVE session in place after a voice-server reconnect. The receiver
must therefore not retain join-time snapshots.
"""

from __future__ import annotations

import os
import struct

import nacl.secret
import nacl.utils
import discord

from plugins.platforms.discord.adapter import VoiceReceiver


class FakeConn:
    def __init__(self) -> None:
        self.secret_key = list(nacl.utils.random(32))
        self.ssrc = 100
        self.dave_session = None
        self.hook = None
        self.ws = discord.utils.MISSING
        self.listeners = []

    def add_socket_listener(self, callback):
        self.listeners.append(callback)

    def remove_socket_listener(self, callback):
        self.listeners.remove(callback)

    def rotate(self) -> None:
        # Match discord.py VoiceConnectionState: mutate in place, do not replace.
        self.secret_key = list(nacl.utils.random(32))
        self.ssrc = 200


class FakeVoiceClient:
    def __init__(self) -> None:
        self._connection = FakeConn()


class DecoderMap(dict):
    """Decoder substitute, so the tests exercise the receiver's decrypt path."""

    def __missing__(self, ssrc):
        class Decoder:
            def decode(self, _payload):
                return b"\0" * 3840

        self[ssrc] = Decoder()
        return self[ssrc]


class StubDecoder:
    def decode(self, _payload):
        return b"\0" * 3840


class DaveNotReady:
    """The real davey error while MLS lacks a sender decryptor."""

    def decrypt(self, *_args):
        raise ValueError("Failed to decrypt: NoDecryptorForUser")


def packet(conn: FakeConn, ssrc: int, sequence: int) -> bytes:
    header = struct.pack(">BBHII", 0x80, 0x78, sequence, sequence * 960, ssrc)
    nonce_suffix = struct.pack(">I", sequence)
    nonce = bytearray(24)
    nonce[:4] = nonce_suffix
    encrypted = nacl.secret.Aead(bytes(conn.secret_key)).encrypt(
        b"\xfc\xff\xfe" + os.urandom(32), header, bytes(nonce)
    ).ciphertext
    return header + encrypted + nonce_suffix


def build_receiver():
    vc = FakeVoiceClient()
    receiver = VoiceReceiver(vc)
    receiver._decoders = DecoderMap()
    receiver._decoders[300] = StubDecoder()
    receiver.start()
    receiver.map_ssrc(300, 42)
    return receiver, vc


def buffered_bytes(receiver: VoiceReceiver) -> int:
    with receiver._lock:
        return sum(len(value) for value in receiver._buffers.values())


def test_receiver_uses_new_key_and_ssrc_after_in_place_rotation():
    receiver, vc = build_receiver()
    receiver._on_packet(packet(vc._connection, 300, 1))
    before_rotation = buffered_bytes(receiver)

    vc._connection.rotate()
    receiver._on_packet(packet(vc._connection, 300, 2))

    assert buffered_bytes(receiver) > before_rotation
    # The previous bot SSRC is no longer self-audio; the new one is.
    receiver._on_packet(packet(vc._connection, 200, 3))
    assert receiver._bot_ssrc == 200


def test_dave_not_ready_falls_through_to_opus_not_a_permanent_drop():
    receiver, vc = build_receiver()
    vc._connection.dave_session = DaveNotReady()

    receiver._on_packet(packet(vc._connection, 300, 1))

    assert buffered_bytes(receiver) > 0


def test_missing_key_during_handshake_does_not_crash_socket_reader():
    receiver, vc = build_receiver()
    encrypted_while_key_was_valid = packet(vc._connection, 300, 1)
    vc._connection.secret_key = discord.utils.MISSING

    # _on_packet should discard the transient frame cleanly, not throw in the
    # long-lived SocketReader thread.
    receiver._on_packet(encrypted_while_key_was_valid)
