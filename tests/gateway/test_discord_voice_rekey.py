"""VoiceReceiver re-key resilience — credentials must track the live connection.

Discord rotates the transport ``secret_key`` on every voice (re)connect
(op 4 SESSION_DESCRIPTION), and ``reinit_dave_session`` REPLACES
``conn.dave_session`` when none existed yet (DAVE negotiating after the
receiver started).  A receiver that snapshots either value once at
``start()`` goes silently deaf — decrypt fails (or "succeeds" into
ciphertext-shredded opus) with nothing actionable in the logs.

Covers: resolve-from-connection, refresh on rotated key, the late
DAVE-session upgrade, the decrypt-failure-streak fallback, the op 4/24
websocket-hook trigger, and the DAVE unmapped-SSRC ciphertext drop.
"""

import asyncio
import struct
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("discord")

from plugins.platforms.discord.adapter import VoiceReceiver


KEY_A = bytes(range(32))
KEY_B = bytes(range(32, 64))


def _make_conn(
    secret_key=KEY_A,
    dave_session=None,
    ssrc=999,
    dave_protocol_version=0,
    dave_downgraded=False,
    dave_pending_transitions=None,
):
    return SimpleNamespace(
        secret_key=secret_key,
        dave_session=dave_session,
        dave_protocol_version=dave_protocol_version,
        dave_downgraded=dave_downgraded,
        dave_pending_transitions=dave_pending_transitions or {},
        ssrc=ssrc,
        hook=None,
        ws=None,
        add_socket_listener=lambda cb: None,
        remove_socket_listener=lambda cb: None,
    )


def _make_vc(conn, members=()):
    channel = SimpleNamespace(members=list(members))
    return SimpleNamespace(_connection=conn, channel=channel, user=SimpleNamespace(id=1))


def _make_receiver(conn, members=()):
    vc = _make_vc(conn, members)
    receiver = VoiceReceiver(vc)
    receiver.start()
    return receiver


def _rtp_packet(ssrc=1234, payload=b"e" * 8):
    """Minimal valid voice RTP packet: v2, PT 0x78, no ext/pad/csrc."""
    header = struct.pack(">BBHII", 0x80, 0x78, 1, 1000, ssrc)
    return header + payload + b"nonc"  # last 4 bytes = nonce suffix


class _FakeAead:
    """Stands in for nacl.secret.Aead — records the key, scripted results."""

    last_key = None
    fail = False
    plaintext = b"\xfc\xff\xfe"  # opus silence-ish frame

    def __init__(self, key):
        _FakeAead.last_key = bytes(key)

    def decrypt(self, encrypted, header, nonce):
        if _FakeAead.fail:
            raise ValueError("decrypt failed (stale key)")
        return _FakeAead.plaintext


@pytest.fixture
def fake_nacl(monkeypatch):
    """Route the in-function ``import nacl.secret`` to the fake."""
    _FakeAead.fail = False
    _FakeAead.last_key = None
    nacl_mod = types.ModuleType("nacl")
    secret_mod = types.ModuleType("nacl.secret")
    secret_mod.Aead = _FakeAead
    nacl_mod.secret = secret_mod
    monkeypatch.setitem(sys.modules, "nacl", nacl_mod)
    monkeypatch.setitem(sys.modules, "nacl.secret", secret_mod)
    return _FakeAead


class TestCredentialResolution:
    def test_start_resolves_from_connection(self):
        conn = _make_conn(secret_key=KEY_A, ssrc=42)
        receiver = _make_receiver(conn)
        assert receiver._secret_key == KEY_A
        assert receiver._bot_ssrc == 42

    def test_refresh_picks_up_rotated_transport_key(self):
        conn = _make_conn(secret_key=KEY_A)
        receiver = _make_receiver(conn)
        conn.secret_key = KEY_B
        receiver.refresh_credentials("test rotation")
        assert receiver._secret_key == KEY_B

    def test_refresh_picks_up_late_dave_session(self):
        """The None -> DaveSession upgrade: reinit_dave_session() CREATES a
        new session object when none existed at receiver start.  A stale
        captured None means every E2EE frame skips DAVE forever."""
        conn = _make_conn(dave_session=None)
        receiver = _make_receiver(conn)
        assert receiver._dave_session is None
        late_session = MagicMock(name="DaveSession")
        conn.dave_session = late_session
        receiver.refresh_credentials("dave negotiated")
        assert receiver._dave_session is late_session

    def test_refresh_after_stop_is_a_noop(self):
        conn = _make_conn(secret_key=KEY_A)
        receiver = _make_receiver(conn)
        receiver.stop()
        conn.secret_key = KEY_B
        receiver.refresh_credentials("late event")
        assert receiver._secret_key == KEY_A


class TestFailureStreakRefresh:
    def test_streak_triggers_refresh_and_reset(self, fake_nacl):
        conn = _make_conn(secret_key=KEY_A)
        receiver = _make_receiver(conn)
        fake_nacl.fail = True
        conn.secret_key = KEY_B  # the rotation the snapshot missed

        packet = _rtp_packet()
        for _ in range(VoiceReceiver.REKEY_FAILURE_STREAK):
            receiver._on_packet(packet)

        # Threshold hit → credentials re-resolved from the live connection
        # and the streak reset for the next retry window.
        assert receiver._secret_key == KEY_B
        assert receiver._nacl_fail_streak == 0
        assert receiver._decode_failed == VoiceReceiver.REKEY_FAILURE_STREAK

    def test_success_resets_streak(self, fake_nacl):
        conn = _make_conn(secret_key=KEY_A)
        receiver = _make_receiver(conn)
        fake_nacl.fail = True
        packet = _rtp_packet()
        for _ in range(5):
            receiver._on_packet(packet)
        assert receiver._nacl_fail_streak == 5
        fake_nacl.fail = False
        with patch("discord.opus.Decoder") as decoder_cls:
            decoder_cls.return_value.decode.return_value = b"\x00" * 3840
            receiver._on_packet(packet)
        assert receiver._nacl_fail_streak == 0


class TestWebsocketHookTrigger:
    def _install_and_get_hook(self, conn):
        vc = _make_vc(conn)
        receiver = VoiceReceiver(vc)
        receiver.start()  # installs the wrapped hook as conn.hook
        assert conn.hook is not None
        return receiver, conn.hook

    def test_op4_refreshes_credentials(self):
        conn = _make_conn(secret_key=KEY_A)
        receiver, hook = self._install_and_get_hook(conn)
        conn.secret_key = KEY_B
        asyncio.run(hook(None, {"op": 4, "d": {}}))
        assert receiver._secret_key == KEY_B

    def test_op24_refreshes_credentials(self):
        conn = _make_conn(dave_session=None)
        receiver, hook = self._install_and_get_hook(conn)
        late_session = MagicMock(name="DaveSession")
        conn.dave_session = late_session
        asyncio.run(hook(None, {"op": 24, "d": {"epoch": 1}}))
        assert receiver._dave_session is late_session

    def test_op5_still_maps_and_does_not_refresh(self):
        conn = _make_conn(secret_key=KEY_A)
        receiver, hook = self._install_and_get_hook(conn)
        conn.secret_key = KEY_B  # must NOT be picked up by op 5
        asyncio.run(hook(None, {"op": 5, "d": {"ssrc": 77, "user_id": 5}}))
        assert receiver._ssrc_to_user[77] == 5
        assert receiver._secret_key == KEY_A

    def test_original_hook_still_called(self):
        conn = _make_conn()
        calls = []

        async def original(ws, msg):
            calls.append(msg)

        conn.hook = original
        vc = _make_vc(conn)
        receiver = VoiceReceiver(vc)
        receiver.start()
        asyncio.run(conn.hook(None, {"op": 4, "d": {}}))
        assert calls == [{"op": 4, "d": {}}]


class TestDaveUnmappedDrop:
    def test_unmapped_ssrc_never_reaches_opus(self, fake_nacl):
        """With E2EE actively on (protocol > 0, no passthrough window), an
        unmapped SSRC's payload is ciphertext — it must be dropped, not fed
        to the opus decoder."""
        conn = _make_conn(
            dave_session=MagicMock(name="DaveSession"), dave_protocol_version=1
        )
        # Two members in channel → the sole-member inference cannot map.
        members = [SimpleNamespace(id=2), SimpleNamespace(id=3)]
        receiver = _make_receiver(conn, members=members)

        with patch("discord.opus.Decoder") as decoder_cls:
            receiver._on_packet(_rtp_packet(ssrc=555))
            decoder_cls.assert_not_called()
        assert receiver._dave_unmapped_dropped == 1
        assert 555 not in receiver._decoders

    def test_downgraded_protocol_zero_decodes_unmapped_plaintext(self, fake_nacl):
        """A non-null session is NOT proof of encryption: after a downgrade
        transition the session object survives with the protocol at 0 and
        senders emit plaintext — unmapped frames must decode, not drop."""
        conn = _make_conn(
            dave_session=MagicMock(name="DaveSession"), dave_protocol_version=0
        )
        members = [SimpleNamespace(id=2), SimpleNamespace(id=3)]
        receiver = _make_receiver(conn, members=members)

        with patch("discord.opus.Decoder") as decoder_cls:
            decoder_cls.return_value.decode.return_value = b"\x00" * 3840
            receiver._on_packet(_rtp_packet(ssrc=555))
            decoder_cls.assert_called_once()
        assert receiver._dave_unmapped_dropped == 0
        assert receiver._decode_ok == 1

    def test_passthrough_window_decodes_unmapped_plaintext(self, fake_nacl):
        """During the op-21 downgrade-pending window (protocol still > 0),
        plaintext passthrough is legitimate — unmapped frames must decode."""
        conn = _make_conn(
            dave_session=MagicMock(name="DaveSession"), dave_protocol_version=1
        )
        members = [SimpleNamespace(id=2), SimpleNamespace(id=3)]
        receiver = _make_receiver(conn, members=members)
        receiver.note_dave_passthrough_window(120.0)

        with patch("discord.opus.Decoder") as decoder_cls:
            decoder_cls.return_value.decode.return_value = b"\x00" * 3840
            receiver._on_packet(_rtp_packet(ssrc=555))
            decoder_cls.assert_called_once()
        assert receiver._dave_unmapped_dropped == 0

    def test_expired_passthrough_window_drops_again(self, fake_nacl):
        conn = _make_conn(
            dave_session=MagicMock(name="DaveSession"), dave_protocol_version=1
        )
        members = [SimpleNamespace(id=2), SimpleNamespace(id=3)]
        receiver = _make_receiver(conn, members=members)
        receiver._dave_passthrough_until = 0.0  # long expired

        with patch("discord.opus.Decoder") as decoder_cls:
            receiver._on_packet(_rtp_packet(ssrc=555))
            decoder_cls.assert_not_called()
        assert receiver._dave_unmapped_dropped == 1

    def test_op21_downgrade_opens_window_and_op22_refreshes(self):
        conn = _make_conn(
            dave_session=MagicMock(name="DaveSession"), dave_protocol_version=1
        )
        vc = _make_vc(conn)
        receiver = VoiceReceiver(vc)
        receiver.start()
        hook = conn.hook

        assert receiver._dave_passthrough_until == 0.0
        asyncio.run(hook(None, {"op": 21, "d": {"protocol_version": 0, "transition_id": 3}}))
        assert receiver._dave_passthrough_until > 0.0

        # op 22 executing the downgrade: the protocol version changed —
        # refresh must pick it up from the connection.
        conn.dave_protocol_version = 0
        conn.dave_downgraded = True
        asyncio.run(hook(None, {"op": 22, "d": {"transition_id": 3}}))
        assert receiver._dave_protocol_version == 0

    def test_op22_upgrade_edge_grants_short_grace_replacing_long_window(self):
        """Only a confirmed upgrade-after-downgrade (dave_downgraded flipping
        True -> False) opens the 10s grace — and it REPLACES a residual 120s
        downgrade window instead of being swallowed by it."""
        conn = _make_conn(
            dave_session=MagicMock(name="DaveSession"),
            dave_protocol_version=0,
            dave_downgraded=True,
        )
        vc = _make_vc(conn)
        receiver = VoiceReceiver(vc)
        receiver.start()
        hook = conn.hook

        # Residual long window from an earlier downgrade-prepare.
        receiver.note_dave_passthrough_window(120.0)
        long_deadline = receiver._dave_passthrough_until

        # Upstream executes the upgrade: protocol back up, downgraded off.
        conn.dave_protocol_version = 1
        conn.dave_downgraded = False
        asyncio.run(hook(None, {"op": 22, "d": {"transition_id": 4}}))

        assert receiver._dave_protocol_version == 1
        # Window replaced by the SHORT grace, not max-extended.
        assert receiver._dave_passthrough_until < long_deadline
        assert receiver._dave_passthrough_until > time.monotonic()

    def test_op22_same_version_transition_opens_no_window(self):
        """Same-version MLS transitions grant no passthrough upstream —
        under active E2EE an op 22 must NOT open a plaintext hole."""
        conn = _make_conn(
            dave_session=MagicMock(name="DaveSession"),
            dave_protocol_version=1,
            dave_downgraded=False,
        )
        vc = _make_vc(conn)
        receiver = VoiceReceiver(vc)
        receiver.start()
        hook = conn.hook

        asyncio.run(hook(None, {"op": 22, "d": {"transition_id": 5}}))
        assert receiver._dave_passthrough_until == 0.0

    def test_op22_duplicate_or_unknown_transition_opens_no_window(self):
        """An op 22 for an unknown/duplicate transition id changes no state
        upstream — no edge, no window."""
        conn = _make_conn(
            dave_session=MagicMock(name="DaveSession"),
            dave_protocol_version=1,
            dave_downgraded=False,
        )
        vc = _make_vc(conn)
        receiver = VoiceReceiver(vc)
        receiver.start()
        hook = conn.hook

        for _ in range(3):
            asyncio.run(hook(None, {"op": 22, "d": {"transition_id": 99}}))
        assert receiver._dave_passthrough_until == 0.0

    def test_start_mid_pending_downgrade_seeds_window(self, fake_nacl):
        """A receiver that starts while a downgrade transition is already
        pending never saw the op 21 — it must seed the passthrough window
        from the connection's physical pending-transition state instead of
        dropping legitimate plaintext for up to 120s."""
        conn = _make_conn(
            dave_session=MagicMock(name="DaveSession"),
            dave_protocol_version=1,
            dave_pending_transitions={3: 0},
        )
        members = [SimpleNamespace(id=2), SimpleNamespace(id=3)]
        receiver = _make_receiver(conn, members=members)

        assert receiver._dave_passthrough_until > 0.0
        with patch("discord.opus.Decoder") as decoder_cls:
            decoder_cls.return_value.decode.return_value = b"\x00" * 3840
            receiver._on_packet(_rtp_packet(ssrc=555))
            decoder_cls.assert_called_once()
        assert receiver._dave_unmapped_dropped == 0

    def test_credentials_swap_is_one_tuple(self):
        """The receive thread reads one consistent generation — key, session,
        protocol version, and downgrade flag always come from the same
        refresh."""
        conn = _make_conn(secret_key=KEY_A, dave_protocol_version=1)
        receiver = _make_receiver(conn)
        before = receiver._creds
        conn.secret_key = KEY_B
        conn.dave_session = MagicMock(name="DaveSession")
        conn.dave_protocol_version = 2
        receiver.refresh_credentials("test")
        after = receiver._creds
        assert before is not after
        assert after == (KEY_B, conn.dave_session, 2, False)

    def test_mapped_user_flows_through_dave_to_opus(self, fake_nacl):
        dave = MagicMock(name="DaveSession")
        dave.decrypt.return_value = b"\xfc\xff\xfe"
        conn = _make_conn(dave_session=dave)
        receiver = _make_receiver(conn)
        receiver.map_ssrc(555, 7)

        davey_mod = types.ModuleType("davey")
        davey_mod.MediaType = SimpleNamespace(audio="audio")
        with patch.dict(sys.modules, {"davey": davey_mod}):
            with patch("discord.opus.Decoder") as decoder_cls:
                decoder_cls.return_value.decode.return_value = b"\x00" * 3840
                receiver._on_packet(_rtp_packet(ssrc=555))

        dave.decrypt.assert_called_once()
        assert receiver._decode_ok == 1
        assert receiver._dave_unmapped_dropped == 0

    def test_sole_member_inference_rescues_unmapped_ssrc(self, fake_nacl):
        """One allowed member in channel → the existing inference maps the
        SSRC and the frame goes through DAVE instead of being dropped."""
        dave = MagicMock(name="DaveSession")
        dave.decrypt.return_value = b"\xfc\xff\xfe"
        conn = _make_conn(dave_session=dave)
        receiver = _make_receiver(conn, members=[SimpleNamespace(id=9)])

        davey_mod = types.ModuleType("davey")
        davey_mod.MediaType = SimpleNamespace(audio="audio")
        with patch.dict(sys.modules, {"davey": davey_mod}):
            with patch("discord.opus.Decoder") as decoder_cls:
                decoder_cls.return_value.decode.return_value = b"\x00" * 3840
                receiver._on_packet(_rtp_packet(ssrc=556))

        assert receiver._ssrc_to_user[556] == 9
        dave.decrypt.assert_called_once()
        assert receiver._dave_unmapped_dropped == 0


class TestTeardownHealthLine:
    def test_stop_logs_decode_health(self, caplog):
        import logging

        conn = _make_conn()
        receiver = _make_receiver(conn)
        receiver._decode_ok = 10
        receiver._decode_failed = 3
        receiver._dave_unmapped_dropped = 2
        with caplog.at_level(logging.INFO):
            receiver.stop()
        line = "\n".join(r.getMessage() for r in caplog.records)
        assert "ok=10" in line
        assert "decrypt_failed=3" in line
        assert "dave_unmapped_dropped=2" in line
