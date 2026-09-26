"""VoiceReceiver: audio from an SSRC that has no user yet (no SPEAKING since the bot joined).

Before the fix, DAVE (E2EE) frames from such an SSRC were Opus-decoded as ciphertext. The resulting
noise was a full "utterance", which Whisper turned into words from its STT prompt, so the first thing
said after every ``/voice join`` came out as junk. Now the SSRC is attributed on its first packet when
one allowed member is in the channel, and an E2EE frame that still can't be attributed is dropped.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SSRC, E2EE_FRAME = 5, b"ciphertext\xfa\xfa"  # DAVE frames end with the 0xFAFA magic marker


@pytest.fixture(autouse=True)
def fake_davey(monkeypatch):
    """The receiver imports ``davey`` lazily; stand in for it so the voice extra isn't required."""
    monkeypatch.setitem(sys.modules, "davey", SimpleNamespace(MediaType=SimpleNamespace(audio="audio")))


def _receiver(member_ids):
    from plugins.platforms.discord.adapter import VoiceReceiver
    vc = MagicMock()
    vc.user.id = 999
    vc.channel.members = [SimpleNamespace(id=m) for m in [999, *member_ids]]
    receiver = VoiceReceiver(vc, allowed_user_ids={"1", "2"})
    receiver._dave_session = MagicMock()
    receiver._dave_session.decrypt.side_effect = lambda uid, media, payload: b"opus"
    decoder = MagicMock()
    decoder.decode.return_value = b"\x01\x00" * 1920
    receiver._decoders[SSRC] = decoder
    return receiver, decoder


def test_first_packet_attributed_to_sole_member_and_decrypted():
    receiver, decoder = _receiver([1])
    receiver._decode_payload(SSRC, E2EE_FRAME)
    assert receiver._ssrc_to_user[SSRC] == 1
    assert receiver._dave_session.decrypt.call_args[0][0] == 1
    decoder.decode.assert_called_once_with(b"opus")
    assert len(receiver._buffers[SSRC]) == 3840


def test_unattributable_e2ee_frame_is_dropped_not_decoded_as_noise():
    receiver, decoder = _receiver([1, 2])  # two allowed members: no sole-member inference
    receiver._decode_payload(SSRC, E2EE_FRAME)
    decoder.decode.assert_not_called()
    assert not receiver._buffers.get(SSRC)


def test_unattributable_plain_opus_still_decoded():
    receiver, decoder = _receiver([1, 2])
    receiver._decode_payload(SSRC, b"plain-opus")  # unencrypted passthrough keeps working
    decoder.decode.assert_called_once_with(b"plain-opus")
    receiver._dave_session.decrypt.assert_not_called()


def test_mapped_ssrc_uses_mapping_without_inference(monkeypatch):
    receiver, decoder = _receiver([1, 2])
    receiver.map_ssrc(SSRC, 2)
    monkeypatch.setattr(receiver, "_infer_user_for_ssrc", MagicMock(side_effect=AssertionError("inferred")))
    receiver._decode_payload(SSRC, E2EE_FRAME)
    assert receiver._dave_session.decrypt.call_args[0][0] == 2
    decoder.decode.assert_called_once_with(b"opus")


def test_inference_retried_at_most_once_a_second(monkeypatch):
    receiver, _ = _receiver([1, 2])
    infer = MagicMock(return_value=0)
    monkeypatch.setattr(receiver, "_infer_user_for_ssrc", infer)
    for _ in range(50):  # ~1 s of 20 ms packets
        receiver._decode_payload(SSRC, E2EE_FRAME)
    assert infer.call_count == 1


def test_stop_clears_inference_state():
    receiver, _ = _receiver([1, 2])
    receiver._decode_payload(SSRC, E2EE_FRAME)
    receiver.stop()
    assert receiver._infer_attempts == {}
