"""Buzz observer telemetry — NIP-44 v2 port and event builders.

The crypto is validated against the complete official paulmillr/nip44
vectors (tests/fixtures/nip44.vectors.json, from
javascript/test/nip44.vectors.json) so the wire format cannot drift.
"""

import importlib.util
import json
import hashlib
import sys
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).parents[2] / "plugins" / "platforms" / "buzz"
_spec = importlib.util.spec_from_file_location(
    "buzz_observer_under_test", _PLUGIN_DIR / "observer.py")
assert _spec is not None and _spec.loader is not None
observer = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = observer
_spec.loader.exec_module(observer)

_VECTORS = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "nip44.vectors.json").read_text())["v2"]["valid"]
_VECTORS_INVALID = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "nip44.vectors.json").read_text())["v2"]["invalid"]

_TEST_PRIV = "01" * 32  # deterministic test key (hex form)


@pytest.mark.parametrize("v", _VECTORS["get_conversation_key"])
def test_conversation_key_vector(v):
    assert observer.get_conversation_key(v["sec1"], v["pub2"]).hex() == v["conversation_key"]


def test_message_keys_vector():
    spec = _VECTORS["get_message_keys"]
    ck = bytes.fromhex(spec["conversation_key"])
    for v in spec["keys"]:
        ckk, cn, mk = observer._get_message_keys(ck, bytes.fromhex(v["nonce"]))
        assert (ckk.hex(), cn.hex(), mk.hex()) == \
            (v["chacha_key"], v["chacha_nonce"], v["hmac_key"])


@pytest.mark.parametrize("n,want", _VECTORS["calc_padded_len"])
def test_calc_padded_len(n, want):
    assert observer._calc_padded_len(n) == want


@pytest.mark.parametrize("v", _VECTORS["encrypt_decrypt"])
def test_encrypt_vector(v):
    pub2 = observer.public_key_hex(v["sec2"])
    ck = observer.get_conversation_key(v["sec1"], pub2)
    assert ck.hex() == v["conversation_key"]
    got = observer.nip44_encrypt(v["plaintext"], ck, bytes.fromhex(v["nonce"]))
    assert got == v["payload"]
    assert observer.nip44_decrypt(got, ck) == v["plaintext"]


def _bare_publisher():
    pub = observer.ObserverPublisher.__new__(observer.ObserverPublisher)
    pub._priv = _TEST_PRIV
    pub.agent_pub = observer.public_key_hex(_TEST_PRIV)
    pub.session_id = "sess-test"
    from itertools import count
    pub._seq = count(1)
    return pub


def _bip340_valid(pubkey_hex: str, msg: bytes, sig: bytes) -> bool:
    """BIP-340 verification, independent of the signer (only curve arithmetic is shared)."""
    na = observer._na
    n = na.CURVE_ORDER
    r, s_ = int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big")
    if r >= observer._FIELD_P or s_ >= n:
        return False
    point = observer._lift_x(pubkey_hex)
    e = int.from_bytes(na._tagged_hash("BIP0340/challenge", sig[:32] + bytes.fromhex(pubkey_hex) + msg),
                       "big") % n
    big_r = na._point_add(na._point_multiply(s_), na._point_multiply(n - e, point))
    return big_r is not None and big_r[1] % 2 == 0 and big_r[0] == r


def _verify_sig(ev):
    """Recompute the NIP-01 id and verify the BIP-340 signature over it."""
    ser = json.dumps([0, ev["pubkey"], ev["created_at"], ev["kind"], ev["tags"], ev["content"]],
                     separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(ser).hexdigest() == ev["id"]
    assert _bip340_valid(ev["pubkey"], bytes.fromhex(ev["id"]), bytes.fromhex(ev["sig"]))


def test_signature_check_rejects_a_tampered_signature():
    ev = _bare_publisher()._presence_event("online")
    sig = bytearray(bytes.fromhex(ev["sig"]))
    sig[40] ^= 1
    assert not _bip340_valid(ev["pubkey"], bytes.fromhex(ev["id"]), bytes(sig))
    assert not _bip340_valid(ev["pubkey"], hashlib.sha256(b"other").digest(), bytes.fromhex(ev["sig"]))


def test_observer_event_shape():
    pub = _bare_publisher()
    ev = pub._observer_event("turn_started", {"source": "channel", "triggeringEventIds": ["e1"]},
                             channel_id="ch", turn_id="t", started_at="2026-11-06T00:00:00+00:00")
    assert ev["kind"] == "turn_started"
    assert ev["channelId"] == "ch" and ev["sessionId"] == "sess-test"
    assert ev["agentIndex"] == 0 and ev["seq"] == 1
    assert ev["payload"]["triggeringEventIds"] == ["e1"]


def test_typing_event_wire():
    ev = _bare_publisher()._typing_event("chan-1", "root-1")
    assert ev["kind"] == 20002 and ev["content"] == ""
    assert ev["tags"] == [["h", "chan-1"], ["e", "root-1", "", "root"]]
    _verify_sig(ev)


def test_presence_event_wire():
    ev = _bare_publisher()._presence_event("offline")
    assert ev["kind"] == 20001 and ev["content"] == "offline"
    assert ev["tags"] == [["status", "offline"]]
    _verify_sig(ev)


class TestNip44InvalidVectors:
    """The fixture's v2.invalid section must be rejected (pubkey validation)."""

    def test_invalid_get_conversation_key_rejects(self):
        vecs = _VECTORS_INVALID["get_conversation_key"]
        assert vecs, "fixture sin vectores invalid"
        for v in vecs:
            with pytest.raises(Exception):
                observer.get_conversation_key(v["sec1"], v["pub2"])


class TestNip44LongMsgVectors:
    """The encrypt_decrypt_long_msg vectors (padding of 64 KB+ plaintexts)."""

    def test_encrypt_decrypt_long_msg(self):
        # stored compressed in the fixture: pattern * repeat, checked by sha256
        for v in _VECTORS["encrypt_decrypt_long_msg"]:
            plaintext = v["pattern"] * v["repeat"]
            assert hashlib.sha256(plaintext.encode()).hexdigest() == v["plaintext_sha256"]
            ck = bytes.fromhex(v["conversation_key"])
            got = observer.nip44_encrypt(plaintext, ck, bytes.fromhex(v["nonce"]))
            assert hashlib.sha256(got.encode()).hexdigest() == v["payload_sha256"]
            assert observer.nip44_decrypt(got, ck) == plaintext


def test_invalid_encrypt_msg_lengths_reject():
    # plaintext LENGTHS encrypt must reject (0 and >= 65536 break NIP-44 padding)
    _v0 = _VECTORS["encrypt_decrypt"][0]
    ck = observer.get_conversation_key(_v0["sec1"], observer.public_key_hex(_v0["sec2"]))
    nonce = bytes(32)
    for length in _VECTORS_INVALID["encrypt_msg_lengths"]:
        with pytest.raises(Exception):
            observer.nip44_encrypt("x" * length, ck, nonce)


def test_invalid_decrypt_rejects():
    # these vectors carry the conversation_key directly (no sec1/pub2)
    for v in _VECTORS_INVALID["decrypt"]:
        ck = bytes.fromhex(v["conversation_key"])
        with pytest.raises(Exception):
            observer.nip44_decrypt(v["payload"], ck)
