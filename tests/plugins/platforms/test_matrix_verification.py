"""Unit tests for the Matrix SAS verification handler crypto primitives.

Covers the pure, spec-critical functions of
``plugins/platforms/matrix/verification.py``: canonical JSON, unpadded
base64, the SAS commitment, the emoji short-auth-string derivation, the
HKDF info strings for SAS/MAC, transaction-id resolution and session TTL
expiry.  No network, no mautrix required — stdlib + pytest + mocks only.
"""

import base64
import hashlib
import json
import time
import types

import asyncio

import pytest

from plugins.platforms.matrix.verification import (
    _EMOJIS,
    _SESSION_TTL_SECONDS,
    _SasSession,
    _bytes_to_emoji_indices,
    _canonical_json,
    _compute_commitment,
    _plain,
    _txid_from,
    _unpadded_base64,
    SasVerificationHandler,
)


# ---------------------------------------------------------------------------
# _canonical_json
# ---------------------------------------------------------------------------


class TestCanonicalJson:
    def test_sorted_keys_compact_separators(self):
        assert _canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_recursive_sorting_and_utf8(self):
        out = _canonical_json({"z": {"y": "ü", "x": 1}, "a": [3, 1]})
        # Keys sorted at every level, no whitespace, non-ASCII left as UTF-8.
        assert out == '{"a":[3,1],"z":{"x":1,"y":"ü"}}'

    def test_empty_and_scalars(self):
        assert _canonical_json({}) == "{}"
        assert _canonical_json({"a": None}) == '{"a":null}'


# ---------------------------------------------------------------------------
# _unpadded_base64
# ---------------------------------------------------------------------------


class TestUnpaddedBase64:
    def test_known_vector(self):
        raw = b"hello"
        expected = base64.b64encode(raw).decode("ascii").rstrip("=")
        assert _unpadded_base64(raw) == expected
        assert "=" not in _unpadded_base64(b"any length data")

    def test_padding_is_always_stripped(self):
        # "hi" base64-encodes to aGH= (one padding char); must be stripped.
        assert _unpadded_base64(b"hi") == "aGk"


# ---------------------------------------------------------------------------
# _plain
# ---------------------------------------------------------------------------


class TestPlain:
    def test_primitives_pass_through(self):
        for v in (None, "x", b"y", 3, 3.5, True):
            assert _plain(v) is v

    def test_dicts_and_lists_recursively(self):
        assert _plain({"a": [1, {"b": None}]}) == {"a": [1, {"b": None}]}

    def test_attr_container_falls_back_to_dict(self):
        obj = types.SimpleNamespace(a=1, b="x")
        assert _plain(obj) == {"a": 1, "b": "x"}

    def test_serialize_is_preferred(self):
        class WithSerialize:
            def serialize(self):
                return {"a": 1}

        assert _plain(WithSerialize()) == {"a": 1}


# ---------------------------------------------------------------------------
# _txid_from
# ---------------------------------------------------------------------------


class TestTxidFrom:
    def test_legacy_transaction_id_wins(self):
        content = {"transaction_id": "legacy-1", "m.relates_to": {"event_id": "$anchor"}}
        assert _txid_from(content) == "legacy-1"

    def test_in_room_anchor_event_id(self):
        content = {"m.relates_to": {"event_id": "$anchor"}}
        assert _txid_from(content) == "$anchor"

    def test_falls_back_to_event_id(self):
        event = types.SimpleNamespace(event_id="$evt")
        assert _txid_from({}, event) == "$evt"

    def test_empty(self):
        assert _txid_from({}) == ""
        assert _txid_from(None) == ""


# ---------------------------------------------------------------------------
# _compute_commitment  (independent known vector)
# ---------------------------------------------------------------------------


class TestComputeCommitment:
    PUBKEY = "Be4Yx8X7d2kPq0sW"
    START_CONTENT = {
        "transaction_id": "t-123",
        "method": "m.sas.v1",
        "from_device": "DEV1",
    }
    # Independently computed: unpadded_b64(SHA256(pubkey + canonical_json(start)))
    EXPECTED = "y/dy80pLdheIvnMMC4W7ScR7dU8goesN23/3Y2G+Re0"

    def test_matches_spec_formula(self):
        assert _compute_commitment(self.PUBKEY, self.START_CONTENT) == self.EXPECTED

    def test_changes_with_key_or_content(self):
        a = _compute_commitment(self.PUBKEY, self.START_CONTENT)
        b = _compute_commitment(self.PUBKEY + "x", self.START_CONTENT)
        c = _compute_commitment(self.PUBKEY, {**self.START_CONTENT, "to": "@u:x"})
        assert len({a, b, c}) == 3


# ---------------------------------------------------------------------------
# _bytes_to_emoji_indices
# ---------------------------------------------------------------------------


class TestBytesToEmojiIndices:
    def test_known_vector(self):
        raw = bytes.fromhex("010203040506")
        assert _bytes_to_emoji_indices(raw) == [0, 16, 8, 3, 1, 0, 20]

    def test_all_zero_bytes(self):
        assert _bytes_to_emoji_indices(b"\x00" * 6) == [0] * 7

    def test_all_ones(self):
        assert _bytes_to_emoji_indices(b"\xff" * 6) == [63] * 7

    def test_only_first_42_bits_used(self):
        # 0xfc = 11111100: the top 6 bits are all ones (first group = 63);
        # everything below is zero, so the remaining groups are 0.
        raw = bytes.fromhex("fc0000000000")
        assert _bytes_to_emoji_indices(raw)[0] == 63
        assert all(i == 0 for i in _bytes_to_emoji_indices(raw)[1:])

    def test_always_seven_indices_in_range(self):
        for raw in (b"\x00" * 6, b"\x12\x34\x56\x78\x9a\xbc", b"\xff" * 6):
            idx = _bytes_to_emoji_indices(raw)
            assert len(idx) == 7
            assert all(0 <= i < 64 for i in idx)


class TestEmojiTable:
    def test_spec_table_64_entries(self):
        assert len(_EMOJIS) == 64
        assert _EMOJIS[0] == ("🐶", "Dog")
        assert _EMOJIS[63] == ("📌", "Pin")

    def test_all_entries_are_emoji_name_pairs(self):
        assert all(isinstance(e, tuple) and len(e) == 2 for e in _EMOJIS)
        assert all(isinstance(e[0], str) and isinstance(e[1], str) for e in _EMOJIS)


# ---------------------------------------------------------------------------
# Handler-level pure helpers (_sas_info / _mac_info_for / expiry / emojis)
# ---------------------------------------------------------------------------


def _make_handler():
    client = types.SimpleNamespace(mxid="@bot:example.org", device_id="BOTDEV")
    return SasVerificationHandler(adapter=None, client=client, olm=None)


def _make_session(we_initiated=False):
    return _SasSession(
        transaction_id="t-1",
        other_user="@alice:example.org",
        other_device="ALICEDEV",
        room_id="!room:example.org",
        our_pubkey="OURKEY",
        their_pubkey="THEIRKEY",
        we_initiated=we_initiated,
    )


class TestSasInfo:
    def test_responder_info_string(self):
        h = _make_handler()
        info = h._sas_info(_make_session(we_initiated=False))
        # start side = user (alice), accept side = bot.
        assert info == (
            "MATRIX_KEY_VERIFICATION_SAS|@alice:example.org|ALICEDEV|THEIRKEY|"
            "@bot:example.org|BOTDEV|OURKEY|t-1"
        )

    def test_initiator_info_string_swaps_sides(self):
        h = _make_handler()
        info = h._sas_info(_make_session(we_initiated=True))
        # start side = bot (we sent start), accept side = user.
        assert info == (
            "MATRIX_KEY_VERIFICATION_SAS|@bot:example.org|BOTDEV|OURKEY|"
            "@alice:example.org|ALICEDEV|THEIRKEY|t-1"
        )


class TestMacInfoFor:
    def test_our_mac_info(self):
        h = _make_handler()
        assert h._mac_info_for(_make_session(), our_side=True) == (
            "MATRIX_KEY_VERIFICATION_MAC@bot:example.orgBOTDEV"
            "@alice:example.orgALICEDEVt-1"
        )

    def test_their_mac_info(self):
        h = _make_handler()
        assert h._mac_info_for(_make_session(), our_side=False) == (
            "MATRIX_KEY_VERIFICATION_MAC@alice:example.orgALICEDEV"
            "@bot:example.orgBOTDEVt-1"
        )


class TestComputeEmojis:
    def test_derives_emojis_from_sas_secret(self):
        h = _make_handler()

        class FakeSas:
            def generate_bytes(self, info, length):
                # Deterministic 6 bytes -> indices [0, 16, 8, 3, 1, 0, 20]
                assert length == 6
                assert "MATRIX_KEY_VERIFICATION_SAS" in info
                return bytes.fromhex("010203040506")

        session = _make_session()
        session.sas = FakeSas()
        emojis = h._compute_emojis(session)
        assert emojis == [_EMOJIS[i] for i in (0, 16, 8, 3, 1, 0, 20)]
        assert len(emojis) == 7

    def test_failure_returns_empty_list(self):
        h = _make_handler()
        session = _make_session()
        session.sas = None  # generate_bytes will raise AttributeError
        assert h._compute_emojis(session) == []


class TestExpireOldSessions:
    def test_removes_only_stale_sessions(self):
        h = _make_handler()
        fresh = _SasSession(
            transaction_id="fresh", other_user="@u:o", other_device="d"
        )
        stale = _SasSession(
            transaction_id="stale", other_user="@u:o", other_device="d"
        )
        stale.created_at = time.time() - _SESSION_TTL_SECONDS - 60
        h._sessions = {"fresh": fresh, "stale": stale}
        h._expire_old_sessions()
        assert list(h._sessions) == ["fresh"]

    def test_fresh_sessions_survive(self):
        h = _make_handler()
        s = _SasSession(transaction_id="t", other_user="@u:o", other_device="d")
        h._sessions = {"t": s}
        h._expire_old_sessions()
        assert "t" in h._sessions


# ---------------------------------------------------------------------------
# MAC verification gate — an unverified session must never count as verified
# ---------------------------------------------------------------------------


class _FakeDevice:
    def __init__(self, signing_key):
        self.signing_key = signing_key


class _FakeCryptoStore:
    def __init__(self, device_key):
        self._device_key = device_key

    async def get_device(self, user_id, device_id):
        if self._device_key is None:
            return None
        return _FakeDevice(self._device_key)


class _FakeOlm:
    """Just enough OlmMachine surface for _verify_user_macs."""

    def __init__(self, device_key="ALICE_ED25519", master_key=None):
        self.crypto_store = _FakeCryptoStore(device_key)
        self._master_key = master_key
        self.key_queries = 0

    async def _fetch_keys(self, user_ids, include_untracked=False):
        self.key_queries += 1

    async def get_cross_signing_public_keys(self, user_id):
        if not self._master_key:
            return None
        return types.SimpleNamespace(master_key=self._master_key)


class _FakeAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, room_id, text):
        self.sent.append((room_id, text))


def _test_mac(key, info):
    """Deterministic stand-in for python-olm's MAC calc (no libolm needed)."""
    return f"MAC<{key}|{info}>"


def _make_mac_handler(device_key="ALICE_ED25519", adapter=None):
    handler = SasVerificationHandler(
        adapter=adapter or _FakeAdapter(),
        client=types.SimpleNamespace(mxid="@bot:example.org", device_id="BOTDEV"),
        olm=_FakeOlm(device_key=device_key),
    )
    handler._mac_func = lambda session: _test_mac
    return handler


def _mac_event(content, sender="@alice:example.org"):
    return types.SimpleNamespace(
        content=content,
        sender=sender,
        room_id="!room:example.org",
        event_id="$evt",
    )


def _their_mac_content(session, handler, key_ids, keys_field=True):
    """Build the MAC event a user's client sends after confirming the emojis."""
    info = handler._mac_info_for(session, our_side=False)
    content = {"transaction_id": session.transaction_id, "mac": {}}
    for key_id in key_ids:
        content["mac"][key_id] = _test_mac("ALICE_ED25519", info + key_id)
    if keys_field:
        content["keys"] = _test_mac(
            ",".join(sorted(content["mac"])), info + "KEY_IDS"
        )
    return content


class TestVerifyUserMacsGate:
    def test_missing_key_ids_mac_is_rejected(self):
        handler = _make_mac_handler()
        session = _make_session()
        content = _their_mac_content(
            session, handler, ["ed25519:ALICEDEV"], keys_field=False
        )
        assert asyncio.run(handler._verify_user_macs(session, content)) is False
        assert session.mac_verified is False

    def test_mac_matching_no_known_key_is_rejected(self):
        handler = _make_mac_handler()
        session = _make_session()
        info = handler._mac_info_for(session, our_side=False)
        content = {
            "transaction_id": session.transaction_id,
            "mac": {"ed25519:UNKNOWNDEV": _test_mac("STRANGER", info + "ed25519:UNKNOWNDEV")},
            "keys": _test_mac("ed25519:UNKNOWNDEV", info + "KEY_IDS"),
        }
        assert asyncio.run(handler._verify_user_macs(session, content)) is False
        assert session.mac_verified is False

    def test_mac_for_known_device_is_accepted(self):
        handler = _make_mac_handler()
        session = _make_session()
        content = _their_mac_content(session, handler, ["ed25519:ALICEDEV"])
        assert asyncio.run(handler._verify_user_macs(session, content)) is True
        assert session.mac_verified is True

    def test_wrong_mac_is_rejected(self):
        handler = _make_mac_handler()
        session = _make_session()
        content = _their_mac_content(session, handler, ["ed25519:ALICEDEV"])
        content["mac"]["ed25519:ALICEDEV"] = "WRONG"
        assert asyncio.run(handler._verify_user_macs(session, content)) is False
        assert session.mac_verified is False


class TestDoneOrdering:
    def _handler_with_session(self, mac_verified=False):
        adapter = _FakeAdapter()
        handler = _make_mac_handler(adapter=adapter)
        session = _make_session()
        session.mac_verified = mac_verified
        handler._sessions[session.transaction_id] = session
        sent = []

        async def _record_send(event_name, s, content):
            sent.append(event_name)

        handler._send = _record_send
        return handler, session, adapter, sent

    def test_done_before_mac_does_not_finalise(self):
        handler, session, adapter, sent = self._handler_with_session()
        asyncio.run(handler._on_done(_mac_event({"transaction_id": "t-1"})))
        assert sent == []
        assert session.done_sent is False
        assert session.done_received is True
        assert handler._sessions.get("t-1") is session  # held, not dropped
        assert adapter.sent == []  # no "verification successful" message

    def test_done_after_mac_finalises(self):
        handler, _session, adapter, sent = self._handler_with_session(mac_verified=True)
        asyncio.run(handler._on_done(_mac_event({"transaction_id": "t-1"})))
        assert sent == ["m.key.verification.done"]
        assert handler._sessions == {}
        assert adapter.sent and "Verification successful" in adapter.sent[0][1]

    def test_mac_after_early_done_finalises_then(self):
        handler, session, adapter, sent = self._handler_with_session()
        session.mac_sent = True  # keep this test off _send_our_mac
        asyncio.run(handler._on_done(_mac_event({"transaction_id": "t-1"})))
        assert handler._sessions.get("t-1") is session

        content = _their_mac_content(session, handler, ["ed25519:ALICEDEV"])
        asyncio.run(handler._on_mac(_mac_event(content)))

        assert session.mac_verified is True
        assert handler._sessions == {}
        assert sent == ["m.key.verification.done"]
        assert adapter.sent and "Verification successful" in adapter.sent[0][1]

    def test_finalize_refuses_unverified_session(self):
        handler, session, adapter, _sent = self._handler_with_session()
        asyncio.run(handler._finalize(session))
        assert handler._sessions.get("t-1") is session
        assert adapter.sent == []
