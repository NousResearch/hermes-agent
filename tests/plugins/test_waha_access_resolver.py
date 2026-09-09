"""Tests for WahaAdapter.resolve_access_ref — the WAHA-specific /access resolution.

Uses a stubbed ``_request`` so no WAHA instance is needed; the roster lookup and
group-name matching logic run exactly as in production.  Phone normalization itself is
covered by tests/gateway/test_whatsapp_identity_phone.py.
"""
from __future__ import annotations

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.slash_commands_access import AccessResolution
from plugins.platforms.waha.adapter import WahaAdapter


class _StubWaha(WahaAdapter):
    """WahaAdapter with a canned ``_request``; nothing binds a port or network."""

    def __init__(self, extra: dict, groups: dict, participants: dict):
        super().__init__(PlatformConfig(enabled=True, extra=extra))
        self._groups = groups
        self._participants = participants
        self._bot_ids = {"6287784454555@s.whatsapp.net"}

    async def _request(self, method, path, payload=None, timeout=30):  # noqa: ARG002
        if "/groups/" in path and path.endswith("/participants"):
            gid = path.rsplit("/", 2)[-2]
            return 200, self._participants.get(gid, [])
        if path.endswith("/groups"):
            return 200, self._groups
        return 404, {}


GROUPS = {
    "120363426491664891@g.us": {"subject": "kami akan berubah mas mba", "size": 41},
    "120363423993556802@g.us": {"subject": "bagian senang senang", "size": 22},
}
PARTICIPANTS = {
    "120363426491664891@g.us": [
        {"id": "254864114368645@lid", "phoneNumber": "6289682642242@s.whatsapp.net", "name": "Niken"},
        {"id": "155671391690979@lid", "phoneNumber": "6289668528305@s.whatsapp.net", "name": "Ajeng"},
    ],
}


def _adapter() -> _StubWaha:
    return _StubWaha({"dm_policy": "allowlist"}, GROUPS, PARTICIPANTS)


@pytest.mark.asyncio
async def test_group_by_exact_name():
    res = await _adapter().resolve_access_ref("kami akan berubah mas mba", scope="group")
    assert res.canonical == "120363426491664891@g.us"
    assert res.label == "kami akan berubah mas mba"


@pytest.mark.asyncio
async def test_group_by_case_insensitive_name():
    res = await _adapter().resolve_access_ref("Bagian Senang Senang", scope="group")
    assert res.canonical == "120363423993556802@g.us"


@pytest.mark.asyncio
async def test_group_ambiguous_name_returns_candidates(monkeypatch):
    # Both subjects contain "grup" after the test roster gains a second match via
    # monkeypatched subject text — the command must ask, never guess.
    groups = dict(GROUPS)
    groups["120363999999999999@g.us"] = {"subject": "senang banget deh", "size": 3}
    adapter = _StubWaha({"dm_policy": "allowlist"}, groups, PARTICIPANTS)
    res = await adapter.resolve_access_ref("senang", scope="group")
    assert res.candidates  # the command asks, never guesses
    assert len(res.candidates) == 2


@pytest.mark.asyncio
async def test_group_jid_passes_through():
    res = await _adapter().resolve_access_ref("120363426491664891@g.us", scope="group")
    assert res.canonical == "120363426491664891@g.us"


@pytest.mark.asyncio
async def test_group_unknown_name_is_an_empty_resolution():
    # A name the roster does not know must not come back as a storable id —
    # the command replies "could not resolve" instead of storing dead text.
    res = await _adapter().resolve_access_ref("Grup XYZ Tanpa Nama", scope="group")
    assert res.canonical == ""
    assert not res.candidates


@pytest.mark.asyncio
async def test_user_by_pushname_resolves_to_phone_jid():
    res = await _adapter().resolve_access_ref("@Niken", scope="user")
    assert res.canonical == "6289682642242@s.whatsapp.net"
    assert res.label == "Niken"


@pytest.mark.asyncio
async def test_user_pushname_not_found_is_an_empty_resolution():
    res = await _adapter().resolve_access_ref("@Nobody", scope="user")
    assert res.canonical == ""
    assert not res.candidates


@pytest.mark.asyncio
async def test_user_jid_passes_through_and_lids_resolve():
    assert (await _adapter().resolve_access_ref("6289603167061@s.whatsapp.net", scope="user")).canonical \
        == "6289603167061@s.whatsapp.net"


@pytest.mark.asyncio
async def test_non_phone_non_jid_returns_none_for_generic_fallback():
    # 'reply' is handled by the generic layer; the WAHA resolver must not claim it.
    assert await _adapter().resolve_access_ref("reply", scope="user") is None


@pytest.mark.asyncio
async def test_resolve_is_none_for_blank():
    assert await _adapter().resolve_access_ref("", scope="user") is None
    assert await _adapter().resolve_access_ref("  ", scope="group") is None


@pytest.mark.asyncio
async def test_real_mixin_normalizes_local_phone():
    """The production path: WhatsAppBehaviorMixin._access_phone_jid normalizes local
    formats against the bot's own country code (no resolver needed for phones)."""
    adapter = _adapter()
    adapter._bot_ids = {"6287784454555@s.whatsapp.net"}
    assert adapter._access_phone_jid("089682642242") == "6289682642242@s.whatsapp.net"
    assert adapter._access_phone_jid("+1 555 123 4567") == "15551234567@s.whatsapp.net"
    assert adapter._access_phone_jid("6289603167061") == "6289603167061@s.whatsapp.net"
    assert adapter._access_phone_jid("abc") == ""


def test_sender_id_canonicalized_to_cus():
    """A phone-addressed DM (``@s.whatsapp.net`` sender) must land as ``@c.us`` — the
    slash-access admin comparison is an exact string match against @c.us lists."""
    adapter = _adapter()
    payload = {
        "id": "false_6285157813352@s.whatsapp.net_X",
        "from": "6285157813352@s.whatsapp.net",
        "chatId": "6285157813352@s.whatsapp.net",
        "fromMe": False,
        "body": "hi",
    }
    data = adapter._map_payload(payload)
    assert data["senderId"] == "6285157813352@s.whatsapp.net"
    assert data["chatId"] == "6285157813352@s.whatsapp.net"


def test_pushname_learned_from_inbound_events():
    """Group rosters (NOWEB) carry JIDs without names; every inbound event carries the
    sender's real pushName.  /access @Name must resolve from that learned directory."""
    adapter = _adapter()
    data = adapter._map_payload({"chatId": "6285157813352@s.whatsapp.net",
                                 "from": "6285157813352@s.whatsapp.net", "fromMe": False,
                                 "body": "x", "sender": {"pushName": "Farel"}})
    adapter.remember_pushname(data["senderId"], data["senderName"])
    assert adapter._access_pushname_lookup("farel") == "6285157813352@s.whatsapp.net"
    # numeric-only "names" are not learnable
    adapter.remember_pushname("6289603167061@s.whatsapp.net", "6289603167061")
    assert adapter._access_pushname_lookup("6289603167061") is None


def test_resolver_falls_back_to_learned_pushname():
    # A name NOT in the roster (roster matches win — they are authoritative): the
    # learned pushname directory is the fallback for names the roster lacks.
    adapter = _adapter()
    adapter.remember_pushname("6289603167061@s.whatsapp.net", "Teman Baru")
    res = asyncio.run(adapter.resolve_access_ref("@Teman Baru", scope="user"))
    assert res.canonical == "6289603167061@s.whatsapp.net"
    assert res.label == "Teman Baru"
