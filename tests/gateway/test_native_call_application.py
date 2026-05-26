from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from gateway.calls.native.application import NativeCallApplication
from gateway.calls.native.ports import (
    NativeCallInvitation,
    NativeMediaOffer,
    NativeMediaStartRequest,
    NativeMediaStartResult,
)


def _source(chat_type="dm"):
    return SimpleNamespace(
        platform=SimpleNamespace(value="simplex"),
        chat_id="42",
        user_id="42",
        chat_type=chat_type,
    )


@dataclass
class FakeSignaling:
    offers: list[tuple[str, NativeMediaOffer]] = field(default_factory=list)
    statuses: list[tuple[str, str]] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    ended: list[str] = field(default_factory=list)

    async def send_offer(self, contact_id: str, offer: NativeMediaOffer) -> None:
        self.offers.append((contact_id, offer))

    async def send_status(self, contact_id: str, status: str) -> None:
        self.statuses.append((contact_id, status))

    async def reject(self, contact_id: str, reason_code: str) -> None:
        self.rejected.append(reason_code)

    async def end(self, contact_id: str) -> None:
        self.ended.append(contact_id)


@dataclass
class FakeMedia:
    result: NativeMediaStartResult
    requests: list[NativeMediaStartRequest] = field(default_factory=list)

    async def start_incoming(self, request: NativeMediaStartRequest) -> NativeMediaStartResult:
        self.requests.append(request)
        return self.result

    async def stop(self, call_id: str) -> None:
        pass


@pytest.mark.asyncio
async def test_incoming_native_call_rejects_group_chat():
    signaling = FakeSignaling()
    media = FakeMedia(NativeMediaStartResult(ok=True))
    app = NativeCallApplication(signaling=signaling, media=media, is_authorized=lambda _s: True)

    result = await app.handle_incoming_invitation(
        _source(chat_type="group"),
        NativeCallInvitation(contact_id="42", media="audio", encrypted=False),
    )

    assert result.ok is False
    assert result.code == "call_private_chat_required"
    assert signaling.rejected == ["call_private_chat_required"]
    assert not signaling.offers


@pytest.mark.asyncio
async def test_incoming_native_call_rejects_unauthorized_contact():
    signaling = FakeSignaling()
    media = FakeMedia(NativeMediaStartResult(ok=True))
    app = NativeCallApplication(signaling=signaling, media=media, is_authorized=lambda _s: False)

    result = await app.handle_incoming_invitation(
        _source(),
        NativeCallInvitation(contact_id="42", media="audio", encrypted=False),
    )

    assert result.ok is False
    assert result.code == "call_auth_denied"
    assert signaling.rejected == ["call_auth_denied"]
    assert not media.requests


@pytest.mark.asyncio
async def test_incoming_native_call_sends_offer_and_connecting_status():
    offer = NativeMediaOffer(
        rtc_session="compressed-offer",
        rtc_ice_candidates="compressed-ice",
        capabilities={"encryption": False},
    )
    signaling = FakeSignaling()
    media = FakeMedia(NativeMediaStartResult(ok=True, offer=offer))
    app = NativeCallApplication(signaling=signaling, media=media, is_authorized=lambda _s: True)

    result = await app.handle_incoming_invitation(
        _source(),
        NativeCallInvitation(contact_id="42", media="audio", encrypted=False, shared_key=None),
    )

    assert result.ok is True
    assert result.code == "call_simplex_native_connecting"
    assert result.call_id.startswith("call_")
    assert media.requests[0].contact_id == "42"
    assert signaling.offers == [("42", offer)]
    assert signaling.statuses == [("42", "connecting")]


@pytest.mark.asyncio
async def test_incoming_native_call_rejects_when_media_start_fails():
    signaling = FakeSignaling()
    media = FakeMedia(
        NativeMediaStartResult(
            ok=False,
            code="call_sidecar_start_failed",
            message="sidecar command is not configured",
        )
    )
    app = NativeCallApplication(signaling=signaling, media=media, is_authorized=lambda _s: True)

    result = await app.handle_incoming_invitation(
        _source(),
        NativeCallInvitation(contact_id="42", media="audio", encrypted=False),
    )

    assert result.ok is False
    assert result.code == "call_sidecar_start_failed"
    assert signaling.rejected == ["call_sidecar_start_failed"]
    assert not signaling.offers
