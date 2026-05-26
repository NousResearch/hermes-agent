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
    fail_offer: bool = False
    fail_status: bool = False
    fail_reject: bool = False

    async def send_offer(self, contact_id: str, offer: NativeMediaOffer) -> None:
        if self.fail_offer:
            raise RuntimeError("offer transport failed")
        self.offers.append((contact_id, offer))

    async def send_status(self, contact_id: str, status: str) -> None:
        if self.fail_status:
            raise RuntimeError("status transport failed")
        self.statuses.append((contact_id, status))

    async def reject(self, contact_id: str, reason_code: str) -> None:
        if self.fail_reject:
            raise RuntimeError("reject transport failed")
        self.rejected.append(reason_code)

    async def end(self, contact_id: str) -> None:
        self.ended.append(contact_id)


@dataclass
class FakeMedia:
    result: NativeMediaStartResult
    requests: list[NativeMediaStartRequest] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    start_error: Exception | None = None

    async def start_incoming(self, request: NativeMediaStartRequest) -> NativeMediaStartResult:
        self.requests.append(request)
        if self.start_error:
            raise self.start_error
        return self.result

    async def stop(self, call_id: str) -> None:
        self.stopped.append(call_id)


@pytest.mark.asyncio
async def test_incoming_native_call_rejects_group_chat(caplog):
    signaling = FakeSignaling()
    media = FakeMedia(NativeMediaStartResult(ok=True))
    app = NativeCallApplication(signaling=signaling, media=media, is_authorized=lambda _s: True)

    caplog.set_level("INFO", logger="gateway.calls.native.application")

    result = await app.handle_incoming_invitation(
        _source(chat_type="group"),
        NativeCallInvitation(contact_id="42", media="audio", encrypted=False),
    )

    assert result.ok is False
    assert result.code == "call_private_chat_required"
    assert signaling.rejected == ["call_private_chat_required"]
    assert not signaling.offers
    assert any(
        record.reason_code == "call_private_chat_required"
        for record in caplog.records
        if record.message == "SimpleX native call rejected"
    )


@pytest.mark.asyncio
async def test_incoming_native_call_rejects_unauthorized_contact(caplog):
    signaling = FakeSignaling()
    media = FakeMedia(NativeMediaStartResult(ok=True))
    app = NativeCallApplication(signaling=signaling, media=media, is_authorized=lambda _s: False)

    caplog.set_level("INFO", logger="gateway.calls.native.application")

    result = await app.handle_incoming_invitation(
        _source(),
        NativeCallInvitation(contact_id="42", media="audio", encrypted=False),
    )

    assert result.ok is False
    assert result.code == "call_auth_denied"
    assert signaling.rejected == ["call_auth_denied"]
    assert not media.requests
    assert any(
        record.reason_code == "call_auth_denied"
        for record in caplog.records
        if record.message == "SimpleX native call rejected"
    )


@pytest.mark.parametrize(
    ("chat_type", "authorized", "expected_code"),
    [
        ("group", True, "call_private_chat_required"),
        ("dm", False, "call_auth_denied"),
    ],
)
@pytest.mark.asyncio
async def test_incoming_native_call_returns_rejection_when_reject_signal_fails(
    chat_type,
    authorized,
    expected_code,
    caplog,
):
    signaling = FakeSignaling(fail_reject=True)
    media = FakeMedia(NativeMediaStartResult(ok=True))
    app = NativeCallApplication(
        signaling=signaling,
        media=media,
        is_authorized=lambda _s: authorized,
    )

    caplog.set_level("INFO", logger="gateway.calls.native.application")

    result = await app.handle_incoming_invitation(
        _source(chat_type=chat_type),
        NativeCallInvitation(contact_id="42", media="audio", encrypted=False),
    )

    assert result.ok is False
    assert result.code == expected_code
    assert signaling.rejected == []
    assert not media.requests
    assert any(
        record.reason_code == expected_code
        for record in caplog.records
        if record.message == "SimpleX native call rejected"
    )
    assert any(
        record.reason_code == expected_code
        for record in caplog.records
        if record.message == "SimpleX native call reject failed"
    )


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


@pytest.mark.asyncio
async def test_incoming_native_call_rejects_when_media_start_raises(caplog):
    signaling = FakeSignaling()
    media = FakeMedia(
        NativeMediaStartResult(ok=True),
        start_error=RuntimeError("sidecar failed with secret-key-material"),
    )
    app = NativeCallApplication(signaling=signaling, media=media, is_authorized=lambda _s: True)

    caplog.set_level("WARNING", logger="gateway.calls.native.application")

    result = await app.handle_incoming_invitation(
        _source(),
        NativeCallInvitation(
            contact_id="42",
            media="audio",
            encrypted=True,
            shared_key="secret-key-material",
        ),
    )

    assert result.ok is False
    assert result.code == "call_simplex_native_media_failed"
    assert signaling.rejected == ["call_simplex_native_media_failed"]
    assert not signaling.offers
    assert "secret-key-material" not in caplog.text
    assert any(
        record.reason_code == "call_simplex_native_media_failed"
        for record in caplog.records
        if record.message == "SimpleX native call media start raised"
    )


@pytest.mark.parametrize("failure", ["offer", "status"])
@pytest.mark.asyncio
async def test_incoming_native_call_stops_media_when_signaling_fails(failure, caplog):
    offer = NativeMediaOffer(
        rtc_session="compressed-offer",
        rtc_ice_candidates="compressed-ice",
        capabilities={"encryption": False},
    )
    signaling = FakeSignaling(
        fail_offer=failure == "offer",
        fail_status=failure == "status",
    )
    media = FakeMedia(NativeMediaStartResult(ok=True, offer=offer))
    app = NativeCallApplication(signaling=signaling, media=media, is_authorized=lambda _s: True)

    caplog.set_level("WARNING", logger="gateway.calls.native.application")

    result = await app.handle_incoming_invitation(
        _source(),
        NativeCallInvitation(contact_id="42", media="audio", encrypted=False),
    )

    assert result.ok is False
    assert result.code == "call_simplex_native_signaling_failed"
    assert result.call_id is not None
    assert media.stopped == [result.call_id]
    assert signaling.rejected == ["call_simplex_native_signaling_failed"]
    assert any(
        record.reason_code == "call_simplex_native_signaling_failed"
        for record in caplog.records
        if record.message == "SimpleX native call signaling failed"
    )
