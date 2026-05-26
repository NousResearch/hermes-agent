from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class NativeCallInvitation:
    contact_id: str
    media: str = "audio"
    encrypted: bool = False
    shared_key: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeMediaOffer:
    rtc_session: str
    rtc_ice_candidates: str
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeMediaStartRequest:
    call_id: str
    contact_id: str
    media: str
    encrypted: bool
    shared_key: str | None = None


@dataclass(frozen=True)
class NativeMediaStartResult:
    ok: bool
    offer: NativeMediaOffer | None = None
    code: str | None = None
    message: str = ""


@dataclass(frozen=True)
class NativeCallResult:
    ok: bool
    code: str
    message: str
    call_id: str | None = None


class NativeCallSignalingPort(Protocol):
    async def send_offer(self, contact_id: str, offer: NativeMediaOffer) -> None:
        raise NotImplementedError

    async def send_status(self, contact_id: str, status: str) -> None:
        raise NotImplementedError

    async def reject(self, contact_id: str, reason_code: str) -> None:
        raise NotImplementedError

    async def end(self, contact_id: str) -> None:
        raise NotImplementedError


class WebRTCMediaPort(Protocol):
    async def start_incoming(self, request: NativeMediaStartRequest) -> NativeMediaStartResult:
        raise NotImplementedError

    async def stop(self, call_id: str) -> None:
        raise NotImplementedError


AuthorizeSource = Callable[[Any], bool]
