"""Public session authority contract; live execution state never crosses this boundary."""

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol

JsonObject = Mapping[str, object]

# Authenticated local WebSocket family, separate from the legacy Desktop API contract.
CANONICAL_GATEWAY_PROTOCOL = 'hermes-gateway-v1'


@dataclass(frozen=True)
class Principal:
    subject: str
    profile_id: str
    capabilities: frozenset[str]
    transport_id: str


@dataclass(frozen=True)
class SessionRef:
    profile_id: str
    session_id: str


@dataclass(frozen=True)
class SessionHandle:
    ref: SessionRef
    instance_id: str
    authority_epoch: int
    revision: int
    execution_generation: int
    execution_state: Literal["idle", "running", "waiting", "unknown", "terminal"]


@dataclass(frozen=True)
class SessionCreate:
    request_id: str
    cwd: str
    source: str
    policy: JsonObject


@dataclass(frozen=True)
class Submission:
    request_id: str
    ref: SessionRef
    payload: JsonObject
    intent: Literal["queue", "steer", "redirect"]


@dataclass(frozen=True)
class AdmissionReceipt:
    admission_id: str
    ref: SessionRef
    sequence: int
    status: Literal["queued", "started", "unknown", "terminal"]
    outcome: str | None
    authority_epoch: int
    execution_generation: int | None


@dataclass(frozen=True)
class PendingAdmission(AdmissionReceipt):
    input_id: str
    text: str


@dataclass(frozen=True)
class SubscriptionSnapshot:
    subscription_id: str
    handle: SessionHandle
    replay_epoch: str
    last_sequence: int
    history: tuple[JsonObject, ...]
    pending: tuple[PendingAdmission, ...]
    prompts: tuple[JsonObject, ...]


class SessionAuthority(Protocol):
    async def create(self, actor: Principal, request: SessionCreate) -> SessionHandle: ...
    async def resolve(self, actor: Principal, ref: SessionRef) -> SessionHandle: ...
    async def attach(self, actor: Principal, ref: SessionRef) -> SubscriptionSnapshot: ...
    async def detach(self, actor: Principal, subscription_id: str) -> None: ...
    async def submit(self, actor: Principal, request: Submission) -> AdmissionReceipt: ...
    async def receipt(self, actor: Principal, ref: SessionRef, admission_id: str) -> AdmissionReceipt: ...
    async def cancel_queued(self, actor: Principal, ref: SessionRef, admission_id: str) -> AdmissionReceipt: ...
    async def interrupt(self, actor: Principal, ref: SessionRef, generation: int) -> SessionHandle: ...
    async def respond(self, actor: Principal, ref: SessionRef, generation: int,
                      prompt_id: str, response: JsonObject) -> JsonObject: ...
    async def mutate(self, actor: Principal, ref: SessionRef, revision: int,
                     request_id: str, action: str, arguments: JsonObject) -> SessionHandle: ...
