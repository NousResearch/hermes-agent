"""Private wire projection for reserving and acknowledging local notifications.

These are client-reported transport receipts, not read receipts or consent.
Candidate content and transport identifiers never leave the profile here.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .delivery import DeliveryReceipt

OpaqueKey = Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
Identifier = Annotated[str, Field(min_length=1, max_length=256, pattern=r"^[^\x00-\x1f\x7f-\x9f]+$")]
RequestId = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class CandidateDeliveryReference(StrictModel):
    kind: Literal["candidate"]
    key: OpaqueKey


class SkillDeliveryReference(StrictModel):
    kind: Literal["skill"]
    skill_id: Identifier
    version: int = Field(ge=1, le=2_147_483_647)
    event_type: Literal["teammate_published", "update_available"]


DeliveryReference = Annotated[CandidateDeliveryReference | SkillDeliveryReference, Field(discriminator="kind")]


class ClientDeliveryClaim(StrictModel):
    request_id: RequestId
    reference: DeliveryReference


class ClientDeliveryReceipt(StrictModel):
    platform: Literal["telegram", "slack", "local"]
    acknowledgement: Literal["provider_accepted", "transport_accepted"]
    message_key: OpaqueKey
    destination_key: OpaqueKey

    @model_validator(mode="after")
    def valid_acknowledgement(self):
        expected = "transport_accepted" if self.platform == "local" else "provider_accepted"
        if self.acknowledgement != expected:
            raise ValueError("Acknowledgement does not match transport")
        return self


class ClientDeliveryResponse(StrictModel):
    org_id: Identifier
    recipient_user_id: Identifier
    event_id: Identifier
    request_id: RequestId
    reference: DeliveryReference
    state: Literal["claimed", "acknowledged", "not_sent", "uncertain", "deferred"]
    lease_until: str | None
    reason: Literal[
        "not_found", "in_progress", "retry_exhausted", "policy_changed",
        "recipient_muted", "notifications_disabled", "reference_invalid", "frequency_cap",
    ] | None

    @model_validator(mode="after")
    def valid_directive(self):
        if self.state == "claimed":
            if not self.lease_until or self.reason is not None:
                raise ValueError("Claim requires a lease and no refusal")
            expiry = datetime.fromisoformat(self.lease_until.replace("Z", "+00:00"))
            if expiry.tzinfo is None or not self.lease_until.endswith("Z"):
                raise ValueError("Lease must be UTC")
        elif self.lease_until is not None:
            raise ValueError("Non-claim cannot grant a lease")
        if (self.state == "deferred") != (self.reason is not None):
            raise ValueError("Only deferrals have a reason")
        return self


def receipt_projection(receipt: DeliveryReceipt) -> ClientDeliveryReceipt:
    if not isinstance(receipt, DeliveryReceipt):
        raise ValueError("A validated local transport receipt is required")

    def key(kind: str, parts: list[str]) -> str:
        data = json.dumps(["wisdom-delivery-v1", kind, *parts], separators=(",", ":"))
        return "sha256:" + hashlib.sha256(data.encode()).hexdigest()

    destination = [receipt.platform, receipt.scope_id, receipt.destination, receipt.thread_id]
    return ClientDeliveryReceipt(
        platform=receipt.platform,
        acknowledgement=receipt.acknowledgement,
        destination_key=key("destination", destination),
        message_key=key("message", [*destination, receipt.message_id]),
    )
