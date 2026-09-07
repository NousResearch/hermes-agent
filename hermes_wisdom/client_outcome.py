"""Private, metadata-only journal outcome reports; never authorization to act."""

from typing import Literal

from pydantic import model_validator

from .client_delivery import Identifier, OpaqueKey, RequestId, StrictModel


class ClientOperationOutcome(StrictModel):
    request_id: RequestId
    operation_key: OpaqueKey
    operation: Literal["share", "publish", "install", "update"]
    state: Literal["queued", "completed", "failed", "stale", "expired", "needs_review"]

    @model_validator(mode="after")
    def valid_result(self):
        if self.state == "queued" and self.operation != "share":
            raise ValueError("Only packaging may be queued")
        if self.state == "completed" and self.operation == "share":
            raise ValueError("Packaging queued is not publication completed")
        return self


class ClientOperationResponse(StrictModel):
    org_id: Identifier
    recipient_user_id: Identifier
    event_id: Identifier
    outcome: ClientOperationOutcome
    attestation: Literal["client_reported"]
    duplicate: bool
