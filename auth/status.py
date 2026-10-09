"""Selection outcomes; distinguish credential state from operational failures."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auth.credentials import CredentialIdentity, CredentialRequest


class CredentialStatus(str, Enum):
    SELECTED = "selected"
    NOT_CONFIGURED = "not_configured"
    EXHAUSTED = "exhausted"
    INVALID = "invalid"
    REFRESH_FAILED = "refresh_failed"
    PERSISTENCE_FAILED = "persistence_failed"


@dataclass(frozen=True)
class CredentialFailure:
    """Unsuccessful resolution with no usable credential material.

    retry_at is an optional epoch timestamp supplied by existing cooldown
    policy, not a new retry strategy. identity identifies a failed refresh or
    write when known; absence must not be interpreted as 'no credentials'.
    detail is untrusted diagnostic text and is excluded from repr.
    """

    request: CredentialRequest
    status: CredentialStatus
    identity: CredentialIdentity | None = None
    code: str | None = None
    relogin_required: bool = False
    retry_at: float | None = None
    detail: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.status, CredentialStatus)
            or self.status is CredentialStatus.SELECTED
        ):
            raise ValueError("a failure requires a non-selected CredentialStatus")
        if self.identity is not None and not self.identity.matches(self.request):
            raise ValueError("failure identity does not match the request scope")
        if self.retry_at is not None and (
            not math.isfinite(self.retry_at) or self.retry_at < 0
        ):
            raise ValueError("retry_at must be a finite epoch timestamp")
