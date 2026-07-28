"""Shared safety contract for authenticated dashboard mutations."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any, Mapping


class MutationRisk(enum.StrEnum):
    READ_ONLY = "read-only"
    LOW_RISK_WRITE = "low-risk-write"
    DESTRUCTIVE = "destructive"
    SERVICE_INTERRUPTING = "service-interrupting"
    SECRET_SENSITIVE = "secret-sensitive"


@dataclass(frozen=True)
class MutationSpec:
    risk: MutationRisk
    confirmation: str | None = None
    action_class: str | None = None


MUTATION_SPECS: Mapping[str, MutationSpec] = {
    "gateway-restart": MutationSpec(
        MutationRisk.SERVICE_INTERRUPTING,
        confirmation="RESTART",
        action_class="service-lifecycle",
    ),
    "hermes-update": MutationSpec(
        MutationRisk.SERVICE_INTERRUPTING,
        confirmation="UPDATE",
        action_class="service-lifecycle",
    ),
    "plugin-remove": MutationSpec(MutationRisk.DESTRUCTIVE, confirmation="REMOVE"),
    "profile-delete": MutationSpec(MutationRisk.DESTRUCTIVE, confirmation="DELETE"),
    "env-reveal": MutationSpec(MutationRisk.SECRET_SENSITIVE, confirmation="REVEAL"),
    "env-delete": MutationSpec(MutationRisk.SECRET_SENSITIVE, confirmation="DELETE"),
    "cron-delete": MutationSpec(MutationRisk.DESTRUCTIVE, confirmation="DELETE"),
    "session-delete": MutationSpec(MutationRisk.DESTRUCTIVE, confirmation="DELETE"),
    "session-delete-all": MutationSpec(
        MutationRisk.DESTRUCTIVE,
        confirmation="DELETE ALL",
    ),
}

_IDEMPOTENCY_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}")


@dataclass(frozen=True)
class MutationRequest:
    action: str
    confirmation: str | None
    idempotency_key: str | None


class MutationRequestError(ValueError):
    pass


def validate_mutation_request(
    action: str,
    body: Any,
    *,
    require_confirmation: bool = False,
) -> MutationRequest:
    """Validate optional rollout fields or require the complete contract."""

    spec = MUTATION_SPECS[action]
    payload = body if isinstance(body, dict) else {}
    confirmation = payload.get("confirmation")
    idempotency_key = payload.get("idempotency_key")

    if confirmation is not None and not isinstance(confirmation, str):
        raise MutationRequestError("confirmation must be a string")
    if idempotency_key is not None and not isinstance(idempotency_key, str):
        raise MutationRequestError("idempotency_key must be a string")
    if idempotency_key is not None and not _IDEMPOTENCY_KEY_RE.fullmatch(idempotency_key):
        raise MutationRequestError("idempotency_key has an invalid format")

    if require_confirmation:
        if confirmation != spec.confirmation:
            raise MutationRequestError(
                f"confirmation must exactly match {spec.confirmation!r}"
            )
        if idempotency_key is None:
            raise MutationRequestError("idempotency_key is required")
    elif confirmation is not None and confirmation != spec.confirmation:
        raise MutationRequestError(
            f"confirmation must exactly match {spec.confirmation!r}"
        )

    return MutationRequest(action, confirmation, idempotency_key)
