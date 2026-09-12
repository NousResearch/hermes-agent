"""Confirmed intents for the dashboard's service-interrupting actions.

Other dashboard mutation classes are outside this contract.
"""

from dataclasses import dataclass
import re
from typing import Any


SERVICE_MUTATION_CONFIRMATIONS = {
    "gateway-restart": "RESTART",
    "hermes-update": "UPDATE",
}
_IDEMPOTENCY_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}")


@dataclass(frozen=True)
class ServiceMutation:
    action: str
    idempotency_key: str


def validate_service_mutation(action: str, body: Any) -> ServiceMutation:
    """Require exact intent and a bounded key before any process is started."""
    if not isinstance(body, dict):
        raise ValueError("body must be an object")
    confirmation = SERVICE_MUTATION_CONFIRMATIONS[action]
    if body.get("confirmation") != confirmation:
        raise ValueError(f"confirmation must exactly match {confirmation!r}")
    key = body.get("idempotency_key")
    if not isinstance(key, str) or not _IDEMPOTENCY_KEY_RE.fullmatch(key):
        raise ValueError("idempotency_key must contain 16-128 letters, digits, or ._:-")
    return ServiceMutation(action, key)
