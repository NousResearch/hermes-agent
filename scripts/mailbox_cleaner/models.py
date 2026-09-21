from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


def _normalise(value: str, *, lower: bool = False) -> str:
    result = " ".join(str(value).split())
    return result.lower() if lower else result


@dataclass(frozen=True, slots=True)
class Observation:
    """A deliberately metadata-only, immutable message observation."""

    account: str
    provider: str
    message_id: str
    sender: str
    subject: str
    received_at: str
    preview: str = ""

    def __post_init__(self) -> None:
        for name in ("account", "provider", "message_id"):
            value = _normalise(getattr(self, name), lower=name != "message_id")
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "sender", _normalise(self.sender, lower=True))
        object.__setattr__(self, "subject", _normalise(self.subject))
        object.__setattr__(self, "received_at", _normalise(self.received_at))
        object.__setattr__(self, "preview", _normalise(self.preview))

    @property
    def fingerprint(self) -> str:
        payload = f"{self.provider}\0{self.account}\0{self.message_id}".encode()
        return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class Classification:
    observation: Observation
    category: str
    confidence: float
    urgent: bool
