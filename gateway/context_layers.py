"""Typed prompt-layer accounting for gateway-created model requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, NewType


ContextLayerId = NewType("ContextLayerId", str)


@dataclass(frozen=True)
class ContextLayer:
    """A named portion of the context request Hermes is about to send."""

    layer_id: ContextLayerId
    source: str
    chars: int
    estimated_tokens: int
    enabled: bool = True
    clipped: bool = False

    @classmethod
    def from_text(
        cls,
        layer_id: str,
        *,
        source: str,
        text: str | None,
        enabled: bool = True,
        clipped: bool = False,
    ) -> "ContextLayer":
        content = text or ""
        return cls(
            layer_id=ContextLayerId(layer_id),
            source=source,
            chars=len(content),
            estimated_tokens=estimate_text_tokens(content),
            enabled=enabled,
            clipped=clipped,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["id"] = payload.pop("layer_id")
        return payload


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_payload_tokens(value: Any) -> int:
    return estimate_text_tokens(str(value) if value is not None else "")


def layers_to_payload(layers: list[ContextLayer]) -> list[dict[str, Any]]:
    return [layer.to_payload() for layer in layers]


def total_estimated_tokens(layers: list[ContextLayer]) -> int:
    return sum(layer.estimated_tokens for layer in layers if layer.enabled)
