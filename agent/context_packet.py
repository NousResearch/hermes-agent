"""Typed Context Packet handoffs for cross-agent delegation.

Context Packets are a small, stable handoff primitive for Bob/Steve/Vera-style
agent handoffs.  They keep the human-readable shape deterministic while giving
runtime seams a validated dict/JSON/fenced-block contract instead of forcing
receivers to parse ad hoc prose.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

from agent.redact import redact_sensitive_text


class ContextPacketValidationError(ValueError):
    """Raised when a Context Packet is missing or has invalid fields."""


_REQUIRED_FIELDS = (
    "role",
    "source",
    "goal",
    "non_goals",
    "safety",
    "allowed_actions",
    "forbidden_actions",
    "context",
    "evidence",
    "verification",
    "output_contract",
    "residual_risks",
)

_FENCE_RE = re.compile(
    (
        r"```(?:context[-_ ]?packet|json\s+context[-_ ]?packet)\s*\n"
        r"(?P<body>[\s\S]*?)\n```"
    ),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContextPacket:
    role: str
    source: str
    goal: str
    non_goals: tuple[str, ...]
    safety: str
    allowed_actions: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    context: tuple[str, ...]
    evidence: tuple[str, ...]
    verification: tuple[str, ...]
    output_contract: str
    residual_risks: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextPacket":
        if not isinstance(data, Mapping):
            raise ContextPacketValidationError(
                f"context packet must be an object/dict, got {type(data).__name__}"
            )
        for field in _REQUIRED_FIELDS:
            if field not in data:
                raise ContextPacketValidationError(f"missing required field: {field}")

        return cls(
            role=_required_text(data, "role"),
            source=_required_text(data, "source"),
            goal=_required_text(data, "goal"),
            non_goals=_text_tuple(data, "non_goals"),
            safety=_required_text(data, "safety"),
            allowed_actions=_text_tuple(data, "allowed_actions"),
            forbidden_actions=_text_tuple(data, "forbidden_actions"),
            context=_text_tuple(data, "context"),
            evidence=_text_tuple(data, "evidence"),
            verification=_text_tuple(data, "verification"),
            output_contract=_required_text(data, "output_contract"),
            residual_risks=_text_tuple(data, "residual_risks"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ContextPacket":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContextPacketValidationError(
                f"context packet JSON is invalid: {exc.msg}"
            ) from exc
        return cls.from_dict(parsed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source": self.source,
            "goal": self.goal,
            "non_goals": list(self.non_goals),
            "safety": self.safety,
            "allowed_actions": list(self.allowed_actions),
            "forbidden_actions": list(self.forbidden_actions),
            "context": list(self.context),
            "evidence": list(self.evidence),
            "verification": list(self.verification),
            "output_contract": self.output_contract,
            "residual_risks": list(self.residual_risks),
        }

    def render_markdown(self) -> str:
        lines = [
            "# Context Packet",
            "",
            f"- role: {self.role}",
            f"- source: {self.source}",
            f"- goal: {self.goal}",
            f"- safety: {self.safety}",
            f"- output_contract: {self.output_contract}",
        ]
        _append_section(lines, "Non-goals", self.non_goals)
        _append_section(lines, "Allowed actions", self.allowed_actions)
        _append_section(lines, "Forbidden actions", self.forbidden_actions)
        _append_section(lines, "Context", self.context)
        _append_section(lines, "Evidence", self.evidence)
        _append_section(lines, "Verification", self.verification)
        _append_section(lines, "Residual risks", self.residual_risks)
        return "\n".join(lines)


def parse_context_packet(value: Any) -> ContextPacket | None:
    """Parse a Context Packet from dict, JSON string, or fenced markdown block.

    Plain natural-language strings return ``None`` for backward compatibility;
    callers can keep using the original prose handoff unchanged.
    """
    if value is None:
        return None
    if isinstance(value, ContextPacket):
        return value
    if isinstance(value, Mapping):
        return ContextPacket.from_dict(value)
    if not isinstance(value, str):
        raise ContextPacketValidationError(
            f"context packet must be dict or string, got {type(value).__name__}"
        )

    raw = value.strip()
    if not raw:
        return None

    match = _FENCE_RE.search(raw)
    if match:
        return ContextPacket.from_json(match.group("body").strip())

    if raw.startswith("{"):
        return ContextPacket.from_json(raw)
    return None


def render_context_packet(value: Any) -> str | None:
    packet = parse_context_packet(value)
    if packet is None:
        return None
    return packet.render_markdown()


def _required_text(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContextPacketValidationError(f"{field} must be a non-empty string")
    return _redact(value.strip())


def _text_tuple(data: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = data.get(field)
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ContextPacketValidationError(
            f"{field} must be a string or list of strings"
        )
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ContextPacketValidationError(
                f"{field} entries must be non-empty strings"
            )
        normalized.append(_redact(item.strip()))
    return tuple(normalized)


def _redact(value: str) -> str:
    return redact_sensitive_text(value, force=True)


def _append_section(lines: list[str], title: str, values: tuple[str, ...]) -> None:
    lines.extend(["", f"## {title}"])
    lines.extend(f"- {value}" for value in values)
