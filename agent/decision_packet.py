"""Structured user-decision packets for finite workflow forks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


NEEDS_CHAD = "NEEDS_CHAD"


@dataclass(frozen=True)
class DecisionPacket:
    """Human-facing stop packet for actions that require Chad's decision."""

    reason: str
    proposed_action: str
    why_this_is_a_fork: str
    safest_default: str
    options: list[str] = field(default_factory=list)
    evidence_summary: str = ""
    status: str = NEEDS_CHAD

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "proposed_action": self.proposed_action,
            "why_this_is_a_fork": self.why_this_is_a_fork,
            "safest_default": self.safest_default,
            "options": list(self.options),
            "evidence_summary": self.evidence_summary,
        }

    def to_text(self) -> str:
        option_lines = "\n".join(f"- {option}" for option in self.options)
        if not option_lines:
            option_lines = "- approve: Allow exactly the proposed action.\n- deny: Do not run it."
        return (
            f"status: {self.status}\n"
            f"reason: {self.reason}\n"
            f"proposed action: {self.proposed_action}\n"
            f"why this is a fork: {self.why_this_is_a_fork}\n"
            f"safest default if no answer: {self.safest_default}\n"
            "exact approve/deny/narrow options:\n"
            f"{option_lines}\n"
            f"evidence summary: {self.evidence_summary}"
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def decision_packet_tool_result(packet: DecisionPacket) -> str:
    """Return a synthetic tool result that callers can detect and halt on."""

    return json.dumps(
        {
            "status": "needs_chad",
            "decision_packet": packet.to_dict(),
            "error": packet.to_text(),
        },
        ensure_ascii=False,
    )


def decision_packet_terminal_result(packet: DecisionPacket) -> str:
    """Return the terminal-tool shaped result for a pre-exec decision stop."""

    return json.dumps(
        {
            "output": "",
            "exit_code": -1,
            "status": "needs_chad",
            "approval_pending": False,
            "decision_packet": packet.to_dict(),
            "error": packet.to_text(),
        },
        ensure_ascii=False,
    )


def extract_decision_packet(value: Any) -> DecisionPacket | None:
    """Parse a DecisionPacket from a synthetic JSON tool result, if present."""

    if isinstance(value, DecisionPacket):
        return value
    if not isinstance(value, str):
        return None
    try:
        data = json.loads(value)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    packet_data = data.get("decision_packet")
    if not isinstance(packet_data, dict):
        return None
    if packet_data.get("status") != NEEDS_CHAD:
        return None
    return DecisionPacket(
        reason=str(packet_data.get("reason") or ""),
        proposed_action=str(packet_data.get("proposed_action") or ""),
        why_this_is_a_fork=str(packet_data.get("why_this_is_a_fork") or ""),
        safest_default=str(packet_data.get("safest_default") or ""),
        options=[
            str(option)
            for option in packet_data.get("options", [])
            if isinstance(option, str)
        ],
        evidence_summary=str(packet_data.get("evidence_summary") or ""),
        status=NEEDS_CHAD,
    )
