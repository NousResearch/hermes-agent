"""Advisor data models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    NIT = "nit"
    CONCERN = "concern"
    BLOCKER = "blocker"


SEVERITY_RANK = {Severity.NIT: 1, Severity.CONCERN: 2, Severity.BLOCKER: 3}


@dataclass
class Advice:
    """A single piece of advice with severity."""

    note: str
    severity: Severity = Severity.NIT

    def tag(self) -> str:
        return f"[{self.severity.value.upper()}]"


@dataclass
class TurnDelta:
    """Data about one agent turn."""

    turn_id: str
    user_message: str
    assistant_response: str
    conversation_history: list  # full message list
    model: str


@dataclass
class AdvisorState:
    """Persistent state for the advisor plugin."""

    enabled: bool = False
    held_notes: list[dict] = field(default_factory=list)
    # Per-advisor model override — empty means inherit primary model
    model: str = ""
    provider: str = ""
    # Keys deduped by note text normalized
    _deduped: set[str] = field(default_factory=set)

    def dedupe_key(self, note: str) -> str:
        return note.strip().replace(r"\s+", " ")

    def has_held(self) -> bool:
        return len(self.held_notes) > 0

    def hold(self, note: str, severity: str):
        """Hold a concern/blocker for reconfirmation."""
        key = self.dedupe_key(note)
        # Check if already held at equal or higher severity
        existing = next((h for h in self.held_notes if self.dedupe_key(h["note"]) == key), None)
        if existing:
            old_rank = SEVERITY_RANK.get(Severity(existing.get("severity", "nit")), 0)
            new_rank = SEVERITY_RANK.get(Severity(severity), 0)
            if new_rank > old_rank:
                existing["severity"] = severity  # escalate
            return  # already noted
        self.held_notes.append({"note": note, "severity": severity})

    def take_held(self) -> list[dict]:
        """Return and clear held notes."""
        notes = list(self.held_notes)
        self.held_notes.clear()
        return notes

    def prune_recanted(self, re_raised: set[str]):
        """Remove held notes that the advisor didn't re-raise (they're resolved)."""
        re_raised_keys = {self.dedupe_key(k) for k in re_raised}
        self.held_notes = [
            h for h in self.held_notes
            if self.dedupe_key(h["note"]) in re_raised_keys
        ]

    def format_reconfirm_preamble(self) -> str:
        """Build a reconfirmation preamble from held notes."""
        if not self.held_notes:
            return ""
        items = "\n".join(
            f"- [{h['severity'].upper()}] {h['note']}"
            for h in self.held_notes
        )
        return (
            "### Held advisories — reconfirm\n\n"
            "You raised these on an earlier step; they were held pending reconfirmation, "
            "because by now the agent may have already addressed them. Re-check each "
            "against the latest activity below.\n\n"
            "For every item that STILL applies, raise it again — same severity, or higher "
            "if it's gotten worse. Say nothing for the rest — silence drops them.\n\n"
            f"{items}\n\n---"
        )

    def parse_response(self, text: str) -> list[Advice]:
        """Parse the advisor model's response into structured advice."""
        if not text or not text.strip():
            return []

        text = text.strip()

        # Check for silence signal
        if "nothing to flag" in text.lower():
            return []

        advice_list: list[Advice] = []
        re_raised_set: set[str] = set()

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            for sev in Severity:
                tag = f"[{sev.value.upper()}]"
                if tag in line:
                    note = line.replace(tag, "").strip().strip(":").strip()
                    if note:
                        advice_list.append(Advice(note=note, severity=sev))
                        re_raised_set.add(note)
                    break

        if not advice_list and "nothing to flag" not in text.lower():
            # Unstructured response — treat as a concern if it has substance
            if len(text) > 20:
                advice_list.append(Advice(note=text, severity=Severity.CONCERN))

        # Update held notes
        for a in advice_list:
            if a.severity in (Severity.CONCERN, Severity.BLOCKER):
                self.hold(a.note, a.severity.value)

        # Prune recanted held notes
        self.prune_recanted(re_raised_set)

        return advice_list

    def serialize(self) -> dict:
        return {
            "enabled": self.enabled,
            "held_notes": self.held_notes,
            "model": self.model,
            "provider": self.provider,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "AdvisorState":
        return cls(
            enabled=data.get("enabled", True),
            held_notes=data.get("held_notes", []),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
        )
