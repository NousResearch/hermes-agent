"""Governance: pre-flight blocking of dangerous actions plus an append-only audit trail."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Union

from .base import Subsystem, now, result

AUDIT_FILE = "governance_audit.json"
MAX_AUDIT_ENTRIES = 5000

_DANGEROUS = (
    ("recursive delete of root", re.compile(r"\brm\s+(-[a-zA-Z]*[rR][a-zA-Z]*\s+)+(--no-preserve-root\s+)?/(\s|\*|$)")),
    ("disk format", re.compile(r"\bmkfs(\.\w+)?\b|\bformat\s+/dev/")),
    ("remote code piped to shell", re.compile(r"\b(curl|wget)\b[^|;\n]*\|\s*(sudo\s+)?(ba|z|da)?sh\b")),
    ("shell=True subprocess", re.compile(r"subprocess\.\w+\([^)]*shell\s*=\s*True")),
    ("system file overwrite", re.compile(r">\s*/etc/(passwd|shadow|sudoers)\b")),
)
_TEXT_FIELDS = ("command", "code", "goal")


class Governance(Subsystem):
    def _text_of(self, action: Union[Dict[str, Any], str]) -> str:
        if isinstance(action, str):
            return action
        return "\n".join(str(action[f]) for f in _TEXT_FIELDS if action.get(f))

    def violations(self, action: Union[Dict[str, Any], str]) -> List[str]:
        text = self._text_of(action)
        return [label for label, rx in _DANGEROUS if rx.search(text)]

    def block_action(self, action: Union[Dict[str, Any], str]) -> bool:
        """True when the action matches a dangerous pattern. Blocked actions are audited."""
        hits = self.violations(action)
        if hits:
            self.record_action(action, "blocked", risk_level="high", details="; ".join(hits))
        return bool(hits)

    def record_action(self, action: Union[Dict[str, Any], str], outcome: str, risk_level: str = "low",
                      quality_score: float = 1.0, details: str = "") -> Dict[str, Any]:
        entry = {
            "ts": now(), "action": action, "result": outcome, "risk_level": risk_level,
            "quality_score": max(0.0, min(1.0, float(quality_score))), "details": details,
        }
        with self.lock(AUDIT_FILE):
            entries = self.load(AUDIT_FILE, default=[])
            if not isinstance(entries, list):
                entries = []
            entries.append(entry)
            self.save(AUDIT_FILE, entries[-MAX_AUDIT_ENTRIES:])
        return entry

    def _entries(self) -> List[Dict[str, Any]]:
        entries = self.load(AUDIT_FILE, default=[])
        return entries if isinstance(entries, list) else []

    def status(self) -> Dict[str, Any]:
        entries = self._entries()
        blocked = sum(1 for e in entries if e.get("result") == "blocked")
        return result(True, f"{len(entries)} audited actions, {blocked} blocked",
                      total=len(entries), blocked=blocked)

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Summarize the audit trail by risk level and average quality."""
        entries = self._entries()
        by_risk: Dict[str, int] = {}
        for e in entries:
            by_risk[e.get("risk_level", "low")] = by_risk.get(e.get("risk_level", "low"), 0) + 1
        avg = sum(e.get("quality_score", 1.0) for e in entries) / len(entries) if entries else None
        return result(True, f"Audited {len(entries)} actions", by_risk=by_risk, avg_quality=avg)
