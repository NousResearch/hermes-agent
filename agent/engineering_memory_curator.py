"""Deterministic screening and human-reviewed state transitions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from agent.engineering_memory_ledger import EngineeringMemoryLedger, EngineeringMemoryLedgerError
from agent.engineering_memory_schema import EngineeringMemoryRecord
from tools.threat_patterns import scan_for_threats

_SECRET_RE = re.compile(
    r"(?:api[_-]?key|token|secret|password)\s*[=:]\s*[\"'](?![A-Z][A-Z0-9_]+[\"'])[A-Za-z0-9+/=_-]{20,}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CuratorDecision:
    record: EngineeringMemoryRecord
    status: str
    reason_codes: tuple[str, ...] = ()
    approved_by: str | None = None

    @property
    def conflict_set(self) -> tuple[str, ...]:
        return self.record.conflict_set


class EngineeringMemoryCurator:
    def __init__(self, ledger: EngineeringMemoryLedger):
        self.ledger = ledger

    @staticmethod
    def _content(record: EngineeringMemoryRecord) -> str:
        value = record.to_mapping()
        value.pop("threat_scan", None)
        value.pop("secret_scan", None)
        value.pop("review", None)
        return "\n".join(str(item) for item in value.values()) + "\n" + record.body

    def _screen(self, record: EngineeringMemoryRecord) -> tuple[str, ...]:
        content = self._content(record)
        reasons: list[str] = []
        findings = scan_for_threats(content, scope="strict")
        if findings:
            reasons.extend("prompt_injection" if item in {"ignore_previous", "system_prompt", "remove_filters"} else item for item in findings)
        if _SECRET_RE.search(content):
            reasons.append("hardcoded_secret")
        return tuple(dict.fromkeys(reasons))

    @staticmethod
    def _conflict_key(record: EngineeringMemoryRecord) -> tuple[str, str, str]:
        return record.repository, record.component, record.task_type

    @staticmethod
    def _content_digest(record: EngineeringMemoryRecord) -> str:
        value = record.to_mapping()
        for field in ("record_id", "status", "threat_scan", "secret_scan", "review", "conflict_set", "related_record_ids"):
            value.pop(field, None)
        import hashlib
        import json
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()

    def stage(self, record: EngineeringMemoryRecord) -> CuratorDecision:
        reasons = list(self._screen(record))
        if reasons:
            updated = record.__class__(**{**record.__dict__, "status": "quarantined", "secret_scan": {"status": "blocked" if "hardcoded_secret" in reasons else "clear"}, "threat_scan": {"status": "blocked", "findings": reasons}})
            self.ledger.write_candidate(record)
            self.ledger.replace(updated)
            return CuratorDecision(updated, updated.status, tuple(reasons))

        existing = [item for item in self.ledger.iter_records() if item.record_id != record.record_id and item.status not in {"rejected", "withdrawn"}]
        duplicate = next((item for item in existing if self._content_digest(item) == self._content_digest(record)), None)
        if duplicate:
            reasons.append("duplicate_digest")
        conflicts = [item for item in existing if self._conflict_key(item) == self._conflict_key(record)]
        conflict_ids = tuple(sorted({record.record_id, *(item.record_id for item in conflicts)}))
        status = "needs_review" if duplicate or conflicts else "candidate"
        updated = record.__class__(**{**record.__dict__, "status": status, "conflict_set": conflict_ids if conflicts else (), "threat_scan": {"status": "clear"}, "secret_scan": {"status": "clear"}})
        self.ledger.write_candidate(record)
        self.ledger.replace(updated)
        if conflicts:
            for item in conflicts:
                revised = item.__class__(**{**item.__dict__, "status": "needs_review", "conflict_set": conflict_ids})
                self.ledger.replace(revised)
        return CuratorDecision(updated, status, tuple(reasons))

    def review(self, record_id: str, action: str, *, reviewer: str, reason: str, evidence_refs: tuple[str, ...] = ()) -> CuratorDecision:
        target = {"approve": "approved", "reject": "rejected", "supersede": "superseded", "withdraw": "withdrawn"}.get(action)
        if target is None:
            raise EngineeringMemoryLedgerError(f"unknown review action: {action}")
        updated = self.ledger.transition(record_id, target, reviewer=reviewer, reason=reason, evidence_refs=evidence_refs)
        return CuratorDecision(updated, updated.status, (), reviewer)
