"""Authoritative Markdown ledger for cross-engineering memory records."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Collection

from agent.engineering_memory_schema import EngineeringMemoryRecord, EngineeringMemorySchemaError, render_markdown_record, parse_markdown_record


class EngineeringMemoryLedgerError(ValueError):
    """Raised when a ledger operation would violate source-record invariants."""


@dataclass(frozen=True)
class LedgerDiagnostic:
    reason_code: str
    source_label: str


_TRANSITIONS = {
    "candidate": {"quarantined", "needs_review", "approved", "rejected", "withdrawn"},
    "quarantined": {"needs_review", "rejected", "withdrawn"},
    "needs_review": {"approved", "rejected", "withdrawn"},
    "approved": {"superseded", "withdrawn"},
    "superseded": set(),
    "rejected": set(),
    "withdrawn": set(),
}


class EngineeringMemoryLedger:
    def __init__(self, root: Path, *, max_record_bytes: int = 16_384):
        self.root = Path(root).expanduser()
        self.max_record_bytes = max_record_bytes
        self.audit_path = self.root / ".audit.jsonl"
        self.diagnostics: list[LedgerDiagnostic] = []

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, record_id: str) -> Path:
        return self.root / f"{record_id}.md"

    def _safe_path(self, path: Path) -> Path:
        root = self.root.resolve()
        try:
            resolved = path.resolve(strict=False)
            if not resolved.is_relative_to(root):
                raise EngineeringMemoryLedgerError("record path outside configured vault")
            return resolved
        except OSError as exc:
            raise EngineeringMemoryLedgerError("record path cannot be resolved") from exc

    def _write(self, record: EngineeringMemoryRecord, *, allow_existing: bool = False) -> None:
        self._ensure_root()
        path = self._safe_path(self._path(record.record_id))
        if path.exists() and not allow_existing:
            raise EngineeringMemoryLedgerError(f"duplicate record_id: {record.record_id}")
        rendered = render_markdown_record(record).encode("utf-8")
        if len(rendered) > self.max_record_bytes:
            raise EngineeringMemoryLedgerError("record exceeds configured byte limit")
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(rendered)
        os.replace(temporary, path)

    def write_candidate(self, record: EngineeringMemoryRecord) -> EngineeringMemoryRecord:
        if record.status != "candidate":
            raise EngineeringMemoryLedgerError("write_candidate requires candidate status")
        self._write(record)
        return record

    def replace(self, record: EngineeringMemoryRecord) -> EngineeringMemoryRecord:
        if not self._path(record.record_id).exists():
            raise EngineeringMemoryLedgerError(f"record not found: {record.record_id}")
        self._write(record, allow_existing=True)
        return record

    def _read_path(self, path: Path) -> EngineeringMemoryRecord | None:
        label = path.name
        try:
            if path.is_symlink():
                self.diagnostics.append(LedgerDiagnostic("symlink_record", label))
                return None
            resolved = self._safe_path(path)
            if not resolved.is_file():
                return None
            if resolved.stat().st_size > self.max_record_bytes:
                self.diagnostics.append(LedgerDiagnostic("record_too_large", label))
                return None
            return parse_markdown_record(resolved.read_text(encoding="utf-8"), source_path=resolved)
        except (OSError, UnicodeDecodeError, EngineeringMemorySchemaError, EngineeringMemoryLedgerError):
            self.diagnostics.append(LedgerDiagnostic("invalid_record", label))
            return None

    def iter_records(self, statuses: Collection[str] | None = None) -> Iterable[EngineeringMemoryRecord]:
        self.diagnostics = []
        if not self.root.exists():
            return iter(())
        records: list[EngineeringMemoryRecord] = []
        for path in sorted(self.root.glob("*.md"), key=lambda item: item.name):
            record = self._read_path(path)
            if record is not None and (statuses is None or record.status in statuses):
                records.append(record)
        return iter(records)

    def get(self, record_id: str) -> EngineeringMemoryRecord | None:
        if not record_id or Path(record_id).name != record_id:
            raise EngineeringMemoryLedgerError("invalid record_id")
        return self._read_path(self._path(record_id))

    def transition(
        self,
        record_id: str,
        status: str,
        *,
        reviewer: str,
        reason: str,
        evidence_refs: tuple[str, ...] = (),
    ) -> EngineeringMemoryRecord:
        record = self.get(record_id)
        if record is None:
            raise EngineeringMemoryLedgerError(f"record not found: {record_id}")
        if status not in _TRANSITIONS.get(record.status, set()):
            raise EngineeringMemoryLedgerError(f"illegal transition {record.status} -> {status}")
        if not reviewer.strip() or not reason.strip():
            raise EngineeringMemoryLedgerError("transition requires reviewer and reason")
        review = dict(record.review)
        review.update({"reviewer": reviewer.strip(), "reason": reason.strip(), "evidence_refs": list(evidence_refs)})
        updated = record.__class__(**{**record.__dict__, "status": status, "review": review})
        self.replace(updated)
        self._ensure_root()
        event = {
            "record_id": record_id,
            "from_status": record.status,
            "to_status": status,
            "reviewer": reviewer.strip(),
            "reason": reason.strip(),
            "evidence_refs": list(evidence_refs),
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return updated
