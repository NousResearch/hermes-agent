from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .models import AdmissionRecord, InspectionReport


class AdmissionStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir).expanduser()
        self.records_dir = self.base_dir / "records"
        self.reports_dir = self.base_dir / "reports"
        self.quarantine_dir = self.base_dir / "quarantine"
        self.approved_dir = self.base_dir / "approved"
        for path in (
            self.records_dir,
            self.reports_dir,
            self.quarantine_dir,
            self.approved_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def record_path(self, record_id: str) -> Path:
        return self.records_dir / f"{record_id}.json"

    def report_path(self, record_id: str) -> Path:
        return self.reports_dir / f"{record_id}.md"

    def candidate_quarantine_path(self, record_id: str, artifact_name: str) -> Path:
        return self.quarantine_dir / record_id / artifact_name

    def candidate_approved_path(self, record_id: str, artifact_name: str) -> Path:
        return self.approved_dir / record_id / artifact_name

    def save_record(self, record: AdmissionRecord) -> Path:
        path = self.record_path(record.record_id)
        payload = _json_ready(asdict(record))
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_record(self, record_id: str) -> AdmissionRecord:
        payload = json.loads(self.record_path(record_id).read_text(encoding="utf-8"))
        return _record_from_dict(payload)

    def list_records(self) -> list[AdmissionRecord]:
        return [
            self.load_record(path.stem)
            for path in sorted(self.records_dir.glob("*.json"))
        ]

    def write_report(self, record_id: str, content: str) -> Path:
        path = self.report_path(record_id)
        path.write_text(content, encoding="utf-8")
        return path


def _json_ready(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _record_from_dict(payload: dict) -> AdmissionRecord:
    from .models import (
        AdmissionStatus,
        CandidateKind,
        CandidateSource,
        IntegrityState,
    )

    source = CandidateSource(**payload["source"])
    integrity = payload.get("integrity")
    if integrity:
        integrity = IntegrityState(
            algorithm=integrity["algorithm"],
            digest=integrity["digest"],
            verified_at=datetime.fromisoformat(integrity["verified_at"]),
        )
    return AdmissionRecord(
        record_id=payload["record_id"],
        kind=CandidateKind(payload["kind"]),
        source=source,
        lineage_id=payload.get("lineage_id"),
        parent_record_id=payload.get("parent_record_id"),
        revision=payload.get("revision", 1),
        source_fingerprint=payload.get("source_fingerprint"),
        status=AdmissionStatus(payload["status"]),
        created_at=datetime.fromisoformat(payload["created_at"]),
        updated_at=datetime.fromisoformat(payload["updated_at"]),
        quarantine_path=payload.get("quarantine_path"),
        approved_path=payload.get("approved_path"),
        integrity=integrity,
        approved_at=datetime.fromisoformat(payload["approved_at"]) if payload.get("approved_at") else None,
        report_path=payload.get("report_path"),
        notes=list(payload.get("notes", [])),
    )
