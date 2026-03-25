from __future__ import annotations

import shutil
from pathlib import Path

from .models import AdmissionRecord, AdmissionStatus
from .store import AdmissionStore


class AdmissionPromoter:
    def __init__(self, store: AdmissionStore) -> None:
        self.store = store

    def promote_record(self, record: AdmissionRecord) -> AdmissionRecord:
        if not record.report_path:
            raise ValueError("cannot promote record without inspection report")
        record.transition_to(AdmissionStatus.APPROVED)
        self.store.save_record(record)
        return record

    def promote_artifact(
        self,
        record: AdmissionRecord,
        source_path: Path,
        artifact_name: str,
    ) -> Path:
        destination = self.store.candidate_approved_path(record.record_id, artifact_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if Path(source_path).is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source_path, destination)
        else:
            shutil.copy2(source_path, destination)
        record.approved_path = str(destination)
        self.store.save_record(record)
        return destination
