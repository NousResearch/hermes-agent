"""Local persistence for Hermes OS architecture artifacts."""

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class LocalRepository:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def save(self, collection: str, record_id: str, record: Any):
        directory = os.path.join(self.root_path, ".hermes", "records", collection)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, _safe_name(record_id) + ".json")
        payload = _to_jsonable(record)
        if isinstance(payload, dict):
            payload.setdefault("id", record_id)
            payload.setdefault("stored_at", datetime.now(timezone.utc).isoformat())
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def get(self, collection: str, record_id: str):
        path = os.path.join(self.root_path, ".hermes", "records", collection, _safe_name(record_id) + ".json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def list(self, collection: str):
        directory = os.path.join(self.root_path, ".hermes", "records", collection)
        if not os.path.isdir(directory):
            return []
        records = []
        for file_name in sorted(os.listdir(directory)):
            if not file_name.endswith(".json"):
                continue
            with open(os.path.join(directory, file_name), "r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        return records

    def latest(self, collection: str):
        records = self.list(collection)
        if not records:
            return None
        return sorted(records, key=lambda item: item.get("stored_at", ""))[-1]


def persist_review_report(repository: LocalRepository, report):
    return repository.save("review-reports", report.project_id, report)


def persist_grill_me_session(repository: LocalRepository, session):
    return repository.save("grill-me-sessions", session.project_id, session)


def persist_decision(repository: LocalRepository, decision_id: str, decision: Dict[str, Any]):
    return repository.save("decisions", decision_id, decision)


def persist_approval(repository: LocalRepository, approval_id: str, approval: Dict[str, Any]):
    return repository.save("approvals", approval_id, approval)


def persist_agent_artifact(repository: LocalRepository, artifact_id: str, artifact: Dict[str, Any]):
    return repository.save("agent-artifacts", artifact_id, artifact)


def _to_jsonable(value: Any):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _safe_name(value: str):
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in str(value))
