"""Inspectable local-filesystem implementation of EngineeringStore."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from engineering.domain import (
    Evidence,
    ReviewResult,
    VerificationResult,
    WorkflowRun,
)

from .base import (
    EngineeringStoreCorruption,
    EngineeringStoreError,
    EvidenceAlreadyExists,
    EvidenceNotFound,
    InvalidWorkflowIdentifier,
    ReviewAlreadyExists,
    ReviewNotFound,
    VerificationAlreadyExists,
    VerificationNotFound,
    WorkflowAlreadyExists,
    WorkflowNotFound,
)
from .records import (
    PersistenceRecordError,
    evidence_from_record,
    evidence_to_record,
    review_from_record,
    review_to_record,
    verification_from_record,
    verification_to_record,
    workflow_from_record,
    workflow_to_record,
)


_SAFE_WORKFLOW_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DomainValue = TypeVar("_DomainValue")


class FileEngineeringStore:
    """Persist Engineering facts beneath an explicitly configured root.

    Snapshot records use atomic same-directory replacement. Evidence is an
    append-only JSONL stream whose appended line is flushed before returning.
    Multi-process concurrent writers and file locking are deferred beyond V1.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        if isinstance(root, str) and not root.strip():
            raise ValueError("root must not be empty")
        self._root = Path(root).resolve(strict=False)
        self._runs_root = self._root / "runs"

    def create_workflow(self, workflow: WorkflowRun) -> None:
        run_dir = self._run_dir(workflow.workflow_run_id)
        if run_dir.exists():
            raise WorkflowAlreadyExists(
                f"workflow already exists: {workflow.workflow_run_id}"
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        self._atomic_write_json(
            run_dir / "workflow.json",
            workflow_to_record(workflow),
        )

    def get_workflow(self, workflow_run_id: str) -> WorkflowRun:
        workflow_path = self._workflow_path(workflow_run_id)
        if not workflow_path.is_file():
            raise WorkflowNotFound(
                f"workflow not found: {workflow_run_id}"
            )
        return self._read_record(
            workflow_path,
            workflow_from_record,
            "workflow",
        )

    def save_workflow(self, workflow: WorkflowRun) -> None:
        workflow_path = self._workflow_path(workflow.workflow_run_id)
        if not workflow_path.is_file():
            raise WorkflowNotFound(
                f"workflow not found: {workflow.workflow_run_id}"
            )
        self._atomic_write_json(
            workflow_path,
            workflow_to_record(workflow),
        )

    def append_evidence(self, evidence: Evidence) -> None:
        self._require_workflow(evidence.workflow_run_id)
        try:
            self.get_evidence(evidence.evidence_id)
        except EvidenceNotFound:
            pass
        else:
            raise EvidenceAlreadyExists(
                f"evidence already exists: {evidence.evidence_id}"
            )
        evidence_path = self._safe_run_path(
            evidence.workflow_run_id, "evidence.jsonl"
        )
        serialized = json.dumps(
            evidence_to_record(evidence),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            with evidence_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(serialized)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise EngineeringStoreError(
                f"failed to append evidence: {evidence.evidence_id}"
            ) from exc

    def list_evidence(
        self,
        workflow_run_id: str,
        attempt: int | None = None,
    ) -> tuple[Evidence, ...]:
        self._require_workflow(workflow_run_id)
        if attempt is not None:
            _validate_attempt(attempt)
        evidence_path = self._safe_run_path(
            workflow_run_id, "evidence.jsonl"
        )
        evidence = self._read_evidence_file(evidence_path)
        if attempt is None:
            return evidence
        return tuple(item for item in evidence if item.attempt == attempt)

    def get_evidence(self, evidence_id: str) -> Evidence:
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string")
        matches: list[Evidence] = []
        if self._runs_root.is_dir():
            for candidate in sorted(self._runs_root.iterdir()):
                if not candidate.is_dir():
                    continue
                evidence_path = self._safe_run_path(
                    candidate.name, "evidence.jsonl"
                )
                matches.extend(
                    evidence
                    for evidence in self._read_evidence_file(evidence_path)
                    if evidence.evidence_id == evidence_id
                )
        if not matches:
            raise EvidenceNotFound(f"evidence not found: {evidence_id}")
        if len(matches) > 1:
            raise EngineeringStoreCorruption(
                f"duplicate evidence identity: {evidence_id}"
            )
        return matches[0]

    def save_verification(self, result: VerificationResult) -> None:
        self._require_workflow(result.workflow_run_id)
        path = self._attempt_path(
            result.workflow_run_id,
            "verifications",
            result.attempt,
        )
        if path.exists():
            raise VerificationAlreadyExists(
                "verification already exists: "
                f"workflow_run_id={result.workflow_run_id}, "
                f"attempt={result.attempt}"
            )
        self._atomic_write_json(
            path,
            verification_to_record(result),
            create_once=True,
            conflict_error=VerificationAlreadyExists,
        )

    def get_verification(
        self,
        workflow_run_id: str,
        attempt: int,
    ) -> VerificationResult:
        self._require_workflow(workflow_run_id)
        path = self._attempt_path(workflow_run_id, "verifications", attempt)
        if not path.is_file():
            raise VerificationNotFound(
                "verification not found: "
                f"workflow_run_id={workflow_run_id}, attempt={attempt}"
            )
        return self._read_record(
            path,
            verification_from_record,
            "verification",
        )

    def save_review(self, result: ReviewResult) -> None:
        self._require_workflow(result.workflow_run_id)
        path = self._attempt_path(
            result.workflow_run_id,
            "reviews",
            result.attempt,
        )
        if path.exists():
            raise ReviewAlreadyExists(
                "review already exists: "
                f"workflow_run_id={result.workflow_run_id}, "
                f"attempt={result.attempt}"
            )
        self._atomic_write_json(
            path,
            review_to_record(result),
            create_once=True,
            conflict_error=ReviewAlreadyExists,
        )

    def get_review(
        self,
        workflow_run_id: str,
        attempt: int,
    ) -> ReviewResult:
        self._require_workflow(workflow_run_id)
        path = self._attempt_path(workflow_run_id, "reviews", attempt)
        if not path.is_file():
            raise ReviewNotFound(
                "review not found: "
                f"workflow_run_id={workflow_run_id}, attempt={attempt}"
            )
        return self._read_record(path, review_from_record, "review")

    def _run_dir(self, workflow_run_id: str) -> Path:
        if not isinstance(workflow_run_id, str) or not _SAFE_WORKFLOW_ID.fullmatch(
            workflow_run_id
        ):
            raise InvalidWorkflowIdentifier(
                f"invalid workflow_run_id: {workflow_run_id!r}"
            )
        candidate = self._runs_root / workflow_run_id
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self._runs_root)
        except ValueError as exc:
            raise InvalidWorkflowIdentifier(
                f"workflow_run_id escapes configured root: {workflow_run_id!r}"
            ) from exc
        return resolved

    def _workflow_path(self, workflow_run_id: str) -> Path:
        return self._safe_run_path(workflow_run_id, "workflow.json")

    def _safe_run_path(self, workflow_run_id: str, *parts: str) -> Path:
        candidate = self._run_dir(workflow_run_id).joinpath(*parts)
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise InvalidWorkflowIdentifier(
                "workflow path escapes configured root: "
                f"{workflow_run_id!r}"
            ) from exc
        return resolved

    def _require_workflow(self, workflow_run_id: str) -> None:
        if not self._workflow_path(workflow_run_id).is_file():
            raise WorkflowNotFound(f"workflow not found: {workflow_run_id}")

    def _attempt_path(
        self,
        workflow_run_id: str,
        collection: str,
        attempt: int,
    ) -> Path:
        _validate_attempt(attempt)
        return self._safe_run_path(
            workflow_run_id,
            collection,
            f"attempt-{attempt}.json",
        )

    def _atomic_write_json(
        self,
        path: Path,
        record: Mapping[str, object],
        *,
        create_once: bool = False,
        conflict_error: type[EngineeringStoreError] = EngineeringStoreError,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open(
                "x", encoding="utf-8", newline="\n"
            ) as stream:
                json.dump(
                    record,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            if create_once and path.exists():
                raise conflict_error(f"record already exists: {path.name}")
            os.replace(temporary, path)
        except EngineeringStoreError:
            raise
        except OSError as exc:
            raise EngineeringStoreError(
                f"failed to persist Engineering record: {path.name}"
            ) from exc
        finally:
            if temporary.exists():
                temporary.unlink()

    def _read_record(
        self,
        path: Path,
        converter: Callable[[Mapping[str, object]], _DomainValue],
        record_kind: str,
    ) -> _DomainValue:
        try:
            with path.open("r", encoding="utf-8") as stream:
                record: Any = json.load(stream)
            if not isinstance(record, Mapping):
                raise TypeError("top-level JSON value must be an object")
            return converter(record)
        except (
            json.JSONDecodeError,
            PersistenceRecordError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise EngineeringStoreCorruption(
                f"corrupt {record_kind} record: {path.name}"
            ) from exc
        except OSError as exc:
            raise EngineeringStoreError(
                f"failed to read Engineering record: {path.name}"
            ) from exc

    def _read_evidence_file(self, path: Path) -> tuple[Evidence, ...]:
        if not path.exists():
            return ()
        evidence: list[Evidence] = []
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        record: Any = json.loads(line)
                        if not isinstance(record, Mapping):
                            raise TypeError(
                                "evidence JSONL value must be an object"
                            )
                        evidence.append(evidence_from_record(record))
                    except (
                        json.JSONDecodeError,
                        PersistenceRecordError,
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        raise EngineeringStoreCorruption(
                            "corrupt evidence record: "
                            f"{path.name}, line={line_number}"
                        ) from exc
        except EngineeringStoreCorruption:
            raise
        except OSError as exc:
            raise EngineeringStoreError(
                f"failed to read Engineering evidence: {path.name}"
            ) from exc
        return tuple(evidence)


def _validate_attempt(attempt: int) -> None:
    if type(attempt) is not int or attempt < 1:
        raise ValueError("attempt must be an integer greater than zero")
