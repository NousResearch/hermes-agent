"""Evidence-backed learning-session orchestration for StudyOS.

``LearningRuntime`` is the narrow lifecycle interface between a conversation
and StudyOS' durable evidence model.  It owns session state and activity
selection, while attempt persistence and competency diagnosis remain injected
dependencies.  Starting a session therefore creates no evidence; only an
explicit, evaluated observation can advance the competency snapshot.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from plugins.study_os.activities import ActivityAdapter, ActivityContext, GeneralActivityAdapter
from plugins.study_os.schemas import (
    EVIDENCE_DIMENSIONS,
    EVALUATOR_KINDS,
    LEARNING_CONTRACT_SCHEMA_VERSION,
    PROJECT_ID_RE,
    SCHEDULE_ID_RE,
    validate_learning_contract,
)


LEARNING_SESSION_SCHEMA_VERSION = "learning_session.v1"
ACTIVITY_SPEC_SCHEMA_VERSION = "study_activity_spec.v1"
COMPETENCY_SNAPSHOT_SCHEMA_VERSION = "competency_snapshot.v1"

AttemptReader = Callable[[str], list[dict[str, Any]]]
AttemptRecorder = Callable[[dict[str, Any]], dict[str, Any]]
SnapshotBuilder = Callable[[list[dict[str, Any]]], dict[str, Any]]
RecommendationBuilder = Callable[[dict[str, Any]], list[dict[str, Any]]]
NowFunction = Callable[[], datetime]

_ACTIVE_VAULTS_BY_CONVERSATION: dict[str, Path] = {}


class LearningRuntimeError(Exception):
    """A stable, model-facing LearningRuntime failure."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LearningRuntimeError("SESSION_STATE_INVALID", f"Invalid JSON in {path.name}") from exc
    if not isinstance(value, dict):
        raise LearningRuntimeError("SESSION_STATE_INVALID", f"{path.name} must contain an object")
    return value


def _runtime_index_path(vault: Path) -> Path:
    return vault / ".StudyOS" / "runtime" / "active-sessions.json"


def active_session_for_conversation(vault: Path, conversation_session_id: str) -> dict[str, Any] | None:
    """Resolve an explicitly-bound active learning session without mutating state."""

    key = str(conversation_session_id or "").strip()
    index_path = _runtime_index_path(vault)
    if not key or not index_path.exists():
        return None
    index = _read_mapping(index_path)
    binding = index.get(key)
    if not isinstance(binding, dict):
        return None
    project_id = str(binding.get("project_id") or "")
    learning_session_id = str(binding.get("learning_session_id") or "")
    if not PROJECT_ID_RE.match(project_id) or not SCHEDULE_ID_RE.match(learning_session_id):
        return None
    project_path = vault / ".StudyOS" / "projects" / project_id
    session_path = project_path / "sessions" / f"{learning_session_id}.json"
    try:
        session_path.resolve().relative_to(project_path.resolve())
    except ValueError:
        return None
    if not session_path.exists():
        return None
    session = _read_mapping(session_path)
    if session.get("status") != "active" or session.get("conversation_session_id") != key:
        return None
    return session


def active_vault_for_conversation(conversation_session_id: str) -> Path | None:
    """Return the process-local vault binding established by ``start``."""

    return _ACTIVE_VAULTS_BY_CONVERSATION.get(str(conversation_session_id or "").strip())


class LearningRuntime:
    """Run one explicit learning contract through evidence-backed activities."""

    def __init__(
        self,
        *,
        vault: Path,
        project: dict[str, Any],
        project_dir: Path,
        attempt_reader: AttemptReader,
        attempt_recorder: AttemptRecorder,
        snapshot_builder: SnapshotBuilder,
        recommendation_builder: RecommendationBuilder,
        activity_adapter: ActivityAdapter | None = None,
        now_fn: NowFunction | None = None,
    ) -> None:
        self.vault = vault
        self.project = project
        self.project_dir = project_dir
        self.attempt_reader = attempt_reader
        self.attempt_recorder = attempt_recorder
        self.snapshot_builder = snapshot_builder
        self.recommendation_builder = recommendation_builder
        self.activity_adapter = activity_adapter or GeneralActivityAdapter()
        self.now_fn = now_fn or (lambda: datetime.now().astimezone())

    def start(
        self,
        *,
        session_id: Any,
        contract: Any,
        conversation_session_id: Any = None,
    ) -> dict[str, Any]:
        """Create an active Session and its first activity without fabricating evidence."""

        resolved_session_id = self._validate_session_id(session_id)
        path = self._session_path(resolved_session_id)
        if path.exists():
            raise LearningRuntimeError("SESSION_EXISTS", f"Learning session already exists: {resolved_session_id}")
        normalized_contract = self._normalize_contract(contract, resolved_session_id)
        conversation_id = str(conversation_session_id or "").strip() or None
        if conversation_id:
            self._assert_conversation_available(conversation_id)
        now = self._now()
        session: dict[str, Any] = {
            "schema_version": LEARNING_SESSION_SCHEMA_VERSION,
            "session_id": resolved_session_id,
            "project_id": self.project["project_id"],
            "contract": normalized_contract,
            "status": "active",
            "started_at": now,
            "updated_at": now,
            "evidence_ids": [],
            "activity_history": [],
        }
        if conversation_id:
            session["conversation_session_id"] = conversation_id
        snapshot = self._competency_snapshot(session)
        session["current_activity"] = self._next_activity(session, snapshot, [])
        self._save_session(session)
        if conversation_id:
            self._bind_conversation(conversation_id, resolved_session_id)
        return {
            "session": session,
            "next_activity": session["current_activity"],
            "competency_snapshot": snapshot,
        }

    def advance(self, *, session_id: Any, observation: Any) -> dict[str, Any]:
        """Record one evaluated observation and select the next activity."""

        session = self._load_session(session_id)
        self._require_active(session)
        if not isinstance(observation, dict):
            raise LearningRuntimeError("VALIDATION_FAILED", "observation must be an object")
        evaluator = observation.get("evaluator")
        if not isinstance(evaluator, dict):
            raise LearningRuntimeError(
                "EVALUATOR_REQUIRED",
                "advance requires observation.evaluator so evidence provenance is explicit",
            )
        if evaluator.get("kind") not in EVALUATOR_KINDS:
            raise LearningRuntimeError(
                "VALIDATION_FAILED",
                f"observation.evaluator.kind must be one of: {', '.join(sorted(EVALUATOR_KINDS))}",
            )

        activity = session.get("current_activity")
        if not isinstance(activity, dict):
            raise LearningRuntimeError("SESSION_STATE_INVALID", "Active session has no current activity")
        issues = self.activity_adapter.validate_observation(activity, observation)
        if issues:
            issue = issues[0]
            raise LearningRuntimeError(issue.code, issue.message)
        attempt_result = self.attempt_recorder(self._attempt_args(session, activity, observation))
        if not attempt_result.get("ok"):
            error_value = attempt_result.get("error")
            error: dict[str, Any] = error_value if isinstance(error_value, dict) else {}
            raise LearningRuntimeError(
                str(error.get("code") or "EVIDENCE_RECORD_FAILED"),
                str(error.get("message") or "Failed to record learning evidence"),
                error.get("details") if isinstance(error.get("details"), dict) else None,
            )
        result_data_value = attempt_result.get("data")
        result_data: dict[str, Any] = result_data_value if isinstance(result_data_value, dict) else {}
        evidence = result_data.get("attempt")
        if not isinstance(evidence, dict):
            raise LearningRuntimeError("EVIDENCE_RECORD_FAILED", "Attempt recorder returned no evidence")

        evidence_id = str(evidence["attempt_id"])
        completed_activity = {
            **activity,
            "status": "completed",
            "completed_at": self._now(),
            "evidence_attempt_id": evidence_id,
        }
        session.setdefault("activity_history", []).append(completed_activity)
        session["evidence_ids"] = list(dict.fromkeys([*session.get("evidence_ids", []), evidence_id]))
        snapshot = self._competency_snapshot(session)
        recommendations = self.recommendation_builder(self._diagnosis_for(session))
        session["current_activity"] = self._next_activity(session, snapshot, recommendations)
        session["updated_at"] = self._now()
        self._save_session(session)
        return {
            "session": session,
            "evidence": evidence,
            "next_activity": session["current_activity"],
            "competency_snapshot": snapshot,
            "recommendations": recommendations,
        }

    def snapshot(self, *, session_id: Any) -> dict[str, Any]:
        """Rebuild the current competency view from immutable evidence."""

        session = self._load_session(session_id)
        snapshot = self._competency_snapshot(session)
        return {"session": session, "competency_snapshot": snapshot}

    def finish(self, *, session_id: Any) -> dict[str, Any]:
        """Close an active Session and report only evidence-supported outcomes."""

        session = self._load_session(session_id)
        self._require_active(session)
        snapshot = self._competency_snapshot(session)
        current = session.get("current_activity")
        if isinstance(current, dict):
            session.setdefault("activity_history", []).append(
                {**current, "status": "not_completed", "completed_at": self._now()}
            )
        session.pop("current_activity", None)
        now = self._now()
        session["status"] = "completed"
        session["completed_at"] = now
        session["updated_at"] = now
        self._save_session(session)
        conversation_id = session.get("conversation_session_id")
        if isinstance(conversation_id, str):
            self._unbind_conversation(conversation_id, str(session["session_id"]))
        required_dimensions = list(session["contract"]["evidence_targets"])
        unverified = [
            dimension
            for dimension in required_dimensions
            if snapshot["dimensions"].get(dimension, {}).get("verification_status") != "independent"
        ]
        observed = [
            dimension
            for dimension in required_dimensions
            if snapshot["dimensions"].get(dimension, {}).get("status") == "observed"
        ]
        outcome = {
            "evidence_count": len(session.get("evidence_ids", [])),
            "evidence_attempt_ids": list(session.get("evidence_ids", [])),
            "observed_dimensions": observed,
            "verified_dimensions": [item for item in required_dimensions if item not in unverified],
            "unverified_dimensions": unverified,
            "competency_snapshot": snapshot,
        }
        return {"session": session, "outcome": outcome}

    def _now(self) -> str:
        return self.now_fn().astimezone().isoformat(timespec="seconds")

    def _validate_session_id(self, value: Any) -> str:
        session_id = str(value or "").strip()
        if not SCHEDULE_ID_RE.match(session_id):
            raise LearningRuntimeError(
                "VALIDATION_FAILED",
                "session_id must match ^[a-z0-9][a-z0-9-]{2,79}$",
            )
        return session_id

    def _session_path(self, session_id: str) -> Path:
        return self.project_dir / "sessions" / f"{session_id}.json"

    def _load_session(self, session_id: Any) -> dict[str, Any]:
        resolved_session_id = self._validate_session_id(session_id)
        path = self._session_path(resolved_session_id)
        if not path.exists():
            raise LearningRuntimeError("SESSION_NOT_FOUND", f"Learning session not found: {resolved_session_id}")
        session = _read_mapping(path)
        if (
            session.get("schema_version") != LEARNING_SESSION_SCHEMA_VERSION
            or session.get("session_id") != resolved_session_id
            or session.get("project_id") != self.project.get("project_id")
        ):
            raise LearningRuntimeError("SESSION_STATE_INVALID", f"Invalid learning session: {resolved_session_id}")
        self._sync_evidence_ids(session)
        return session

    def _save_session(self, session: dict[str, Any]) -> None:
        _atomic_write_json(self._session_path(str(session["session_id"])), session)

    @staticmethod
    def _require_active(session: dict[str, Any]) -> None:
        if session.get("status") != "active":
            raise LearningRuntimeError(
                "SESSION_NOT_ACTIVE",
                f"Learning session is not active: {session.get('session_id')}",
            )

    def _normalize_contract(self, value: Any, session_id: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise LearningRuntimeError("VALIDATION_FAILED", "contract must be an object")
        contract = {
            **value,
            "schema_version": value.get("schema_version") or LEARNING_CONTRACT_SCHEMA_VERSION,
            "contract_id": value.get("contract_id") or f"contract-{uuid4().hex[:16]}",
            "project_id": value.get("project_id") or self.project["project_id"],
            "created_at": value.get("created_at") or self._now(),
        }
        ok, validated = validate_learning_contract(contract, project=self.project)
        if not ok:
            validation_errors = validated if isinstance(validated, list) else ["Invalid learning contract"]
            raise LearningRuntimeError(
                "VALIDATION_FAILED",
                "; ".join(validation_errors),
                {"errors": validation_errors, "session_id": session_id},
            )
        if not isinstance(validated, dict):
            raise LearningRuntimeError("VALIDATION_FAILED", "Learning contract validator returned invalid data")
        return validated

    def _relevant_attempts(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        attempts = self.attempt_reader(str(self.project["project_id"]))
        objective_ids = set(session["contract"].get("objective_ids", []))
        if objective_ids:
            return [
                attempt
                for attempt in attempts
                if objective_ids.intersection(str(item) for item in attempt.get("objective_ids", []))
            ]
        learning_session_id = str(session["session_id"])
        return [attempt for attempt in attempts if attempt.get("session_id") == learning_session_id]

    def _sync_evidence_ids(self, session: dict[str, Any]) -> None:
        recorded = [
            str(attempt.get("attempt_id"))
            for attempt in self.attempt_reader(str(self.project["project_id"]))
            if attempt.get("session_id") == session.get("session_id") and attempt.get("attempt_id")
        ]
        session["evidence_ids"] = list(dict.fromkeys([*session.get("evidence_ids", []), *recorded]))

    def _diagnosis_for(self, session: dict[str, Any]) -> dict[str, Any]:
        return self.snapshot_builder(self._relevant_attempts(session))

    def _competency_snapshot(self, session: dict[str, Any]) -> dict[str, Any]:
        attempts = self._relevant_attempts(session)
        diagnosis = self.snapshot_builder(attempts)
        dimensions = diagnosis.get("mastery_dimensions", {})
        provenance: Counter[str] = Counter()
        for attempt in attempts:
            evaluator = attempt.get("evaluator")
            if isinstance(evaluator, dict) and evaluator.get("kind"):
                provenance[str(evaluator["kind"])] += 1
            else:
                provenance["unprovenanced"] += 1
        return {
            "schema_version": COMPETENCY_SNAPSHOT_SCHEMA_VERSION,
            "project_id": self.project["project_id"],
            "objective_ids": list(session["contract"].get("objective_ids", [])),
            "built_at": self._now(),
            "evidence_count": len(attempts),
            "evidence_attempt_ids": [str(item.get("attempt_id")) for item in attempts],
            "dimensions": dimensions,
            "concepts": diagnosis.get("concepts", []),
            "diagnosis_clusters": diagnosis.get("diagnosis_clusters", []),
            "calibration": diagnosis.get("calibration", {}),
            "score_delta_earlier_to_later": diagnosis.get("score_delta_earlier_to_later"),
            "evaluator_provenance": dict(provenance),
            "unverified_dimensions": [
                dimension
                for dimension in EVIDENCE_DIMENSIONS
                if dimensions.get(dimension, {}).get("verification_status") != "independent"
            ],
        }

    def _objective_details(self, contract: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
        objective_ids = set(contract.get("objective_ids", []))
        criteria: list[str] = []
        anchors: list[dict[str, Any]] = []
        seen_anchors: set[str] = set()
        for objective in self.project.get("objectives", []):
            if not isinstance(objective, dict) or objective.get("objective_id") not in objective_ids:
                continue
            criteria.extend(str(item) for item in objective.get("success_criteria", []))
            for anchor in objective.get("source_anchors", []):
                fingerprint = json.dumps(anchor, ensure_ascii=False, sort_keys=True)
                if isinstance(anchor, dict) and fingerprint not in seen_anchors:
                    anchors.append(anchor)
                    seen_anchors.add(fingerprint)
        return list(dict.fromkeys(criteria)), anchors

    def _next_activity(
        self,
        session: dict[str, Any],
        snapshot: dict[str, Any],
        recommendations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        contract = session["contract"]
        dimensions = snapshot.get("dimensions", {})
        pending_targets = [
            target
            for target in contract["evidence_targets"]
            if dimensions.get(target, {}).get("status") != "observed"
        ]
        selected_recommendation = recommendations[0] if recommendations else None
        if pending_targets:
            target = pending_targets[0]
            reason = f"The learning contract still needs {target} evidence."
            activity_recommendation = None
            assistance_level = contract["assistance_level"]
        elif selected_recommendation:
            intervention = str(selected_recommendation.get("intervention") or "retention_probe")
            if intervention == "near_transfer_probe":
                target = "near_transfer"
            elif intervention == "independence_probe":
                target = str(selected_recommendation.get("evidence_dimension") or contract["evidence_targets"][-1])
            else:
                target = contract["evidence_targets"][-1]
            reason = str(selected_recommendation.get("reason") or "Verify the current competency estimate.")
            activity_recommendation = selected_recommendation
            assistance_level = "independent" if intervention == "independence_probe" else contract["assistance_level"]
        else:
            target = contract["evidence_targets"][-1]
            reason = "Verify the current competency estimate with a fresh, independently evaluated response."
            activity_recommendation = None
            assistance_level = "independent"
        criteria, anchors = self._objective_details(contract)
        sequence = len(session.get("activity_history", [])) + 1
        activity = {
            "schema_version": ACTIVITY_SPEC_SCHEMA_VERSION,
            "activity_id": f"activity-{session['session_id']}-{sequence:03d}",
            "session_id": session["session_id"],
            "project_id": self.project["project_id"],
            "kind": "evidence_probe",
            "objective": contract["objective"],
            "objective_ids": list(contract.get("objective_ids", [])),
            "evidence_target": target,
            "assistance_level": assistance_level,
            "instructions": f"Produce learner-authored {target} evidence for: {contract['objective']}",
            "response_policy": "Collect the learner's response before feedback or evaluator judgment.",
            "rubric_requirements": criteria or ["valid result", "reasoning made explicit", "independent contribution identified"],
            "source_anchors": anchors,
            "reason": reason,
            "status": "pending",
            "created_at": self._now(),
        }
        activity.update(
            self.activity_adapter.build(
                ActivityContext(
                    project=self.project,
                    contract=contract,
                    evidence_target=target,
                    recommendation=activity_recommendation,
                    success_criteria=criteria,
                    source_anchors=anchors,
                )
            )
        )
        return activity

    def _attempt_args(
        self,
        session: dict[str, Any],
        activity: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        contract = session["contract"]
        assistance_value = observation.get("assistance")
        assistance = dict(assistance_value) if isinstance(assistance_value, dict) else {}
        hints_used = assistance.get("hints_used", observation.get("hints_used"))
        if hints_used is None:
            hints_used = 0
        assistance["level"] = (
            assistance.get("level")
            or observation.get("assistance_level")
            or activity.get("assistance_level")
            or contract["assistance_level"]
        )
        assistance["hints_used"] = hints_used
        return {
            "vault_path": str(self.vault),
            "project_id": self.project["project_id"],
            "attempt_id": observation.get("attempt_id"),
            "item_id": observation.get("item_id") or activity["activity_id"],
            "occurred_at": observation.get("occurred_at") or self._now(),
            "response": observation.get("response"),
            "result": observation.get("result"),
            "score": observation.get("score"),
            "duration_seconds": observation.get("duration_seconds"),
            "hints_used": hints_used,
            "self_confidence": observation.get("self_confidence"),
            "evaluator_confidence": observation.get("evaluator_confidence"),
            "evaluator": observation.get("evaluator"),
            "assistance": assistance,
            "transfer_level": observation.get("transfer_level") or activity["evidence_target"],
            "concepts": observation.get("concepts", []),
            "patterns": observation.get("patterns", []),
            "objective_ids": list(contract.get("objective_ids", [])),
            "diagnoses": observation.get("diagnoses", []),
            "source_anchors": observation.get("source_anchors", activity.get("source_anchors", [])),
            "artifact_refs": observation.get("artifact_refs"),
            "activity_kind": activity.get("kind"),
            "source": activity["activity_id"],
            "session_id": session["session_id"],
        }

    def _read_active_index(self) -> dict[str, Any]:
        path = _runtime_index_path(self.vault)
        return _read_mapping(path) if path.exists() else {}

    def _assert_conversation_available(self, conversation_id: str) -> None:
        binding = self._read_active_index().get(conversation_id)
        if isinstance(binding, dict):
            raise LearningRuntimeError(
                "CONVERSATION_SESSION_ACTIVE",
                f"Conversation already has an active learning session: {binding.get('learning_session_id')}",
            )

    def _bind_conversation(self, conversation_id: str, learning_session_id: str) -> None:
        index = self._read_active_index()
        index[conversation_id] = {
            "project_id": self.project["project_id"],
            "learning_session_id": learning_session_id,
        }
        _atomic_write_json(_runtime_index_path(self.vault), index)
        _ACTIVE_VAULTS_BY_CONVERSATION[conversation_id] = self.vault

    def _unbind_conversation(self, conversation_id: str, learning_session_id: str) -> None:
        index = self._read_active_index()
        binding = index.get(conversation_id)
        if isinstance(binding, dict) and binding.get("learning_session_id") == learning_session_id:
            index.pop(conversation_id, None)
            _atomic_write_json(_runtime_index_path(self.vault), index)
        if _ACTIVE_VAULTS_BY_CONVERSATION.get(conversation_id) == self.vault:
            _ACTIVE_VAULTS_BY_CONVERSATION.pop(conversation_id, None)
