"""Pure, fail-closed admission decisions for Kanban state transitions.

This module deliberately has no configuration, filesystem, database, process, or
network access.  Storage, authentication, ownership, and dispatcher wiring are
separate layers; callers must supply the exact policy and execution identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, cast
from uuid import UUID

_SUPPORTED_SCHEMA: Final = 1
_MODES: Final = frozenset({"frozen", "manual_allowlist", "open"})
_DIAGNOSTIC_ACTIONS: Final = frozenset({"diagnostic_read"})
_EXECUTION_ACTIONS: Final = frozenset(
    {"claim", "spawn", "promote", "unblock", "retry"}
)
_ALWAYS_FORBIDDEN_ACTIONS: Final = frozenset(
    {"default_assignment", "route_fallback"}
)
_POLICY_KEYS: Final = frozenset(
    {"schema_version", "generation", "mode", "paused", "admissions"}
)
_ADMISSION_KEYS: Final = frozenset(
    {"board_uuid", "task_id", "run_generation"}
)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _canonical_uuid(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        return None
    canonical = str(parsed)
    return canonical if value == canonical else None


def _task_id(value: object) -> str | None:
    if not isinstance(value, str) or not value or value.strip() != value:
        return None
    return value


@dataclass(frozen=True, slots=True)
class AdmissionIdentity:
    board_uuid: str
    task_id: str
    run_generation: int


@dataclass(frozen=True, slots=True)
class AdmissionRequest:
    action: str
    policy_generation: int
    board_uuid: str | None = None
    task_id: str | None = None
    run_generation: int | None = None
    force: bool = False


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    allowed: bool
    reason: str
    execution_enabling: bool


@dataclass(frozen=True, slots=True)
class AdmissionPolicy:
    schema_version: int
    generation: int
    mode: str
    paused: bool
    admissions: frozenset[AdmissionIdentity]
    valid: bool = True

    @classmethod
    def invalid(cls) -> AdmissionPolicy:
        return cls(
            schema_version=_SUPPORTED_SCHEMA,
            generation=0,
            mode="frozen",
            paused=True,
            admissions=frozenset(),
            valid=False,
        )

    @classmethod
    def from_mapping(cls, raw: object) -> AdmissionPolicy:
        if not isinstance(raw, Mapping) or set(raw) != _POLICY_KEYS:
            return cls.invalid()

        schema_version = raw.get("schema_version")
        generation = raw.get("generation")
        mode = raw.get("mode")
        paused = raw.get("paused")
        raw_admissions = raw.get("admissions")

        if type(schema_version) is not int or schema_version != _SUPPORTED_SCHEMA:
            return cls.invalid()
        if not _positive_int(generation):
            return cls.invalid()
        if not isinstance(mode, str) or mode not in _MODES:
            return cls.invalid()
        if not isinstance(paused, bool):
            return cls.invalid()
        if not isinstance(raw_admissions, list):
            return cls.invalid()

        admissions: set[AdmissionIdentity] = set()
        for item in raw_admissions:
            if not isinstance(item, Mapping) or set(item) != _ADMISSION_KEYS:
                return cls.invalid()
            board_uuid = _canonical_uuid(item.get("board_uuid"))
            task_id = _task_id(item.get("task_id"))
            run_generation = item.get("run_generation")
            if (
                board_uuid is None
                or task_id is None
                or not _positive_int(run_generation)
            ):
                return cls.invalid()
            identity = AdmissionIdentity(
                board_uuid, task_id, cast(int, run_generation)
            )
            if identity in admissions:
                return cls.invalid()
            admissions.add(identity)

        return cls(
            schema_version=schema_version,
            generation=cast(int, generation),
            mode=mode,
            paused=paused,
            admissions=frozenset(admissions),
        )

    def _is_structurally_valid(self) -> bool:
        if self.valid is not True:
            return False
        if type(self.schema_version) is not int or self.schema_version != _SUPPORTED_SCHEMA:
            return False
        if not _positive_int(self.generation):
            return False
        if type(self.mode) is not str or self.mode not in _MODES:
            return False
        if type(self.paused) is not bool:
            return False
        if not isinstance(self.admissions, frozenset):
            return False
        for identity in self.admissions:
            if not isinstance(identity, AdmissionIdentity):
                return False
            if _canonical_uuid(identity.board_uuid) is None:
                return False
            if _task_id(identity.task_id) is None:
                return False
            if not _positive_int(identity.run_generation):
                return False
        return True

    def decide(self, request: AdmissionRequest) -> AdmissionDecision:
        if not self._is_structurally_valid():
            action = request.action if isinstance(request, AdmissionRequest) else None
            execution_enabling = not (
                type(action) is str and action in _DIAGNOSTIC_ACTIONS
            )
            return AdmissionDecision(False, "invalid_policy", execution_enabling)
        if not isinstance(request, AdmissionRequest) or type(request.action) is not str:
            return AdmissionDecision(False, "unknown_action", True)

        execution_enabling = request.action not in _DIAGNOSTIC_ACTIONS
        if request.action in _DIAGNOSTIC_ACTIONS:
            return AdmissionDecision(True, "diagnostic_read", False)
        if request.action in _ALWAYS_FORBIDDEN_ACTIONS:
            return AdmissionDecision(False, "forbidden_action", True)
        if request.action not in _EXECUTION_ACTIONS:
            return AdmissionDecision(False, "unknown_action", True)
        if self.paused:
            return AdmissionDecision(False, "paused", True)
        if self.mode == "frozen":
            return AdmissionDecision(False, "frozen", True)
        if (
            not _positive_int(request.policy_generation)
            or request.policy_generation != self.generation
        ):
            return AdmissionDecision(False, "policy_generation_mismatch", True)

        board_uuid = _canonical_uuid(request.board_uuid)
        task_id = _task_id(request.task_id)
        if (
            board_uuid is None
            or task_id is None
            or not _positive_int(request.run_generation)
        ):
            return AdmissionDecision(False, "identity_not_admitted", True)

        identity = AdmissionIdentity(
            board_uuid, task_id, cast(int, request.run_generation)
        )
        if self.mode == "manual_allowlist" and identity not in self.admissions:
            return AdmissionDecision(False, "identity_not_admitted", True)
        if self.mode == "open":
            return AdmissionDecision(True, "open", True)
        if self.mode == "manual_allowlist":
            return AdmissionDecision(True, "exact_admission", True)

        # Defensive against state constructed outside from_mapping().
        return AdmissionDecision(False, "invalid_policy", True)
