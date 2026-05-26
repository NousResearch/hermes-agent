"""Pydantic models for contract-ledger autonomous PM execution.

These models intentionally cover the minimum v1 shape from the autonomous
contract PM spec. They are strict enough to prevent raw-prose execution drift,
but small enough to stabilize before adding CLI/profile-factory code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NonEmptyStr = str
ContractStatus = Literal["draft", "approved", "executing", "superseded", "archived"]
GateType = Literal["human_approval", "ci_check", "ledger_condition", "external_service", "credential", "combined"]
GateOwner = Literal["galt", "benjamin", "pm_profile", "worker", "external"]
GateSeverity = Literal["blocking", "warning"]
VerificationType = Literal["command", "file_exists", "manual", "ledger_condition", "review_artifact", "none"]
SprintState = Literal[
    "not_started",
    "ready",
    "packet_generated",
    "dispatched",
    "in_progress",
    "review_required",
    "verification_required",
    "blocked_galt",
    "blocked_human",
    "failed",
    "completed",
    "completed_with_warnings",
    "skipped_by_galt_decision",
    "superseded",
]
WorkerRole = Literal["implementer", "reviewer", "dogfood", "research"]
CleanupState = Literal["active_needed", "closed", "archived", "retained_with_reason", "orphaned_blocker"]
CleanupType = Literal[
    "tmux_session",
    "worktree",
    "process",
    "port",
    "discord_thread",
    "kanban_card",
    "cron_job",
    "container",
    "temp_file",
    "browser_session",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectSpec(StrictModel):
    id: NonEmptyStr
    name: NonEmptyStr
    publicName: str | None = None
    repoPath: NonEmptyStr
    primaryBranch: NonEmptyStr = "main"
    supervisorProfile: NonEmptyStr = "galt"
    pmProfile: NonEmptyStr


class ChainOfCommand(StrictModel):
    owner: NonEmptyStr
    supervisor: NonEmptyStr
    projectManager: NonEmptyStr
    escalationPolicy: Literal["galt_first"] = "galt_first"
    benjaminEscalationAllowedOnlyBy: Literal["galt"] = "galt"


class CommunicationPolicy(StrictModel):
    primary: NonEmptyStr = "discord"
    profileHomeChannel: NonEmptyStr
    supervisorMention: NonEmptyStr
    botToBotPolicy: NonEmptyStr = "mentions_required"
    ackPolicy: NonEmptyStr = "reaction_for_ack_text_for_substance"
    unavailabilityPolicy: NonEmptyStr = "local_log_retry_5m_then_block_to_galt_when_available"
    messageClasses: list[NonEmptyStr] = Field(default_factory=list)


class CredentialsPolicy(StrictModel):
    inheritancePolicy: NonEmptyStr
    claudeCode: NonEmptyStr | bool = False
    codex: NonEmptyStr | bool = False
    github: NonEmptyStr | bool = False
    mcp: NonEmptyStr | bool = False
    secretsPolicy: NonEmptyStr = "do_not_print_or_commit"
    smokeTestsRequired: bool = True


class AuthorityPolicy(StrictModel):
    allowed: list[NonEmptyStr] = Field(default_factory=list)
    gated: list[NonEmptyStr] = Field(default_factory=list)
    forbidden: list[NonEmptyStr] = Field(default_factory=list)


class ModelPolicy(StrictModel):
    defaultImplementer: NonEmptyStr
    defaultReviewer: NonEmptyStr
    codexModel: NonEmptyStr | None = None
    codexReasoningEffort: NonEmptyStr | None = None
    claudeModel: NonEmptyStr | None = None
    claudeEffort: NonEmptyStr | None = None
    overrideAllowedBy: NonEmptyStr = "galt"
    localModelUse: NonEmptyStr = "scratch_or_low_risk_only"


class BudgetPolicy(StrictModel):
    maxWorkerCallsPerSprint: int = Field(ge=0)
    maxReviewerCallsPerSprint: int = Field(ge=0)
    maxSprintWallClockMinutes: int = Field(gt=0)
    maxDiffFilesPerSprint: int = Field(gt=0)
    maxDiffLinesPerSprint: int = Field(gt=0)
    maxRetriesPerWorkerPacket: int = Field(ge=0)
    maxConsecutiveFailedSprints: int = Field(ge=0)


class KanbanPolicy(StrictModel):
    enabled: bool = False
    platform: str | None = None
    role: str | None = None
    sourceOfTruth: Literal["ledger"] = "ledger"
    conflictPolicy: str | None = None


class CleanupPolicy(StrictModel):
    closeLedgerItems: bool = True
    closeKanbanProjectionItems: bool = True
    summarizeDiscordThreadsWhenDone: bool = True
    killIdleWorkerSessions: bool = True
    archiveOrRemoveTempWorktrees: bool = True
    reconcileBackgroundProcesses: bool = True
    reconcileOpenPorts: bool = True
    reconcileProjectCronJobs: bool = True
    reconcileContainers: bool = True
    recordCleanupEvidence: bool = True


class Gate(StrictModel):
    id: NonEmptyStr
    type: GateType
    owner: GateOwner
    severity: GateSeverity = "blocking"
    description: NonEmptyStr
    blocksSprintIds: list[NonEmptyStr] = Field(default_factory=list)
    resolutionCondition: NonEmptyStr
    evidenceRequired: list[NonEmptyStr] = Field(default_factory=list)
    expiresAfter: str | None = None


class Section(StrictModel):
    id: NonEmptyStr
    title: NonEmptyStr
    objective: NonEmptyStr
    order: int = Field(ge=0)


class AcceptanceCriterion(StrictModel):
    id: NonEmptyStr
    text: NonEmptyStr
    verification: VerificationType
    command: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def require_verification_payload(self) -> "AcceptanceCriterion":
        if self.verification == "command" and not self.command:
            raise ValueError("command verification requires command")
        if self.verification == "file_exists" and not self.path:
            raise ValueError("file_exists verification requires path")
        return self


class ReviewPolicy(StrictModel):
    required: bool = False


class CloseoutPolicy(StrictModel):
    requiresCleanupAudit: bool = True
    requiresGaltGate: bool = False


class Sprint(StrictModel):
    id: NonEmptyStr
    section: NonEmptyStr
    title: NonEmptyStr
    order: int = Field(ge=0)
    dependsOn: list[NonEmptyStr] = Field(default_factory=list)
    priority: int = 0
    parallelSafe: bool = False
    materialType: NonEmptyStr
    allowedPaths: list[NonEmptyStr] = Field(default_factory=list)
    forbiddenPaths: list[NonEmptyStr] = Field(default_factory=list)
    objective: NonEmptyStr
    requiredInputs: list[NonEmptyStr] = Field(default_factory=list)
    requiredContext: list[NonEmptyStr] = Field(default_factory=list)
    implementationRequirements: list[NonEmptyStr] = Field(default_factory=list)
    acceptance: list[AcceptanceCriterion] = Field(default_factory=list)
    gates: list[NonEmptyStr] = Field(default_factory=list)
    stopConditions: list[dict[str, Any]] = Field(default_factory=list)
    evidenceRequired: list[NonEmptyStr] = Field(default_factory=list)
    review: ReviewPolicy = Field(default_factory=ReviewPolicy)
    closeout: CloseoutPolicy = Field(default_factory=CloseoutPolicy)

    @field_validator("acceptance")
    @classmethod
    def require_acceptance(cls, value: list[AcceptanceCriterion]) -> list[AcceptanceCriterion]:
        if not value:
            raise ValueError("sprint must define at least one acceptance criterion")
        return value


class Contract(StrictModel):
    schemaVersion: Literal["autonomous-contract/v1"]
    contractId: NonEmptyStr
    contractVersion: NonEmptyStr
    contractStatus: ContractStatus
    project: ProjectSpec
    chainOfCommand: ChainOfCommand
    communication: CommunicationPolicy
    credentials: CredentialsPolicy
    authority: AuthorityPolicy
    modelPolicy: ModelPolicy
    budgets: BudgetPolicy
    kanban: KanbanPolicy = Field(default_factory=KanbanPolicy)
    cleanupPolicy: CleanupPolicy = Field(default_factory=CleanupPolicy)
    gates: list[Gate] = Field(default_factory=list)
    sections: list[Section]
    sprints: list[Sprint]


class ContractLock(StrictModel):
    contractId: NonEmptyStr
    contractVersion: NonEmptyStr
    schemaVersion: Literal["autonomous-contract/v1"]
    contractSha256: NonEmptyStr
    approvedBy: NonEmptyStr
    approvedAt: str
    ledgerInitializedAt: str
    active: bool = True

    @classmethod
    def new(cls, contract: Contract, contractSha256: str, approvedBy: str, approvedAt: datetime | None = None) -> "ContractLock":
        now = approvedAt or datetime.now(timezone.utc)
        iso = now.isoformat()
        return cls(
            contractId=contract.contractId,
            contractVersion=contract.contractVersion,
            schemaVersion=contract.schemaVersion,
            contractSha256=contractSha256,
            approvedBy=approvedBy,
            approvedAt=iso,
            ledgerInitializedAt=iso,
            active=True,
        )


class LedgerSprintRecord(StrictModel):
    sprintId: NonEmptyStr
    state: SprintState = "not_started"
    section: NonEmptyStr
    title: NonEmptyStr
    order: int = Field(ge=0)
    priority: int = 0
    parallelSafe: bool = False
    materialType: NonEmptyStr
    objective: NonEmptyStr
    dependsOn: list[NonEmptyStr] = Field(default_factory=list)
    gates: list[NonEmptyStr] = Field(default_factory=list)
    requiredInputs: list[NonEmptyStr] = Field(default_factory=list)
    requiredContext: list[NonEmptyStr] = Field(default_factory=list)
    implementationRequirements: list[NonEmptyStr] = Field(default_factory=list)
    acceptance: list[AcceptanceCriterion] = Field(default_factory=list)
    stopConditions: list[dict[str, Any]] = Field(default_factory=list)
    evidenceRequired: list[NonEmptyStr] = Field(default_factory=list)
    allowedPaths: list[NonEmptyStr] = Field(default_factory=list)
    forbiddenPaths: list[NonEmptyStr] = Field(default_factory=list)
    review: ReviewPolicy = Field(default_factory=ReviewPolicy)
    closeout: CloseoutPolicy = Field(default_factory=CloseoutPolicy)


class LedgerGateRecord(StrictModel):
    gateId: NonEmptyStr
    type: GateType
    owner: GateOwner
    severity: GateSeverity = "blocking"
    description: NonEmptyStr
    blocksSprintIds: list[NonEmptyStr] = Field(default_factory=list)
    resolutionCondition: NonEmptyStr
    evidenceRequired: list[NonEmptyStr] = Field(default_factory=list)
    resolved: bool = False
    resolvedBy: str | None = None
    resolvedAt: str | None = None
    evidence: list[NonEmptyStr] = Field(default_factory=list)


class CleanupRecord(StrictModel):
    id: NonEmptyStr
    type: CleanupType
    createdBy: NonEmptyStr
    sprintId: NonEmptyStr
    createdAt: str
    state: CleanupState
    identifier: NonEmptyStr
    owner: NonEmptyStr
    closeCondition: NonEmptyStr
    closedAt: str | None = None
    notes: str | None = None


class CleanupRegistry(StrictModel):
    schemaVersion: Literal["cleanup-registry/v1"] = "cleanup-registry/v1"
    records: list["CleanupRecord"] = Field(default_factory=list)


class LedgerSeed(StrictModel):
    schemaVersion: Literal["contract-ledger/v1"] = "contract-ledger/v1"
    contractId: NonEmptyStr
    contractVersion: NonEmptyStr
    contractSha256: NonEmptyStr
    projectId: NonEmptyStr
    contractLock: ContractLock | None = None
    sprints: list[LedgerSprintRecord]
    gates: list[LedgerGateRecord]
    cleanupRegistry: CleanupRegistry = Field(default_factory=CleanupRegistry)


class WorkerPacket(StrictModel):
    schemaVersion: Literal["worker-packet/v1"] = "worker-packet/v1"
    packetId: NonEmptyStr
    projectId: NonEmptyStr
    sprintId: NonEmptyStr
    workerRole: WorkerRole
    assignedWorker: NonEmptyStr
    sessionId: NonEmptyStr
    allowedPaths: list[NonEmptyStr]
    forbiddenPaths: list[NonEmptyStr]
    mission: NonEmptyStr
    context: dict[str, Any] = Field(default_factory=dict)
    acceptanceCriteria: list[AcceptanceCriterion]
    verificationCommands: list[NonEmptyStr] = Field(default_factory=list)
    stopConditions: list[dict[str, Any]] = Field(default_factory=list)
    outputRequirements: list[NonEmptyStr]
