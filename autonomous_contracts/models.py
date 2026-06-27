"""Pydantic models for contract-ledger autonomous PM execution.

These models intentionally cover the minimum v1 shape from the autonomous
contract PM spec. They are strict enough to prevent raw-prose execution drift,
but small enough to stabilize before adding CLI/profile-factory code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NonEmptyStr = Annotated[str, Field(min_length=1)]
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
CleanupState = Literal[
    "active_needed",
    "open",
    "closed",
    "archived",
    "retained_with_reason",
    "orphaned_blocker",
]
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
    "generated_artifact",
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


class McpGrant(StrictModel):
    server: NonEmptyStr
    clients: list[NonEmptyStr] = Field(default_factory=list)
    access: NonEmptyStr
    required: bool = True
    purpose: NonEmptyStr
    allowedSprintCategories: list[NonEmptyStr] = Field(default_factory=list)
    sideEffects: NonEmptyStr = "none"


class McpRuntime(StrictModel):
    policy: NonEmptyStr
    activeBaselineServers: list[NonEmptyStr] = Field(default_factory=list)
    clientHomes: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    grantSchema: str | None = None
    smokeTestScript: str | None = None
    forbiddenByDefault: list[NonEmptyStr] = Field(default_factory=list)


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
    model_config = ConfigDict(extra="allow", validate_assignment=True)

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
    mcpGrants: list[McpGrant] = Field(default_factory=list)
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
    mcpRuntime: McpRuntime | None = None
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
    amendmentReason: str | None = None

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
    mcpGrants: list[McpGrant] = Field(default_factory=list)
    review: ReviewPolicy = Field(default_factory=ReviewPolicy)
    closeout: CloseoutPolicy = Field(default_factory=CloseoutPolicy)
    workerPacket: str | None = None
    handoffArtifact: str | None = None
    handoff: str | None = None
    verificationEvidence: list[NonEmptyStr] = Field(default_factory=list)
    amendmentEvidence: list[NonEmptyStr] = Field(default_factory=list)
    cleanupEvidence: list[NonEmptyStr] = Field(default_factory=list)
    reviewEvidence: dict[str, Any] | None = None
    completedBy: str | None = None
    completedAt: str | None = None
    startedBy: str | None = None
    startedAt: str | None = None
    closedBy: str | None = None
    reviewException: str | None = None


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
    mcpRuntime: McpRuntime | None = None
    amendments: list[dict[str, Any]] = Field(default_factory=list)


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
    mcpGrants: list[McpGrant] = Field(default_factory=list)
    stopConditions: list[dict[str, Any]] = Field(default_factory=list)
    outputRequirements: list[NonEmptyStr]
    reviewPolicy: ReviewPolicy | None = None


EvidenceProvenance = Literal[
    "worker_authored",
    "control_plane_captured",
    "pm_rerun",
    "git_derived",
    "sqlite_derived",
    "supervisor_authored",
]
WarningClass = Literal[
    "informational",
    "deferred_non_blocking",
    "requires_next_sprint_ticket",
    "requires_benjamin_acceptance",
]
WorkerCloseoutStatus = Literal["completed", "completed_with_warnings", "failed", "action_required"]


class VerificationEvidenceRecord(StrictModel):
    id: NonEmptyStr
    type: VerificationType
    provenance: EvidenceProvenance
    passed: bool
    commandId: str | None = None
    command: str | None = None
    path: str | None = None
    exitCode: int | None = None
    observedAt: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_typed_payload(self) -> "VerificationEvidenceRecord":
        if self.type == "command":
            if not self.commandId or not self.command:
                raise ValueError("command evidence requires commandId and command")
        if self.type == "file_exists" and not self.path:
            raise ValueError("file_exists evidence requires path")
        return self


class CloseoutWarning(StrictModel):
    id: NonEmptyStr
    warningClass: WarningClass
    message: NonEmptyStr
    provenance: EvidenceProvenance
    commandId: str | None = None
    blocker: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)


class ArtifactRecord(StrictModel):
    path: NonEmptyStr
    kind: NonEmptyStr = "artifact"
    sha256: str | None = None
    provenance: EvidenceProvenance = "worker_authored"
    metadata: dict[str, Any] = Field(default_factory=dict)


class NoLiveDbMutationProof(StrictModel):
    provenance: EvidenceProvenance
    observedAt: NonEmptyStr
    evidence: dict[str, Any] = Field(default_factory=dict)


class WorkerCloseoutEnvelope(StrictModel):
    schemaVersion: Literal["worker-closeout/v1"] = "worker-closeout/v1"
    sprintId: NonEmptyStr
    workerPacketSha256: NonEmptyStr
    resultStatus: WorkerCloseoutStatus
    verificationEvidence: list[VerificationEvidenceRecord] = Field(default_factory=list)
    warnings: list[CloseoutWarning] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    noLiveDbMutationProof: NoLiveDbMutationProof | None = None
    handoffArtifact: ArtifactRecord | None = None
    sourceCompletedAt: str | None = None
    summary: str | None = None

    @model_validator(mode="after")
    def enforce_terminal_success_semantics(self) -> "WorkerCloseoutEnvelope":
        if self.resultStatus in {"completed", "completed_with_warnings"}:
            if self.noLiveDbMutationProof is None:
                raise ValueError("terminal-success closeout requires noLiveDbMutationProof")
            blockers = [warning.id for warning in self.warnings if warning.blocker]
            if blockers:
                raise ValueError(f"terminal-success closeout cannot contain blockers: {', '.join(blockers)}")
            benjamin = [warning.id for warning in self.warnings if warning.warningClass == "requires_benjamin_acceptance"]
            if benjamin:
                raise ValueError(
                    "requires_benjamin_acceptance warnings cannot be imported as terminal success without separate acceptance: "
                    + ", ".join(benjamin)
                )
        return self
