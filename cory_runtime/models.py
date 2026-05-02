from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RequestType = Literal[
    "advisory_discussion",
    "governed_change_request",
    "execution_task",
    "km_candidate",
]
WorkflowState = Literal[
    "received",
    "interpretation_pending",
    "needs_clarification",
    "draft_ready",
    "awaiting_approval",
    "approved",
    "rejected",
    "in_progress",
    "blocked",
    "completed",
    "archived",
]
InterpretationStatus = Literal["completed", "needs_human_review"]


class CoryBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SkillRequirement(CoryBaseModel):
    id: str
    required: bool
    why: str


class PromptBlueprint(CoryBaseModel):
    systemIntent: list[str] = Field(default_factory=list)
    operatingRules: list[str] = Field(default_factory=list)
    taskPrompt: str
    deliverables: list[str] = Field(default_factory=list)


class OutputField(CoryBaseModel):
    key: str
    required: bool
    description: str


class WorkflowStateOption(CoryBaseModel):
    state: WorkflowState
    useWhen: str


class OutputContract(CoryBaseModel):
    completeEndpoint: str
    failEndpoint: str
    responseLanguage: str
    artifactKeyLanguage: str
    interpretationFields: list[OutputField] = Field(default_factory=list)
    nextWorkflowStateOptions: list[WorkflowStateOption] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class HarnessContextSummary(CoryBaseModel):
    sourceType: str
    requestType: RequestType
    workflowState: WorkflowState
    interpretationCount: int
    openClarificationCount: int
    linkedExecutionTask: bool


class CoryHermesHarness(CoryBaseModel):
    version: str
    runtime: str
    agent: str
    mode: str
    scenario: str
    preferredResponseLanguage: str
    artifactKeyLanguage: str
    contextSummary: HarnessContextSummary
    skills: list[SkillRequirement] = Field(default_factory=list)
    prompt: PromptBlueprint
    guardrails: list[str] = Field(default_factory=list)
    outputContract: OutputContract


class JobRecord(CoryBaseModel):
    id: str
    requestId: str | None = None
    interpreter: str | None = None
    triggerSource: str | None = None
    requestedBy: str | None = None
    status: str


class RequestRecord(CoryBaseModel):
    id: str
    title: str | None = None
    routingText: str | None = None
    sourceType: str
    sourceUrl: str | None = None
    sourceEventId: str | None = None
    sourcePayload: Any = None
    projectId: str | None = None
    repoId: str | None = None
    activeInterpretationId: str | None = None
    requestType: RequestType
    workflowState: WorkflowState
    requestedBy: str | None = None


class InterpretationRecord(CoryBaseModel):
    id: str
    producedBy: str | None = None
    interpretationStatus: str | None = None
    summary: str | None = None
    proposedRequestType: RequestType | None = None
    proposedScope: str | None = None
    proposedNonGoals: str | None = None
    proposedClarifications: list[str] = Field(default_factory=list)
    rawResponse: dict[str, Any] | None = None


class ClarificationRecord(CoryBaseModel):
    id: str
    prompt: str
    status: str
    response: str | None = None
    answeredBy: str | None = None


class LinkedTaskRecord(CoryBaseModel):
    id: str
    title: str | None = None
    status: str | None = None


class ClaimedJobEnvelope(CoryBaseModel):
    ok: bool
    claimed: bool
    job: JobRecord | None = None
    request: RequestRecord | None = None
    interpretations: list[InterpretationRecord] = Field(default_factory=list)
    clarifications: list[ClarificationRecord] = Field(default_factory=list)
    linkedTask: LinkedTaskRecord | None = None
    harness: CoryHermesHarness | None = None

    @model_validator(mode="after")
    def validate_claim(self) -> "ClaimedJobEnvelope":
        if self.claimed and (self.job is None or self.request is None or self.harness is None):
            raise ValueError("claimed response must include job, request, and harness")
        return self


class InterpretationPayload(CoryBaseModel):
    producedBy: str = "cory_hermes"
    interpretationStatus: InterpretationStatus
    summary: str
    proposedRequestType: RequestType
    proposedProjectId: str | None = None
    proposedRepoId: str | None = None
    proposedScope: str | None = None
    proposedNonGoals: str | None = None
    proposedClarifications: list[str] = Field(default_factory=list)
    proposedBrief: dict[str, Any] | None = None
    rawResponse: dict[str, Any]

    @model_validator(mode="after")
    def validate_strings(self) -> "InterpretationPayload":
        self.producedBy = self.producedBy.strip()
        self.summary = self.summary.strip()
        if not self.producedBy:
            raise ValueError("interpretation.producedBy must be non-empty")
        if not self.summary:
            raise ValueError("interpretation.summary must be non-empty")

        normalized: list[str] = []
        for item in self.proposedClarifications:
            value = item.strip()
            if not value:
                raise ValueError("interpretation.proposedClarifications cannot contain blank items")
            normalized.append(value)
        self.proposedClarifications = normalized
        return self


class InterpretationSubmission(CoryBaseModel):
    interpretation: InterpretationPayload
    nextWorkflowState: WorkflowState | None = None
