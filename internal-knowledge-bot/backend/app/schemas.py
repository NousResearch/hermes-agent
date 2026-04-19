from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    company_name: str
    name: str
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    password: str = Field(min_length=8)
    role: str = Field(pattern="^(admin|manager|employee|viewer)$")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    tenant_id: int
    email: EmailStr
    name: str
    role: str


class GroupCreateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=128)
    description: str = ""


class GroupAddMemberByEmailRequest(BaseModel):
    email: EmailStr


class GroupOut(BaseModel):
    id: int
    name: str
    description: str
    member_count: int
    created_at: datetime


class DocumentTextCreate(BaseModel):
    title: str
    text: str
    roles_allowed: list[str] = Field(default_factory=lambda: ["admin", "manager", "employee", "viewer"])
    groups_allowed: list[int] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    classification: str = "internal"
    source_url: str = ""
    freshness_score: float = 0.5
    auto_refresh_enabled: bool = False
    freshness_check_interval_hours: int = Field(default=24, ge=1, le=24 * 30)
    freshness_stale_after_hours: int = Field(default=168, ge=1, le=24 * 365)
    citation_anchor_mode: str = Field(default="char_offsets", pattern="^(char_offsets|page_section)$")


class DocumentOut(BaseModel):
    id: int
    title: str
    source_type: str
    roles_allowed: list[str]
    groups_allowed: list[int]
    tags: list[str]
    classification: str
    source_url: str
    freshness_score: float
    freshness_last_checked_at: datetime | None = None
    freshness_last_updated_at: datetime | None = None
    auto_refresh_enabled: bool
    freshness_check_interval_hours: int
    freshness_stale_after_hours: int
    citation_anchor_mode: str
    chunk_count: int
    created_at: datetime


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    idempotency_key: str | None = None


class Citation(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    snippet: str
    score: float
    semantic_score: float
    keyword_score: float
    classification: str
    source_url: str
    chunk_index: int
    start_char: int
    end_char: int
    page_number: int | None = None
    section_label: str = ""


class AskResponse(BaseModel):
    answer: str
    confidence: float
    citations: list[Citation]
    handoff_recommended: bool
    matched_policy_keywords: list[str]
    matched_policy_rules: list[str]
    abstained: bool
    query_log_id: int
    run_id: int | None = None
    run_status: str | None = None
    budget_enforced: bool = False


class FeedbackCreate(BaseModel):
    query_log_id: int
    rating: str = Field(pattern="^(up|down)$")
    reason: str = ""


class HandoffCreate(BaseModel):
    question: str
    context: str = ""
    query_log_id: int | None = None
    sla_target_minutes: int | None = Field(default=None, ge=1, le=60 * 24 * 30)


class HandoffResolve(BaseModel):
    resolution: str


class HandoffOut(BaseModel):
    id: int
    question: str
    status: str
    resolution: str
    due_at: datetime | None = None
    first_response_at: datetime | None = None
    breached_at: datetime | None = None
    created_at: datetime
    resolved_at: datetime | None = None


class IntegrationConnectRequest(BaseModel):
    provider: str = Field(pattern="^(slack|notion|gdrive|webhook|hermes|public_api)$")
    display_name: str
    config: dict[str, Any] = Field(default_factory=dict)


class PublicApiProviderOut(BaseModel):
    id: int
    name: str
    category: str
    base_url: str
    auth_type: str
    docs_url: str
    cors: str
    enabled: bool
    tenant_scope: str
    default_timeout_seconds: int
    rate_limit_hint: str
    normalization_strategy: str
    sample_query: dict[str, Any]
    allowed_for_tenant: bool


class PublicApiFetchRequest(BaseModel):
    provider: str = Field(min_length=2, max_length=128)
    path: str = Field(default="")
    query: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int | None = Field(default=None, ge=1, le=60)
    retries: int = Field(default=1, ge=0, le=2)
    idempotency_key: str | None = None


class PublicApiFetchResponse(BaseModel):
    success: bool
    provider: str
    url: str
    status_code: int
    item_count: int
    items: list[dict[str, Any]]
    raw: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    run_id: int | None = None
    run_status: str | None = None
    budget_enforced: bool = False
    external_api_budget_remaining: int | None = None


class HermesToolInvokeRequest(BaseModel):
    tool_name: str = Field(min_length=1, max_length=128)
    args: dict[str, Any] = Field(default_factory=dict)
    task_id: str | None = None


class HermesToolInvokeResponse(BaseModel):
    success: bool
    tool_name: str
    task_id: str
    result: dict[str, Any]


class TenantPolicyRule(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    field: str = Field(pattern="^(question|answer|confidence|classification|role)$")
    op: str = Field(pattern="^(contains|equals|lt|lte|gt|gte|in|regex)$")
    value: str | float | int | bool | list[str]
    action: str = Field(pattern="^(handoff|allow|deny|redact)$")
    reason: str = ""
    enabled: bool = True


class TenantPolicyOut(BaseModel):
    min_confidence: float
    force_handoff_keywords: list[str]
    pii_redaction_enabled: bool
    max_citations: int
    rules: list[TenantPolicyRule]
    policy_pack: str
    daily_query_budget: int
    daily_run_budget: int
    daily_cost_budget_usd: float
    max_top_k: int
    max_question_chars: int


class TenantPolicyUpdate(BaseModel):
    min_confidence: float | None = None
    force_handoff_keywords: list[str] | None = None
    pii_redaction_enabled: bool | None = None
    max_citations: int | None = None
    rules: list[TenantPolicyRule] | None = None
    policy_pack: str | None = Field(default=None, pattern="^(safe|balanced|aggressive)$")
    daily_query_budget: int | None = Field(default=None, ge=1, le=100000)
    daily_run_budget: int | None = Field(default=None, ge=1, le=100000)
    daily_cost_budget_usd: float | None = Field(default=None, ge=0.01, le=100000.0)
    max_top_k: int | None = Field(default=None, ge=1, le=20)
    max_question_chars: int | None = Field(default=None, ge=32, le=16000)


class PolicyValidationRequest(BaseModel):
    question: str
    confidence: float
    classification: str = "internal"
    role: str = "employee"


class PolicyValidationResponse(BaseModel):
    handoff_recommended: bool
    matched_keywords: list[str]
    matched_rules: list[str]
    final_action: str


class IngestionJobCreate(BaseModel):
    title: str
    text: str
    roles_allowed: list[str] = Field(default_factory=lambda: ["admin", "manager", "employee", "viewer"])
    groups_allowed: list[int] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    classification: str = "internal"
    source_url: str = ""
    freshness_score: float = 0.5
    max_attempts: int = Field(default=3, ge=1, le=10)


class IngestionJobOut(BaseModel):
    id: int
    status: str
    attempts: int
    max_attempts: int
    last_error: str
    document_id: int | None = None
    next_attempt_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class FreshnessRunOut(BaseModel):
    scanned: int
    updated: int
    skipped: int
    errors: int


class AgentRunStepOut(BaseModel):
    id: int
    step_order: int
    name: str
    status: str
    metadata: dict[str, Any]
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int


class AgentRunOut(BaseModel):
    id: int
    endpoint: str
    status: str
    idempotency_key: str
    replay_of_run_id: int | None = None
    request: dict[str, Any]
    response: dict[str, Any]
    error: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int
    steps: list[AgentRunStepOut] = Field(default_factory=list)


class ReplayRunResponse(BaseModel):
    success: bool
    source_run_id: int
    replay_run_id: int


class UsageOverviewOut(BaseModel):
    day_utc: str
    queries_today: int
    runs_today: int
    estimated_cost_today_usd: float
    query_budget_remaining: int
    run_budget_remaining: int
    cost_budget_remaining_usd: float
