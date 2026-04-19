from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from .database import Base


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(String(32), default="employee", nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Group(Base):
    __tablename__ = "groups"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_group_name_per_tenant"),)

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, default="", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class GroupMember(Base):
    __tablename__ = "group_members"
    __table_args__ = (UniqueConstraint("group_id", "user_id", name="uq_group_member"),)

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    group_id = Column(Integer, ForeignKey("groups.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class TenantPolicy(Base):
    __tablename__ = "tenant_policies"
    __table_args__ = (UniqueConstraint("tenant_id", name="uq_policy_per_tenant"),)

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)

    # Retrieval & policy
    min_confidence = Column(Float, default=0.22, nullable=False)
    force_handoff_keywords_json = Column(Text, default='["legal","complaint","incident","breach"]', nullable=False)
    pii_redaction_enabled = Column(Boolean, default=True, nullable=False)
    max_citations = Column(Integer, default=5, nullable=False)
    policy_rules_json = Column(Text, default="[]", nullable=False)
    policy_pack = Column(String(32), default="balanced", nullable=False)

    # Budget & safety guardrails
    daily_query_budget = Column(Integer, default=1000, nullable=False)
    daily_run_budget = Column(Integer, default=1000, nullable=False)
    daily_cost_budget_usd = Column(Float, default=25.0, nullable=False)
    max_top_k = Column(Integer, default=8, nullable=False)
    max_question_chars = Column(Integer, default=4000, nullable=False)

    # External/public API policy controls
    daily_external_api_budget = Column(Integer, default=200, nullable=False)
    external_api_timeout_cap_seconds = Column(Integer, default=8, nullable=False)
    public_api_allowlist_json = Column(Text, default="[]", nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    source_type = Column(String(50), default="upload", nullable=False)
    roles_allowed_json = Column(Text, default='["admin","manager","employee","viewer"]', nullable=False)
    raw_text = Column(Text, nullable=False)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DocumentPolicy(Base):
    __tablename__ = "document_policies"
    __table_args__ = (UniqueConstraint("document_id", name="uq_doc_policy"),)

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    groups_allowed_json = Column(Text, default="[]", nullable=False)
    tags_json = Column(Text, default="[]", nullable=False)
    classification = Column(String(32), default="internal", nullable=False)
    source_url = Column(String(1000), default="", nullable=False)
    freshness_score = Column(Float, default=0.5, nullable=False)
    freshness_last_checked_at = Column(DateTime, nullable=True)
    freshness_last_updated_at = Column(DateTime, nullable=True)
    freshness_check_interval_hours = Column(Integer, default=24, nullable=False)
    freshness_stale_after_hours = Column(Integer, default=168, nullable=False)
    auto_refresh_enabled = Column(Boolean, default=False, nullable=False)
    citation_anchor_mode = Column(String(32), default="char_offsets", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding_json = Column(Text, nullable=False)
    roles_allowed_json = Column(Text, nullable=False)
    start_char = Column(Integer, default=0, nullable=False)
    end_char = Column(Integer, default=0, nullable=False)
    page_number = Column(Integer, nullable=True)
    section_label = Column(String(255), default="", nullable=False)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    endpoint = Column(String(64), default="/api/ask", nullable=False)
    status = Column(String(32), default="running", nullable=False)

    idempotency_key = Column(String(255), default="", nullable=False)
    replay_of_run_id = Column(Integer, ForeignKey("agent_runs.id"), nullable=True)

    request_json = Column(Text, default="{}", nullable=False)
    response_json = Column(Text, default="{}", nullable=False)
    error = Column(Text, default="", nullable=False)

    estimated_input_tokens = Column(Integer, default=0, nullable=False)
    estimated_output_tokens = Column(Integer, default=0, nullable=False)
    estimated_cost_usd = Column(Float, default=0.0, nullable=False)

    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, default=0, nullable=False)


class AgentRunStep(Base):
    __tablename__ = "agent_run_steps"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("agent_runs.id"), nullable=False, index=True)
    step_order = Column(Integer, nullable=False)
    name = Column(String(64), nullable=False)
    status = Column(String(32), default="running", nullable=False)
    metadata_json = Column(Text, default="{}", nullable=False)

    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, default=0, nullable=False)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    run_id = Column(Integer, ForeignKey("agent_runs.id"), nullable=True, index=True)

    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)
    was_answered = Column(Boolean, default=True, nullable=False)

    estimated_input_tokens = Column(Integer, default=0, nullable=False)
    estimated_output_tokens = Column(Integer, default=0, nullable=False)
    estimated_cost_usd = Column(Float, default=0.0, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    query_log_id = Column(Integer, ForeignKey("query_logs.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    rating = Column(String(16), nullable=False)  # up/down
    reason = Column(Text, default="", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class HandoffTicket(Base):
    __tablename__ = "handoff_tickets"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    query_log_id = Column(Integer, ForeignKey("query_logs.id"), nullable=True)
    question = Column(Text, nullable=False)
    context = Column(Text, default="", nullable=False)
    status = Column(String(32), default="open", nullable=False)
    resolution = Column(Text, default="", nullable=False)
    sla_target_minutes = Column(Integer, default=1440, nullable=False)
    due_at = Column(DateTime, nullable=True)
    first_response_at = Column(DateTime, nullable=True)
    breached_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime, nullable=True)


class IntegrationConnection(Base):
    __tablename__ = "integration_connections"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    provider = Column(String(64), nullable=False)
    display_name = Column(String(255), nullable=False)
    status = Column(String(32), default="connected", nullable=False)
    config_json = Column(Text, default="{}", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    actor_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(128), nullable=False)
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(128), default="", nullable=False)
    metadata_json = Column(Text, default="{}", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class PublicApiProvider(Base):
    __tablename__ = "public_api_providers"
    __table_args__ = (
        UniqueConstraint("name", name="uq_public_api_provider_name"),
    )

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False, index=True)
    category = Column(String(64), default="open-data", nullable=False)
    base_url = Column(String(1000), nullable=False)
    auth_type = Column(String(32), default="none", nullable=False)
    docs_url = Column(String(1000), default="", nullable=False)
    cors = Column(String(32), default="unknown", nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    tenant_scope = Column(String(32), default="global", nullable=False)
    default_timeout_seconds = Column(Integer, default=5, nullable=False)
    rate_limit_hint = Column(String(255), default="", nullable=False)
    normalization_strategy = Column(String(64), default="auto", nullable=False)
    sample_query_json = Column(Text, default="{}", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    title = Column(String(500), nullable=False)
    source_type = Column(String(50), default="text", nullable=False)
    raw_text = Column(Text, nullable=False)
    roles_allowed_json = Column(Text, default='["admin","manager","employee","viewer"]', nullable=False)
    groups_allowed_json = Column(Text, default="[]", nullable=False)
    tags_json = Column(Text, default="[]", nullable=False)
    classification = Column(String(32), default="internal", nullable=False)
    source_url = Column(String(1000), default="", nullable=False)
    freshness_score = Column(Float, default=0.5, nullable=False)

    status = Column(String(32), default="queued", nullable=False)
    attempts = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    last_error = Column(Text, default="", nullable=False)
    next_attempt_at = Column(DateTime, nullable=True)

    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        UniqueConstraint("tenant_id", "endpoint", "idempotency_key", name="uq_idempotency_tenant_endpoint_key"),
    )

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    endpoint = Column(String(128), nullable=False)
    idempotency_key = Column(String(255), nullable=False)
    request_hash = Column(String(128), nullable=False)

    response_json = Column(Text, default="", nullable=False)
    status_code = Column(Integer, default=0, nullable=False)
    run_id = Column(Integer, ForeignKey("agent_runs.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
