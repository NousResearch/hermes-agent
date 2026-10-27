"""Reviewed field policy for a complete, lossless Hermes state export.

This module is deliberately independent of the exporter and of any output sink.  The
schema census is the official ``521`` source identity; the writer call-site identity
is current ``188`` source.  Keeping those identities separate is required because the
DDL census and the current writer tree are different evidence inputs.

The policy is a closed allowlist.  A producer must validate the live SQLite schema
against it before opening rows.  Unknown tables/columns are a refusal, not an
implicit omission.  FTS/SQLite implementation tables are retained only as separate
derived evidence and are never folded into business records.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

OFFICIAL_SOURCE_ID = "official-521-schema31"
CURRENT_WRITER_SOURCE_ID = "current-writer-188"
OFFICIAL_SCHEMA_VERSION = 31
SCHEMA_CENSUS_SHA256 = "6861f3d75ef33363a1a54a1b2a6ffa0663d05778df7b6696cc4c0cb69ffa3d37"
FIELD_POLICY_VERSION = "agc1663-field-policy-package/v1"
FIELD_POLICY_REVIEW_SHA256 = "aa7756db0873e40505634d72dd62dede3818ce9078e43d5b2e06a014eaf839fd"

# Version 32 retains the v31 field layout, measured through the stock constructor
# and optional Telegram migration. Its independent field/writer review binds the
# separate v32 provenance without changing the accepted historical v30/v31 identities.
SCHEMA32_SCHEMA_VERSION = 32
SCHEMA32_SOURCE_ID = "official-e003-schema32"
SCHEMA32_WRITER_SOURCE_ID = "current-writer-18c525b669b6ea4ba62e6967d228af21d89aa9d3"
SCHEMA32_SCHEMA_CENSUS_SHA256 = "a9c9c9f8bb8c3e537a0bfd47adaff7ac76c42e23f514d2a26127d15f96ed26ff"
SCHEMA32_FIELD_POLICY_REVIEW_SHA256 = "dbe0e921cbfea7ff3308046630b3b90c7862cc0ef9ebf8abce70f5d688d5d7a4"

BUSINESS_PAYLOAD = "business_payload"
BUSINESS_METADATA = "business_metadata"
CONTROL_RECOVERY = "control_recovery"
DELIVERY_METADATA = "delivery_metadata"
SECURITY_IDENTITY = "security_identity"
MIXED_REQUIRES_PROJECTION = "mixed_requires_projection"
DERIVED_INTERNAL = "derived_internal"
CREDENTIAL_SECURITY_EXCLUDED = "credential_security_excluded"

BUSINESS_DISPOSITIONS = frozenset((BUSINESS_PAYLOAD, BUSINESS_METADATA))
EXACT_VALUE_DISPOSITIONS = frozenset(
    (BUSINESS_PAYLOAD, BUSINESS_METADATA, CONTROL_RECOVERY, DELIVERY_METADATA,
     SECURITY_IDENTITY, MIXED_REQUIRES_PROJECTION, DERIVED_INTERNAL)
)

# Extension owners provide a closed field disposition, but Core cannot apply a
# credential projection to an addon-owned value.  The mixed disposition is
# therefore deliberately excluded from the typed extension seam; an extension
# must either retain the exact scalar/raw value or refuse its own capture.
EXTENSION_EXACT_DISPOSITIONS = frozenset(
    EXACT_VALUE_DISPOSITIONS - {MIXED_REQUIRES_PROJECTION}
)


def extension_disposition(value: Any) -> str:
    """Normalize one source-owned extension disposition through the closed vocabulary."""
    raw = getattr(value, "value", value)
    if not isinstance(raw, str) or raw not in EXTENSION_EXACT_DISPOSITIONS:
        raise ValueError("extension field disposition is outside the exact closed policy")
    return raw


@dataclass(frozen=True)
class ColumnPolicy:
    name: str
    disposition: str
    raw_value_retention: str = "exact"

    @property
    def requires_projection(self) -> bool:
        return self.disposition == MIXED_REQUIRES_PROJECTION

    @property
    def retains_business_bytes(self) -> bool:
        return self.disposition in BUSINESS_DISPOSITIONS or self.requires_projection


@dataclass(frozen=True)
class TablePolicy:
    name: str
    columns: tuple[ColumnPolicy, ...]
    table_disposition: str
    writer_fence_status: str
    optional: bool = False

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def mixed_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.requires_projection)


@dataclass(frozen=True)
class VersionedSchemaPolicy:
    """Immutable field/schema mapping selected for one observed database version."""

    schema_version: int
    official_schema_source: str
    writer_source: str
    table_policies: Mapping[str, TablePolicy]
    schema_census_sha256: str = ""
    field_policy_review_sha256: str = ""


@dataclass(frozen=True)
class SchemaValidation:
    schema_version: int | None
    present_tables: tuple[str, ...]
    optional_tables: tuple[str, ...]
    unknown_tables: tuple[str, ...]
    missing_tables: tuple[str, ...]
    column_mismatches: Mapping[str, tuple[str, ...]]
    relation_type_mismatches: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    missing_extension_tables: tuple[str, ...] = ()
    missing_extension_relations: tuple[str, ...] = ()
    derived_relation_mismatches: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    absent_optional_derived_relations: tuple[str, ...] = ()
    missing_required_derived_relations: tuple[str, ...] = ()
    expected_schema_version: int = OFFICIAL_SCHEMA_VERSION

    @property
    def valid(self) -> bool:
        return not (
            self.unknown_tables
            or self.missing_tables
            or self.column_mismatches
            or self.relation_type_mismatches
            or self.missing_extension_tables
            or self.missing_extension_relations
            or self.derived_relation_mismatches
            or self.missing_required_derived_relations
            or self.schema_version != self.expected_schema_version
        )

    def refusal_reason(self) -> str:
        if self.valid:
            return ""
        pieces: list[str] = []
        if self.schema_version != self.expected_schema_version:
            pieces.append(
                f"schema_version={self.schema_version!r} (expected {self.expected_schema_version})"
            )
        if self.missing_tables:
            pieces.append("missing tables: " + ", ".join(self.missing_tables))
        if self.unknown_tables:
            pieces.append("unknown tables: " + ", ".join(self.unknown_tables))
        if self.column_mismatches:
            pieces.extend(
                f"{table} columns: {', '.join(columns)}"
                for table, columns in sorted(self.column_mismatches.items())
            )
        if self.relation_type_mismatches:
            pieces.extend(
                f"{relation} relation type: {', '.join(details)}"
                for relation, details in sorted(self.relation_type_mismatches.items())
            )
        if self.missing_extension_tables:
            pieces.append(
                "missing extension tables: " + ", ".join(self.missing_extension_tables)
            )
        if self.missing_extension_relations:
            pieces.append(
                "missing extension relations: " + ", ".join(self.missing_extension_relations)
            )
        if self.derived_relation_mismatches:
            pieces.extend(
                f"{relation} derived relation: {', '.join(details)}"
                for relation, details in sorted(self.derived_relation_mismatches.items())
            )
        if self.missing_required_derived_relations:
            pieces.append(
                "missing required derived relations: "
                + ", ".join(self.missing_required_derived_relations)
            )
        return "; ".join(pieces)


@dataclass(frozen=True)
class ModelConfigProjection:
    """The only reviewed mixed security projection in the core policy.

    ``value`` is the bytes that may enter an export sink.  When the known top-level
    ``api_key`` member is present, the original whole-column bytes are forbidden and
    ``value`` is a typed JSON projection with that member removed.  Unknown members
    are preserved; this function does not recursively scan or redact user text.
    """

    value: bytes
    raw_value_allowed: bool
    excluded_json_paths: tuple[str, ...]
    source_was_bytes: bool


class ModelConfigProjectionRefused(ValueError):
    """The reviewed typed projection cannot safely represent model_config."""


# Generated from the immutable reviewed policy package.  Do not append a table here
# without a new field-policy review and a corresponding writer-fence decision.
_TABLE_SPECS = {'async_delegations': {'optional': False, 'table_disposition': 'business_and_control', 'writer_fence_status': 'RUNTIME_LEDGER_PARTIAL; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('delegation_id', 'security_identity'), ('origin_session', 'security_identity'), ('origin_ui_session_id', 'security_identity'), ('parent_session_id', 'security_identity'), ('state', 'control_recovery'), ('dispatched_at', 'control_recovery'), ('completed_at', 'control_recovery'), ('updated_at', 'control_recovery'), ('event_json', 'mixed_requires_projection'), ('result_json', 'mixed_requires_projection'), ('delivery_state', 'control_recovery'), ('delivery_attempts', 'control_recovery'), ('delivered_at', 'control_recovery'), ('owner_pid', 'control_recovery'), ('owner_started_at', 'control_recovery'), ('task_json', 'mixed_requires_projection'), ('owner_execution_id', 'control_recovery'), ('delivery_claim', 'control_recovery'), ('delivery_claimed_at', 'control_recovery'), ('origin_session_id', 'security_identity')]}, 'compression_locks': {'optional': False, 'table_disposition': 'control_recovery', 'writer_fence_status': 'CORE_DB_TRANSACTION_ONLY; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('session_id', 'control_recovery'), ('holder', 'control_recovery'), ('acquired_at', 'control_recovery'), ('expires_at', 'control_recovery')]}, 'conversation_generations': {'optional': False, 'table_disposition': 'control_recovery', 'writer_fence_status': 'CORE_DB_TRANSACTION_ONLY; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('source', 'control_recovery'), ('session_key', 'control_recovery'), ('generation', 'control_recovery')]}, 'gateway_heartbeats': {'optional': False, 'table_disposition': 'control_recovery', 'writer_fence_status': 'GATEWAY_WRITERS_UNQUIESCED', 'columns': [('backend_id', 'security_identity'), ('pid', 'control_recovery'), ('started_at', 'control_recovery'), ('last_heartbeat', 'control_recovery'), ('profile', 'security_identity'), ('host', 'security_identity')]}, 'gateway_hygiene_state': {'optional': False, 'table_disposition': 'control_recovery', 'writer_fence_status': 'GATEWAY_WRITERS_UNQUIESCED', 'columns': [('session_key', 'control_recovery'), ('failure_streak', 'control_recovery')]}, 'gateway_routing': {'optional': False, 'table_disposition': 'business_and_control', 'writer_fence_status': 'GATEWAY_AND_LOCAL_WRITERS_UNQUIESCED', 'columns': [('scope', 'security_identity'), ('session_key', 'security_identity'), ('entry_json', 'mixed_requires_projection'), ('updated_at', 'control_recovery')]}, 'messages': {'optional': False, 'table_disposition': 'business_and_control', 'writer_fence_status': 'CORE_DB_TRANSACTION_ONLY; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('id', 'control_recovery'), ('session_id', 'security_identity'), ('role', 'business_payload'), ('content', 'business_payload'), ('tool_call_id', 'business_payload'), ('tool_calls', 'business_payload'), ('tool_name', 'business_payload'), ('effect_disposition', 'control_recovery'), ('timestamp', 'delivery_metadata'), ('token_count', 'business_payload'), ('finish_reason', 'business_payload'), ('reasoning', 'business_payload'), ('reasoning_content', 'business_payload'), ('reasoning_details', 'business_payload'), ('codex_reasoning_items', 'business_payload'), ('codex_message_items', 'business_payload'), ('platform_message_id', 'delivery_metadata'), ('observed', 'control_recovery'), ('_compressed_summary', 'control_recovery'), ('active', 'control_recovery'), ('compacted', 'control_recovery'), ('api_content', 'delivery_metadata'), ('display_kind', 'business_payload'), ('display_metadata', 'mixed_requires_projection'), ('display_identity', 'derived_internal'), ('display_order', 'derived_internal'), ('message_uid', 'derived_internal'), ('absorbed_message_uids', 'control_recovery'), ('tool_call_uids', 'control_recovery'), ('tool_call_uid', 'control_recovery')]}, 'runtime_epoch': {'optional': False, 'table_disposition': 'control_recovery', 'writer_fence_status': 'RUNTIME_EPOCH_PARTIAL; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('singleton', 'control_recovery'), ('epoch', 'control_recovery'), ('instance_id', 'control_recovery')]}, 'schema_version': {'optional': False, 'table_disposition': 'control_recovery', 'writer_fence_status': 'SCHEMA_MARKER_ONLY', 'columns': [('version', 'control_recovery')]}, 'session_admissions': {'optional': False, 'table_disposition': 'business_and_control', 'writer_fence_status': 'RUNTIME_ADMISSION_LEDGER_PARTIAL; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('seq', 'control_recovery'), ('admission_id', 'security_identity'), ('request_id', 'security_identity'), ('principal_id', 'security_identity'), ('target_session_id', 'security_identity'), ('lineage_json', 'mixed_requires_projection'), ('payload_json', 'mixed_requires_projection'), ('payload_digest', 'security_identity'), ('intent', 'control_recovery'), ('status', 'control_recovery'), ('outcome', 'control_recovery'), ('owner_epoch', 'control_recovery'), ('generation', 'control_recovery')]}, 'session_model_usage': {'optional': False, 'table_disposition': 'business_and_control', 'writer_fence_status': 'CORE_DB_TRANSACTION_ONLY; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('session_id', 'security_identity'), ('model', 'security_identity'), ('billing_provider', 'security_identity'), ('billing_base_url', 'security_identity'), ('billing_mode', 'security_identity'), ('task', 'business_metadata'), ('api_call_count', 'business_metadata'), ('input_tokens', 'business_metadata'), ('output_tokens', 'business_metadata'), ('cache_read_tokens', 'business_metadata'), ('cache_write_tokens', 'business_metadata'), ('reasoning_tokens', 'business_metadata'), ('estimated_cost_usd', 'business_metadata'), ('actual_cost_usd', 'business_metadata'), ('cost_status', 'business_metadata'), ('cost_source', 'business_metadata'), ('first_seen', 'business_metadata'), ('last_seen', 'business_metadata')]}, 'session_turn_leases': {'optional': False, 'table_disposition': 'control_recovery', 'writer_fence_status': 'CORE_DB_TRANSACTION_ONLY; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('conversation_id', 'security_identity'), ('holder', 'security_identity'), ('acquired_at', 'control_recovery'), ('expires_at', 'control_recovery')]}, 'sessions': {'optional': False, 'table_disposition': 'business_and_control', 'writer_fence_status': 'CORE_DB_TRANSACTION_ONLY; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('id', 'security_identity'), ('runtime_revision', 'control_recovery'), ('runtime_generation', 'control_recovery'), ('source', 'security_identity'), ('created_source', 'security_identity'), ('user_id', 'security_identity'), ('session_key', 'security_identity'), ('chat_id', 'security_identity'), ('chat_type', 'security_identity'), ('thread_id', 'security_identity'), ('display_name', 'business_metadata'), ('origin_json', 'mixed_requires_projection'), ('expiry_finalized', 'control_recovery'), ('model', 'business_metadata'), ('model_config', 'mixed_requires_projection'), ('system_prompt', 'business_metadata'), ('system_prompt_hash', 'derived_internal'), ('parent_session_id', 'security_identity'), ('started_at', 'control_recovery'), ('ended_at', 'control_recovery'), ('end_reason', 'control_recovery'), ('message_count', 'control_recovery'), ('tool_call_count', 'control_recovery'), ('input_tokens', 'business_metadata'), ('output_tokens', 'business_metadata'), ('cache_read_tokens', 'business_metadata'), ('cache_write_tokens', 'business_metadata'), ('reasoning_tokens', 'business_metadata'), ('cwd', 'business_metadata'), ('git_branch', 'business_metadata'), ('git_repo_root', 'business_metadata'), ('git_metadata_generation', 'control_recovery'), ('billing_provider', 'security_identity'), ('billing_base_url', 'security_identity'), ('billing_mode', 'security_identity'), ('estimated_cost_usd', 'business_metadata'), ('actual_cost_usd', 'business_metadata'), ('cost_status', 'business_metadata'), ('cost_source', 'business_metadata'), ('pricing_version', 'business_metadata'), ('title', 'business_metadata'), ('title_source', 'business_metadata'), ('last_activity_at', 'business_metadata'), ('last_activity_description', 'business_metadata'), ('last_activity_provenance', 'business_metadata'), ('api_call_count', 'business_metadata'), ('handoff_state', 'control_recovery'), ('handoff_platform', 'security_identity'), ('handoff_error', 'control_recovery'), ('compression_failure_cooldown_until', 'control_recovery'), ('compression_failure_error', 'control_recovery'), ('compression_fallback_streak', 'control_recovery'), ('compression_ineffective_count', 'control_recovery'), ('compression_recovery_deadline', 'control_recovery'), ('compression_overload_streak', 'control_recovery'), ('profile_name', 'security_identity'), ('transport_profile', 'security_identity'), ('rewind_count', 'control_recovery'), ('archived', 'control_recovery'), ('auto_archived', 'control_recovery'), ('pinned', 'control_recovery'), ('hidden', 'control_recovery'), ('last_read_at', 'control_recovery'), ('tool_names', 'business_metadata')]}, 'sqlite_sequence': {'optional': False, 'table_disposition': 'derived_internal', 'writer_fence_status': 'SQLITE_INTERNAL_DERIVED', 'columns': [('name', 'derived_internal'), ('seq', 'derived_internal')]}, 'state_meta': {'optional': False, 'table_disposition': 'business_and_control', 'writer_fence_status': 'KEY_NAMESPACE_WRITERS_UNQUIESCED', 'columns': [('key', 'security_identity'), ('value', 'mixed_requires_projection')]}, 'system_prompts': {'optional': False, 'table_disposition': 'business_and_control', 'writer_fence_status': 'CORE_DB_TRANSACTION_ONLY; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('hash', 'derived_internal'), ('prompt', 'business_payload')]}, 'worker_executions': {'optional': False, 'table_disposition': 'control_recovery', 'writer_fence_status': 'RUNTIME_LEDGER_PARTIAL; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('execution_id', 'security_identity'), ('session_id', 'security_identity'), ('kind', 'control_recovery'), ('owner_epoch', 'security_identity'), ('generation', 'control_recovery'), ('status', 'control_recovery'), ('adoption_digest', 'security_identity'), ('last_sequence', 'control_recovery')]}, 'worker_receipts': {'optional': False, 'table_disposition': 'business_and_control', 'writer_fence_status': 'RUNTIME_LEDGER_PARTIAL; COMPLETE_EXPORT_FENCE_MISSING', 'columns': [('execution_id', 'security_identity'), ('sequence', 'control_recovery'), ('payload_digest', 'security_identity'), ('result_json', 'mixed_requires_projection')]}, 'telegram_dm_topic_bindings': {'optional': True, 'table_disposition': 'control_recovery', 'writer_fence_status': 'TELEGRAM_WRITERS_UNQUIESCED', 'columns': [('profile_name', 'security_identity'), ('chat_id', 'security_identity'), ('thread_id', 'security_identity'), ('user_id', 'security_identity'), ('session_key', 'security_identity'), ('session_id', 'security_identity'), ('managed_mode', 'control_recovery'), ('linked_at', 'control_recovery'), ('updated_at', 'control_recovery')]}, 'telegram_dm_topic_mode': {'optional': True, 'table_disposition': 'control_recovery', 'writer_fence_status': 'TELEGRAM_WRITERS_UNQUIESCED', 'columns': [('profile_name', 'security_identity'), ('chat_id', 'security_identity'), ('user_id', 'security_identity'), ('enabled', 'control_recovery'), ('activated_at', 'control_recovery'), ('updated_at', 'control_recovery'), ('has_topics_enabled', 'control_recovery'), ('allows_users_to_create_topics', 'control_recovery'), ('capability_checked_at', 'control_recovery'), ('intro_message_id', 'control_recovery'), ('pinned_message_id', 'control_recovery')]}}


def _make_table_policy(name: str, spec: Mapping[str, Any]) -> TablePolicy:
    columns = tuple(ColumnPolicy(column_name, disposition) for column_name, disposition in spec["columns"])
    if len({column.name for column in columns}) != len(columns):
        raise RuntimeError(f"duplicate field policy column in {name}")
    return TablePolicy(
        name=name,
        columns=columns,
        table_disposition=str(spec["table_disposition"]),
        writer_fence_status=str(spec["writer_fence_status"]),
        optional=bool(spec["optional"]),
    )


TABLE_POLICIES = {name: _make_table_policy(name, spec) for name, spec in _TABLE_SPECS.items()}
CORE_TABLE_POLICIES = {name: policy for name, policy in TABLE_POLICIES.items() if not policy.optional}
OPTIONAL_TELEGRAM_TABLE_POLICIES = {
    name: policy for name, policy in TABLE_POLICIES.items() if policy.optional
}

SCHEMA30_SOURCE_ID = "desktop-d337-schema30"
SCHEMA30_WRITER_SOURCE_ID = "d337-pinned-source-complete"

# Pinned d337 schema-30 columns, in the independent DDL order.  The disposition
# groups are copied from the reviewed version-scoped field policy; this mapping is
# immutable and is never merged into the schema-31 global policy.
_SCHEMA30_DISPOSITIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "async_delegations": (
        ("delegation_id", SECURITY_IDENTITY), ("origin_session", SECURITY_IDENTITY),
        ("origin_ui_session_id", SECURITY_IDENTITY), ("parent_session_id", SECURITY_IDENTITY),
        ("state", CONTROL_RECOVERY), ("dispatched_at", CONTROL_RECOVERY),
        ("completed_at", CONTROL_RECOVERY), ("updated_at", CONTROL_RECOVERY),
        ("event_json", MIXED_REQUIRES_PROJECTION), ("result_json", MIXED_REQUIRES_PROJECTION),
        ("delivery_state", CONTROL_RECOVERY), ("delivery_attempts", CONTROL_RECOVERY),
        ("delivered_at", CONTROL_RECOVERY), ("owner_pid", CONTROL_RECOVERY),
        ("owner_started_at", CONTROL_RECOVERY), ("task_json", MIXED_REQUIRES_PROJECTION),
        ("delivery_claim", CONTROL_RECOVERY), ("delivery_claimed_at", CONTROL_RECOVERY),
        ("origin_session_id", SECURITY_IDENTITY),
    ),
    "compression_locks": (
        ("session_id", CONTROL_RECOVERY), ("holder", CONTROL_RECOVERY),
        ("acquired_at", CONTROL_RECOVERY), ("expires_at", CONTROL_RECOVERY),
    ),
    "conversation_generations": (
        ("source", CONTROL_RECOVERY), ("session_key", CONTROL_RECOVERY),
        ("generation", CONTROL_RECOVERY),
    ),
    "gateway_heartbeats": (
        ("backend_id", SECURITY_IDENTITY), ("pid", CONTROL_RECOVERY),
        ("started_at", CONTROL_RECOVERY), ("last_heartbeat", CONTROL_RECOVERY),
        ("profile", SECURITY_IDENTITY), ("host", SECURITY_IDENTITY),
    ),
    "gateway_hygiene_state": (
        ("session_key", CONTROL_RECOVERY), ("failure_streak", CONTROL_RECOVERY),
    ),
    "gateway_routing": (
        ("scope", SECURITY_IDENTITY), ("session_key", SECURITY_IDENTITY),
        ("entry_json", MIXED_REQUIRES_PROJECTION), ("updated_at", CONTROL_RECOVERY),
    ),
    "messages": (
        ("id", CONTROL_RECOVERY), ("session_id", SECURITY_IDENTITY),
        ("role", BUSINESS_PAYLOAD), ("content", BUSINESS_PAYLOAD),
        ("tool_call_id", BUSINESS_PAYLOAD), ("tool_calls", BUSINESS_PAYLOAD),
        ("tool_name", BUSINESS_PAYLOAD), ("effect_disposition", CONTROL_RECOVERY),
        ("timestamp", DELIVERY_METADATA), ("token_count", BUSINESS_PAYLOAD),
        ("finish_reason", BUSINESS_PAYLOAD), ("reasoning", BUSINESS_PAYLOAD),
        ("reasoning_content", BUSINESS_PAYLOAD), ("reasoning_details", BUSINESS_PAYLOAD),
        ("codex_reasoning_items", BUSINESS_PAYLOAD), ("codex_message_items", BUSINESS_PAYLOAD),
        ("platform_message_id", DELIVERY_METADATA), ("observed", CONTROL_RECOVERY),
        ("_compressed_summary", CONTROL_RECOVERY), ("active", CONTROL_RECOVERY),
        ("compacted", CONTROL_RECOVERY), ("api_content", DELIVERY_METADATA),
        ("display_kind", BUSINESS_PAYLOAD), ("display_metadata", MIXED_REQUIRES_PROJECTION),
        ("display_identity", DERIVED_INTERNAL), ("display_order", DERIVED_INTERNAL),
    ),
    "schema_version": (("version", CONTROL_RECOVERY),),
    "session_model_usage": (
        ("session_id", SECURITY_IDENTITY), ("model", SECURITY_IDENTITY),
        ("billing_provider", SECURITY_IDENTITY), ("billing_base_url", SECURITY_IDENTITY),
        ("billing_mode", SECURITY_IDENTITY), ("task", BUSINESS_METADATA),
        ("api_call_count", BUSINESS_METADATA), ("input_tokens", BUSINESS_METADATA),
        ("output_tokens", BUSINESS_METADATA), ("cache_read_tokens", BUSINESS_METADATA),
        ("cache_write_tokens", BUSINESS_METADATA), ("reasoning_tokens", BUSINESS_METADATA),
        ("estimated_cost_usd", BUSINESS_METADATA), ("actual_cost_usd", BUSINESS_METADATA),
        ("cost_status", BUSINESS_METADATA), ("cost_source", BUSINESS_METADATA),
        ("first_seen", BUSINESS_METADATA), ("last_seen", BUSINESS_METADATA),
    ),
    "session_turn_leases": (
        ("conversation_id", SECURITY_IDENTITY), ("holder", SECURITY_IDENTITY),
        ("acquired_at", CONTROL_RECOVERY), ("expires_at", CONTROL_RECOVERY),
    ),
    "sessions": (
        ("id", SECURITY_IDENTITY), ("source", SECURITY_IDENTITY),
        ("user_id", SECURITY_IDENTITY), ("session_key", SECURITY_IDENTITY),
        ("chat_id", SECURITY_IDENTITY), ("chat_type", SECURITY_IDENTITY),
        ("thread_id", SECURITY_IDENTITY), ("display_name", BUSINESS_METADATA),
        ("origin_json", MIXED_REQUIRES_PROJECTION), ("expiry_finalized", CONTROL_RECOVERY),
        ("model", BUSINESS_METADATA), ("model_config", MIXED_REQUIRES_PROJECTION),
        ("system_prompt", BUSINESS_METADATA), ("system_prompt_hash", DERIVED_INTERNAL),
        ("parent_session_id", SECURITY_IDENTITY), ("started_at", CONTROL_RECOVERY),
        ("ended_at", CONTROL_RECOVERY), ("end_reason", CONTROL_RECOVERY),
        ("message_count", CONTROL_RECOVERY), ("tool_call_count", CONTROL_RECOVERY),
        ("input_tokens", BUSINESS_METADATA), ("output_tokens", BUSINESS_METADATA),
        ("cache_read_tokens", BUSINESS_METADATA), ("cache_write_tokens", BUSINESS_METADATA),
        ("reasoning_tokens", BUSINESS_METADATA), ("cwd", BUSINESS_METADATA),
        ("git_branch", BUSINESS_METADATA), ("git_repo_root", BUSINESS_METADATA),
        ("git_metadata_generation", CONTROL_RECOVERY), ("billing_provider", SECURITY_IDENTITY),
        ("billing_base_url", SECURITY_IDENTITY), ("billing_mode", SECURITY_IDENTITY),
        ("estimated_cost_usd", BUSINESS_METADATA), ("actual_cost_usd", BUSINESS_METADATA),
        ("cost_status", BUSINESS_METADATA), ("cost_source", BUSINESS_METADATA),
        ("pricing_version", BUSINESS_METADATA), ("title", BUSINESS_METADATA),
        ("title_source", BUSINESS_METADATA), ("last_activity_at", BUSINESS_METADATA),
        ("last_activity_description", BUSINESS_METADATA), ("last_activity_provenance", BUSINESS_METADATA),
        ("api_call_count", BUSINESS_METADATA), ("handoff_state", CONTROL_RECOVERY),
        ("handoff_platform", SECURITY_IDENTITY), ("handoff_error", CONTROL_RECOVERY),
        ("compression_failure_cooldown_until", CONTROL_RECOVERY),
        ("compression_failure_error", CONTROL_RECOVERY),
        ("compression_fallback_streak", CONTROL_RECOVERY),
        ("compression_ineffective_count", CONTROL_RECOVERY),
        ("compression_recovery_deadline", CONTROL_RECOVERY), ("profile_name", SECURITY_IDENTITY),
        ("transport_profile", SECURITY_IDENTITY), ("rewind_count", CONTROL_RECOVERY),
        ("archived", CONTROL_RECOVERY), ("pinned", CONTROL_RECOVERY),
        ("hidden", CONTROL_RECOVERY), ("last_read_at", CONTROL_RECOVERY),
        ("tool_names", BUSINESS_METADATA),
    ),
    "sqlite_sequence": (("name", DERIVED_INTERNAL), ("seq", DERIVED_INTERNAL)),
    "state_meta": (("key", SECURITY_IDENTITY), ("value", MIXED_REQUIRES_PROJECTION)),
    "system_prompts": (("hash", DERIVED_INTERNAL), ("prompt", BUSINESS_PAYLOAD)),
}


def _make_schema30_table_policies() -> dict[str, TablePolicy]:
    policies: dict[str, TablePolicy] = {}
    for table, columns in _SCHEMA30_DISPOSITIONS.items():
        current = TABLE_POLICIES[table]
        policies[table] = TablePolicy(
            name=table,
            columns=tuple(ColumnPolicy(name, disposition) for name, disposition in columns),
            table_disposition=current.table_disposition,
            writer_fence_status=current.writer_fence_status,
            optional=False,
        )
    # d337 creates these reviewed Telegram tables only after opt-in; absent tables
    # stay out of the schema-30 manifest and therefore preserve the legacy wire.
    policies.update(OPTIONAL_TELEGRAM_TABLE_POLICIES)
    return policies


SCHEMA30_TABLE_POLICIES = _make_schema30_table_policies()
# Hermes v32 explicitly has no data migration: its source keeps the v31 table
# layout and only advances the durable version after the v31 ledger columns exist.
# Keep that structural relationship explicit while giving v32 its own provenance slot.
SCHEMA32_TABLE_POLICIES = dict(TABLE_POLICIES)
SCHEMA_POLICIES: dict[int, VersionedSchemaPolicy] = {
    30: VersionedSchemaPolicy(30, SCHEMA30_SOURCE_ID, SCHEMA30_WRITER_SOURCE_ID, SCHEMA30_TABLE_POLICIES),
    31: VersionedSchemaPolicy(
        schema_version=31,
        official_schema_source=OFFICIAL_SOURCE_ID,
        writer_source=CURRENT_WRITER_SOURCE_ID,
        table_policies=TABLE_POLICIES,
        schema_census_sha256=SCHEMA_CENSUS_SHA256,
        field_policy_review_sha256=FIELD_POLICY_REVIEW_SHA256,
    ),
    32: VersionedSchemaPolicy(
        schema_version=SCHEMA32_SCHEMA_VERSION,
        official_schema_source=SCHEMA32_SOURCE_ID,
        writer_source=SCHEMA32_WRITER_SOURCE_ID,
        table_policies=SCHEMA32_TABLE_POLICIES,
        schema_census_sha256=SCHEMA32_SCHEMA_CENSUS_SHA256,
        field_policy_review_sha256=SCHEMA32_FIELD_POLICY_REVIEW_SHA256,
    ),
}

# SQLite's FTS virtual table and shadow tables are implementation state.  They are
# intentionally enumerated rather than accepted by a prefix so a new derived table
# cannot disappear silently from an export.
DERIVED_SQLITE_TABLES = frozenset({
    "messages_fts_src", "messages_fts_trigram_src", "messages_fts_cjk_src",
    "messages_fts", "messages_fts_data", "messages_fts_idx", "messages_fts_content",
    "messages_fts_docsize", "messages_fts_config",
    "messages_fts_trigram", "messages_fts_trigram_data", "messages_fts_trigram_idx",
    "messages_fts_trigram_content", "messages_fts_trigram_docsize", "messages_fts_trigram_config",
    "messages_fts_cjk", "messages_fts_cjk_data", "messages_fts_cjk_idx",
    "messages_fts_cjk_content", "messages_fts_cjk_docsize", "messages_fts_cjk_config",
})
DERIVED_SQLITE_TABLE_POLICIES = frozenset({"sqlite_sequence"}) | DERIVED_SQLITE_TABLES

# Derived relations are closed as well.  Their values are emitted as separate
# evidence, so a relation with a reviewed name but the wrong SQLite object or
# column shape must not be accepted as an equivalent implementation relation.
# FTS5 shadow layouts are stable for the external-content tables used by Hermes.
_DERIVED_RELATION_SPECS: dict[str, tuple[str, tuple[str, ...], bool]] = {
    "sqlite_sequence": ("table", ("name", "seq"), False),
    "messages_fts_src": ("view", ("id", "content", "tool_name", "tool_calls"), True),
    "messages_fts_trigram_src": ("view", ("id", "role", "content", "tool_name"), True),
    "messages_fts_cjk_src": (
        "view",
        ("id", "role", "content", "tool_name", "tool_calls"),
        True,
    ),
    "messages_fts": ("table", ("content", "tool_name", "tool_calls"), True),
    "messages_fts_trigram": ("table", ("content", "tool_name"), True),
    "messages_fts_cjk": ("table", ("content", "tool_name", "tool_calls"), True),
}
for _fts_name in ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"):
    _DERIVED_RELATION_SPECS.update(
        {
            f"{_fts_name}_data": ("table", ("id", "block"), True),
            f"{_fts_name}_idx": ("table", ("segid", "term", "pgno"), True),
            f"{_fts_name}_content": (
                "table",
                ("id",) + tuple(
                    f"c{index}"
                    for index in range(
                        3 if _fts_name in {"messages_fts", "messages_fts_cjk"} else 2
                    )
                ),
                True,
            ),
            f"{_fts_name}_docsize": ("table", ("id", "sz"), True),
            f"{_fts_name}_config": ("table", ("k", "v"), True),
        }
    )

DERIVED_REQUIRED_RELATIONS = frozenset(
    relation for relation, (_, _, optional) in _DERIVED_RELATION_SPECS.items() if not optional
)

def policy_counts() -> dict[str, int]:
    """Return observed counts from this policy mapping for manifests/diagnostics.

    Tests compare the mapping to the live schema field-by-field.  There is no
    hard-coded count assertion here: a schema change must be reviewed against the
    actual policy and writer census rather than hidden behind a change detector.
    """
    core_policies = tuple(CORE_TABLE_POLICIES.values())
    core_columns = tuple(column for table in core_policies for column in table.columns)
    optional_policies = tuple(OPTIONAL_TELEGRAM_TABLE_POLICIES.values())
    optional_columns = tuple(column for table in optional_policies for column in table.columns)
    columns = core_columns
    return {
        "tables": len(core_policies),
        "columns": len(columns),
        "optional_tables": len(optional_policies),
        "optional_columns": len(optional_columns),
        "business_payload_columns": sum(column.disposition == BUSINESS_PAYLOAD for column in columns),
        "business_metadata_columns": sum(column.disposition == BUSINESS_METADATA for column in columns),
        "control_recovery_columns": sum(column.disposition == CONTROL_RECOVERY for column in columns),
        "delivery_metadata_columns": sum(column.disposition == DELIVERY_METADATA for column in columns),
        "security_identity_columns": sum(column.disposition == SECURITY_IDENTITY for column in columns),
        "mixed_projection_columns": sum(column.disposition == MIXED_REQUIRES_PROJECTION for column in columns),
        "derived_internal_columns": sum(column.disposition == DERIVED_INTERNAL for column in columns),
    }


def table_policy(table: str) -> TablePolicy:
    return table_policy_for_mapping(table, TABLE_POLICIES)


def table_policy_for_mapping(
    table: str,
    policies: Mapping[str, TablePolicy],
) -> TablePolicy:
    try:
        return policies[table]
    except KeyError as exc:
        raise KeyError(f"table is outside the reviewed export policy: {table}") from exc


def column_policy(
    table: str,
    column: str,
    *,
    table_policies: Mapping[str, TablePolicy] | None = None,
) -> ColumnPolicy:
    policy = table_policy_for_mapping(table, table_policies or TABLE_POLICIES)
    for item in policy.columns:
        if item.name == column:
            return item
    raise KeyError(f"column is outside the reviewed export policy: {table}.{column}")


def _table_columns(conn: Any, table: str) -> tuple[str, ...]:
    # Names come exclusively from the closed policy; no caller-supplied identifier is
    # interpolated into this query.
    quoted = table.replace('"', '""')
    return tuple(str(row[1]) for row in conn.execute(f'PRAGMA table_info("{quoted}")'))


def _schema_objects(conn: Any) -> tuple[dict[str, str], set[str]]:
    rows = conn.execute(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
    ).fetchall()
    object_types = {str(row[0]): str(row[1]) for row in rows}
    return object_types, set(object_types)


def _column_shape_details(expected: Sequence[str], actual: tuple[str, ...]) -> tuple[str, ...]:
    if actual == tuple(expected):
        return ()
    missing_columns = sorted(set(expected) - set(actual))
    extra_columns = sorted(set(actual) - set(expected))
    order_drift = actual != tuple(expected) and not missing_columns and not extra_columns
    details = [f"missing={','.join(missing_columns)}"] if missing_columns else []
    if extra_columns:
        details.append(f"extra={','.join(extra_columns)}")
    if order_drift:
        details.append(f"order={actual!r}")
    return tuple(details)


def _validate_table_shapes(
    conn: Any,
    object_types: Mapping[str, str],
    present: set[str],
    policies: Mapping[str, TablePolicy],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    mismatches: dict[str, tuple[str, ...]] = {}
    relation_types: dict[str, tuple[str, ...]] = {}
    for table, policy in policies.items():
        if table not in present:
            continue
        if object_types[table] != "table":
            relation_types[table] = (f"expected=table,actual={object_types[table]}",)
        details = _column_shape_details(policy.column_names, _table_columns(conn, table))
        if details:
            mismatches[table] = details
    return mismatches, relation_types


def _validate_extension_shapes(
    conn: Any,
    object_types: Mapping[str, str],
    present: set[str],
    extensions: Mapping[str, Sequence[str]],
    relation_types: dict[str, tuple[str, ...]],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    mismatches: dict[str, tuple[str, ...]] = {}
    for table, expected_columns in sorted(extensions.items()):
        if table not in present:
            continue
        if object_types[table] != "table":
            relation_types[table] = (f"expected=table,actual={object_types[table]}",)
        details = _column_shape_details(expected_columns, _table_columns(conn, table))
        if details:
            mismatches[table] = details
    missing = tuple(sorted(set(extensions) - present))
    return mismatches, missing


def _validate_derived_shapes(
    conn: Any,
    object_types: Mapping[str, str],
    present: set[str],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...], tuple[str, ...]]:
    mismatches: dict[str, tuple[str, ...]] = {}
    absent_optional: list[str] = []
    missing_required: list[str] = []
    for relation in sorted(DERIVED_SQLITE_TABLE_POLICIES):
        if relation not in present:
            (missing_required if relation in DERIVED_REQUIRED_RELATIONS else absent_optional).append(relation)
            continue
        spec = _DERIVED_RELATION_SPECS.get(relation)
        if spec is None:
            mismatches[relation] = ("no reviewed relation shape",)
            continue
        expected_type, expected_columns, _optional = spec
        details: list[str] = []
        actual_type = object_types[relation]
        if actual_type != expected_type:
            details.append(f"expected_type={expected_type},actual_type={actual_type}")
        actual_columns = _table_columns(conn, relation)
        if actual_columns != expected_columns:
            details.append(f"expected_columns={expected_columns!r},actual_columns={actual_columns!r}")
        if details:
            mismatches[relation] = tuple(details)
    return mismatches, tuple(absent_optional), tuple(missing_required)


def _read_schema_version(conn: Any) -> int | None:
    try:
        rows = conn.execute("SELECT version FROM schema_version ORDER BY rowid").fetchall()
    except sqlite3.Error:
        return None
    if len(rows) != 1:
        return None
    value = rows[0][0]
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _schema_presence(
    policies: Mapping[str, TablePolicy],
    present: set[str],
    extension_tables: Mapping[str, Sequence[str]],
    extension_relations: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    policy_names = set(policies)
    required = {name for name, policy in policies.items() if not policy.optional}
    missing = tuple(sorted(required - present))
    unknown = tuple(sorted(
        present
        - policy_names
        - DERIVED_SQLITE_TABLE_POLICIES
        - set(extension_tables)
        - extension_relations
    ))
    optional = tuple(sorted(
        table for table, policy in policies.items() if policy.optional and table in present
    ))
    return missing, unknown, optional


def _schema_shape_validation(
    conn: Any,
    object_types: Mapping[str, str],
    present: set[str],
    policies: Mapping[str, TablePolicy],
    extension_tables: Mapping[str, Sequence[str]],
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    mismatches, relation_types = _validate_table_shapes(conn, object_types, present, policies)
    extension_mismatches, missing_extension_tables = _validate_extension_shapes(
        conn, object_types, present, extension_tables, relation_types
    )
    mismatches.update(extension_mismatches)
    derived_mismatches, absent_optional, missing_required = _validate_derived_shapes(
        conn, object_types, present
    )
    return (
        mismatches,
        relation_types,
        missing_extension_tables,
        derived_mismatches,
        absent_optional,
        missing_required,
    )


def validate_schema(
    conn: Any,
    *,
    schema_policy: VersionedSchemaPolicy | None = None,
    registered_extension_tables: Mapping[str, Sequence[str]] | None = None,
    registered_extension_relations: Iterable[str] = (),
) -> SchemaValidation:
    """Validate a live SQLite schema before any row or external source is opened."""
    schema_policy = schema_policy or SCHEMA_POLICIES[OFFICIAL_SCHEMA_VERSION]
    policies = schema_policy.table_policies
    registered_extension_tables = dict(registered_extension_tables or {})
    registered_relations = set(registered_extension_relations)
    object_types, present = _schema_objects(conn)
    missing, unknown, optional = _schema_presence(
        policies, present, registered_extension_tables, registered_relations
    )
    (
        mismatches,
        relation_types,
        missing_extension_tables,
        derived_mismatches,
        absent_optional,
        missing_required,
    ) = _schema_shape_validation(
        conn, object_types, present, policies, registered_extension_tables
    )
    missing_extension_relations = tuple(sorted(registered_relations - present))
    version = _read_schema_version(conn) if "schema_version" in present else None
    return SchemaValidation(
        schema_version=version,
        present_tables=tuple(sorted(present)),
        optional_tables=optional,
        unknown_tables=unknown,
        missing_tables=missing,
        column_mismatches=mismatches,
        relation_type_mismatches=relation_types,
        missing_extension_tables=missing_extension_tables,
        missing_extension_relations=missing_extension_relations,
        derived_relation_mismatches=derived_mismatches,
        absent_optional_derived_relations=absent_optional,
        missing_required_derived_relations=missing_required,
        expected_schema_version=schema_policy.schema_version,
    )


def _skip_json_ws(text: str, position: int) -> int:
    while position < len(text) and text[position] in " \t\r\n":
        position += 1
    return position


def _raw_json_value(text: str, decoder: json.JSONDecoder, position: int) -> tuple[Any, int]:
    try:
        return decoder.raw_decode(text, position)
    except (ValueError, RecursionError) as exc:
        raise ModelConfigProjectionRefused("model_config contains ambiguous JSON") from exc


def _parse_model_config_object(
    text: str,
    decoder: json.JSONDecoder,
) -> tuple[Any, int, tuple[tuple[str, int, int], ...]]:
    start = _skip_json_ws(text, 0)
    value, end = _raw_json_value(text, decoder, start)
    if _skip_json_ws(text, end) != len(text):
        raise ModelConfigProjectionRefused("model_config contains trailing data")
    if not isinstance(value, dict) or text[start:start + 1] != "{":
        return value, start, ()
    position = start + 1
    members: list[tuple[str, int, int]] = []
    while True:
        member_start = _skip_json_ws(text, position)
        if member_start >= len(text):
            raise ModelConfigProjectionRefused("model_config object is incomplete")
        if text[member_start] == "}":
            object_end = member_start + 1
            if _skip_json_ws(text, object_end) != len(text):
                raise ModelConfigProjectionRefused("model_config contains trailing data")
            return value, start, tuple(members)
        key, key_end = _raw_json_value(text, decoder, member_start)
        if not isinstance(key, str):
            raise ModelConfigProjectionRefused("model_config object key is not text")
        colon = _skip_json_ws(text, key_end)
        if colon >= len(text) or text[colon] != ":":
            raise ModelConfigProjectionRefused("model_config object member is malformed")
        value_start = _skip_json_ws(text, colon + 1)
        _, value_end = _raw_json_value(text, decoder, value_start)
        members.append((key, member_start, value_end))
        position = _skip_json_ws(text, value_end)
        if position >= len(text):
            raise ModelConfigProjectionRefused("model_config object is incomplete")
        if text[position] == ",":
            position += 1
            continue
        if text[position] != "}":
            raise ModelConfigProjectionRefused("model_config object member separator is malformed")
        object_end = position + 1
        if _skip_json_ws(text, object_end) != len(text):
            raise ModelConfigProjectionRefused("model_config contains trailing data")
        return value, start, tuple(members)


def _remove_model_config_credential(
    raw_bytes: bytes,
    text: str,
    members: tuple[tuple[str, int, int], ...],
    source_was_bytes: bool,
) -> ModelConfigProjection:
    credential_members = [member for member in members if member[0] == "api_key"]
    if not credential_members:
        return ModelConfigProjection(raw_bytes, True, (), source_was_bytes)
    if len(credential_members) != 1:
        raise ModelConfigProjectionRefused("model_config has an ambiguous duplicate $.api_key path")

    _, credential_start, credential_end = credential_members[0]
    previous = [member for member in members if member[2] < credential_start]
    following = [member for member in members if member[1] > credential_end]
    if following:
        comma = _skip_json_ws(text, credential_end)
        if comma >= len(text) or text[comma] != ",":
            raise ModelConfigProjectionRefused("model_config credential member separator is malformed")
        deletion = (credential_start, comma + 1)
    elif previous:
        comma = _skip_json_ws(text, previous[-1][2])
        if comma >= len(text) or text[comma] != ",":
            raise ModelConfigProjectionRefused("model_config credential member separator is malformed")
        deletion = (comma, credential_end)
    else:
        deletion = (credential_start, credential_end)
    projected_bytes = (text[:deletion[0]] + text[deletion[1]:]).encode("utf-8")
    return ModelConfigProjection(projected_bytes, False, ("$.api_key",), source_was_bytes)


def project_model_config(raw: str | bytes | None) -> ModelConfigProjection | None:
    """Apply the reviewed ``sessions.model_config`` credential boundary."""
    if raw is None:
        return None
    source_was_bytes = isinstance(raw, bytes)
    raw_bytes = bytes(raw) if source_was_bytes else str(raw).encode("utf-8")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelConfigProjectionRefused("model_config is not valid UTF-8 JSON") from exc
    decoder = json.JSONDecoder(
        parse_int=lambda value: value,
        parse_float=lambda value: value,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )
    value, _start, members = _parse_model_config_object(text, decoder)
    if not isinstance(value, dict):
        return ModelConfigProjection(raw_bytes, True, (), source_was_bytes)
    return _remove_model_config_credential(raw_bytes, text, members, source_was_bytes)


def policy_table_names(
    *,
    include_optional: bool = True,
    table_policies: Mapping[str, TablePolicy] | None = None,
) -> tuple[str, ...]:
    selected = table_policies or TABLE_POLICIES
    policies = selected.values()
    if not include_optional:
        policies = (policy for policy in policies if not policy.optional)
    return tuple(policy.name for policy in policies)


def schema_policy_for_version(schema_version: int) -> VersionedSchemaPolicy:
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(f"unsupported Hermes schema version: {schema_version!r}")
    try:
        return SCHEMA_POLICIES[schema_version]
    except KeyError as exc:
        raise ValueError(f"unsupported Hermes schema version: {schema_version}") from exc


__all__ = [
    "BUSINESS_METADATA",
    "BUSINESS_PAYLOAD",
    "CONTROL_RECOVERY",
    "CREDENTIAL_SECURITY_EXCLUDED",
    "CURRENT_WRITER_SOURCE_ID",
    "DELIVERY_METADATA",
    "DERIVED_INTERNAL",
    "DERIVED_REQUIRED_RELATIONS",
    "DERIVED_SQLITE_TABLES",
    "DERIVED_SQLITE_TABLE_POLICIES",
    "EXTENSION_EXACT_DISPOSITIONS",
    "FIELD_POLICY_REVIEW_SHA256",
    "FIELD_POLICY_VERSION",
    "MIXED_REQUIRES_PROJECTION",
    "OFFICIAL_SCHEMA_VERSION",
    "OFFICIAL_SOURCE_ID",
    "OPTIONAL_TELEGRAM_TABLE_POLICIES",
    "SCHEMA30_SOURCE_ID",
    "SCHEMA30_TABLE_POLICIES",
    "SCHEMA30_WRITER_SOURCE_ID",
    "SCHEMA32_FIELD_POLICY_REVIEW_SHA256",
    "SCHEMA32_SCHEMA_CENSUS_SHA256",
    "SCHEMA32_SCHEMA_VERSION",
    "SCHEMA32_SOURCE_ID",
    "SCHEMA32_TABLE_POLICIES",
    "SCHEMA32_WRITER_SOURCE_ID",
    "SCHEMA_CENSUS_SHA256",
    "SCHEMA_POLICIES",
    "SECURITY_IDENTITY",
    "TABLE_POLICIES",
    "ColumnPolicy",
    "ModelConfigProjection",
    "ModelConfigProjectionRefused",
    "SchemaValidation",
    "TablePolicy",
    "VersionedSchemaPolicy",
    "column_policy",
    "extension_disposition",
    "policy_counts",
    "policy_table_names",
    "project_model_config",
    "schema_policy_for_version",
    "table_policy",
    "table_policy_for_mapping",
    "validate_schema",
]
