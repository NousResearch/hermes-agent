"""Immutable PostgreSQL state-store core v1-v25 baseline.

This revision intentionally contains the complete core catalog only.  It is a
fresh-tenant baseline, never a historical replay or an autogeneration artifact.
"""

from __future__ import annotations

from alembic import op

from state_store_alembic.migration_helpers import V25_CORE_REVISION, qualified_name

revision = V25_CORE_REVISION
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = op.get_context().config.attributes["tenant_schema"]
    q = lambda relation: qualified_name(schema, relation)

    op.execute(
        f"CREATE TABLE {q('sessions')} ("
        "id text PRIMARY KEY, source text NOT NULL, started_at double precision NOT NULL, "
        "ended_at double precision, end_reason text, user_id text, session_key text, "
        "chat_id text, chat_type text, thread_id text, display_name text, origin_json text, "
        "model text, model_config jsonb, parent_session_id text, cwd text, profile_name text, "
        "git_repo_root text, title text, title_source text, hidden boolean NOT NULL DEFAULT false, "
        "archived boolean NOT NULL DEFAULT false, pinned boolean NOT NULL DEFAULT false, "
        "system_prompt_hash text, input_tokens bigint NOT NULL DEFAULT 0, "
        "output_tokens bigint NOT NULL DEFAULT 0, cache_read_tokens bigint NOT NULL DEFAULT 0, "
        "cache_write_tokens bigint NOT NULL DEFAULT 0, reasoning_tokens bigint NOT NULL DEFAULT 0, "
        "estimated_cost_usd double precision, actual_cost_usd double precision, cost_status text, "
        "cost_source text, pricing_version text, billing_provider text, billing_base_url text, "
        "billing_mode text, api_call_count bigint NOT NULL DEFAULT 0, git_branch text, "
        "git_metadata_generation bigint NOT NULL DEFAULT 0, last_activity_at double precision, "
        "last_activity_description text NOT NULL DEFAULT '', "
        "last_activity_provenance text NOT NULL DEFAULT 'unknown', "
        "compression_failure_cooldown_until double precision, compression_failure_error text, "
        "compression_fallback_streak bigint NOT NULL DEFAULT 0, "
        "compression_ineffective_count bigint NOT NULL DEFAULT 0, "
        "compression_recovery_deadline double precision, rewind_count bigint NOT NULL DEFAULT 0)"
    )
    op.execute(
        f"CREATE TABLE {q('messages')} ("
        "id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, session_id text NOT NULL "
        f"REFERENCES {q('sessions')}(id), role text NOT NULL, content text, "
        "created_at double precision NOT NULL, tool_call_id text, tool_calls jsonb, tool_name text, "
        "effect_disposition text, token_count bigint, finish_reason text, reasoning text, "
        "reasoning_content text, reasoning_details text, codex_reasoning_items text, "
        "codex_message_items text, platform_message_id text, observed boolean NOT NULL DEFAULT false, "
        "_compressed_summary boolean NOT NULL DEFAULT false, active boolean NOT NULL DEFAULT true, "
        "compacted boolean NOT NULL DEFAULT false, api_content text, display_kind text, "
        "display_metadata jsonb, display_identity text, search_document tsvector GENERATED ALWAYS AS "
        "(to_tsvector('simple', coalesce(content, '') || ' ' || coalesce(tool_name, '') || ' ' || "
        "coalesce(tool_calls::text, ''))) STORED)"
    )
    op.execute(f"CREATE TABLE {q('system_prompts')} (hash text PRIMARY KEY, prompt text NOT NULL)")
    op.execute(
        f"ALTER TABLE {q('sessions')} ADD CONSTRAINT sessions_parent_session_id_fkey "
        f"FOREIGN KEY (parent_session_id) REFERENCES {q('sessions')}(id) NOT VALID"
    )
    op.execute(
        f"ALTER TABLE {q('sessions')} ADD CONSTRAINT sessions_system_prompt_hash_fkey "
        f"FOREIGN KEY (system_prompt_hash) REFERENCES {q('system_prompts')}(hash)"
    )
    op.execute(
        f"CREATE TABLE {q('session_model_usage')} ("
        f"session_id text NOT NULL REFERENCES {q('sessions')}(id) ON DELETE CASCADE, "
        "model text NOT NULL, billing_provider text NOT NULL DEFAULT '', "
        "billing_base_url text NOT NULL DEFAULT '', billing_mode text NOT NULL DEFAULT '', "
        "task text NOT NULL DEFAULT '', api_call_count bigint NOT NULL DEFAULT 0, "
        "input_tokens bigint NOT NULL DEFAULT 0, output_tokens bigint NOT NULL DEFAULT 0, "
        "cache_read_tokens bigint NOT NULL DEFAULT 0, cache_write_tokens bigint NOT NULL DEFAULT 0, "
        "reasoning_tokens bigint NOT NULL DEFAULT 0, estimated_cost_usd double precision NOT NULL DEFAULT 0, "
        "actual_cost_usd double precision NOT NULL DEFAULT 0, cost_status text, cost_source text, "
        "first_seen double precision, last_seen double precision, "
        "PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task))"
    )
    op.execute(
        f"CREATE TABLE {q('conversation_generations')} ("
        "source text NOT NULL, session_key text NOT NULL, generation bigint NOT NULL DEFAULT 0, "
        "PRIMARY KEY (source, session_key))"
    )
    op.execute(
        f"CREATE TABLE {q('search_index_maintenance')} ("
        "singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton), "
        "last_success_at double precision, last_error text)"
    )
    op.execute(f"INSERT INTO {q('search_index_maintenance')} (singleton) VALUES (true)")
    op.execute(
        f"CREATE TABLE {q('session_runtime_owners')} ("
        "namespace text NOT NULL DEFAULT '', session_id text NOT NULL, installation_id text NOT NULL, "
        "host text NOT NULL, process_generation text NOT NULL, fence bigint NOT NULL CHECK (fence > 0), "
        "expires_at double precision NOT NULL, updated_at double precision NOT NULL, "
        "PRIMARY KEY (namespace, session_id))"
    )
    op.execute(
        f"CREATE TABLE {q('session_runtime_turns')} ("
        "namespace text NOT NULL DEFAULT '', session_id text NOT NULL, turn_id text NOT NULL, "
        "state text NOT NULL CHECK (state IN ('running', 'indeterminate', 'settled')), "
        "owner_fence bigint NOT NULL CHECK (owner_fence > 0), receipt_json jsonb, "
        "created_at double precision NOT NULL, updated_at double precision NOT NULL, "
        "PRIMARY KEY (namespace, session_id, turn_id))"
    )
    op.execute(
        f"CREATE TABLE {q('compression_locks')} ("
        f"session_id text PRIMARY KEY REFERENCES {q('sessions')}(id) ON DELETE CASCADE, "
        "holder text NOT NULL, fence bigint NOT NULL CHECK (fence > 0), "
        "expires_at double precision NOT NULL, updated_at double precision NOT NULL)"
    )
    op.execute(
        f"CREATE TABLE {q('session_turn_leases')} ("
        "conversation_id text PRIMARY KEY, holder text NOT NULL, fence bigint NOT NULL CHECK (fence > 0), "
        "expires_at double precision NOT NULL, updated_at double precision NOT NULL)"
    )
    op.execute(
        f"CREATE TABLE {q('compression_rotation_receipts')} ("
        "request_id text PRIMARY KEY, parent_session_id text NOT NULL "
        f"REFERENCES {q('sessions')}(id), child_session_id text NOT NULL REFERENCES {q('sessions')}(id), "
        "holder text NOT NULL, fence bigint NOT NULL CHECK (fence > 0), committed_at double precision NOT NULL)"
    )
    op.execute(
        f"CREATE TABLE {q('session_control_state')} ("
        f"session_id text NOT NULL REFERENCES {q('sessions')}(id) ON DELETE CASCADE, "
        "control_kind text NOT NULL CHECK (control_kind IN ('goal', 'heartbeat', 'loop')), "
        "status text NOT NULL, payload jsonb NOT NULL, revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0), "
        "updated_at double precision NOT NULL, PRIMARY KEY (session_id, control_kind))"
    )
    op.execute(
        f"CREATE TABLE {q('rewind_receipts')} ("
        "request_id text PRIMARY KEY, session_id text NOT NULL REFERENCES "
        f"{q('sessions')}(id), conversation_root_id text NOT NULL, target_message_id bigint NOT NULL, "
        "turn_holder text, turn_fence bigint, compression_holder text, compression_fence bigint, "
        "replacement_message_id bigint, retired_count bigint NOT NULL, active_prefix_ids jsonb NOT NULL, "
        "committed_at double precision NOT NULL)"
    )
    op.execute(
        f"CREATE TABLE {q('foreign_import_receipts')} ("
        "origin_fingerprint text PRIMARY KEY, session_id text NOT NULL REFERENCES "
        f"{q('sessions')}(id), origin_json jsonb NOT NULL, committed_at double precision NOT NULL)"
    )
    op.execute(
        f"CREATE TABLE {q('gateway_session_routes')} ("
        "tenant_namespace text NOT NULL, session_key text NOT NULL, session_id text NOT NULL "
        f"REFERENCES {q('sessions')}(id), generation bigint NOT NULL DEFAULT 1 CHECK (generation > 0), "
        "flags jsonb NOT NULL DEFAULT '{}'::jsonb, metadata jsonb NOT NULL DEFAULT '{}'::jsonb, "
        "created_at double precision NOT NULL, updated_at double precision NOT NULL, "
        "PRIMARY KEY (tenant_namespace, session_key))"
    )
    op.execute(f"CREATE INDEX messages_session_id_id ON {q('messages')} (session_id, id)")
    op.execute(f"CREATE INDEX sessions_source_session_key ON {q('sessions')} (source, session_key)")
    op.execute(f"CREATE INDEX sessions_parent_session_id ON {q('sessions')} (parent_session_id)")
    op.execute(f"CREATE UNIQUE INDEX sessions_title_unique ON {q('sessions')} (title) WHERE title IS NOT NULL")
    op.execute(f"CREATE INDEX sessions_visibility_started_at ON {q('sessions')} (archived, hidden, started_at DESC)")
    op.execute(f"CREATE INDEX sessions_pinned_started_at ON {q('sessions')} (pinned, started_at DESC) WHERE pinned")
    op.execute(f"CREATE INDEX messages_resume_projection ON {q('messages')} (session_id, active, id)")
    op.execute(f"CREATE INDEX session_model_usage_session ON {q('session_model_usage')} (session_id)")
    op.execute(f"CREATE INDEX session_model_usage_model ON {q('session_model_usage')} (model)")
    op.execute(f"CREATE INDEX messages_search_document_gin ON {q('messages')} USING GIN (search_document)")
    op.execute(
        f"CREATE INDEX sessions_effective_activity ON {q('sessions')} "
        "(archived, hidden, last_activity_at DESC, started_at DESC, id DESC)"
    )
    op.execute(f"CREATE INDEX session_runtime_owners_expires ON {q('session_runtime_owners')} (expires_at)")
    op.execute(f"CREATE INDEX session_runtime_turns_state ON {q('session_runtime_turns')} (namespace, session_id, state)")
    op.execute(f"CREATE INDEX compression_locks_expires ON {q('compression_locks')} (expires_at)")
    op.execute(f"CREATE INDEX session_turn_leases_expires ON {q('session_turn_leases')} (expires_at)")
    op.execute(
        f"CREATE UNIQUE INDEX compression_rotation_receipts_child_unique ON "
        f"{q('compression_rotation_receipts')} (child_session_id)"
    )
    op.execute(
        f"CREATE INDEX compression_rotation_receipts_parent ON "
        f"{q('compression_rotation_receipts')} (parent_session_id, committed_at)"
    )
    op.execute(f"CREATE INDEX session_control_state_kind_status ON {q('session_control_state')} (control_kind, status)")
    op.execute(f"CREATE INDEX messages_active_target ON {q('messages')} (session_id, id) WHERE active")
    op.execute(f"CREATE INDEX rewind_receipts_session_committed ON {q('rewind_receipts')} (session_id, committed_at)")
    op.execute(
        f"CREATE UNIQUE INDEX foreign_import_receipts_session_unique ON "
        f"{q('foreign_import_receipts')} (session_id)"
    )
    op.execute(
        f"CREATE UNIQUE INDEX gateway_session_routes_session_unique ON "
        f"{q('gateway_session_routes')} (tenant_namespace, session_id)"
    )
    op.execute(
        f"CREATE UNIQUE INDEX messages_platform_message_id_unique ON {q('messages')} "
        "(platform_message_id) WHERE platform_message_id IS NOT NULL"
    )
    op.execute(
        f"CREATE INDEX messages_session_platform_message_id ON {q('messages')} "
        "(session_id, platform_message_id) WHERE platform_message_id IS NOT NULL"
    )


def downgrade() -> None:
    raise RuntimeError("The immutable state-store core v25 baseline cannot be downgraded")
