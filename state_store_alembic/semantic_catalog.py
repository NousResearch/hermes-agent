"""Read-only semantic validation for the immutable core and current catalogs.

Alembic is the only DDL authority.  This module compares PostgreSQL catalogs
with exact revision contracts after bootstrap and from operational health checks.
It never creates, alters, stamps, or repairs database objects.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from state_store_alembic.errors import BaselineMigrationContractError
from state_store_alembic.migration_helpers import TENANT_SCHEMA_PATTERN, V25_CORE_REVISION

V26_SQLITE_IMPORT_MANIFEST_REVISION = "state_store_v26_sqlite_import"

# Compact form: name:type[:notnull][:default].  Types are format_type() results.
_CORE_TABLES: Mapping[str, tuple[str, ...]] = {
    "sessions": (
        "id:text:notnull", "source:text:notnull", "started_at:double precision:notnull", "ended_at:double precision", "end_reason:text",
        "user_id:text", "session_key:text", "chat_id:text", "chat_type:text", "thread_id:text", "display_name:text", "origin_json:text",
        "model:text", "model_config:jsonb", "parent_session_id:text", "cwd:text", "profile_name:text", "git_repo_root:text", "title:text", "title_source:text",
        "hidden:boolean:notnull:false", "archived:boolean:notnull:false", "pinned:boolean:notnull:false", "system_prompt_hash:text",
        "input_tokens:bigint:notnull:0", "output_tokens:bigint:notnull:0", "cache_read_tokens:bigint:notnull:0", "cache_write_tokens:bigint:notnull:0", "reasoning_tokens:bigint:notnull:0",
        "estimated_cost_usd:double precision", "actual_cost_usd:double precision", "cost_status:text", "cost_source:text", "pricing_version:text", "billing_provider:text", "billing_base_url:text", "billing_mode:text", "api_call_count:bigint:notnull:0",
        "git_branch:text", "git_metadata_generation:bigint:notnull:0", "last_activity_at:double precision", "last_activity_description:text:notnull:''", "last_activity_provenance:text:notnull:'unknown'", "compression_failure_cooldown_until:double precision", "compression_failure_error:text", "compression_fallback_streak:bigint:notnull:0", "compression_ineffective_count:bigint:notnull:0", "compression_recovery_deadline:double precision", "rewind_count:bigint:notnull:0",
    ),
    "messages": (
        "id:bigint:notnull:identity", "session_id:text:notnull", "role:text:notnull", "content:text", "created_at:double precision:notnull", "tool_call_id:text", "tool_calls:jsonb", "tool_name:text", "effect_disposition:text", "token_count:bigint", "finish_reason:text", "reasoning:text", "reasoning_content:text", "reasoning_details:text", "codex_reasoning_items:text", "codex_message_items:text", "platform_message_id:text", "observed:boolean:notnull:false", "_compressed_summary:boolean:notnull:false", "active:boolean:notnull:true", "compacted:boolean:notnull:false", "api_content:text", "display_kind:text", "display_metadata:jsonb", "display_identity:text", "search_document:tsvector:generated",
    ),
    "system_prompts": ("hash:text:notnull", "prompt:text:notnull"),
    "session_model_usage": ("session_id:text:notnull", "model:text:notnull", "billing_provider:text:notnull:''", "billing_base_url:text:notnull:''", "billing_mode:text:notnull:''", "task:text:notnull:''", "api_call_count:bigint:notnull:0", "input_tokens:bigint:notnull:0", "output_tokens:bigint:notnull:0", "cache_read_tokens:bigint:notnull:0", "cache_write_tokens:bigint:notnull:0", "reasoning_tokens:bigint:notnull:0", "estimated_cost_usd:double precision:notnull:0", "actual_cost_usd:double precision:notnull:0", "cost_status:text", "cost_source:text", "first_seen:double precision", "last_seen:double precision"),
    "conversation_generations": ("source:text:notnull", "session_key:text:notnull", "generation:bigint:notnull:0"),
    "search_index_maintenance": ("singleton:boolean:notnull:true", "last_success_at:double precision", "last_error:text"),
    "session_runtime_owners": ("namespace:text:notnull:''", "session_id:text:notnull", "installation_id:text:notnull", "host:text:notnull", "process_generation:text:notnull", "fence:bigint:notnull", "expires_at:double precision:notnull", "updated_at:double precision:notnull"),
    "session_runtime_turns": ("namespace:text:notnull:''", "session_id:text:notnull", "turn_id:text:notnull", "state:text:notnull", "owner_fence:bigint:notnull", "receipt_json:jsonb", "created_at:double precision:notnull", "updated_at:double precision:notnull"),
    "compression_locks": ("session_id:text:notnull", "holder:text:notnull", "fence:bigint:notnull", "expires_at:double precision:notnull", "updated_at:double precision:notnull"),
    "session_turn_leases": ("conversation_id:text:notnull", "holder:text:notnull", "fence:bigint:notnull", "expires_at:double precision:notnull", "updated_at:double precision:notnull"),
    "compression_rotation_receipts": ("request_id:text:notnull", "parent_session_id:text:notnull", "child_session_id:text:notnull", "holder:text:notnull", "fence:bigint:notnull", "committed_at:double precision:notnull"),
    "session_control_state": ("session_id:text:notnull", "control_kind:text:notnull", "status:text:notnull", "payload:jsonb:notnull", "revision:bigint:notnull:1", "updated_at:double precision:notnull"),
    "rewind_receipts": ("request_id:text:notnull", "session_id:text:notnull", "conversation_root_id:text:notnull", "target_message_id:bigint:notnull", "turn_holder:text", "turn_fence:bigint", "compression_holder:text", "compression_fence:bigint", "replacement_message_id:bigint", "retired_count:bigint:notnull", "active_prefix_ids:jsonb:notnull", "committed_at:double precision:notnull"),
    "foreign_import_receipts": ("origin_fingerprint:text:notnull", "session_id:text:notnull", "origin_json:jsonb:notnull", "committed_at:double precision:notnull"),
    "gateway_session_routes": ("tenant_namespace:text:notnull", "session_key:text:notnull", "session_id:text:notnull", "generation:bigint:notnull:1", "flags:jsonb:notnull:'{}'", "metadata:jsonb:notnull:'{}'", "created_at:double precision:notnull", "updated_at:double precision:notnull"),
}
_MANIFEST_TABLES: Mapping[str, tuple[str, ...]] = {
    "sqlite_import_manifests": ("import_id:text:notnull", "source_fingerprint:text:notnull", "source_counts:jsonb:notnull", "source_schema:jsonb:notnull", "pre_import_target:jsonb:notnull", "destination_counts:jsonb", "status:text:notnull", "error:text", "created_at:double precision:notnull", "updated_at:double precision:notnull"),
}
_CURRENT_TABLES = {**_CORE_TABLES, **_MANIFEST_TABLES}

_CORE_PKS = {
    "alembic_version": ("version_num",),
    "sessions": ("id",), "messages": ("id",), "system_prompts": ("hash",),
    "session_model_usage": ("session_id", "model", "billing_provider", "billing_base_url", "billing_mode", "task"),
    "conversation_generations": ("source", "session_key"), "search_index_maintenance": ("singleton",),
    "session_runtime_owners": ("namespace", "session_id"), "session_runtime_turns": ("namespace", "session_id", "turn_id"),
    "compression_locks": ("session_id",), "session_turn_leases": ("conversation_id",), "compression_rotation_receipts": ("request_id",),
    "session_control_state": ("session_id", "control_kind"), "rewind_receipts": ("request_id",), "foreign_import_receipts": ("origin_fingerprint",), "gateway_session_routes": ("tenant_namespace", "session_key"),
}
_PKS = {**_CORE_PKS, "sqlite_import_manifests": ("import_id",)}
# table, columns, target table, target columns, delete action, update action,
# match type, validated, deferrable, initially deferred.  PostgreSQL stores the
# action/match fields as their catalog single-character codes.
_FKS = {
    ("messages", "messages_session_id_fkey"): (("session_id",), "sessions", ("id",), "a", "a", "s", True, False, False),
    ("sessions", "sessions_parent_session_id_fkey"): (("parent_session_id",), "sessions", ("id",), "a", "a", "s", False, False, False),
    ("sessions", "sessions_system_prompt_hash_fkey"): (("system_prompt_hash",), "system_prompts", ("hash",), "a", "a", "s", True, False, False),
    ("session_model_usage", "session_model_usage_session_id_fkey"): (("session_id",), "sessions", ("id",), "c", "a", "s", True, False, False),
    ("compression_locks", "compression_locks_session_id_fkey"): (("session_id",), "sessions", ("id",), "c", "a", "s", True, False, False),
    ("compression_rotation_receipts", "compression_rotation_receipts_parent_session_id_fkey"): (("parent_session_id",), "sessions", ("id",), "a", "a", "s", True, False, False),
    ("compression_rotation_receipts", "compression_rotation_receipts_child_session_id_fkey"): (("child_session_id",), "sessions", ("id",), "a", "a", "s", True, False, False),
    ("session_control_state", "session_control_state_session_id_fkey"): (("session_id",), "sessions", ("id",), "c", "a", "s", True, False, False),
    ("rewind_receipts", "rewind_receipts_session_id_fkey"): (("session_id",), "sessions", ("id",), "a", "a", "s", True, False, False),
    ("foreign_import_receipts", "foreign_import_receipts_session_id_fkey"): (("session_id",), "sessions", ("id",), "a", "a", "s", True, False, False),
    ("gateway_session_routes", "gateway_session_routes_session_id_fkey"): (("session_id",), "sessions", ("id",), "a", "a", "s", True, False, False),
}
# CHECK name -> canonical pg_get_constraintdef() expression and convalidated state.
_CORE_CHECKS = {
    ("search_index_maintenance", "search_index_maintenance_singleton_check"): ("CHECK (singleton)", True),
    ("session_runtime_owners", "session_runtime_owners_fence_check"): ("CHECK (fence > 0)", True),
    ("session_runtime_turns", "session_runtime_turns_state_check"): ("CHECK (state = ANY (ARRAY['running', 'indeterminate', 'settled']))", True),
    ("session_runtime_turns", "session_runtime_turns_owner_fence_check"): ("CHECK (owner_fence > 0)", True),
    ("compression_locks", "compression_locks_fence_check"): ("CHECK (fence > 0)", True),
    ("session_turn_leases", "session_turn_leases_fence_check"): ("CHECK (fence > 0)", True),
    ("compression_rotation_receipts", "compression_rotation_receipts_fence_check"): ("CHECK (fence > 0)", True),
    ("session_control_state", "session_control_state_control_kind_check"): ("CHECK (control_kind = ANY (ARRAY['goal', 'heartbeat', 'loop']))", True),
    ("session_control_state", "session_control_state_revision_check"): ("CHECK (revision > 0)", True),
    ("gateway_session_routes", "gateway_session_routes_generation_check"): ("CHECK (generation > 0)", True),
}
_CHECKS = {**_CORE_CHECKS, ("sqlite_import_manifests", "sqlite_import_manifests_status_check"): ("CHECK (status = ANY (ARRAY['running', 'failed', 'complete']))", True)}
# name: table, access method, (key column, indoption) pairs, unique, predicate.
_INDEXES = {
    "messages_session_id_id": ("messages", "btree", (("session_id", 0), ("id", 0)), False, None),
    "sessions_source_session_key": ("sessions", "btree", (("source", 0), ("session_key", 0)), False, None),
    "sessions_parent_session_id": ("sessions", "btree", (("parent_session_id", 0),), False, None),
    "sessions_title_unique": ("sessions", "btree", (("title", 0),), True, "title is not null"),
    "sessions_visibility_started_at": ("sessions", "btree", (("archived", 0), ("hidden", 0), ("started_at", 3)), False, None),
    "sessions_pinned_started_at": ("sessions", "btree", (("pinned", 0), ("started_at", 3)), False, "pinned"),
    "messages_resume_projection": ("messages", "btree", (("session_id", 0), ("active", 0), ("id", 0)), False, None),
    "session_model_usage_session": ("session_model_usage", "btree", (("session_id", 0),), False, None), "session_model_usage_model": ("session_model_usage", "btree", (("model", 0),), False, None),
    "messages_search_document_gin": ("messages", "gin", (("search_document", 0),), False, None),
    "sessions_effective_activity": ("sessions", "btree", (("archived", 0), ("hidden", 0), ("last_activity_at", 3), ("started_at", 3), ("id", 3)), False, None),
    "session_runtime_owners_expires": ("session_runtime_owners", "btree", (("expires_at", 0),), False, None), "session_runtime_turns_state": ("session_runtime_turns", "btree", (("namespace", 0), ("session_id", 0), ("state", 0)), False, None),
    "compression_locks_expires": ("compression_locks", "btree", (("expires_at", 0),), False, None), "session_turn_leases_expires": ("session_turn_leases", "btree", (("expires_at", 0),), False, None),
    "compression_rotation_receipts_child_unique": ("compression_rotation_receipts", "btree", (("child_session_id", 0),), True, None), "compression_rotation_receipts_parent": ("compression_rotation_receipts", "btree", (("parent_session_id", 0), ("committed_at", 0)), False, None),
    "session_control_state_kind_status": ("session_control_state", "btree", (("control_kind", 0), ("status", 0)), False, None), "messages_active_target": ("messages", "btree", (("session_id", 0), ("id", 0)), False, "active"),
    "rewind_receipts_session_committed": ("rewind_receipts", "btree", (("session_id", 0), ("committed_at", 0)), False, None), "foreign_import_receipts_session_unique": ("foreign_import_receipts", "btree", (("session_id", 0),), True, None),
    "gateway_session_routes_session_unique": ("gateway_session_routes", "btree", (("tenant_namespace", 0), ("session_id", 0)), True, None),
    "messages_platform_message_id_unique": ("messages", "btree", (("platform_message_id", 0),), True, "platform_message_id is not null"), "messages_session_platform_message_id": ("messages", "btree", (("session_id", 0), ("platform_message_id", 0)), False, "platform_message_id is not null"),
}


def _norm(value: str | None) -> str:
    return re.sub(r"\s+|::(?:[a-z ]+)(?:\[\])?", "", (value or "").lower()).replace("(", "").replace(")", "")


def _fail(detail: str) -> None:
    raise BaselineMigrationContractError(f"PostgreSQL State Store catalog drift: {detail}; formal reinitialization is required")


def _parse(spec: str) -> tuple[str, str, bool, str | None]:
    name, typ, *rest = spec.split(":")
    notnull = "notnull" in rest
    default = next((part for part in rest if part != "notnull"), None)
    return name, typ, notnull, default


def _expected_index_signatures() -> dict[str, tuple[Any, ...]]:
    """Immutable index semantics, including all default-only catalog options."""
    return {
        name: (*value[:4], True, True, True, False, True, True, True, (), _norm(value[4]))
        for name, value in _INDEXES.items()
    }


def _validate_identity_sequence(cursor: Any, schema: str) -> None:
    """Prove the sole identity sequence is internal, owned, and unmodified."""
    cursor.execute(
        "SELECT sequence_relation.relkind, sequence_relation.relpersistence, "
        "format_type(sequence_row.seqtypid, NULL), sequence_row.seqstart, "
        "sequence_row.seqincrement, sequence_row.seqmax, sequence_row.seqmin, "
        "sequence_row.seqcache, sequence_row.seqcycle, "
        "EXISTS (SELECT 1 FROM pg_catalog.pg_depend dependency "
        "JOIN pg_catalog.pg_class table_relation ON table_relation.oid=dependency.refobjid "
        "JOIN pg_catalog.pg_namespace table_namespace ON table_namespace.oid=table_relation.relnamespace "
        "JOIN pg_catalog.pg_attribute column_row ON column_row.attrelid=table_relation.oid "
        "AND column_row.attnum=dependency.refobjsubid "
        "WHERE dependency.classid='pg_class'::regclass AND dependency.objid=sequence_relation.oid "
        "AND dependency.refclassid='pg_class'::regclass AND dependency.deptype='i' "
        "AND table_namespace.nspname=%s AND table_relation.relname='messages' "
        "AND column_row.attname='id') "
        "FROM pg_catalog.pg_class sequence_relation "
        "JOIN pg_catalog.pg_namespace namespace ON namespace.oid=sequence_relation.relnamespace "
        "JOIN pg_catalog.pg_sequence sequence_row ON sequence_row.seqrelid=sequence_relation.oid "
        "WHERE namespace.nspname=%s AND sequence_relation.relname='messages_id_seq'",
        (schema, schema),
    )
    if cursor.fetchall() != [("S", "p", "bigint", 1, 1, 9223372036854775807, 1, 1, False, True)]:
        _fail("messages.id identity sequence ownership or options differ")


def _validate_version_table(cursor: Any, schema: str, revision: str) -> None:
    cursor.execute(
        "SELECT attribute.attname, format_type(attribute.atttypid, attribute.atttypmod), "
        "attribute.attnotnull FROM pg_catalog.pg_attribute attribute "
        "WHERE attribute.attrelid=(%s || '.' || 'alembic_version')::regclass "
        "AND attribute.attnum>0 AND NOT attribute.attisdropped ORDER BY attribute.attnum",
        (schema,),
    )
    if cursor.fetchall() != [("version_num", "character varying(32)", True)]:
        _fail("malformed alembic_version table")
    cursor.execute("SELECT version_num FROM " + '"' + schema + '".alembic_version')
    if cursor.fetchall() != [(revision,)]:
        _fail(f"Alembic version table does not contain exactly {revision}")


def _validate_absent_behavior_objects(cursor: Any, schema: str) -> None:
    checks = (
        ("SELECT procedure.proname FROM pg_catalog.pg_proc procedure JOIN pg_catalog.pg_namespace namespace ON namespace.oid=procedure.pronamespace WHERE namespace.nspname=%s", "functions"),
        ("SELECT trigger.tgname FROM pg_catalog.pg_trigger trigger JOIN pg_catalog.pg_class relation ON relation.oid=trigger.tgrelid JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace WHERE namespace.nspname=%s AND NOT trigger.tgisinternal", "triggers"),
        ("SELECT rule.rulename FROM pg_catalog.pg_rewrite rule JOIN pg_catalog.pg_class relation ON relation.oid=rule.ev_class JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace WHERE namespace.nspname=%s", "rules"),
        ("SELECT policy.polname FROM pg_catalog.pg_policy policy JOIN pg_catalog.pg_class relation ON relation.oid=policy.polrelid JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace WHERE namespace.nspname=%s", "RLS policies"),
        ("SELECT type_row.typname FROM pg_catalog.pg_type type_row JOIN pg_catalog.pg_namespace namespace ON namespace.oid=type_row.typnamespace WHERE namespace.nspname=%s AND type_row.typrelid=0 AND type_row.typelem=0 AND type_row.typtype <> 'p'", "types"),
        ("SELECT coll.collname FROM pg_catalog.pg_collation coll JOIN pg_catalog.pg_namespace namespace ON namespace.oid=coll.collnamespace WHERE namespace.nspname=%s", "collations"),
        ("SELECT config.cfgname FROM pg_catalog.pg_ts_config config JOIN pg_catalog.pg_namespace namespace ON namespace.oid=config.cfgnamespace WHERE namespace.nspname=%s", "text-search configurations"),
        ("SELECT dictionary.dictname FROM pg_catalog.pg_ts_dict dictionary JOIN pg_catalog.pg_namespace namespace ON namespace.oid=dictionary.dictnamespace WHERE namespace.nspname=%s", "text-search dictionaries"),
        ("SELECT parser.prsname FROM pg_catalog.pg_ts_parser parser JOIN pg_catalog.pg_namespace namespace ON namespace.oid=parser.prsnamespace WHERE namespace.nspname=%s", "text-search parsers"),
        ("SELECT tmpl.tmplname FROM pg_catalog.pg_ts_template tmpl JOIN pg_catalog.pg_namespace namespace ON namespace.oid=tmpl.tmplnamespace WHERE namespace.nspname=%s", "text-search templates"),
        ("SELECT relation.relname FROM pg_catalog.pg_class relation JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace WHERE namespace.nspname=%s AND (relation.relrowsecurity OR relation.relforcerowsecurity)", "RLS relation settings"),
    )
    for statement, label in checks:
        cursor.execute(statement, (schema,))
        if cursor.fetchall():
            _fail(f"unexpected {label}")


def _validate_catalog(cursor: Any, schema: str, *, revision: str, tables: Mapping[str, tuple[str, ...]], pks: Mapping[str, tuple[str, ...]], checks: Mapping[tuple[str, str], tuple[str, bool]]) -> None:
    _validate_version_table(cursor, schema, revision)
    cursor.execute("SELECT relation.relname, relation.relkind FROM pg_catalog.pg_class relation JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace WHERE namespace.nspname=%s AND relation.relkind IN ('r','p','v','m','S','f') ORDER BY relation.relname", (schema,))
    allowed_relations = {**{name: "r" for name in tables}, "alembic_version": "r", "messages_id_seq": "S"}
    relations = {str(name): str(kind) for name, kind in cursor.fetchall()}
    if relations != allowed_relations:
        _fail(f"unexpected, missing, or invalid relations {sorted(set(relations) ^ set(allowed_relations))}")
    _validate_absent_behavior_objects(cursor, schema)
    for table, specs in tables.items():
        cursor.execute("SELECT attribute.attname, format_type(attribute.atttypid, attribute.atttypmod), attribute.attnotnull, attribute.attidentity, attribute.attgenerated, COALESCE(pg_get_expr(default_value.adbin, default_value.adrelid), ''), attribute.attcollation=type_row.typcollation FROM pg_catalog.pg_attribute attribute JOIN pg_catalog.pg_type type_row ON type_row.oid=attribute.atttypid LEFT JOIN pg_catalog.pg_attrdef default_value ON default_value.adrelid=attribute.attrelid AND default_value.adnum=attribute.attnum WHERE attribute.attrelid=(%s || '.' || %s)::regclass AND attribute.attnum>0 AND NOT attribute.attisdropped ORDER BY attribute.attnum", (schema, table))
        actual = {str(row[0]): row[1:] for row in cursor.fetchall()}
        expected = {_parse(spec)[0]: _parse(spec)[1:] for spec in specs}
        if set(actual) != set(expected):
            _fail(f"{table} columns differ")
        for column, (typ, notnull, default) in expected.items():
            actual_type, actual_notnull, identity, generated, actual_default, default_collation = actual[column]
            if actual_type != typ or bool(actual_notnull) != notnull:
                _fail(f"{table}.{column} type or nullability differs")
            if not default_collation:
                _fail(f"{table}.{column} collation differs")
            if default == "identity":
                if identity != "a" or generated:
                    _fail(f"{table}.{column} identity contract differs")
            elif default == "generated":
                expected_expression = "to_tsvector('simple', coalesce(content, '') || ' ' || coalesce(tool_name, '') || ' ' || coalesce(tool_calls::text, ''))"
                if generated != "s" or _norm(actual_default) != _norm(expected_expression):
                    _fail(f"{table}.{column} generated expression contract differs")
            elif identity or generated or _norm(actual_default) != _norm(default):
                _fail(f"{table}.{column} default/generated contract differs")
    _validate_identity_sequence(cursor, schema)
    cursor.execute("SELECT constraint_table.relname, constraint_row.conname, constraint_row.contype, array_agg(source_column.attname ORDER BY source_key.ordinality), target_table.relname, array_agg(target_column.attname ORDER BY target_key.ordinality), constraint_row.confdeltype, constraint_row.confupdtype, constraint_row.confmatchtype, constraint_row.convalidated, constraint_row.condeferrable, constraint_row.condeferred FROM pg_catalog.pg_constraint constraint_row JOIN pg_catalog.pg_class constraint_table ON constraint_table.oid=constraint_row.conrelid LEFT JOIN pg_catalog.pg_class target_table ON target_table.oid=constraint_row.confrelid LEFT JOIN unnest(constraint_row.conkey) WITH ORDINALITY source_key(attnum, ordinality) ON true LEFT JOIN pg_catalog.pg_attribute source_column ON source_column.attrelid=constraint_row.conrelid AND source_column.attnum=source_key.attnum LEFT JOIN unnest(constraint_row.confkey) WITH ORDINALITY target_key(attnum, ordinality) ON target_key.ordinality=source_key.ordinality LEFT JOIN pg_catalog.pg_attribute target_column ON target_column.attrelid=constraint_row.confrelid AND target_column.attnum=target_key.attnum JOIN pg_catalog.pg_namespace namespace ON namespace.oid=constraint_table.relnamespace WHERE namespace.nspname=%s AND constraint_row.contype IN ('p','u','f') GROUP BY constraint_table.relname, constraint_row.conname, constraint_row.contype, target_table.relname, constraint_row.confdeltype, constraint_row.confupdtype, constraint_row.confmatchtype, constraint_row.convalidated, constraint_row.condeferrable, constraint_row.condeferred", (schema,))
    actual_pks: dict[str, tuple[str, ...]] = {}
    actual_fks: dict[tuple[str, str], tuple[tuple[str, ...], str, tuple[str, ...], str, str, str, bool, bool, bool]] = {}
    for table, name, kind, source, target, target_columns, delete, update, match, validated, deferrable, deferred in cursor.fetchall():
        if kind == "p":
            actual_pks[str(table)] = tuple(source)
        elif kind == "f":
            actual_fks[(str(table), str(name))] = (
                tuple(source), str(target), tuple(target_columns), str(delete), str(update), str(match),
                bool(validated), bool(deferrable), bool(deferred),
            )
        else:
            _fail(f"unexpected unique constraint {name}")
    if actual_pks != pks or actual_fks != _FKS:
        _fail("primary-key, unique, or foreign-key semantics differ")
    cursor.execute("SELECT constraint_table.relname, constraint_row.conname, pg_get_constraintdef(constraint_row.oid, true), constraint_row.convalidated FROM pg_catalog.pg_constraint constraint_row JOIN pg_catalog.pg_class constraint_table ON constraint_table.oid=constraint_row.conrelid JOIN pg_catalog.pg_namespace namespace ON namespace.oid=constraint_table.relnamespace WHERE namespace.nspname=%s AND constraint_row.contype='c'", (schema,))
    actual_checks = {(str(table), str(name)): (str(definition), bool(validated)) for table, name, definition, validated in cursor.fetchall()}
    expected_checks = {key: (_norm(definition), validated) for key, (definition, validated) in checks.items()}
    normalized_checks = {key: (_norm(definition), validated) for key, (definition, validated) in actual_checks.items()}
    if normalized_checks != expected_checks:
        _fail(f"check constraint name, expression, or validation semantics differ: expected {expected_checks}, actual {normalized_checks}")
    cursor.execute("SELECT index_relation.relname, table_relation.relname, access_method.amname, index_row.indisunique, index_row.indisvalid, index_row.indisready, index_row.indislive, index_row.indnullsnotdistinct, index_row.indnkeyatts=index_row.indnatts, bool_and(attribute.attname IS NOT NULL AND operator_class.opcdefault AND operator_class.opcintype=attribute.atttypid AND operator_class.opcmethod=index_relation.relam), bool_and(COALESCE(index_collation.oid, 0)=attribute.attcollation), COALESCE(index_relation.reloptions, ARRAY[]::text[]), array_agg(COALESCE(attribute.attname::text, '<expression:' || pg_get_indexdef(index_row.indexrelid, key_column.ordinality::int, true) || '>') ORDER BY key_column.ordinality), array_agg(key_option.option ORDER BY key_column.ordinality), pg_get_expr(index_row.indpred, index_row.indrelid) FROM pg_catalog.pg_index index_row JOIN pg_catalog.pg_class index_relation ON index_relation.oid=index_row.indexrelid JOIN pg_catalog.pg_class table_relation ON table_relation.oid=index_row.indrelid JOIN pg_catalog.pg_namespace namespace ON namespace.oid=table_relation.relnamespace JOIN pg_catalog.pg_am access_method ON access_method.oid=index_relation.relam JOIN unnest(index_row.indkey) WITH ORDINALITY key_column(attnum, ordinality) ON true JOIN unnest(index_row.indoption) WITH ORDINALITY key_option(option, ordinality) ON key_option.ordinality=key_column.ordinality JOIN unnest(index_row.indclass) WITH ORDINALITY key_class(opclass, ordinality) ON key_class.ordinality=key_column.ordinality JOIN unnest(index_row.indcollation) WITH ORDINALITY key_collation(collation_oid, ordinality) ON key_collation.ordinality=key_column.ordinality LEFT JOIN pg_catalog.pg_attribute attribute ON attribute.attrelid=index_row.indrelid AND attribute.attnum=key_column.attnum LEFT JOIN pg_catalog.pg_opclass operator_class ON operator_class.oid=key_class.opclass LEFT JOIN pg_catalog.pg_collation index_collation ON index_collation.oid=key_collation.collation_oid WHERE namespace.nspname=%s AND NOT index_row.indisprimary GROUP BY index_relation.relname, table_relation.relname, access_method.amname, index_row.indisunique, index_row.indisvalid, index_row.indisready, index_row.indislive, index_row.indnullsnotdistinct, index_row.indnkeyatts, index_row.indnatts, index_relation.reloptions, index_row.indpred, index_row.indrelid", (schema,))
    indexes = {str(name): (str(table), str(method), tuple(zip(keys, (int(option) for option in options))), bool(unique), bool(valid), bool(ready), bool(live), bool(nulls_not_distinct), bool(keys_only), bool(default_opclasses), bool(default_collations), tuple(sorted(reloptions)), _norm(predicate)) for name, table, method, unique, valid, ready, live, nulls_not_distinct, keys_only, default_opclasses, default_collations, reloptions, keys, options, predicate in cursor.fetchall()}
    if indexes != _expected_index_signatures():
        _fail("index method, key direction/options, opclasses, collations, NULLS NOT DISTINCT, reloptions, predicate, unique, or validity semantics differ")


def validate_v25_core_catalog_cursor(cursor: Any, schema: str) -> None:
    """Validate precisely the immutable v25 core before a child migration writes."""
    if not TENANT_SCHEMA_PATTERN.fullmatch(schema):
        _fail("untrusted tenant schema")
    _validate_catalog(cursor, schema, revision=V25_CORE_REVISION, tables=_CORE_TABLES, pks=_CORE_PKS, checks=_CORE_CHECKS)


def validate_current_catalog_cursor(cursor: Any, schema: str) -> None:
    """Validate precisely the current Alembic head, including child receipts."""
    if not TENANT_SCHEMA_PATTERN.fullmatch(schema):
        _fail("untrusted tenant schema")
    _validate_catalog(cursor, schema, revision=V26_SQLITE_IMPORT_MANIFEST_REVISION, tables=_CURRENT_TABLES, pks=_PKS, checks=_CHECKS)


def validate_core_v25_catalog_cursor(cursor: Any, schema: str) -> None:
    """Compatibility validator for a v25 tenant or the current child revision."""
    if not TENANT_SCHEMA_PATTERN.fullmatch(schema):
        _fail("untrusted tenant schema")
    cursor.execute("SELECT version_num FROM " + '"' + schema + '".alembic_version')
    versions = cursor.fetchall()
    if versions == [(V25_CORE_REVISION,)]:
        validate_v25_core_catalog_cursor(cursor, schema)
    elif versions == [(V26_SQLITE_IMPORT_MANIFEST_REVISION,)]:
        validate_current_catalog_cursor(cursor, schema)
    else:
        _fail("Alembic version table contains an unsupported revision")


def _validate_with_cursor(connection: Any, schema: str, validator: Any) -> None:
    if hasattr(connection, "exec_driver_sql"):
        raw = connection.connection.driver_connection
        with raw.cursor() as cursor:
            validator(cursor, schema)
    else:
        with connection.cursor() as cursor:
            validator(cursor, schema)


def validate_v25_core_catalog(connection: Any, schema: str) -> None:
    _validate_with_cursor(connection, schema, validate_v25_core_catalog_cursor)


def validate_current_catalog(connection: Any, schema: str) -> None:
    _validate_with_cursor(connection, schema, validate_current_catalog_cursor)


def validate_core_v25_catalog(connection: Any, schema: str) -> None:
    """Compatibility wrapper used by existing runtime health checks."""
    _validate_with_cursor(connection, schema, validate_core_v25_catalog_cursor)
