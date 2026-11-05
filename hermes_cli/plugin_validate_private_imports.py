"""Admission lint: name a plugin's imports of private Hermes core modules (a warning, never a refusal).

Catalog rule 9 ("public surfaces only") is enforced for runtime rebinds by
``plugin_validate_core_override``; an import of a core internal (``tools.tirith_security``,
``hermes_cli.cli_config_load._parse_service_tier_config``) passes that check yet breaks the plugin on
the release that moves or removes the internal. This static pass lists those imports, with
file:line, so reviewers see the coupling at admission. Imports of the plugin's own modules, the
standard library and third-party packages are never flagged; neither are ``TYPE_CHECKING`` imports.
"""

from __future__ import annotations

import ast
import json
import tomllib
from functools import lru_cache
from pathlib import Path

from hermes_cli.plugin_validate_core_override import _hermes_top_level, _python_files

_PLUGIN_API = Path(__file__).resolve().parent / "data" / "plugin_api.json"


@lru_cache(maxsize=1)
def _public_api() -> frozenset:
    """The published plugin API (``hermes_cli/data/plugin_api.json``): ``module`` entries allow the
    module (underscore names excepted), ``module:Name`` entries allow one name. The docs page
    ``developer-guide/plugins/public-api.md`` is generated from the same file."""
    data = json.loads(_PLUGIN_API.read_text(encoding="utf-8-sig"))
    return frozenset(entry for group in data["groups"] for entry in group["entries"])


_SHOWN = 10


@lru_cache(maxsize=1)
def _hermes_packages() -> frozenset:
    """Hermes-owned top-level names: pyproject's ``packages.find`` includes plus the root modules
    setup.py ships as ``py-modules`` (every root ``*.py``)."""
    root = Path(__file__).resolve().parents[1]
    try:
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8-sig"))
        packages = {name.split(".")[0] for name in pyproject["tool"]["setuptools"]["packages"]["find"]["include"]}
    except (OSError, KeyError, ValueError):
        packages = set(_hermes_top_level()) - {"hermes_plugins", "tests", "evals"}
    return frozenset((packages | {p.stem for p in root.glob("*.py")}) - {"setup", "conftest"})


def _local_modules(rels) -> set[str]:
    """Every dotted name under which one of the plugin's own files could be imported."""
    names = set()
    for rel in rels:
        parts = rel.with_suffix("").parts
        parts = parts[:-1] if parts[-1] == "__init__" else parts
        names |= {".".join(parts[start:end]) for start in range(len(parts)) for end in range(start + 1, len(parts) + 1)}
    return names


def _type_checking_nodes(tree: ast.AST) -> set[int]:
    skipped: set[int] = set()
    for node in ast.walk(tree):
        test = getattr(node, "test", None)
        if isinstance(node, ast.If) and (getattr(test, "id", None) or getattr(test, "attr", None)) == "TYPE_CHECKING":
            skipped |= {id(n) for stmt in node.body for n in ast.walk(stmt)}
    return skipped


def _imported(node: ast.AST) -> list[tuple[str, str]]:
    """``(module, name)`` pairs an import node binds; ``name`` is "" for ``import module``."""
    if isinstance(node, ast.Import):
        return [(alias.name, "") for alias in node.names]
    if isinstance(node, ast.ImportFrom) and not node.level and node.module:
        return [(node.module, alias.name) for alias in node.names]
    if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
            and isinstance(node.args[0].value, str) and ast.unparse(node.func) in ("importlib.import_module", "import_module"):
        return [(node.args[0].value, "")]
    return []


@lru_cache(maxsize=512)
def _core_module_exists(module: str) -> bool:
    path = Path(__file__).resolve().parents[1].joinpath(*module.split("."))
    return path.with_suffix(".py").is_file() or path.is_dir()


def _is_own_module_path(module: str, local: set[str]) -> bool:
    """A Hermes-shaped path that core does not ship but whose tail is one of the plugin's own modules:
    the fallback spelling of the plugin's install location (``plugins.memory.openltm._db`` beside
    ``from . import _db``). A path core DOES ship stays a core import."""
    parts = module.split(".")
    return any(".".join(parts[i:]) in local for i in range(1, len(parts))) and not _core_module_exists(module)


def _private_target(module: str, name: str, core: frozenset, local: set[str]) -> str:
    """The dotted target to report, or "" when the import is public, local or not Hermes's."""
    target = f"{module}.{name}" if name else module
    own = _is_own_module_path(module, local) or bool(name) and _is_own_module_path(target, local)
    hermes = module.split(".")[0] in core and module not in local and not own
    public_api = _public_api()
    public = target in public_api or f"{module}:{name}" in public_api \
        or (module in public_api and not name.startswith("_"))  # ``from gateway.platforms import base`` too
    return target if hermes and not public else ""


def _parsed_files(plugin_dir: Path) -> list[tuple[Path, ast.AST]]:
    """``(rel, tree)`` for every plugin module that can load inside Hermes. Hidden dirs (``.github/``
    CI helpers) never do; tests and vendored envs are skipped by ``_python_files``."""
    parsed = []
    for path, rel in _python_files(plugin_dir):
        if any(p.startswith(".") for p in rel.parts[:-1]):
            continue
        try:
            parsed.append((rel, ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))))
        except (SyntaxError, ValueError):
            continue
    return parsed


def private_import_findings(plugin_dir: Path) -> list[str]:
    """``["<rel>:<line> <module>[.<name>]", ...]`` for every import of a private Hermes core module:
    the first site of each distinct module first (so a truncated list still shows the breadth),
    then the rest, each group in file/line order."""
    files = _parsed_files(Path(plugin_dir))
    local = _local_modules(rel for rel, _tree in files)
    core = _hermes_packages()
    found: set[tuple[str, int, str, str]] = set()
    for rel, tree in files:
        skipped = _type_checking_nodes(tree)
        for node in ast.walk(tree):
            if id(node) in skipped:
                continue
            for module, name in _imported(node):
                target = _private_target(module, name, core, local)
                if target:
                    found.add((rel.as_posix(), getattr(node, "lineno", 0), target, module))
    seen: set[str] = set()
    first = [site for site in sorted(found) if site[3] not in seen and not seen.add(site[3])]
    rest = sorted(found.difference(first))
    return [f"{rel}:{line} {target}" for rel, line, target, _module in first + rest]


# The core objects a plugin holds a live handle to — ``SessionDB`` (``ctx`` session APIs, the
# dashboard), ``AIAgent`` and ``GatewayRunner`` (hook kwargs), ``PluginContext`` — split across their
# facade + sibling files. Calling one of their private methods (``db._execute_write(...)``,
# ``runner._evict_cached_agent(...)``) couples the plugin to core internals exactly like a private
# import, but no import names it, so the import pass above cannot see it.
_CORE_HANDLE_SOURCES = ("run_agent.py", "agent/turn_*.py", "gateway/run.py", "gateway/run_*.py",
                        "hermes_cli/plugins.py")
# The removed session store's private method names (main's ``hermes_state.py`` and ``hermes_state_*.py``); they go
# with the ``hermes_state`` stand-in.
_REMOVED_STORE_PRIVATE_METHODS = (
    "_acquire_path_permit", "_acquire_process_permit", "_active_clause", "_active_transcript_counts",
    "_advance_message_uid_backfill", "_append_detached_result", "_append_messages_in_transaction",
    "_apply_claimed_batch", "_apply_token_batch", "_archive_and_compact_on_conn", "_archive_named_rows",
    "_attach_import_parents", "_auto_archive_lineage", "_backfill_gateway_metadata_from_sessions_json",
    "_backfill_progress", "_bump_conversation_generation", "_bump_session_counters", "_canonical_replayed_user_content",
    "_capture_retired_generation", "_carried_tail_ids", "_carried_tail_slots", "_carry_parent_timestamps",
    "_chain_search_where", "_check_transcript_write_guards", "_checkout_read_conn", "_claim_stored_copy",
    "_clone_message_rows", "_close_conn_logged", "_close_connection_quietly", "_close_read_conn",
    "_coalesce_token_deltas", "_coerce_or", "_compact_session_cols", "_compile_like_boolean_query",
    "_compression_tip_on", "_connect_and_init", "_connect_and_init_with_lock_patience", "_connect_read_only",
    "_contains_cjk", "_copy_of_current_row", "_corrupt_error", "_count_cjk", "_db_file_was_replaced",
    "_db_has_legacy_inline_fts", "_db_has_trigram_tool_calls_projection", "_db_needs_fts_storage_upgrade",
    "_decode_content", "_decode_display_metadata", "_decoded_repair_row", "_dedupe_display_generations",
    "_dedupe_legacy_system_prompts", "_dedupe_replayed_user", "_defer_stale_fts_for_holders",
    "_delete_routing_entries_for_sessions", "_delete_unreferenced_system_prompts", "_demote_legacy_fts_to_trash",
    "_describe_search_path", "_disable_close_time_checkpoint", "_display_dedupe_key", "_display_identity",
    "_display_messages_from_conn", "_display_rows_from_conn", "_display_set", "_drain_token_queue_at_exit",
    "_drop_all_fts_triggers",
    "_drop_fts_triggers", "_drop_shadowed_checkpoint_rows", "_encode_content", "_encode_display_metadata",
    "_end_and_bump", "_ensure_db_file_generation", "_ensure_display_order", "_ensure_fts_schema",
    "_ensure_unique_title_index", "_ensure_v23_fts_tables",
    "_enter_fts_fail_open", "_evict_one_idle_read_conn", "_exact_replayed_user_clone_key",
    "_execute_ddl_script_transactional", "_execute_ddl_skipping_settled_triggers", "_execute_transcript_write",
    "_execute_write", "_expand_compression_lineage", "_fetch_conversation_rows", "_finalize_search_matches",
    "_find_duplicate_replayed_user_message", "_find_foreign_import_on_conn", "_fold_summary_carrier",
    "_foreign_state_db_holders", "_freelist_ratio",
    "_fts_external_index_empty_with_messages", "_fts_index_is_misaligned_source", "_fts_index_known_empty",
    "_fts_match_sql", "_fts_rebuild_finish", "_fts_table_exists", "_fts_table_probe", "_fts_teardown_start_table",
    "_fts_teardown_trash_step", "_fts_triggers_missing", "_fts_update_trigger_needs_narrowing", "_get_read_conn",
    "_get_session_rich_row", "_get_session_rich_rows_batch", "_guarded_ids", "_halt_db_corrupt",
    "_halt_if_db_generation_changed", "_handle_quarantine_if_invalid", "_has_fts_trash",
    "_has_short_cjk_token", "_heal_gateway_routing_pk", "_heal_session_model_usage_pk", "_import_float_or_finite",
    "_import_int_or_bounded", "_import_int_or_none", "_import_json_object_or_none", "_import_load_json",
    "_import_non_finite_path", "_import_session_row", "_import_text_or_none", "_index_held_dicts",
    "_inherit_parent_session_metadata", "_init_fts", "_init_schema", "_insert_message_rows", "_insert_row",
    "_insert_session_row", "_is_compression_ancestor", "_is_compression_child_row", "_is_explicit_branch_session",
    "_is_explicit_fork_child_row", "_is_fts5_unavailable_error", "_is_fts_write_corruption_error", "_is_model_only_row",
    "_is_structural_corruption_error", "_is_trigram_unavailable_error", "_ledgered_session_ids", "_legacy_display_page",
    "_like_rows", "_list_row", "_live_pk_columns", "_loaded_view_content", "_match_rows", "_matching_active_ids",
    "_matching_retired_ids", "_merge_fts_incrementally", "_merge_model_config_json", "_merged_user_run",
    "_message_column_names", "_message_row_params", "_migrate_broad_fts_update_triggers",
    "_migrate_misaligned_fts_source", "_migrate_trigram_cron_exclusion", "_migrate_v22_session_model_usage",
    "_next_chunk_rows", "_normalize_import_message", "_normalize_import_session", "_open_read_only", "_open_writer",
    "_open_writer_conn", "_optimize_drive", "_optimize_settle", "_optimize_vacuum", "_or_relaxed_query",
    "_own_profile_name", "_page_pragmas", "_parse_schema_columns", "_pin_connection", "_pinned_read_conn",
    "_plan_refused_retention", "_present_fts_tables",
    "_proved_coverage", "_prune_filter_where", "_prune_shadowed_checkpoints", "_prune_where",
    "_publish_child_session_row", "_publish_compression_child_on_conn",
    "_quarantine_reason", "_raise_if_db_corrupt", "_raise_if_db_replaced", "_reaction_list", "_reaction_row_query",
    "_read_all", "_read_ctx", "_read_one", "_read_retrying_ioerr", "_read_rows", "_read_session_number",
    "_reap_inactive_orphan_desktop_holders", "_reasoning_json_text", "_rebuild_fts_indexes", "_rebuild_table",
    "_rebuild_table_statements", "_reconcile_columns",
    "_reconcile_display_orders", "_record_db_file_identity", "_record_model_usage", "_recover_stale_fts",
    "_recover_stale_fts_locked", "_refresh", "_refresh_fts_stale_state", "_remove_session_files",
    "_removed_todo_copies", "_reopen_after_close_locked", "_repair_optimize_bookkeeping", "_replays_as",
    "_reseed_missing_progress", "_reset_fts_index_to_empty", "_resolve_carried_row_ids", "_resolve_title_conflict",
    "_resume_count_scope", "_resume_lineage_ids", "_retire_canonical_bot_chat_in_transaction", "_retire_donor_segment", "_retire_undrained_queue_rows_conn",
    "_rewrite_explains", "_rewritten_carry", "_rich_row", "_routing_entries_for_sessions", "_row_identity",
    "_row_to_message_dict", "_rows_to_conversation", "_run", "_run_admitted_startup_rebuild", "_run_data_migrations",
    "_sanitize_fts5_query", "_search_cjk", "_search_message_fields", "_search_messages_impl",
    "_search_messages_like_fallback", "_search_unindexed_gap", "_seed_fts_rebuild_markers", "_serialized_message_row",
    "_session_lineage_root_to_tip", "_session_row_dict", "_session_turn_lease_key", "_session_turn_lease_key_on_conn",
    "_set_lineage_column", "_set_lineage_column_in_transaction", "_set_session_title",
    "_set_session_title_in_transaction", "_settle_lost_generation_locked", "_sleep_before_write_retry",
    "_split_rewind_target", "_sqlite_supports_fts5", "_sqlite_table_exists", "_stamp_kept_live_prefix",
    "_stamp_settled_fts_layout", "_stamp_tool_call_uids", "_stop_token_writer", "_store_system_prompt",
    "_tail_originals", "_tail_rows_after_watermark", "_title_rank", "_todo_copies", "_todo_copy_removed",
    "_token_writer_loop", "_topic_read", "_topic_read_all", "_topic_read_one", "_transcript_cursor_slot",
    "_transcript_durable_firsts", "_trigram_eligible_tokens",
    "_trigram_route_ok", "_trigram_tokenizer_available", "_try_checkpoint", "_try_incremental_merge_fts",
    "_try_wal_checkpoint", "_uids_active_outside", "_unarchive_auto_archived_lineage", "_uncounted_merged_rows",
    "_update_token_counts_in_transaction", "_validate_import_message_content", "_validate_import_payload",
    "_validate_import_session", "_wal_generation_was_lost", "_warn_fts5_unavailable", "_warn_trigram_unavailable",
    "_with_messages", "_with_removed_identities", "_write_generation", "_write_guards_reject",
    "_write_model_config_patch",
    "_write_rowcount", "_write_session_column", "_write_sql", "_write_sql_logged",
)


def _private_method_names(root: Path, patterns) -> frozenset:
    names: set[str] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            except (OSError, SyntaxError, ValueError):
                continue
            names |= {item.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef) for item in node.body
                      if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and item.name.startswith("_") and not item.name.startswith("__")}
    return frozenset(names)


@lru_cache(maxsize=1)
def _core_private_methods() -> frozenset:
    root = Path(__file__).resolve().parents[1]
    return _private_method_names(root, _CORE_HANDLE_SOURCES) | frozenset(_REMOVED_STORE_PRIVATE_METHODS)


def _own_names(files: list[tuple[Path, ast.AST]]) -> set[str]:
    """Every name the plugin defines or assigns, so its own ``obj._helper()`` never reads as core."""
    own: set[str] = set()
    for _rel, tree in files:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                own.add(node.name)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                own.add(node.attr)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                own.add(node.id)
    return own


def private_attribute_findings(plugin_dir: Path) -> list[str]:
    """``["<rel>:<line> <expr>", ...]`` for every access to a private method of a core handle class
    that the plugin does not define itself (``self``/``cls`` receivers excluded)."""
    files = _parsed_files(Path(plugin_dir))
    private = _core_private_methods() - _own_names(files)
    found = {(rel.as_posix(), node.lineno, ast.unparse(node)[:80]) for rel, tree in files for node in ast.walk(tree)
             if isinstance(node, ast.Attribute) and node.attr in private
             and not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"))}
    return [f"{rel}:{line} {expr}" for rel, line, expr in sorted(found)]


def _warn_sites(report, label: str, hits: list[str]) -> None:
    more = f"; and {len(hits) - _SHOWN} more" if len(hits) > _SHOWN else ""
    report.warn(
        f"{label} (catalog rule 9: public surfaces only; these can break on any release): "
        + "; ".join(hits[:_SHOWN]) + more
        + ". Use ctx.* / documented plugin APIs, or open an issue for the seam you need."
    )


def check_private_imports(report, plugin_dir: Path) -> None:
    """Warn (never fail) when the plugin imports private Hermes core modules or calls private
    methods on the core objects it is handed."""
    if hits := private_import_findings(plugin_dir):
        _warn_sites(report, "private core imports", hits)
    if hits := private_attribute_findings(plugin_dir):
        _warn_sites(report, "private core attribute access", hits)
