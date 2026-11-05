from . import Copy

# The delete and archive methods return what they removed or newly archived; the caller runs the post-commit work
# (the on_session_delete hook, retired-media collection, transcript-file removal) through session_retention.
# A delete lists the removed ids as get_session_delete_targets orders them: the lineage, then the delegates.
_DELETE_TAIL = (
    "        deleted = self._execute_write(_do)\n"
    "        if deleted:\n"
    "            emit_session_deleted(self, removed_ids, reason=\"delete\", primary=session_id, sessions_dir=sessions_dir)\n"
    "        from hermes_state_media import collect_retired_media\n"
    "        collect_retired_media(self)\n"
    "        for sid in removed_ids:\n"
    "            self._remove_session_files(sessions_dir, sid)\n"
    "        return bool(deleted)"
)
_DELETE_MANY_TAIL = (
    "        count = self._execute_write(_do)\n"
    "        if count:\n"
    "            emit_session_deleted(self, removed_ids, reason=\"delete\", primary=unique_ids[0] if len(unique_ids) == 1 else None,\n"
    "                                 sessions_dir=sessions_dir)\n"
    "        from hermes_state_media import collect_retired_media\n"
    "        collect_retired_media(self)\n"
    "        for sid in removed_ids:\n"
    "            self._remove_session_files(sessions_dir, sid)\n"
    "        return count"
)
_DELETE_IF_EMPTY_TAIL = (
    "        deleted = self._execute_write(_do)\n"
    "        if deleted:\n"
    "            emit_session_deleted(self, [session_id], reason=\"empty_sweep\", sessions_dir=sessions_dir)\n"
    "        from hermes_state_media import collect_retired_media\n"
    "        collect_retired_media(self)\n"
    "        if deleted:\n"
    "            self._remove_session_files(sessions_dir, session_id)\n"
    "        return deleted"
)
_DELETE_EMPTY_TAIL = (
    "        count = self._execute_write(_do)\n"
    "        if count:\n"
    "            emit_session_deleted(self, removed_ids, reason=\"empty_sweep\", sessions_dir=sessions_dir)\n"
    "        from hermes_state_media import collect_retired_media\n"
    "        collect_retired_media(self)\n"
    "        for sid in removed_ids:\n"
    "            self._remove_session_files(sessions_dir, sid)\n"
    "        return count"
)
_ARCHIVE_TAIL = (
    "        ids, newly_archived = self._execute_write(_do)\n"
    "        if newly_archived and notify:\n"
    "            emit_session_deleted(self, ids, reason=\"archive\", primary=session_id)\n"
    "        return bool(ids)"
)

SUBSTITUTIONS = {
    # The column-parse cache is read from the folder that holds the store.
    'list_sessions_rich': (
        ("self._compact_session_cols()", "self._compact_session_cols(self.db_path.parent)"),
    ),
    '_delete_delegate_children': (
        ("def _delete_delegate_children(conn, parent_ids: list[str]) -> list[str]:",
         "def _delete_delegate_children(conn, parent_ids: list[str], *, media_root: str) -> list[str]:"),
        ("    retire_sessions(conn, ids)\n", "    retire_sessions(conn, ids, media_root=media_root)\n"),
    ),
    'discard_unadmitted_session': (
        ("(self, session_id: str) -> bool:", "(self, session_id: str, *, media_root: str) -> tuple[str, ...]:"),
        ("return self.delete_session(session_id)", "return self.delete_session(session_id, media_root=media_root)"),
        ("            return cur.rowcount > 0", "            return (session_id,) if cur.rowcount > 0 else ()"),
        ("        return bool(result)", "        return result"),
    ),
    'delete_session': (
        ("        include_compression_chain: bool = False,\n    ) -> bool:",
         "        include_compression_chain: bool = False, *, media_root: str,\n    ) -> tuple[str, ...]:"),
        ("            retire_sessions(conn, scope)\n", "            retire_sessions(conn, scope, media_root=media_root)\n"),
        ("            removed_ids.extend(_delete_delegate_children(conn, scope))",
         "            delegate_ids = _delete_delegate_children(conn, scope, media_root=media_root)"),
        ("            removed_ids.extend(scope)",
         "            removed_ids.extend([session_id, *sorted(set(scope) - {session_id}), *sorted(delegate_ids)])"),
        ("                include_compression_chain=True,\n            ) > 0",
         "                include_compression_chain=True, media_root=media_root,\n            )[1]"),
        (_DELETE_TAIL, "        self._execute_write(_do)\n        return tuple(removed_ids)"),
    ),
    'delete_sessions': (
        ("        include_compression_chain: bool = False,\n    ) -> int:",
         "        include_compression_chain: bool = False, *, media_root: str,\n    ) -> tuple[int, tuple[str, ...]]:"),
        ("            retire_sessions(conn, expanded)\n", "            retire_sessions(conn, expanded, media_root=media_root)\n"),
        ("            removed_ids.extend(_delete_delegate_children(conn, expanded))",
         "            delegate_ids = _delete_delegate_children(conn, expanded, media_root=media_root)"),
        ("            removed_ids.extend(expanded)", "            removed_ids.extend([*expanded, *delegate_ids])"),
        ("            return 0\n        removed_ids: list[str] = []",
         "            return 0, ()\n        removed_ids: list[str] = []"),
        (_DELETE_MANY_TAIL, "        count = self._execute_write(_do)\n        return count, tuple(removed_ids)"),
    ),
    'delete_session_if_empty': (
        ("sessions_dir: Optional[Path] = None) -> bool:",
         "sessions_dir: Optional[Path] = None, *, media_root: str) -> tuple[str, ...]:"),
        ("retire_sessions(conn, [session_id], expiring=[session_id])",
         "retire_sessions(conn, [session_id], expiring=[session_id], media_root=media_root)"),
        (_DELETE_IF_EMPTY_TAIL, "        return (session_id,) if self._execute_write(_do) else ()"),
    ),
    'delete_empty_sessions': (
        ("sessions_dir: Optional[Path] = None) -> int:",
         "sessions_dir: Optional[Path] = None, *, media_root: str) -> tuple[str, ...]:"),
        ("retire_prunable(conn, sorted(session_ids))", "retire_prunable(conn, sorted(session_ids), media_root=media_root)"),
        (_DELETE_EMPTY_TAIL, "        self._execute_write(_do)\n        return tuple(removed_ids)"),
    ),
    'set_session_archived': (
        (") -> bool:", ") -> tuple[bool, tuple[str, ...]]:"),
        (_ARCHIVE_TAIL,
         "        ids, newly_archived = self._execute_write(_do)\n"
         "        return bool(ids), tuple(ids) if newly_archived and notify else ()"),
    ),
    'unarchive_recoverable_session': (
        ("if not self.set_session_archived(session_id, False):",
         "if not self.set_session_archived(session_id, False)[0]:"),
    ),
}

COPY = Copy(
    legacy='hermes_state_sessions',
    doc='Session rows: create, end and reopen, metadata and model config, lifecycle flags, listing and counting, delete.',
    names=(
        'logger',
        '_delegate_from_json',
        '_MODEL_CONFIG_ROW_MISSING',
        '_LINEAGE_CTE_SQL',
        '_parse_model_config',
        '_cwd_prefix_clause',
        '_workspace_key_clause',
        '_where_sql',
        '_session_filter_where',
        '_expand_compression_lineage_ids',
        '_collect_delegate_child_ids',
        '_delete_delegate_children',
        'INTERNAL_LISTING_SOURCES',
        'SESSION_STATUS_COMPLETE',
        'SESSION_STATUS_INTERRUPTED',
        'SESSION_STATUS_ERROR',
        '_ERROR_FINISH_REASONS',
        'classify_session_status',
        '_SAME_KEY_NAMESPACE_SQL',
        '_UPSERT_KEEP_EXISTING_SQL',
        '_inherit_col_sql',
        '_INHERIT_SEP',
        '_INHERIT_PARENT_META_SQL',
        '_FORK_EDGE_EXCLUSION_SQL',
        '_INHERIT_PARENT_ROUTING_SQL',
    ),
    legacy_class='SessionSessionsMixin',
    members=(
        '_ORPHAN_ADOPTION_MAX_GAP_S',
        '_NON_CONTINUATION_CHILD_FILTER_SQL',
        'RECOVERABLE_END_REASONS',
        '_SESSION_COMPACT_EXCLUDED',
        '_EMPTY_SESSION_WHERE',
        '_own_profile_name',
        '_inherit_parent_session_metadata',
        '_insert_session_row',
        'create_session',
        'end_session',
        '_end_and_bump',
        '_retire_undrained_queue_rows_conn',
        'retire_undrained_queue_rows',
        'reopen_session',
        'promote_to_session_reset',
        'update_session_cwd',
        'publish_session_git_metadata',
        'backfill_repo_roots',
        'touch_session_activity',
        'clear_session_activity_labels',
        'update_session_meta',
        'update_system_prompt',
        'update_session_tool_names',
        'update_session_model',
        '_write_model_config_patch',
        '_merge_model_config_json',
        'patch_session_model_config',
        'get_session_model_config_value',
        'update_session_runtime_lock',
        'set_session_yolo',
        'session_yolo_enabled',
        'get_session',
        'get_session_list_row',
        'get_recent_session_model_route',
        'resolve_session_id',
        'backfill_null_session_profiles',
        'backfill_acp_session_cwd',
        '_set_lineage_column',
        '_set_lineage_column_in_transaction',
        'set_session_archived',
        '_auto_archive_lineage',
        '_unarchive_auto_archived_lineage',
        'unarchive_recoverable_session',
        'set_session_pinned',
        'set_session_hidden',
        'set_session_read',
        'session_unread',
        '_chain_search_where',
        'list_recent_sessions_bounded',
        '_list_row',
        'list_sessions_rich',
        'session_lifecycle_statuses',
        '_is_explicit_branch_session',
        '_session_lineage_root_to_tip',
        'search_sessions',
        'session_count',
        'session_count_ge',
        'session_count_by_source',
        'declared_scope_identity',
        'get_session_delete_targets',
        '_expand_compression_lineage',
        'delete_session',
        'discard_unadmitted_session',
        'delete_session_if_empty',
        'delete_sessions',
        'count_empty_sessions',
        'delete_empty_sessions',
    ),
    substitutions=SUBSTITUTIONS,
)
