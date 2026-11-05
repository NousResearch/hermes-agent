from . import Copy

COPY = Copy(
    legacy='hermes_state',
    doc='Session row decoding, prompt cleanup and the kanban ownership reads.',
    names=(
        'logger',
        '_MAX_SAFE_MESSAGES',
        'SessionResumeTooLargeError',
        'SessionExportTooLargeError',
        '_compression_lock_holder_process_is_dead',
        '_REVIEW_HARNESS_PREFIXES',
        '_is_background_review_harness_message',
        '_strip_background_review_harness',
        '_STALE_TOOL_CALL_MARKER_RE',
        '_is_stale_tool_call_marker_message',
        '_strip_stale_tool_call_markers',
    ),
    legacy_class='SessionDB',
    # The legacy logger is named after its module, which the copy is not.
    substitutions={'logger': (('logging.getLogger(__name__)', 'logging.getLogger("hermes_state")'),)},
    members=(
        '_AUTO_PRUNE_STALE_OPEN_SOURCES',
        '_IMPORT_MAX_SESSIONS',
        '_IMPORT_MAX_SESSION_BYTES',
        'MAX_TITLE_LENGTH',
        'CANONICAL_BOT_CHAT_TITLE',
        '_CONTENT_JSON_PREFIX',
        'REACTIONS_METADATA_KEY',
        '_CONVERSATION_ROW_COLUMNS',
        '_delete_unreferenced_system_prompts',
        '_session_row_dict',
        'retag_kanban_worker_sessions',
        'is_kanban_owned_session',
    ),
)
