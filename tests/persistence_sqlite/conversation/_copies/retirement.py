from . import Copy

COPY = Copy(
    legacy='hermes_state_mutation_retirement',
    doc='The deletion fence: retired-id checks, retirement of sessions with their receipts and routes.',
    names=(
        'RETIRED_PREFIX',
        'VIEWER_CLOSE_PREFIX',
        'retired_in',
        'create_conflict',
        '_TRANSCRIPT_FIELDS',
        '_compact_result',
        'retire_terminal_receipts',
        '_LIVE_LEDGER_SQL',
        'retire_sessions',
        '_MUTATION_TRANSCRIPT_FIELDS',
        '_RECEIPT_SESSION_PATHS',
        'retire_mutation_receipts',
        '_DERIVED_LOCAL_PREFIXES',
        '_DERIVED_API_SQL',
        '_local_conversations',
        '_derived_api_sessions',
        'retire_prunable',
        'entry_session_id',
        '_mark_retired',
        'retire_routes',
    ),
    substitutions={
        # Every retiring path passes the caller's media root down to retire_media.
        'retire_terminal_receipts': (
            ("def retire_terminal_receipts(conn, session_ids):", "def retire_terminal_receipts(conn, session_ids, *, media_root):"),
            ("retire_media(conn, json.loads(row['payload_json']), row['request_id'])",
             "retire_media(conn, json.loads(row['payload_json']), row['request_id'], media_root=media_root)"),
        ),
        'retire_sessions': (
            ("def retire_sessions(conn, session_ids, *, expiring=()):",
             "def retire_sessions(conn, session_ids, *, expiring=(), media_root):"),
            ("    retire_terminal_receipts(conn, session_ids)\n",
             "    retire_terminal_receipts(conn, session_ids, media_root=media_root)\n"),
        ),
        'retire_prunable': (
            ("def retire_prunable(conn, session_ids):", "def retire_prunable(conn, session_ids, *, media_root):"),
            ("retire_sessions(conn, quiet, expiring=[sid for sid in quiet if sid not in derived])",
             "retire_sessions(conn, quiet, expiring=[sid for sid in quiet if sid not in derived], media_root=media_root)"),
        ),
    },
)
