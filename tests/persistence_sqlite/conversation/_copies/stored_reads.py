from . import Copy

COPY = Copy(
    legacy='hermes_state_stored_reads',
    doc='Reads of a stored session without attaching: the transcript cursor, segment pages, handoffs and picker facts.',
    names=(
        '_HANDOFF_ROW_SQL',
        '_SHOWN_ROWS_SQL',
        '_HOLDER_ROWS_SQL',
        '_HANDOFF_COPIES_SQL',
        '_HANDOFF_SHOWN_BEFORE_SQL',
        '_CURSOR_ROW_SQL',
        '_CURSOR_COPY_SQL',
        '_CURSOR_CARRIER_SQL',
        '_LAST_REPLIES_SQL',
        '_ids',
    ),
    legacy_class='SessionStoredReadsMixin',
    members=(
        'transcript_lineage_ids',
        'transcript_cursor_state',
        'transcript_display_index_complete',
        'transcript_handoff_rows',
        'transcript_handoffs_shown_before',
        'transcript_segment_page',
        '_transcript_cursor_slot',
        '_transcript_durable_firsts',
        'transcript_tool_calls_before',
        'sessions_in_workspace',
        'picker_session_facts',
        'session_current_segments',
        'subagent_session_ids',
    ),
)
