from . import Copy

COPY = Copy(
    legacy='hermes_state_transcript_repair',
    doc='Transcript batch repair and inserted-row snapshots, run inside the transcript write transactions.',
    names=(
        '_NON_PAYLOAD_COLUMNS',
        '_REPAIR_COLUMNS',
        '_METADATA_COLUMNS',
        '_OWNED_COLUMNS',
        '_REPLAY_COLUMNS',
        'REPLAY_SIDECARS',
        '_LIVE_MISSING_METADATA',
        '_METADATA_ONLY',
        '_CONTENT_ONLY',
        'transcript_row_snapshot',
        'stamp_inserted_row_snapshots',
        '_active_logical_message_row',
        '_active_uid_dedupe',
        '_RESTORE_MAY_OMIT',
        '_same_as_stored',
        'resolve_and_repair_transcript_batch',
        '_rewrite_row',
        '_restore_display_index',
        '_active_message_row',
    ),
)
