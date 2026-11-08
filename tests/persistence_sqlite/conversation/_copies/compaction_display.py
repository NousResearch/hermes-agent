from . import Copy

COPY = Copy(
    legacy='hermes_state_compaction_display',
    doc='Display reads paint a compacted tool result once, as its archived original.',
    names=(
        '_ORIGINAL_VISIBLE_SQL',
        'not_a_later_tool_copy_sql',
        'later_tool_copy_ids',
    ),
)
