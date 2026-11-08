from . import Copy

COPY = Copy(
    legacy='hermes_state_timeline',
    doc='The session timeline and prompt pages, read in one snapshot.',
    names=(
        '_SYNTHETIC_PROMPT',
        '_prompt_preview',
        '_snapshot',
        '_display_rows_sql',
        '_register_functions',
        'get_session_messages_around',
        'get_session_timeline',
    ),
)
