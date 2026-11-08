from . import Copy

COPY = Copy(
    legacy='hermes_state_worker_lifecycle',
    doc='The session row insert run inside a write transaction.',
    names=(
        'insert_session_row_in_transaction',
    ),
)
