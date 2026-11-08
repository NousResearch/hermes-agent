from . import Copy

COPY = Copy(
    legacy='hermes_state_terminal',
    doc='The terminal-admission key prefixes and the stored-result codec.',
    names=(
        'RESULT_PREFIX',
        'ADMISSION_PREFIX',
        'IDENTITY_PREFIX',
        'WORKER_PREFIX',
        '_STORED',
        'encode_result',
        'load_result',
        'identity_key',
    ),
)
