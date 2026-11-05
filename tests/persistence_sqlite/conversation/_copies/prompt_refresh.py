from . import Copy

COPY = Copy(
    legacy='hermes_state_prompt_refresh',
    doc='Operator-requested system prompt refresh epochs, kept in state_meta.',
    names=(
        'PROFILE_KEY',
        'SESSION_KEY_PREFIX',
    ),
    legacy_class='SessionPromptRefreshMixin',
    members=(
        'request_prompt_refresh',
        'prompt_refresh_epoch',
        'count_sessions_with_stored_prompt',
    ),
)
