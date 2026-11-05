from . import Copy

COPY = Copy(
    legacy='hermes_state_sessions_model_config',
    doc='Conversation model-config values at the compression tip.',
    names=(
        'update_conversation_model_config_value',
        'session_ids_with_model_config_key',
    ),
)
