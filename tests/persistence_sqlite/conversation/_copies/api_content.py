from . import Copy

COPY = Copy(
    legacy='hermes_state_api_content',
    doc='Provider-facing content kept beside a stored message.',
    legacy_class='SessionApiContentMixin',
    members=(
        'set_latest_user_api_content',
        'set_message_api_content',
    ),
)
