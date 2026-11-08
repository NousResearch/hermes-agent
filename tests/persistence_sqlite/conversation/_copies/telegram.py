from . import Copy

COPY = Copy(
    legacy='hermes_state_telegram',
    doc='Telegram sessions not yet linked to a topic.',
    names=(
        '_normalize_telegram_topic_profile_name',
        '_UNLINKED_SELECT_HEAD',
        '_UNLINKED_SELECT_TAIL',
        '_UNLINKED_SCOPE_CLAUSES',
    ),
    legacy_class='SessionTelegramTopicsMixin',
    members=(
        'list_unlinked_telegram_sessions_for_user',
    ),
)
