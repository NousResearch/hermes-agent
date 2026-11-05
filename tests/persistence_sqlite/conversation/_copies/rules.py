from . import Copy

COPY = (
    Copy(
        legacy='agent.message_metadata',
        doc='Pure helpers and constants of the application modules the store functions run, each block equal to '
            'the module its comment names.',
        names=('DB_ROW_SNAPSHOT', 'CANONICAL_ROW', 'TOOL_CALL_UID', 'DATA_USE', 'lift_data_use', 'stamp_message_uid'),
    ),
    Copy(legacy='agent.conversation_compression_archive', names=('ABSORBED_ROW_IDS',)),
    Copy(legacy='agent.context_compressor', names=('_is_checkpoint_item', '_newest_checkpoint_carrier')),
    Copy(legacy='agent.message_sanitization', names=('_SURROGATE_RE', '_sanitize_surrogates')),
    Copy(legacy='agent.skill_commands', names=('SKILL_SCAFFOLD_SQL_LIKE', 'AUTO_LOAD_SCAFFOLD_SQL_LIKE')),
    Copy(
        legacy='gateway.message_timestamps',
        names=('_TIMESTAMP_PREFIX_RE', '_localize', '_parse_iso', '_parse_timestamp_match',
               'strip_leading_message_timestamps'),
    ),
    Copy(
        legacy='hermes_cli.timefmt',
        names=('logger', 'EPOCH_MIN', 'EPOCH_MAX', 'coerce_epoch'),
        # The warning keeps the logger name of main's hermes_cli.timefmt.
        substitutions={'logger': (('logging.getLogger(__name__)', 'logging.getLogger("hermes_cli.timefmt")'),)},
    ),
    Copy(legacy='tools.todo_tool', names=('MAX_TODO_RESULT_CHARS',)),
    Copy(legacy='tools.tool_result_storage', names=('_SPILLOVER_NAME_RE', 'SPILLOVER_SQL_FILTER', 'spillover_names_in_text')),
    Copy(legacy='agent.message_metadata', names=('RESTORED_FROM_STORE',)),
    Copy(legacy='agent.context_compressor', names=('_IMAGE_PART_TYPES', '_is_image_part')),
    Copy(legacy='agent.session_persistence', names=('_PERSIST_AFTER_ADMISSION_INTERRUPT',)),
)
