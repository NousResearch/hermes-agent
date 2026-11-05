from . import Copy

COPY = (Copy(
    legacy='hermes_state_portability',
    doc='Rich session rows, cron runs, skill-scaffolded sessions and working directories.',
    names=(
        '_rich_select',
        '_PROMPT_RESOLVED_SQL',
        '_CRON_POLICY_JOB_SQL',
        '_CRON_ADMISSION_SQL',
        '_CRON_RUN_IDS_SQL',
        '_CRON_EXECUTION_SQL',
    ),
    legacy_class='SessionPortabilityMixin',
    members=(
        'get_session_rich_row',
        '_SQLITE_INT_MIN',
        '_SQLITE_INT_MAX',
        '_compact_session_cols',
        '_rich_row',
        '_read_rows',
        'distinct_session_cwds',
        'list_cron_job_runs',
        'cron_run_identity',
        '_get_session_rich_row',
        '_get_session_rich_rows_batch',
        'list_skill_scaffolded_sessions',
    ),
    # The column-parse cache is read from the folder that holds the store, as the schema setup reads it.
    substitutions={
        '_compact_session_cols': (
            ("def _compact_session_cols(cls) -> str:", "def _compact_session_cols(cls, home) -> str:"),
            ("cls._parse_schema_columns(SCHEMA_SQL)", "cls._parse_schema_columns(SCHEMA_SQL, home)"),
        ),
        '_get_session_rich_rows_batch': (
            ("self._compact_session_cols()", "self._compact_session_cols(self.db_path.parent)"),
        ),
    },
), Copy(
    # The column list memo ``_compact_session_cols`` fills, beside its one reader.
    legacy='hermes_state_sessions',
    legacy_class='SessionSessionsMixin',
    members=('_session_compact_cols_sql',),
))
