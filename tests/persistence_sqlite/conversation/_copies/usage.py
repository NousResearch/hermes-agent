from . import Copy

COPY = Copy(
    legacy='hermes_state_usage',
    doc='Usage accounting: the counters transaction the accounting queue applies, auxiliary usage, billing route and totals.',
    names=(
        '_TOKEN_COUNTERS',
        '_token_update_sql',
        '_TOKEN_UPDATE_ABSOLUTE_SQL',
        '_TOKEN_UPDATE_DELTA_SQL',
        '_MODEL_USAGE_UPSERT_SQL',
        '_MODEL_USAGE_FIELDS',
    ),
    legacy_class='SessionUsageMixin',
    members=(
        'update_session_billing_route',
        'update_token_counts',
        '_update_token_counts_in_transaction',
        '_record_model_usage',
        'record_auxiliary_usage',
        'auxiliary_usage_by_task',
        'usage_totals',
    ),
    # The copy's receiver parameter is ``holder``, so that is the name ``locals()`` holds for it.
    substitutions={'update_token_counts': (("    values.pop('self')", "    values.pop('holder')"),)},
)
