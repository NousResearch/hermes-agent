from . import Copy

COPY = Copy(
    legacy='hermes_state_detached_results',
    doc='Idempotent detached-result appends: delegation deliveries and completion notes.',
    names=(
        '_LINEAGE_MATCH_SQL',
        '_DELEGATION_DELIVERY_MATCH',
    ),
    legacy_class='SessionDetachedResultsMixin',
    members=(
        'append_delegation_delivery',
        'append_completion_note',
        '_append_detached_result',
    ),
)
