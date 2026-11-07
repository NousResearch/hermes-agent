from . import Copy

COPY = Copy(
    legacy='hermes_state_prune_rewrite',
    doc='Held-row staleness checks and the in-place prune rewrite.',
    names=(
        '_MESSAGE_ROW_COLUMNS',
        '_PRUNE_FIXED_COLUMNS',
    ),
    legacy_class='SessionPruneRewriteMixin',
    members=(
        'rewritten_held_row_ids',
        'held_row_mismatches',
        'rewrite_pruned_rows',
    ),
)
