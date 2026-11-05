from . import Copy

COPY = Copy(
    legacy='hermes_state_coverage',
    doc='Carried-row coverage proofs for compaction.',
    names=(
        '_text_of',
        '_holds_as_block',
        '_images_replaced',
    ),
    legacy_class='SessionCoverageMixin',
    members=(
        '_carried_tail_ids',
        '_index_held_dicts',
        '_claim_stored_copy',
        '_rewritten_carry',
        '_replays_as',
        '_copy_of_current_row',
        '_rewrite_explains',
        '_uids_active_outside',
        '_carried_tail_slots',
        '_resolve_carried_row_ids',
        '_matching_active_ids',
        '_matching_retired_ids',
        '_merged_user_run',
        '_proved_coverage',
        '_uncounted_merged_rows',
        '_tail_originals',
    ),
)
