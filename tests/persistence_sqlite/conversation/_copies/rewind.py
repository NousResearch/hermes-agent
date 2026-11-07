from . import Copy

COPY = Copy(
    legacy='hermes_state_rewind',
    doc='User-turn rewind: the plan reads and the rewind write.',
    names=(
        '_HISTORY_CHANGED',
        'RewindTargetUnavailableError',
        'RewindPlan',
        'RewindOutcome',
        '_user_indices',
        '_comparison_content',
    ),
    legacy_class='SessionRewindMixin',
    members=(
        'rewind_user_turn',
        'plan_user_turn_rewind',
    ),
    # The legacy class was imported only to reach a member that is now a copied function.
    substitutions={'_comparison_content': (("    from hermes_state_messages import SessionMessagesMixin\n", ""),)},
)
