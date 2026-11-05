from . import Copy

COPY = Copy(
    legacy='hermes_state_titles',
    doc='Session titles: set with conflict resolution, auto titles, title reads and lineage numbering.',
    names=(
        '_TITLE_CONTROL_RE',
        '_TITLE_INVISIBLE_RE',
    ),
    legacy_class='SessionTitlesMixin',
    members=(
        'sanitize_title',
        '_is_compression_ancestor',
        '_set_session_title',
        '_set_session_title_in_transaction',
        '_resolve_title_conflict',
        'set_session_title',
        'set_auto_title',
        'get_session_title',
        'get_session_title_source',
        'set_session_title_source',
        'get_session_by_title',
        'resolve_session_by_title',
        'get_next_title_in_lineage',
    ),
    # The title limit is the copied ``MAX_TITLE_LENGTH``, so the legacy class is not imported for it.
    substitutions={'sanitize_title': (("        from hermes_state import SessionDB\n", ""),)},
)
