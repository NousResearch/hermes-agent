from . import Copy

COPY = Copy(
    legacy='hermes_state_local_lineage',
    doc='Local lineage validation and the routed target advance.',
    names=(
        'validate_local_lineage',
        '_routed_entry',
        '_commit_local_target',
        'advance_local_target',
    ),
    # The expected session id is the binding contract's ``local_session_id``, the derivation ``local_identity`` makes.
    substitutions={'validate_local_lineage': (
        ("    from persistence.sqlite._state.keys import local_identity\n", "    from persistence.authority.bindings import local_session_id\n"),
        ("receipt['session_id'] != local_identity(", "receipt['session_id'] != local_session_id("),
    )},
)
