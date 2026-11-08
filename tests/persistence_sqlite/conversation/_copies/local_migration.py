from . import Copy

COPY = Copy(
    legacy='hermes_state_local_migration',
    doc='The root of a legacy local lineage.',
    names=(
        'LEGACY_PREFIX',
        'legacy_lineage_root',
    ),
)
