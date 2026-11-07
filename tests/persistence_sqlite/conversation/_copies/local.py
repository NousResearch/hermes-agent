from . import Copy

COPY = Copy(
    legacy='hermes_state_local',
    doc='Local conversation lineages: owners, segments and receipts.',
    names=(
        'POLICY_PREFIX',
        'API_BINDING_PREFIX',
        'API_DECLARED_PREFIX',
        '_policy_target',
        'retire_local_receipts',
        'local_lineage_owner',
        'owned_lineage_ids',
        'local_conversation_ids',
        'local_lineage_index',
        '_lineage_version',
        '_copy_lineage_index',
        '_cached_local_lineage_index',
        'exclude_superseded_segments',
        'annotate_local_lineages',
    ),
)
