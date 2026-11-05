from . import Copy

COPY = Copy(
    legacy='hermes_state_identity',
    doc='The codec between the live message-identity keys and the identity columns of a message row.',
    names=(
        '_live_or_column',
        '_uid_list',
        '_uid_map',
        '_tool_call_uid_map',
        '_absorbed_uids_json',
        '_tool_call_uids_json',
        '_tool_call_uid_or_none',
        '_live_extends',
        '_fill_missing_tool_call_uids',
        '_restore_row_identity',
        '_restore_identity_columns',
        '_stable_tool_key',
    ),
)
