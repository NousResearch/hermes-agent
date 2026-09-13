"""Private provenance for already-authorized local operator input.

Operator access is profile-local, not account linking or a change of authorship.
It never claims unowned legacy history: that still requires the native migration
path. Only SessionAuthority.submit writes this envelope after authorization;
public submission parameters cannot supply it.
"""
from hermes_state_runtime import RuntimeStoreError


def check_local_input(authority, ref, row):
    payload = row['payload']
    allowed = {'text', 'attachments_v1', 'finite', 'surface_v1'}
    if 'local_operator_v1' in payload:
        allowed.add('local_operator_v1')
        if payload['local_operator_v1'] != {
                'profile_id': authority.profile_id, 'session_id': ref.session_id,
                'principal_id': row['principal_id']}:
            raise RuntimeStoreError('permission_denied')
    elif row['principal_id'] != authority.sessions[ref.session_id].source.user_id:
        raise RuntimeStoreError('permission_denied')
    if not {'text'} <= set(payload) <= allowed:
        raise RuntimeStoreError('permission_denied')
