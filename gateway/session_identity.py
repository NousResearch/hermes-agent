"""Stable owner keys from server-authenticated identity, never RPC parameters."""
import json


def authenticated_subject(identity):
    """Keep private native receipts stable; namespace every dashboard identity.

    Only the private bootstrap stamps profile/instance/capabilities. A dashboard
    provider named 'local' (or a subject resembling a UID) is not native provenance.
    Historical remote bare-subject receipts cannot be assigned to a first caller.
    """
    subject = str(identity.get('user_id') or 'unbound')
    if (identity.get('provider') == 'local' and identity.get('profile_id')
            and identity.get('instance_id') and 'capabilities' in identity):
        return subject
    return 'auth:v1:' + json.dumps(
        [identity.get('provider') or '', identity.get('issuer') or '', subject],
        separators=(',', ':'), ensure_ascii=True)
