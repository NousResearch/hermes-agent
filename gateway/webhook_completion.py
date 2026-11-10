"""Verified webhook lineage for selected native completion admissions."""
from copy import deepcopy
import json

from hermes_state_runtime import RuntimeStoreError, get_session_admission

FIELDS = ('webhook_delivery', 'webhook_route', 'webhook_target')


def final_identity(identity):
    try:
        value = json.loads(identity)
    except (ValueError, TypeError):
        return False
    return (isinstance(value, list) and len(value) == 3 and bool(value[1])
            and ((value[0] == 'completion' and value[2] is not None)
                 or (value[0] == 'async_delegation' and value[2] == '')))


def inherit_webhook(authority, entry, identity):
    """Use the committed original's pin, including after two automation hops/finalization."""
    from gateway.webhook_target import is_bot_chat
    from hermes_state_runtime import list_session_admissions
    if not final_identity(identity):
        return {}
    rows = list_session_admissions(authority.db, session_id=authority.logical_owner(entry.session_id), pending_only=False)
    roots = [(row, row['payload'].get('native_text_v1', {})) for row in rows]
    roots = [(row, native) for row, native in roots if native.get('source', {}).get('platform') == 'webhook'
             and native.get('automation') is None and is_bot_chat(native.get('webhook_delivery', {}).get('deliver'))]
    if not roots:
        return {}
    if len(roots) != 1:
        raise RuntimeStoreError('admission_conflict')
    row, native = roots[0]
    if not native.get('webhook_target'):
        raise RuntimeStoreError('permission_denied')
    return dict(webhook_root=row['admission_id'], **{key: deepcopy(native[key]) for key in FIELDS})


def verify_root(authority, row, native):
    if not final_identity(native['automation']['identity']):
        raise RuntimeStoreError('permission_denied')
    root = get_session_admission(authority.db, admission_id=native.get('webhook_root'))
    original = (root or {}).get('payload', {}).get('native_text_v1', {})
    if (root is None or root['target_session_id'] != row['target_session_id'] or original.get('automation') is not None
            or original.get('source') != native.get('source') or original.get('provenance') != native.get('provenance')
            or any(original.get(key) != native.get(key) for key in FIELDS)):
        raise RuntimeStoreError('permission_denied')
