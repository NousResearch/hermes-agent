"""Editor policy lives with execution; an ACP connection is only a viewer."""
from contextlib import contextmanager
from dataclasses import asdict
import uuid

from hermes_state_runtime import RuntimeStoreError


def validate_editor(source, editor):
    """The launch's editor policy, or a refusal that names the key it read (create path totality).

    An ACP connection is a viewer; the policy itself lives with execution. Every branch below refuses
    with its own field, so a rejected editor launch says which key was wrong instead of a bare code.
    """
    if editor is None:
        return
    if source != 'acp':
        raise RuntimeStoreError('invalid_params', field='editor',
                                detail=f'an editor policy is only accepted from an ACP launch, not {source}')
    if not isinstance(editor, dict):
        raise RuntimeStoreError('invalid_params', field='editor',
                                detail=f'must be a mapping, got {type(editor).__name__}')
    unknown = sorted(set(editor) - {'mcp_servers', 'edit_approval_policy'})
    if unknown:
        raise RuntimeStoreError('invalid_params', field=f'editor.{unknown[0]}',
                                detail='unknown editor key')
    if editor.get('edit_approval_policy', 'ask') != 'ask':
        raise RuntimeStoreError('invalid_params', field='editor.edit_approval_policy',
                                detail="must be 'ask': every edit is approved one at a time")
    if not isinstance(editor.get('mcp_servers', []), list):
        raise RuntimeStoreError('invalid_params', field='editor.mcp_servers',
                                detail=f'must be a list, got {type(editor.get("mcp_servers")).__name__}')
    from gateway.session_local_mcp import validate_servers
    validate_servers(editor.get('mcp_servers', []))


def request_editor_edit(proposal):
    from tools import approval
    from tools.approval_context import get_current_session_key
    from tools.approval_gateway_wait import _await_gateway_decision
    route = get_current_session_key()
    with approval._lock:
        notify = approval._gateway_notify_cbs.get(route)
    if notify is None:
        return False
    # Each diff is a one-use consent, never a command-pattern or session grant.
    data = {'command': f'Edit {proposal.path}', 'description': 'Approve proposed file change',
            'pattern_key': 'editor-' + uuid.uuid4().hex,
            'allow_session': False, 'allow_permanent': False, 'edit': asdict(proposal)}
    data['pattern_keys'] = [data['pattern_key']]
    decision = _await_gateway_decision(route, notify, data, surface='acp')
    return decision.get('resolved') and decision.get('choice') == 'once'


@contextmanager
def editor_scope(policy):
    from acp_adapter.edit_approval import set_edit_approval_requester, reset_edit_approval_requester
    token = set_edit_approval_requester(request_editor_edit if policy.source == 'acp' else None)
    try:
        yield
    finally:
        reset_edit_approval_requester(token)
