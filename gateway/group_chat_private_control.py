"""Thin registered private Group Stop/approval presentation and exact dispatch."""
import re

from gateway.group_chat_policy import PRIVATE_ADMIN_REQUIRED, private_admin_event
from gateway.session_group_messaging_control import (
    attest_room_control, authorize_room_control, pending_room_approvals,
)
from gateway.session_group_controls import dispatch_group_control
from gateway.group_chat_private_send import _raw_group_args
from hermes_state_runtime import RuntimeStoreError

UNAVAILABLE = 'Private Group control is unavailable or access changed.'
UNCERTAIN = ('Control outcome is uncertain. Check the Group status; '
             'do not repeat the request to force execution.')
INVALID = 'Use /group N stop or /group N approve pa-<selector> once|deny (from /group N).'


def parse_control(args):
    match = re.fullmatch(r'\s*([0-9]{1,19})[ \t]+(stop|approve)(?:[ \t]+(pa-[0-9a-f]{64})[ \t]+(once|deny))?\s*', args)
    if match is None:
        raise ValueError('invalid control')
    ref = int(match.group(1))
    if not 1 <= ref <= 2**63 - 2:
        raise ValueError('invalid reference')
    operation, selector, choice = match.group(2), match.group(3), match.group(4)
    if (operation == 'stop' and selector is not None) or (operation == 'approve' and
            selector is None):
        raise ValueError('invalid choice')
    return ref, operation, selector, choice


def exact_pending(status, selector):
    actions = status.get('pending_actions') if type(status) is dict else None
    if type(actions) is not list or len(actions) > 128:
        raise RuntimeStoreError('permission_denied')
    approvals = [a for a in actions if type(a) is dict and a.get('kind') == 'approval']
    if len(approvals) > 12:
        raise RuntimeStoreError('permission_denied')
    selected = [action for action in approvals if action.get('selector') == selector]
    if len(selected) != 1:
        raise RuntimeStoreError('permission_denied')
    action = selected[0]
    fields = ('member_id', 'task_id', 'request_id')
    if (any(type(action.get(k)) is not str or not action[k] for k in fields)
            or type(action.get('execution_generation')) is not int
            or action['execution_generation'] < 1):
        raise RuntimeStoreError('permission_denied')
    return action


async def handle_private_group_control(runner, event):
    adapter = private_admin_event(runner, event)
    if adapter is None:
        return PRIVATE_ADMIN_REQUIRED
    source = event.source
    target = source.chat_id, source.thread_id
    try:
        room_ref, operation, selector, choice = parse_control(_raw_group_args(event))
    except ValueError:
        return INVALID
    attempted = False
    try:
        # Consent must be checked before even selecting a pending action.
        scope = 'stop' if operation == 'stop' else 'approval'
        read = authorize_room_control(runner, event, room_ref, scope)
        params = {'room_id': read.room_id}
        if operation == 'stop':
            from gateway.session_group_messaging_send import _stable_message_identity
            recipient = read.inventory._require_context()
            _, digest = _stable_message_identity(event, recipient)
            params['cancel_id'] = 'messaging-stop:' + digest
        else:
            read, actions = pending_room_approvals(runner, event, room_ref)
            action = exact_pending({'pending_actions': actions}, selector)
            params.update({key: action[key] for key in
                           ('member_id', 'task_id', 'execution_generation', 'request_id')})
            params['choice'] = choice
        context = attest_room_control(runner, event, room_ref, scope, params)
        method = 'groups.stop' if operation == 'stop' else 'groups.approve'
        attempted = True
        result = await dispatch_group_control(context, method, params)
    except RuntimeStoreError:
        return UNAVAILABLE
    except Exception:
        if not attempted:
            return UNAVAILABLE
        # A committed reservation or failed external RPC is never retried here.
        body = UNCERTAIN
        result = None
    else:
        if operation == 'stop':
            body = (f"Stop requested for Group {room_ref} ({result['cancelled']} task(s) selected). "
                    'This does not prove interruption has finished.')
        else:
            body = f"Group {room_ref} approval request processed ({choice})."
    # Both settled and uncertain handoffs require the captured operation grant.
    try:
        context.require_current(method=method, params=params)
    except RuntimeStoreError:
        return ''
    if (private_admin_event(runner, event) is not adapter or event.source is not source
            or (source.chat_id, source.thread_id) != target):
        return ''
    metadata = {'_interim_send': True}
    if target[1] is not None:
        metadata['thread_id'] = target[1]
    try:
        await adapter.send(target[0], body, reply_to=None, metadata=metadata)
    except Exception:
        pass  # Never log transport exceptions with private payloads or repeat mutations.
    return ''
