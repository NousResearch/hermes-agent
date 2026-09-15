"""Gmail-only single-message cleanup adapter. Plan is local; execute MUTATES.

Uses common review, target verification, process evidence and local state.
No regex decisions, automatic retries, credential access or thread operations.
"""
import argparse
import json
from pathlib import Path
try:
    from .operation_support import successful_json, run_recorded, json_digest, now, save_new
    from .backend_operations import validate_target, base_argv, target_get_argv
    from .task_support import atomic_json, attempt_path, task_lock, stop_check
    from .review_support import bind, validate_binding, load_review
except ImportError:
    from operation_support import successful_json, run_recorded, json_digest, now, save_new
    from backend_operations import validate_target, base_argv, target_get_argv
    from task_support import atomic_json, attempt_path, task_lock, stop_check
    from review_support import bind, validate_binding, load_review


def get_argv(runtime, account, mid):
    return base_argv(runtime, account, 'gmail') + [
        'gmail', 'messages', 'get', '--format', 'metadata', '--', mid]


def move_argv(runtime, account, mid, destination):
    return base_argv(runtime, account, 'gmail') + [
        'gmail', 'messages', 'modify', '--add-label', destination,
        '--remove-label', 'INBOX', '--', mid]


def labels(message):
    value = message.get('label-ids')
    if (not isinstance(message.get('id'), str) or not message['id']
            or not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value)
            or len(set(value)) != len(value)):
        raise ValueError('Invalid Gmail metadata identity/labels')
    return set(value)


def protected(message, destination):
    value = labels(message)
    permitted = {'INBOX', 'UNREAD', 'CATEGORY_PERSONAL', 'CATEGORY_PROMOTIONS',
                 'CATEGORY_SOCIAL', 'CATEGORY_FORUMS', 'CATEGORY_UPDATES', destination}
    if 'INBOX' not in value or value-permitted:
        raise ValueError('Source absent, protected, user-labelled or unknown label')


def metadata_capture(directory, runtime, account, mid=None):
    invocation, _, message = successful_json(directory)
    labels(message)
    if mid is not None and message['id'] != mid:
        raise ValueError('Returned message ID differs')
    if (invocation['argv'] != get_argv(runtime, account, message['id'])
            or invocation['cwd'] != runtime['cwd']):
        raise ValueError('Metadata capture runtime/account/command mismatch')
    return message


def fetch(runtime, account, mid, parent, stopped, runner=run_recorded):
    directory = attempt_path(parent, 'get')
    runner(get_argv(runtime, account, mid), runtime['cwd'], directory, timeout=30, stopped=stopped)
    return metadata_capture(directory, runtime, account, mid), str(directory)


def journal_read(path, account):
    if not Path(path).exists():
        return {'schema_version': 1, 'account': account, 'backend': 'gmail', 'operations': {}}
    state = json.loads(Path(path).read_bytes())
    if (state.get('schema_version'), state.get('account'), state.get('backend')) != (1, account, 'gmail'):
        raise ValueError('Journal account/backend/schema mismatch')
    return state


def plan(journal, record, operation, target_path, source_capture, review_path, assessment_path, authorized):
    if authorized is not True:
        raise ValueError('Existing authorization must cover this concrete cleanup')
    target = json.loads(Path(target_path).read_bytes())
    account, runtime = target['account'], target['runtime']
    destination = validate_target(target, account, 'gmail', check_files=True)
    # Only ordinary user recovery labels; never a Gmail system label.
    if target['target'].get('type') != 'user':
        raise ValueError('Recovery must be a verified custom Gmail label')
    message = metadata_capture(source_capture, runtime, account)
    protected(message, destination)
    proof = bind(review_path, assessment_path, account, 'gmail', message['id'])
    review = load_review(review_path)
    if json.loads(Path(review['metadata_path']).read_bytes()) != message:
        raise ValueError('Review metadata differs from source capture')
    proposal = {'schema_version': 1, 'record': record, 'operation': operation,
                'account': account, 'runtime': runtime, 'message': message,
                'destination': target, 'source_capture': str(Path(source_capture).resolve()),
                'review': proof, 'authorized': True,
                'argv': move_argv(runtime, account, message['id'], destination)}
    with task_lock(str(journal)+'.lock'):
        state = journal_read(journal, account)
        if operation in state['operations']:
            raise ValueError('Operation ID already exists')
        for previous in state['operations'].values():
            if previous['status'] in ('planned', 'submitted', 'unknown'):
                raise ValueError('Resolve the pending operation before planning more work')
            old = previous['plan']
            if old['message']['id'] == message['id'] and old['record'] != record:
                raise ValueError('Reuse the original logical record')
        state['operations'][operation] = {'plan': proposal, 'plan_sha256': json_digest(proposal),
                                         'status': 'planned', 'history': [{'state': 'planned', 'at': now()}]}
        atomic_json(journal, state)
    return proposal


def validate_plan(op):
    proposal = op['plan']
    if json_digest(proposal) != op['plan_sha256'] or proposal['authorized'] is not True:
        raise ValueError('Plan changed or lacks authorization')
    account, runtime, message = proposal['account'], proposal['runtime'], proposal['message']
    destination = validate_target(proposal['destination'], account, 'gmail', check_files=True)
    if runtime != proposal['destination']['runtime']:
        raise ValueError('Target runtime differs')
    if metadata_capture(proposal['source_capture'], runtime, account, message['id']) != message:
        raise ValueError('Source evidence changed')
    validate_binding(proposal['review'], account, 'gmail', message['id'])
    protected(message, destination)
    if proposal['argv'] != move_argv(runtime, account, message['id'], destination):
        raise ValueError('Plan argv differs from verified records')
    return proposal, destination


def transition(journal, state, op, status, **evidence):
    op['status'] = status
    op['history'].append({'state': status, 'at': now(), **evidence})
    atomic_json(journal, state)


def execute(journal, account, operation, captures, stopped, runner=run_recorded):
    captures = Path(captures)
    captures.mkdir(parents=True, exist_ok=True)
    with task_lock(str(journal)+'.lock'):
        state = journal_read(journal, account)
        op = state['operations'][operation]
        if op['status'] != 'planned':
            raise ValueError('Only a new planned operation can execute; never retry an unknown')
        proposal, destination = validate_plan(op)
        if stopped():
            return {'status': 'stopped_before_preflight'}
        current, preflight = fetch(proposal['runtime'], account, proposal['message']['id'], captures, stopped, runner)
        if current != proposal['message']:
            raise ValueError('Current metadata differs; reconcile and prepare a new review/plan')
        protected(current, destination)
        target_check = attempt_path(captures, 'target')
        runner(target_get_argv(proposal['runtime'], account, 'gmail', destination),
               proposal['runtime']['cwd'], target_check, timeout=30, stopped=stopped)
        _, _, target_response = successful_json(target_check)
        rows = target_response.get('labels')
        if (not isinstance(rows, list) or len(rows) != 1
                or any(rows[0].get(k) != proposal['destination']['target'].get(k) for k in ('id', 'name', 'type'))):
            raise ValueError('Destination changed; reverify it before a new plan')
        if stopped():
            return {'status': 'stopped_before_launch'}
        directory = attempt_path(captures, 'move')
        def before_launch(invocation, path):
            transition(journal, state, op, 'submitted', capture=str(directory), preflight=preflight)
        try:
            runner(proposal['argv'], proposal['runtime']['cwd'], directory, timeout=30,
                   stopped=stopped, before_launch=before_launch)
        finally:
            if op['status'] == 'submitted':
                transition(journal, state, op, 'unknown', capture=str(directory))
        return {'status': op['status'], 'capture': str(directory)}


def reconcile(journal, account, operation, captures, stopped, runner=run_recorded):
    """Read-only mailbox get; update local journal only from exact label evidence."""
    captures = Path(captures)
    captures.mkdir(parents=True, exist_ok=True)
    with task_lock(str(journal)+'.lock'):
        state = journal_read(journal, account)
        op = state['operations'][operation]
        if op['status'] not in ('submitted', 'unknown'):
            raise ValueError('Only submitted/unknown operations need reconciliation')
        proposal = op['plan']
        if json_digest(proposal) != op['plan_sha256']:
            raise ValueError('Plan evidence changed')
        message, capture = fetch(proposal['runtime'], account, proposal['message']['id'], captures, stopped, runner)
        before, after = labels(proposal['message']), labels(message)
        expected = before | {proposal['destination']['target']['id']}
        expected.discard('INBOX')
        identity_fields = ('headers', 'internal-date', 'thread-id')
        identity_ok = all(message.get(k) == proposal['message'].get(k) for k in identity_fields)
        if identity_ok and after == expected:
            status = 'confirmed'
        elif identity_ok and after == before:
            status = 'failed_no_effect_verified'
        else:
            status = 'unknown'
        transition(journal, state, op, status, verification=capture)
        return {'status': status}


def cancel(journal, account, operation, reason):
    """Local only: cancel an unsubmitted plan, never an unknown operation."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Cancellation reason required')
    with task_lock(str(journal)+'.lock'):
        state = journal_read(journal, account)
        op = state['operations'][operation]
        if op['status'] != 'planned':
            raise ValueError('Only unsubmitted plans can be cancelled; reconcile attempts')
        transition(journal, state, op, 'cancelled', reason=reason)
        return {'status': 'cancelled'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    get = sub.add_parser('get', help='Read-only metadata capture; select ID by local scan index')
    get.add_argument('--scan', required=True, type=Path)
    get.add_argument('--index', required=True, type=int)
    get.add_argument('--target', required=True, type=Path)
    get.add_argument('--captures', required=True, type=Path)
    get.add_argument('--stop-file', required=True, type=Path)
    body = sub.add_parser('body', help='Read-only shared JSON body; ID loaded from metadata capture')
    body.add_argument('--source-capture', required=True, type=Path)
    body.add_argument('--target', required=True, type=Path)
    body.add_argument('--captures', required=True, type=Path)
    body.add_argument('--stop-file', required=True, type=Path)
    p = sub.add_parser('plan', help='Local planning from verified records')
    for flag in ('journal', 'target', 'source-capture', 'review', 'assessment'):
        p.add_argument('--'+flag, required=True, type=Path)
    p.add_argument('--record', required=True)
    p.add_argument('--operation', required=True)
    p.add_argument('--authorized', action='store_true')
    cancel_parser = sub.add_parser('cancel', help='Local only: cancel an unsubmitted plan')
    cancel_parser.add_argument('--journal', required=True, type=Path)
    cancel_parser.add_argument('--account', required=True)
    cancel_parser.add_argument('--operation', required=True)
    cancel_parser.add_argument('--reason', required=True)
    for command in ('execute', 'reconcile'):
        p = sub.add_parser(command, help='MUTATING' if command == 'execute' else 'Read-only mailbox verification')
        p.add_argument('--journal', required=True, type=Path)
        p.add_argument('--account', required=True)
        p.add_argument('--operation', required=True)
        p.add_argument('--captures', required=True, type=Path)
        p.add_argument('--stop-file', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        proposal = plan(args.journal, args.record, args.operation, args.target, args.source_capture,
                        args.review, args.assessment, args.authorized)
        result = {'status': 'planned', 'operation': proposal['operation']}
    elif args.command == 'cancel':
        result = cancel(args.journal, args.account, args.operation, args.reason)
    elif args.command == 'body':
        target = json.loads(args.target.read_bytes())
        validate_target(target, target['account'], 'gmail', check_files=True)
        message = metadata_capture(args.source_capture, target['runtime'], target['account'])
        args.captures.mkdir(parents=True, exist_ok=True)
        directory = attempt_path(args.captures, 'body')
        argv = base_argv(target['runtime'], target['account'], 'gmail') + [
            'message', 'read', '--', message['id']]
        run_recorded(argv, target['runtime']['cwd'], directory, timeout=60, stopped=stop_check(args.stop_file))
        successful_json(directory)
        result = {'body_json': str(directory/'stdout.bin'), 'body_capture': str(directory)}
    elif args.command == 'get':
        target = json.loads(args.target.read_bytes())
        validate_target(target, target['account'], 'gmail', check_files=True)
        scan = json.loads(args.scan.read_bytes())
        if (scan.get('schema_version') != 1 or not scan.get('complete')
                or scan['scope']['account'] != target['account'] or 'INBOX' not in scan['scope']['labels']
                or scan['runtime']['executable'] != target['runtime']['executable']
                or scan['scope'].get('config') not in (target['runtime'].get('config'), None if not target['runtime'].get('config') else '')):
            raise ValueError('Need a complete matching Inbox scan')
        if not 0 <= args.index < len(scan['ids']):
            raise ValueError('Index outside scan')
        args.captures.mkdir(parents=True, exist_ok=True)
        _, directory = fetch(target['runtime'], target['account'], scan['ids'][args.index],
                             args.captures, stop_check(args.stop_file))
        result = {'metadata_capture': directory}
    else:
        result = globals()[args.command](args.journal, args.account, args.operation, args.captures,
                                         stop_check(args.stop_file))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
