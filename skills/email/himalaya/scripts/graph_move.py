"""Prepare a verified Graph plan, or explicitly execute one recorded attempt.

Plan preparation is local. The execute subcommand MUTATES mail only after a
passing authorized plan; it never retries and leaves the outcome unconfirmed.
"""
import argparse
import json
from pathlib import Path

try:
    from .cleanup_records import append_event, replay, locked
    from .backend_operations import validate_target, validate_graph_plan, graph_move_argv
    from .operation_support import digest, now, run_recorded, load_capture
    from .graph_scan import snapshot_hash
    from .review_support import bind, load_review
except ImportError:
    from cleanup_records import append_event, replay, locked
    from backend_operations import validate_target, validate_graph_plan, graph_move_argv
    from operation_support import digest, now, run_recorded, load_capture
    from graph_scan import snapshot_hash
    from review_support import bind, load_review


def read_events(journal):
    return json.loads(Path(journal).read_text(encoding='utf-8'))


def prepare_plan(journal, record_id, target_path, preflight_path, operation_id, authorized, action='cleanup',
                 review_path=None, assessment_path=None):
    records, _ = replay(read_events(journal))
    record = records[record_id]
    target = json.loads(Path(target_path).read_bytes())
    destination = validate_target(target, record['account'], 'msgraph', check_files=True)
    raw = Path(preflight_path).read_bytes()
    scan = json.loads(raw)
    matches = [m for m in scan.get('messages', []) if m.get('id') == record['message']['id']]
    if matches != [record['message']]:
        raise ValueError('Reconcile changed source snapshot before planning')
    source = {'path': str(Path(preflight_path).resolve()), 'sha256': digest(raw),
              'account': scan.get('account'), 'snapshot': matches[0],
              'metadata_evidence': scan.get('metadata_evidence', {}).get(matches[0]['id'])}
    plan = {'event_id': operation_id+'-plan', 'record_id': record_id, 'at': now(),
            'type': 'plan_move', 'plan_schema': 2, 'operation_id': operation_id, 'action': action,
            'decision_revision': record['decision_revision'], 'authorized': authorized,
            'source_id': record['message']['id'], 'source_folder': record['message']['parentFolderId'],
            'destination_folder': destination, 'destination_evidence': target,
            'source_evidence': source, 'source_snapshot_sha256': snapshot_hash(record['message']),
            'membership_evidence': str(Path(preflight_path).resolve()), 'runtime': target['runtime']}
    plan['argv'] = graph_move_argv(plan['runtime'], record['account'], plan['source_id'], destination)
    if action == 'cleanup':
        if review_path is None or assessment_path is None:
            raise ValueError('Cleanup planning requires complete shared review evidence')
        plan['review'] = bind(review_path, assessment_path, record['account'], 'msgraph', plan['source_id'])
        reviewed_metadata = json.loads(Path(load_review(review_path)['metadata_path']).read_bytes())
        if reviewed_metadata != record['message']:
            raise ValueError('Review metadata differs from registered Graph snapshot')
    validate_graph_plan(plan, record, check_files=True)
    append_event(journal, plan)
    return plan


def execute_plan(journal, operation_id, directory, timeout=30, stopped=lambda: False):
    journal, directory = Path(journal), Path(directory)
    # Held across validation, capture, submission and the process. The journal's
    # short write lock remains separate. A stale execution lock needs diagnosis.
    with locked(journal.with_name(journal.name+'.execute.lock')):
        records, operations = replay(read_events(journal))
        operation = operations[operation_id]
        if operation['state'] != 'planned':
            raise ValueError('Operation is not a new planned attempt; reconcile before any retry')
        record = records[operation['record_id']]
        plan = operation['plan']
        args = validate_graph_plan(plan, record, check_files=True)
        submitted = [False]

        def before_launch(invocation, path):
            append_event(journal, {'event_id': operation_id+'-submitted', 'record_id': operation['record_id'],
                                  'at': now(), 'type': 'submitted', 'operation_id': operation_id,
                                  'execution': {'argv': invocation['argv'], 'cwd': invocation['cwd'],
                                                'invocation_path': str(path),
                                                'invocation_sha256': digest(path.read_bytes())}})
            submitted[0] = True

        try:
            run_recorded(args, plan['runtime']['cwd'], directory, timeout, stopped, before_launch)
        finally:
            if submitted[0]:
                # Exit 0, error, timeout or interruption all need mailbox
                # reconciliation. Never equate a CLI status to verified effect.
                append_event(journal, {'event_id': operation_id+'-unknown', 'record_id': operation['record_id'],
                                      'at': now(), 'type': 'unknown', 'operation_id': operation_id,
                                      'evidence': str(directory), 'reason': 'Recorded attempt requires outcome verification'})
        _, result, _ = load_capture(directory)
        return result


def cancel_plan(journal, operation_id, reason):
    journal = Path(journal)
    with locked(journal.with_name(journal.name+'.execute.lock')):
        _, operations = replay(read_events(journal))
        operation = operations[operation_id]
        if operation['state'] != 'planned' or not reason:
            raise ValueError('Only unsubmitted plans can be cancelled with a reason')
        append_event(journal, {'event_id': operation_id+'-cancelled', 'record_id': operation['record_id'],
                              'at': now(), 'type': 'cancelled', 'operation_id': operation_id, 'reason': reason})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='Local only; IDs loaded from records, never typed')
    plan.add_argument('--journal', required=True, type=Path)
    plan.add_argument('--record', required=True)
    plan.add_argument('--target', required=True, type=Path)
    plan.add_argument('--preflight', required=True, type=Path)
    plan.add_argument('--operation', required=True)
    plan.add_argument('--authorized', action='store_true', help='Use only when authorization already covers this move')
    plan.add_argument('--action', choices=['cleanup', 'rescue'], default='cleanup')
    plan.add_argument('--review', type=Path, help='Required for cleanup; shared review_support output')
    plan.add_argument('--assessment', type=Path, help='Required for cleanup; completed chunk assessments')
    execute = sub.add_parser('execute', help='MUTATING: launch one already authorized plan')
    execute.add_argument('--journal', required=True, type=Path)
    execute.add_argument('--operation', required=True)
    execute.add_argument('--directory', required=True, type=Path)
    execute.add_argument('--timeout', type=float, default=30)
    execute.add_argument('--stop-file', type=Path)
    cancel = sub.add_parser('cancel', help='Local only: cancel an unsubmitted plan')
    cancel.add_argument('--journal', required=True, type=Path)
    cancel.add_argument('--operation', required=True)
    cancel.add_argument('--reason', required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        result = prepare_plan(args.journal, args.record, args.target, args.preflight,
                              args.operation, args.authorized, args.action, args.review, args.assessment)
        print(json.dumps({'operation_id': result['operation_id'], 'state': 'planned',
                          'note': 'Exact argv and target evidence are retained privately in the journal'}))
    elif args.command == 'cancel':
        cancel_plan(args.journal, args.operation, args.reason)
        print(json.dumps({'operation_id': args.operation, 'state': 'cancelled'}))
    else:
        if args.stop_file and not args.stop_file.is_absolute():
            parser.error('Stop file must use an absolute native path')
        result = execute_plan(args.journal, args.operation, args.directory, args.timeout,
                              lambda: bool(args.stop_file and args.stop_file.exists()))
        print(json.dumps({'status': result['status'], 'returncode': result['returncode'],
                          'note': 'Inspect journal and reconcile; this is not move confirmation'}))
        raise SystemExit(0 if result['status'] == 'completed' and result['returncode'] == 0 else 2)


if __name__ == '__main__':
    main()
