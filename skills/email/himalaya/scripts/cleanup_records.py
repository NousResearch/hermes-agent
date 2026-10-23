"""Graph-specific offline cleanup gate and logical event journal. No mailbox I/O.

The journal rewrites its event array atomically under an exclusive lock and
derives the current view. Inputs are assessments/verification evidence supplied
by the agent, not independently verified by this helper.
"""
import argparse
import copy
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

try:
    from .graph_scan import atomic_write, category_check
    from .backend_operations import validate_graph_plan
    from .operation_support import digest
except ImportError:
    from graph_scan import atomic_write, category_check
    from backend_operations import validate_graph_plan
    from operation_support import digest

PROTECTED = {'invoice', 'payment', 'purchase', 'refund', 'security', 'trial_conversion',
             'order_update', 'financial_record', 'commitment'}


def decision_gate(message, assessment, proposed, *, metadata_evidence=None, account=None, backend='msgraph'):
    if backend != 'msgraph':
        raise ValueError('This cleanup gate implements Graph protections only; do not normalize Gmail into Graph')
    if proposed not in ('KEEP', 'REVIEW', 'REMOVE'):
        raise ValueError('Unknown decision')
    if proposed != 'REMOVE':
        return proposed, []
    reasons = []
    if assessment.get('fraud_status') not in ('not_suspected', 'suspected', 'confirmed'):
        reasons.append('Fraud assessment missing')
    if assessment.get('fraud_status') in ('suspected', 'confirmed'):
        reasons.append('Use the user-authorized fraud preservation/disposition workflow')
    kinds = assessment.get('content_kinds')
    if not isinstance(kinds, list) or not kinds or any(not isinstance(k, str) for k in kinds):
        reasons.append('Content assessment missing')
        kinds = []
    if set(kinds) & PROTECTED:
        reasons.append('Contains a protected money/account/commitment record')
    if message.get('isDraft') is not False:
        reasons.append('Draft state is protected or unknown')
    if message.get('importance') != 'normal':
        reasons.append('Importance is non-normal or unknown')
    flag = message.get('flag')
    if not isinstance(flag, dict) or flag.get('flagStatus') != 'notFlagged':
        reasons.append('Follow-up flag is protected or unknown')
    if category_check(message, metadata_evidence, account)['state'] != 'empty':
        reasons.append('Categories are protected or unknown')
    if assessment.get('protection_fields_verified') is not True:
        reasons.append('Protection field selection/presence has not been verified')
    if assessment.get('body_reviewed') is not True:
        reasons.append('Read and review this candidate body before moving it')
    if assessment.get('remove_authorized') is not True:
        reasons.append('Removal from this source into the chosen destination is not authorized')
    if assessment.get('reason') in (None, ''):
        reasons.append('A per-message reason is required')
    if assessment.get('content_only_removable') is not True:
        reasons.append('Mixed or unresolved content must not inherit a sender-level removal rule')
    if reasons:
        # Authentication passing never overrides suspected fraud or ambiguity.
        verdict = 'KEEP' if set(kinds) & PROTECTED and assessment.get('fraud_status') == 'not_suspected' else 'REVIEW'
        return verdict, reasons
    return 'REMOVE', []


def replay(events):
    records, operations, seen = {}, {}, {}
    for event in events:
        key = event.get('event_id')
        if not isinstance(key, str) or not key:
            raise ValueError('Every event needs a unique event_id')
        if key in seen:
            if event != seen[key]:
                raise ValueError('Conflicting reuse of event_id')
            continue
        seen[key] = event
        kind, rid = event.get('type'), event.get('record_id')
        if not event.get('at') or not isinstance(rid, str) or not rid:
            raise ValueError('Events need timestamp and stable record_id, never a row index')
        if kind == 'register':
            if event.get('backend', 'msgraph') != 'msgraph':
                raise ValueError('This journal implements Graph move semantics only')
            if rid in records:
                raise ValueError('Record already registered')
            msg = copy.deepcopy(event.get('message'))
            if not isinstance(msg, dict) or not isinstance(msg.get('id'), str) or not msg['id']:
                raise ValueError('Register needs a message with an opaque string ID')
            if not event.get('account') or not msg.get('parentFolderId'):
                raise ValueError('Register needs verified account and parentFolderId')
            records[rid] = {'account': event['account'], 'backend': 'msgraph', 'message': msg, 'original_id': msg['id'],
                            'original_folder': msg['parentFolderId'], 'decision': 'REVIEW',
                            'assessment': {}, 'decision_revision': 0, 'pending_operation': None}
            continue
        if rid not in records:
            raise ValueError('Unregistered record')
        record = records[rid]
        if kind == 'decide':
            if record['pending_operation']:
                raise ValueError('Resolve pending/unknown operation before revising a decision')
            evidence = event.get('metadata_evidence')
            verdict, reasons = decision_gate(record['message'], event.get('assessment', {}), event.get('proposed'),
                                            metadata_evidence=evidence, account=record['account'])
            record.update(decision=verdict, assessment=copy.deepcopy(event.get('assessment', {})),
                          metadata_evidence=copy.deepcopy(evidence),
                          category_check=category_check(record['message'], evidence, record['account']),
                          gate_reasons=reasons, decision_revision=record['decision_revision']+1)
        elif kind == 'plan_move':
            op = event.get('operation_id')
            action = event.get('action')
            if not isinstance(op, str) or not op or op in operations or record['pending_operation']:
                raise ValueError('New operation ID required and previous operation must be resolved')
            if action not in ('cleanup', 'rescue'):
                raise ValueError('Action must be cleanup or rescue')
            if event.get('decision_revision') != record['decision_revision']:
                raise ValueError('Plan must reference the current decision revision')
            if action == 'cleanup' and record['decision'] != 'REMOVE':
                raise ValueError('Cleanup requires a passing REMOVE decision')
            if action == 'rescue' and record['decision'] not in ('KEEP', 'REVIEW'):
                raise ValueError('Correct the decision before planning rescue')
            destination = event.get('destination_folder')
            if not destination or destination == record['message']['parentFolderId']:
                raise ValueError('Resolve a different destination folder ID')
            if event.get('source_id') != record['message']['id'] or event.get('source_folder') != record['message']['parentFolderId']:
                raise ValueError('Plan references stale message ID or source folder')
            if type(record['message'].get('isRead')) is not bool:
                raise ValueError('Capture the source read state before a move')
            if not event.get('membership_evidence') or event.get('authorized') is not True:
                raise ValueError('Verify current source membership and existing authorization')
            if event.get('plan_schema') == 2:
                validate_graph_plan(event, record)
            operations[op] = dict(record_id=rid, action=action, state='planned',
                                  source_id=event['source_id'], source_folder=event['source_folder'],
                                  destination_folder=destination, plan_schema=event.get('plan_schema', 1),
                                  plan=copy.deepcopy(event))
            record['pending_operation'] = op
        elif kind in ('submitted', 'unknown', 'failed', 'confirmed', 'cancelled'):
            op = event.get('operation_id')
            if op not in operations or operations[op]['record_id'] != rid or record['pending_operation'] != op:
                raise ValueError('No matching pending operation')
            operation = operations[op]
            if kind == 'cancelled':
                if operation['state'] != 'planned' or not event.get('reason'):
                    raise ValueError('Only unsubmitted plans can be cancelled with a reason')
                operation['state'] = 'cancelled'
                record['pending_operation'] = None
            elif kind == 'submitted':
                if operation['state'] != 'planned':
                    raise ValueError('Do not submit an operation twice')
                if operation['plan_schema'] == 2:
                    execution = event.get('execution', {})
                    if (execution.get('argv') != operation['plan']['argv']
                            or execution.get('cwd') != operation['plan']['runtime']['cwd']
                            or not execution.get('invocation_path') or not execution.get('invocation_sha256')):
                        raise ValueError('Submission must reference the actual recorded plan argv/cwd')
                    operation['execution'] = copy.deepcopy(execution)
                operation['state'] = 'submitted'
            elif kind == 'unknown':
                operation['state'] = 'unknown'
            elif kind == 'failed':
                if event.get('no_effect_verified') is not True or not event.get('evidence'):
                    raise ValueError('A failure needs verified no-effect evidence; otherwise mark unknown')
                operation['state'] = 'failed'
                record['pending_operation'] = None
            else:
                # Verification may discover a completed operation after a crash
                # even if only its plan was durably saved.
                msg = event.get('destination_message')
                if not isinstance(msg, dict) or not isinstance(msg.get('id'), str) or not msg['id']:
                    raise ValueError('Confirmation requires the destination message ID')
                if msg.get('parentFolderId') != operation['destination_folder']:
                    raise ValueError('Destination folder does not match plan')
                if event.get('source_absent') is not True or event.get('match_unique') is not True or not event.get('evidence'):
                    raise ValueError('Require source absence and uniquely matched destination evidence')
                old = record['message']
                if type(msg.get('isRead')) is not bool or msg['isRead'] != old['isRead']:
                    raise ValueError('Read-state preservation is unknown or changed; investigate before confirmation')
                for field in ('internetMessageId', 'receivedDateTime', 'subject'):
                    if old.get(field) is not None and msg.get(field) != old[field]:
                        raise ValueError('Destination identity evidence does not match')
                record['message'] = copy.deepcopy(msg)
                record['pending_operation'] = None
                operation['state'] = 'confirmed'
        else:
            raise ValueError('Unknown journal event type')
    return records, operations


def summary(events):
    records, operations = replay(events)
    decisions = Counter(r['decision'] for r in records.values())
    confirmed = [o for o in operations.values() if o['state'] == 'confirmed']
    return {'unique_messages': len(records), 'decisions': dict(decisions),
            'confirmed_cleanup_moves': sum(o['action'] == 'cleanup' for o in confirmed),
            'confirmed_rescue_moves': sum(o['action'] == 'rescue' for o in confirmed),
            'pending_or_unknown_operations': sum(o['state'] in ('planned', 'submitted', 'unknown') for o in operations.values()),
            'current_folder_counts': dict(Counter(r['message']['parentFolderId'] for r in records.values())),
            'note': 'Folder counts describe the last verified journal state, not a fresh mailbox snapshot'}


@contextmanager
def locked(path):
    path = Path(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump({'pid': os.getpid()}, stream)
        yield
    finally:
        path.unlink()


def append_event(path, event):
    path = Path(path)
    with locked(path.with_name(path.name+'.lock')):
        events = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        if not isinstance(events, list):
            raise ValueError('Journal must contain an event array')
        candidate = events + [event]
        replay(candidate)
        is_new = not any(e['event_id'] == event['event_id'] for e in events)
        if is_new and event.get('type') in ('plan_move', 'submitted'):
            records, operations = replay(events)
            if event['type'] == 'plan_move':
                validate_graph_plan(event, records[event['record_id']], check_files=True)
            else:
                operation = operations[event['operation_id']]
                if operation['plan_schema'] != 2:
                    raise ValueError('Legacy plans may be reconciled but not newly submitted')
                execution = event.get('execution', {})
                raw = Path(execution['invocation_path']).read_bytes()
                invocation = json.loads(raw)
                if (digest(raw) != execution.get('invocation_sha256')
                        or invocation.get('argv') != operation['plan']['argv']
                        or invocation.get('cwd') != operation['plan']['runtime']['cwd']
                        or invocation.get('shell') is not False):
                    raise ValueError('Submission capture does not match plan')
        if is_new:
            atomic_write(path, candidate)
        return summary(candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--event', type=Path, help='Append this JSON event; omit to print a derived summary')
    args = parser.parse_args()
    if args.event:
        result = append_event(args.journal, json.loads(args.event.read_text(encoding='utf-8')))
    else:
        result = summary(json.loads(args.journal.read_text(encoding='utf-8')))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
