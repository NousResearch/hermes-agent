"""Operator and agent CLI for the profile's interactive task delivery ledger."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hermes_cli.task_delivery.store import DeliveryError, WorkLedger, fingerprint


def _read_json(path):
    file = Path(path)
    if file.stat().st_size > 256 * 1024:
        raise DeliveryError('Report exceeds 256 KiB')
    return json.loads(file.read_text(encoding='utf-8-sig'))


def _evidence(args):
    report = _read_json(args.evidence_file)
    if not isinstance(report, dict) or report.get('candidate_digest') != args.candidate:
        raise DeliveryError('Evidence must identify the current candidate_digest')
    if args.stage in {'verification', 'acceptance'} and report.get('passed') is not True:
        raise DeliveryError('Failed or unknown verification/acceptance cannot advance delivery')
    required = {'integration': 'revision', 'verification': 'command', 'acceptance': 'scenario'}[args.stage]
    if not isinstance(report.get(required), str) or not report[required].strip():
        raise DeliveryError(f'{args.stage} evidence must describe {required}')
    if args.stage == 'verification' and (type(report.get('exit_code')) is not int or report['exit_code'] != 0):
        raise DeliveryError('Verification must record exit_code=0')
    return json.dumps({'artifact': fingerprint(args.evidence_file), 'report': report}, ensure_ascii=False)


def _inbox(db, args):
    from hermes_state_runtime import get_session_admission
    row = get_session_admission(db, admission_id=args.admission)
    if row is None or row['target_session_id'] != args.session:
        raise DeliveryError('Admission not found in this session/profile')
    with db._read_ctx() as conn:
        record = conn.execute('SELECT value FROM state_meta WHERE key=?',
                              ('gateway.completion_exposure.v1:' + args.admission,)).fetchone()
    return {'admission_id': args.admission, 'untrusted_payload': row['payload'],
            'exposure': json.loads(record[0]) if record else None, 'accepted': False}


def _context(work, args):
    item = work.get(args.session, args.task)
    ref = {'session_id': args.session, 'task_id': args.task, 'role': args.role,
           'base': args.base, 'files': args.files}
    return ('[HERMES_WORK_ITEM_V1]\n' + json.dumps(ref, ensure_ascii=False)
            + '\n\nAcceptance: ' + item['acceptance_criteria'])


def execute(args, db):
    work = WorkLedger(db)
    operations = {
        'init': lambda: work.register(args.session, args.task, title=args.title, acceptance=args.acceptance),
        'list': lambda: work.projection(args.session),
        'show': lambda: work.get(args.session, args.task),
        'claim': lambda: work.claim(args.session, args.task, owner=args.owner, role=args.role,
            request_id=args.request_id, base=args.base, files=args.files, lease_seconds=args.lease_seconds),
        'renew': lambda: work.renew(args.session, args.task, owner=args.owner, generation=args.generation,
            lease_seconds=args.lease_seconds),
        'release': lambda: work.release(args.session, args.task, owner=args.owner, generation=args.generation,
            reason=args.reason),
        'deliver': lambda: work.deliver(args.session, args.task, owner=args.owner, generation=args.generation,
            path=args.file, summary=args.summary),
        'review': lambda: work.record_review(args.session, args.task, _read_json(args.file),
            owner=args.owner, generation=args.generation),
        'advance': lambda: work.advance(args.session, args.task, stage=args.stage,
            candidate_digest=args.candidate, evidence=_evidence(args)),
        'reopen': lambda: work.reopen(args.session, args.task, reason=args.reason),
        'decision': lambda: work.decision(args.session, args.task, reason=args.reason, acceptance=args.acceptance),
        'context': lambda: _context(work, args),
        'inbox': lambda: _inbox(db, args),
    }
    return operations[args.action]()


def run(args):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    try:
        with SessionDB(db_path=get_hermes_home() / 'state.db') as db:
            result = execute(args, db)
        print(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False))
        return 0
    except (DeliveryError, OSError, ValueError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 2


def configure(parser):
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ('init', 'list', 'show', 'claim', 'renew', 'release', 'deliver', 'review', 'advance', 'reopen', 'decision', 'context', 'inbox'):
        sub = actions.add_parser(name)
        sub.add_argument('--session', required=True, help='Owning logical session in this profile')
        if name not in {'list', 'inbox'}:
            sub.add_argument('--task', required=True, help='Stable work-item ID')
        _arguments(sub, name)
    parser.set_defaults(func=run)
    return parser


def _arguments(parser, action):
    specs = {
        'init': [('title', {}), ('acceptance', {})],
        'claim': [('owner', {}), ('request-id', {}), ('lease-seconds', {'type': int, 'default': 1800})],
        'renew': [('owner', {}), ('generation', {'type': int}), ('lease-seconds', {'type': int, 'default': 1800})],
        'release': [('owner', {}), ('generation', {'type': int}), ('reason', {})],
        'deliver': [('owner', {}), ('generation', {'type': int}), ('file', {}), ('summary', {})],
        'review': [('file', {}), ('owner', {'default': None}), ('generation', {'type': int, 'default': None})],
        'advance': [('stage', {'choices': ['integration', 'verification', 'acceptance']}),
                    ('candidate', {}), ('evidence-file', {})],
        'reopen': [('reason', {})], 'decision': [('reason', {}), ('acceptance', {})], 'inbox': [('admission', {})],
    }
    for flag, options in specs.get(action, []):
        parser.add_argument('--' + flag, required='default' not in options, **options)
    if action in {'claim', 'context'}:
        parser.add_argument('--role', choices=['implement', 'review'], default='implement')
        parser.add_argument('--base', default='')
        parser.add_argument('--files', nargs='*', default=[])


def main(argv=None):
    return run(configure(argparse.ArgumentParser(prog='hermes work')).parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
