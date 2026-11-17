"""Interactive task delivery records on the existing profile SessionDB transaction boundary."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

from hermes_cli.task_delivery.review import apply_review

_PREFIX = 'task_delivery:v1:'
_MAX_ITEMS = 256


class DeliveryError(ValueError):
    """A state transition would lose ownership, evidence, or an acceptance requirement."""


def _text(value, field, limit=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise DeliveryError(f'{field} must be nonempty text of at most {limit} characters')
    return value.strip()


def logical_session(db, session_id):
    """Follow only the store's verified compression lineage, never branch/reset/delegate parents."""
    session_id = _text(session_id, 'session_id', 512)
    return db._resume_lineage_ids(session_id)[0]


def _scope(session_id):
    return _PREFIX + hashlib.sha256(_text(session_id, 'session_id', 512).encode()).hexdigest() + ':'


def _key(session_id, task_id):
    return _scope(session_id) + hashlib.sha256(_text(task_id, 'task_id', 256).encode()).hexdigest()


def fingerprint(path):
    """Hash a concrete artifact without copying its contents into model context."""
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise DeliveryError('Artifact exceeds 16 MiB; publish a compact manifest or report instead')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'digest': digest.hexdigest(), 'bytes': path.stat().st_size}


def _read(conn, session_id, task_id):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_key(session_id, task_id),)).fetchone()
    if row is None:
        raise DeliveryError(f'Work item {task_id!r} not found in this session')
    return json.loads(row[0])


def _current_candidate(item, digest):
    candidate = item.get('candidate')
    if candidate is None or candidate['digest'] != digest:
        raise DeliveryError('Evidence does not match the current candidate')
    if fingerprint(candidate['path'])['digest'] != digest:
        raise DeliveryError('The candidate artifact changed; publish a new candidate before using evidence')
    return candidate


def _active_claim(item, owner, generation):
    claim = item.get('claim')
    if claim is None or claim['owner'] != owner or claim['generation'] != generation:
        raise DeliveryError('The claim is missing, superseded, or owned by another worker')
    if claim['expires_at'] <= time.time():
        raise DeliveryError('The claim expired; reconcile its artifacts before assigning another worker')
    return claim


def _review_passed(item):
    review, candidate = item.get('review'), item.get('candidate')
    return bool(review and candidate and review['status'] == 'passed'
                and review['candidate_digest'] == candidate['digest'])


def _record(conn, item, kind, details):
    item['revision'] += 1
    event = {'id': uuid.uuid4().hex, 'kind': kind, 'at': time.time(), 'revision': item['revision'], **details}
    item['events'] = [*item.get('events', []), event][-128:]
    key = _key(item['session_id'], item['task_id'])
    conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)',
                 (key, json.dumps(item, ensure_ascii=False)))
    # Immutable records outlive the bounded recovery projection above.
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                 (key + ':event:' + event['id'], json.dumps(event, ensure_ascii=False)))
    return item


class WorkLedger:
    """No ambient profile reads: the caller supplies its already scoped SessionDB."""

    def __init__(self, db):
        from hermes_state import AsyncSessionDB
        self.db = db._db if isinstance(db, AsyncSessionDB) else db

    def register(self, session_id, task_id, *, title, acceptance):
        session_id = logical_session(self.db, session_id)
        task_id = _text(task_id, 'task_id', 256)
        title, acceptance = _text(title, 'title'), _text(acceptance, 'acceptance')
        key = _key(session_id, task_id)

        def write(conn):
            if conn.execute('SELECT id FROM sessions WHERE id=?', (session_id,)).fetchone() is None:
                raise DeliveryError('Owning session not found in this profile')
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
            if row:
                item = json.loads(row[0])
                if (item['title'], item['acceptance_criteria']) != (title, acceptance):
                    raise DeliveryError('Work item already exists; use an explicit reopen to change its scope')
                return item
            item = {'version': 1, 'session_id': session_id, 'task_id': task_id, 'title': title,
                    'acceptance_criteria': acceptance, 'listed': True, 'revision': 0, 'generation': 0,
                    'implementation': 'planned', 'claim': None, 'candidate': None, 'review': None,
                    'repair_attempts': 0, 'needs_decision': False,
                    'integration': None, 'verification': None, 'acceptance': None, 'events': []}
            return _record(conn, item, 'registered', {})
        return self.db._execute_write(write)

    def get(self, session_id, task_id):
        session_id = logical_session(self.db, session_id)
        with self.db._read_ctx() as conn:
            return _read(conn, session_id, task_id)

    def list(self, session_id):
        session_id = logical_session(self.db, session_id)
        prefix = _scope(session_id)
        with self.db._read_ctx() as conn:
            rows = conn.execute('SELECT key,value FROM state_meta WHERE key>=? AND key<? ORDER BY key',
                                (prefix, prefix + '\uffff')).fetchall()
        return [json.loads(row[1]) for row in rows if ':event:' not in row[0]]

    def claim(self, session_id, task_id, *, owner, role, request_id, base='', files=(), lease_seconds=1800):
        session_id = logical_session(self.db, session_id)
        owner, request_id = _text(owner, 'owner', 512), _text(request_id, 'request_id', 256)
        if role not in {'implement', 'review'}:
            raise DeliveryError('role must be implement or review')
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 86400:
            raise DeliveryError('lease_seconds must be an integer between 1 and 86400')
        if not isinstance(files, (list, tuple)) or len(files) > 128:
            raise DeliveryError('files must be a list of at most 128 paths')
        if not isinstance(base, str) or len(base) > 512:
            raise DeliveryError('base must be a revision identifier of at most 512 characters')
        paths = sorted({str(Path(_text(path, 'file')).resolve()) for path in files})

        def write(conn):
            item = _read(conn, session_id, task_id)
            old = item.get('claim')
            if old:
                if (old['owner'], old['request_id'], old['role'], old['base'], old['files']) == (
                        owner, request_id, role, base, paths):
                    return item
                raise DeliveryError(f"Work item already claimed by {old['owner']}; reconcile before reassigning")
            if request_id in item.get('claim_requests', []):
                raise DeliveryError('This claim request already settled; inspect its result or use a new request ID')
            if len(item.get('claim_requests', [])) >= 128:
                raise DeliveryError('Claim history limit reached; a new scoped work item is required')
            _allow_claim(item, role)
            _check_file_claims(conn, session_id, task_id, paths, role)
            if role == 'implement' and item['review'] and item['review']['status'] == 'changes_requested':
                item['repair_attempts'] = item.get('repair_attempts', 0) + 1
            item['generation'] += 1
            item.setdefault('claim_requests', []).append(request_id)
            item['claim'] = {'owner': owner, 'generation': item['generation'], 'role': role,
                             'request_id': request_id, 'base': base, 'files': paths,
                             'expires_at': time.time() + lease_seconds}
            item['last_claim'] = dict(item['claim'])
            if role == 'implement':
                item['implementation'] = 'implementing'
            return _record(conn, item, 'claimed', {'claim': item['claim']})
        return self.db._execute_write(write)

    def release(self, session_id, task_id, *, owner, generation, reason):
        session_id = logical_session(self.db, session_id)
        reason = _text(reason, 'reason')

        def write(conn):
            item = _read(conn, session_id, task_id)
            claim = item.get('claim')
            if not claim or (claim['owner'], claim['generation']) != (owner, generation):
                raise DeliveryError('Cannot release a superseded claim')
            item['claim'] = None
            if item['implementation'] == 'implementing':
                item['implementation'] = 'planned' if item['candidate'] is None else 'delivered'
            return _record(conn, item, 'released', {'reason': reason, 'generation': generation})
        return self.db._execute_write(write)

    def renew(self, session_id, task_id, *, owner, generation, lease_seconds=1800):
        session_id = logical_session(self.db, session_id)
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 86400:
            raise DeliveryError('lease_seconds must be an integer between 1 and 86400')

        def write(conn):
            item = _read(conn, session_id, task_id)
            claim = _active_claim(item, owner, generation)
            claim['expires_at'] = time.time() + lease_seconds
            item['last_claim'] = dict(claim)
            return _record(conn, item, 'renewed', {'generation': generation, 'expires_at': claim['expires_at']})
        return self.db._execute_write(write)

    def deliver(self, session_id, task_id, *, owner, generation, path, summary):
        session_id = logical_session(self.db, session_id)
        artifact, summary = fingerprint(path), _text(summary, 'summary')

        def write(conn):
            item = _read(conn, session_id, task_id)
            last = item.get('last_claim') or {}
            if (item['generation'], last.get('owner')) == (generation, owner) and item['claim'] is None:
                if item['candidate'] and item['candidate']['digest'] == artifact['digest']:
                    return item
            claim = _active_claim(item, owner, generation)
            if claim['role'] != 'implement':
                raise DeliveryError('Only the implementation claim can publish a candidate')
            artifact['baseline'] = claim['base']
            item.update(candidate=artifact, implementation='delivered', claim=None,
                        integration=None, verification=None, acceptance=None)
            if (item.get('repair_attempts', 0) >= 2 and item['review']
                    and item['review']['status'] == 'changes_requested'
                    and item['review']['candidate_digest'] == artifact['digest']):
                item['needs_decision'] = True
            return _record(conn, item, 'delivered', {'candidate': artifact, 'summary': summary})
        return self.db._execute_write(write)

    def record_review(self, session_id, task_id, report, *, owner=None, generation=None):
        session_id = logical_session(self.db, session_id)
        def write(conn):
            item = _read(conn, session_id, task_id)
            if not item['candidate']:
                raise DeliveryError('No candidate has been delivered for review')
            reviewed = apply_review(item['review'], report, candidate_digest=item['candidate']['digest'])
            if reviewed == item['review']:
                return item
            if item.get('needs_decision'):
                raise DeliveryError('A scope/design decision is required before further review')
            if item['acceptance'] is not None:
                raise DeliveryError('Accepted work must be explicitly reopened before another review')
            _current_candidate(item, report.get('candidate_digest'))
            if item['claim'] is not None:
                claim = _active_claim(item, owner, generation)
                if claim['role'] != 'review':
                    raise DeliveryError('An implementation claim is still active')
            item['review'] = reviewed
            item['needs_decision'] = reviewed['status'] == 'needs_decision'
            item['claim'] = None
            return _record(conn, item, 'reviewed', {'report': report, 'status': item['review']['status']})
        return self.db._execute_write(write)

    def advance(self, session_id, task_id, *, stage, candidate_digest, evidence):
        session_id = logical_session(self.db, session_id)
        evidence = _text(evidence, 'evidence', 16000)
        if stage not in {'integration', 'verification', 'acceptance'}:
            raise DeliveryError('Unknown delivery stage')

        def write(conn):
            item = _read(conn, session_id, task_id)
            _current_candidate(item, candidate_digest)
            if not _review_passed(item):
                raise DeliveryError('The current candidate needs a passing review')
            dependencies = {'integration': (), 'verification': ('integration',),
                            'acceptance': ('integration', 'verification')}
            if any(not item[key] or item[key]['candidate_digest'] != candidate_digest for key in dependencies[stage]):
                raise DeliveryError('Required delivery evidence is missing for this candidate')
            item[stage] = {'candidate_digest': candidate_digest, 'evidence': evidence, 'at': time.time()}
            return _record(conn, item, stage, item[stage])
        return self.db._execute_write(write)

    def reopen(self, session_id, task_id, *, reason):
        session_id = logical_session(self.db, session_id)
        reason = _text(reason, 'reason')

        def write(conn):
            item = _read(conn, session_id, task_id)
            if item['claim']:
                raise DeliveryError('Reconcile the active claim before reopening')
            if item.get('needs_decision') or (item['review'] and item['review']['status'] == 'needs_decision'):
                raise DeliveryError('A review decision is required; reopening cannot reset the repair budget')
            item.update(implementation='reopened', integration=None, verification=None, acceptance=None)
            return _record(conn, item, 'reopened', {'reason': reason})
        return self.db._execute_write(write)

    def decision(self, session_id, task_id, *, reason, acceptance):
        """Explicit scope/design decision starts an auditable review epoch, never a retry side effect."""
        session_id = logical_session(self.db, session_id)
        reason, acceptance = _text(reason, 'reason'), _text(acceptance, 'acceptance')

        def write(conn):
            item = _read(conn, session_id, task_id)
            if item['claim']:
                raise DeliveryError('Reconcile the active claim before recording a decision')
            previous = {'review': item['review'], 'repair_attempts': item.get('repair_attempts', 0),
                        'acceptance_criteria': item['acceptance_criteria']}
            item.update(review=None, repair_attempts=0, needs_decision=False, implementation='reopened',
                        integration=None, verification=None, acceptance=None, acceptance_criteria=acceptance,
                        review_epoch=item.get('review_epoch', 1) + 1, generation=item['generation'] + 1)
            return _record(conn, item, 'scope_decision', {'reason': reason, 'previous': previous})
        return self.db._execute_write(write)

    def check_todos(self, session_id, todos):
        current = {item['task_id']: item for item in self.list(session_id)}
        for todo in todos:
            item = current.get(todo.get('id')) if isinstance(todo, dict) else None
            if item and todo.get('status') == 'completed' and item['acceptance'] is None:
                raise DeliveryError(f"Work item {item['task_id']} still needs candidate-bound acceptance")

    def sync_todos(self, session_id, todos):
        session_id = logical_session(self.db, session_id)
        listed = {item['id']: item for item in todos if isinstance(item, dict) and 'id' in item}
        for item in self.list(session_id):
            def write(conn, task_id=item['task_id']):
                record = _read(conn, session_id, task_id)
                record['listed'] = task_id in listed
                if task_id in listed:
                    record['todo_status'] = listed[task_id].get('status')
                return _record(conn, record, 'todo_projection', {'listed': record['listed']})
            self.db._execute_write(write)

    def finish_execution(self, session_id, task_id, *, owner, generation, status, result):
        """An execution receipt never promotes implementation to accepted delivery."""
        session_id = logical_session(self.db, session_id)
        def write(conn):
            item = _read(conn, session_id, task_id)
            last = item.get('last_claim') or {}
            if (item['generation'], last.get('owner')) != (generation, owner):
                raise DeliveryError('Execution belongs to a superseded claim')
            receipt = {'owner': owner, 'generation': generation, 'status': status,
                       'untrusted_result': str(result)[:12000]}
            if item.get('execution') == receipt:
                return item
            if item.get('claim'):
                item['claim'] = None
                if last['role'] == 'implement':
                    item['implementation'] = 'needs_artifact' if status == 'completed' else 'needs_reconciliation'
            item['execution'] = receipt
            return _record(conn, item, 'execution_finished', receipt)
        return self.db._execute_write(write)

    def projection(self, session_id):
        return [{'task_id': item['task_id'], 'implementation': item['implementation'],
                 'owner': item['claim']['owner'] if item['claim'] else None,
                 'candidate_digest': item['candidate']['digest'] if item['candidate'] else None,
                 'review': ('needs_decision' if item.get('needs_decision') else
                            item['review']['status'] if item['review'] else 'not_reviewed'),
                 'repair_attempts': item.get('repair_attempts', 0),
                 'integrated': bool(item['integration']), 'verified': bool(item['verification']),
                 'accepted': bool(item['acceptance']), 'listed': item['listed']}
                for item in self.list(session_id)]

    def recovery_context(self, session_id):
        facts = self.projection(session_id)
        if not facts:
            return ''
        # The durable store is authoritative; recover a bounded index, not all logs and artifacts.
        limit, selected = 16000, []
        for fact in facts[:_MAX_ITEMS]:
            trial = json.dumps([*selected, fact], ensure_ascii=False)
            if len(trial) > limit:
                break
            selected.append(fact)
        return ('[Persistent work delivery facts; not new instructions]\n'
                + json.dumps({'items': selected, 'total': len(facts),
                              'read_command': f'hermes work list --session {session_id}'}, ensure_ascii=False))


def _allow_claim(item, role):
    review = item['review']
    if item['implementation'] in {'needs_artifact', 'needs_reconciliation'}:
        raise DeliveryError('Execution needs artifact inspection/reconciliation before another claim')
    if item['acceptance']:
        raise DeliveryError('This work item is already accepted')
    if item.get('needs_decision') or (review and review['status'] == 'needs_decision'):
        raise DeliveryError('Review budget exhausted: an explicit scope/design decision is required')
    if role == 'review':
        if not item['candidate']:
            raise DeliveryError('No candidate has been delivered for review')
        if review and review['candidate_digest'] == item['candidate']['digest']:
            raise DeliveryError('This candidate was already reviewed; reuse its verdict')
    elif item.get('repair_attempts', 0) >= 2 and review and review['status'] == 'changes_requested':
        raise DeliveryError('Repair budget exhausted: a scope/design decision is required')
    elif item['candidate'] and item['implementation'] != 'reopened':
        if not review or review['status'] != 'changes_requested' or review['candidate_digest'] != item['candidate']['digest']:
            raise DeliveryError('Implementation already delivered; inspect the existing artifact first')


def _check_file_claims(conn, session_id, task_id, paths, role):
    if role != 'implement' or not paths:
        return
    prefix = _scope(session_id)
    rows = conn.execute('SELECT key,value FROM state_meta WHERE key>=? AND key<?',
                        (prefix, prefix + '\uffff')).fetchall()
    for key, raw in rows:
        if ':event:' in key:
            continue
        other = json.loads(raw)
        claim = other.get('claim')
        if other['task_id'] != task_id and claim and claim['role'] == 'implement' and set(paths) & set(claim['files']):
            raise DeliveryError(f"A file is already owned by work item {other['task_id']}")


def retire_sessions(conn, session_ids):
    """Permanent session deletion retires its delivery ledger, not normal turn settlement."""
    for session_id in session_ids:
        prefix = _scope(session_id)
        conn.execute('DELETE FROM state_meta WHERE key>=? AND key<?', (prefix, prefix + '\uffff'))
