"""Trusted lineage and bounded final reports for background work started by a room turn."""
import json
from contextlib import closing
from pathlib import Path
from dataclasses import asdict

from gateway.hosted_room_driver import TaskIdentity
from gateway.owner_identity import same_owner
from hermes_state_runtime import get_session_admission

COMPLETION_KINDS = frozenset({'async_delegation', 'completion'})


def room_origin(authority, session_id, row):
    """Follow producer-captured admission ids, never the most recent room turn or model text."""
    seen = set()
    for _ in range(32):
        if not isinstance(row, dict) or row.get('target_session_id') != session_id:
            return None
        if row['admission_id'] in seen:
            return None
        seen.add(row['admission_id'])
        if row['request_id'].startswith('hosted:'):
            try:
                task, generation = json.loads(row['request_id'][7:])
                task = TaskIdentity(**task)
                if type(generation) is not int or generation < 1:
                    return None
            except (ValueError, TypeError, KeyError):
                return None
            return row, task, generation
        descriptor = row['payload'].get('local_automation_v1') or {}
        try:
            kind = json.loads(descriptor.get('identity', 'null'))[0]
        except (ValueError, TypeError, IndexError):
            return None
        if (kind not in COMPLETION_KINDS or descriptor.get('owner') != session_id
                or descriptor.get('identity') != row['request_id']
                or row['principal_id'] != 'automation:' + str(descriptor.get('route', ''))):
            return None
        origin_id = descriptor.get('origin_admission_id')
        if not isinstance(origin_id, str) or not origin_id:
            return None
        row = get_session_admission(authority.db, admission_id=origin_id)
    return None


def completion_page(rpc, after_seq):
    """Only parent final responses are exported; raw child/process payloads stay private."""
    from hermes_state_runtime import RuntimeStoreError
    from gateway.session_results import admission_result
    if type(after_seq) is not int or after_seq < 0:
        raise RuntimeStoreError('invalid_params')
    rpc.authority.authorize(rpc.principal, rpc.ref, 'session:read')
    with rpc.authority.db._read_ctx() as conn:
        ids = conn.execute('SELECT admission_id FROM session_admissions WHERE target_session_id=? AND seq>? '
                           'ORDER BY seq LIMIT 64', (rpc.ref.session_id, after_seq)).fetchall()
    cursor, reports, used = after_seq, [], 0
    for (admission_id,) in ids:
        row = get_session_admission(rpc.authority.db, admission_id=admission_id)
        if row is None:
            break
        if row['status'] != 'terminal':
            break
        previous_cursor, cursor = cursor, row['seq']
        if 'local_automation_v1' not in row['payload']:
            continue
        origin = room_origin(rpc.authority, rpc.ref.session_id, row)
        if origin is None:
            continue
        parent, task, generation = origin
        if task.room_id != rpc.room_id or not same_owner(parent['principal_id'], rpc.principal.subject):
            continue
        saved = admission_result(rpc.authority.db, admission_id)
        text = ((saved or {}).get('result') or {}).get('final_response', '')
        from gateway.response_filters import is_intentional_silence_response
        if (row['outcome'] != 'completed' or not text or is_intentional_silence_response(text)
                or text.strip() == '(pass)'):
            continue
        from tui_gateway.hosted_room_driver import MAX_TERMINAL_TEXT_BYTES, _truncate_utf8
        text, _ = _truncate_utf8(text, max_bytes=MAX_TERMINAL_TEXT_BYTES)
        report = {'admission_id': admission_id, 'task': asdict(task),
                  'execution_generation': generation, 'text': text}
        size = len(json.dumps(report).encode('utf-8'))
        if reports and used + size > 384 * 1024:
            cursor = previous_cursor
            break
        reports.append(report)
        used += size
    return {'cursor': cursor, 'reports': reports}


def _current_report(conn, room_id, selector, report):
    """Validate under the room writer: Stop, membership and publication share one snapshot."""
    from gateway import hosted_rooms
    task = TaskIdentity(**report['task'])
    room = conn.execute('SELECT * FROM hosted_rooms WHERE room_id=?', (room_id,)).fetchone()
    if room is None or room['disbanded_at'] is not None or task.room_id != room_id:
        return None
    members = json.loads(room['members_json'])
    member = next((m for m in members if m['member_id'] == selector['member_id']
                   and m['profile'] == selector['profile']
                   and (m.get('target') or {}).get('kind', 'local') == 'local'), None)
    stored = conn.execute('SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?',
                          (room_id, task.task_id)).fetchone()
    if stored is None or member is None:
        return None
    payload = json.loads(stored['payload_json'])
    if (stored['thread_id'] != task.thread_id or stored['turn_id'] != task.turn_id
            or stored['execution_generation'] != report['execution_generation']
            or stored['status'] not in {'running', 'settled'}
            or payload.get('target_member_id', payload['target_profile']) != selector['member_id']
            or payload['target_profile'] != selector['profile']):
        return None
    source = conn.execute('SELECT authority_epoch FROM hosted_room_events WHERE room_id=? AND seq=?',
                          (room_id, stored['source_event_seq'])).fetchone()
    if source is None or source['authority_epoch'] != room['authority_epoch']:
        return None
    if conn.execute("SELECT 1 FROM hosted_room_events WHERE room_id=? AND seq>? AND "
                    "(kind='room.stop_requested' OR (kind='message.user' AND "
                    "json_extract(actor_json,'$.kind')='user' AND json_extract(payload_json,'$.thread_id')=?)) LIMIT 1",
                    (room_id, stored['source_event_seq'], task.thread_id)).fetchone():
        return None
    if room['authority_gateway_id'] != hosted_rooms.local_authority_gateway_id():
        return None
    return room, member


class _ReportRevoked(Exception):
    pass


class _ReportPending(Exception):
    """The durable report waits for the prior discussion, without advancing its read cursor."""


def _require_idle_thread(conn, room_id, thread_id):
    source = conn.execute("SELECT event_id,seq FROM hosted_room_events WHERE room_id=? AND kind='message.user' "
        "AND json_extract(payload_json,'$.thread_id')=? ORDER BY seq DESC LIMIT 1", (room_id, thread_id)).fetchone()
    if source is None:
        raise _ReportPending()
    settled = conn.execute("SELECT MAX(seq) FROM hosted_room_events WHERE room_id=? AND kind='room.activity' "
        "AND json_extract(payload_json,'$.discussion_event_id')=? "
        "AND json_extract(payload_json,'$.status')='settled'", (room_id, source['event_id'])).fetchone()[0]
    if settled is None or conn.execute("SELECT 1 FROM hosted_room_events WHERE room_id=? AND seq>? "
        "AND kind='room.continued' AND json_extract(payload_json,'$.discussion_event_id')=? LIMIT 1",
        (room_id, settled, source['event_id'])).fetchone():
        raise _ReportPending()
    if conn.execute("SELECT 1 FROM hosted_room_driver_tasks WHERE room_id=? AND thread_id=? "
        "AND status IN ('queued','running','stopping','indeterminate','deferred') LIMIT 1",
        (room_id, thread_id)).fetchone():
        raise _ReportPending()


def completion_allowed(authority, ref, row):
    """Reauthorize the room at the parent wake, including a member hosted by another profile."""
    origin = room_origin(authority, ref.session_id, row)
    if origin is None:
        return False
    parent, task, generation = origin
    from gateway.session_hosted_transport import _BINDING, _attest, _principal
    with authority.db._read_ctx() as conn:
        remote = conn.execute('SELECT value FROM state_meta WHERE key=?', (_BINDING + ref.session_id,)).fetchone()
    params = {'task': asdict(task), 'execution_generation': generation}
    if remote is not None:
        binding = json.loads(remote[0])
        if not same_owner(parent['principal_id'], _principal(authority, binding).subject):
            return False
        return _attest(binding, 'completion', params).get('allowed') is True
    service = getattr(authority, 'hosted_room_service', None)
    if service is None or not same_owner(service._owner(task.room_id), parent['principal_id']):
        return False
    from gateway.hosted_room_driver import get_task, TaskConflictError
    try:
        stored = get_task(service.db_path, task)
    except TaskConflictError:
        return False
    selector = {'room_id': task.room_id, 'member_id': stored['payload'].get(
        'target_member_id', stored['payload']['target_profile']), 'profile': stored['payload']['target_profile']}
    if service.profile_homes().get(selector['profile']) != Path(authority.profile_id):
        return False
    with authority.db._read_ctx() as conn:
        return _current_report(conn, task.room_id, selector, params) is not None


def publish_report(service, selector, report):
    """Idempotent member-authored discussion input, routed only by explicit peer mentions."""
    import hashlib
    from gateway import hosted_rooms
    room_id = selector['room_id']
    digest = hashlib.sha256(json.dumps([selector, report['admission_id']], sort_keys=True).encode()).hexdigest()
    event_id = 'completion:' + digest
    with closing(hosted_rooms._read_connection(service.db_path)) as conn:
        existing = hosted_rooms._load_event(conn, room_id, event_id)
        if existing is not None:
            return hosted_rooms._event_from_row(existing, idempotent=True)
        current = _current_report(conn, room_id, selector, report)
    if current is None:
        return None
    room, member = current
    actor = {'kind': 'member', 'id': selector['member_id'], 'profile': selector['profile'],
             **({'display_name': member['display_name']} if member.get('display_name') else {})}

    def guarded(conn):
        if _current_report(conn, room_id, selector, report) is None:
            raise _ReportRevoked()
        _require_idle_thread(conn, room_id, report['task']['thread_id'])
        return {'text': report['text'], 'thread_id': report['task']['thread_id']}
    try:
        return hosted_rooms.append_event(service.db_path, room_id=room_id, event_id=event_id,
            kind='message.user', actor=actor, payload=guarded,
            authority_gateway_id=room['authority_gateway_id'], authority_epoch=room['authority_epoch'])
    except _ReportRevoked:
        return None


def sweep_reports(service, binding):
    """Recover reports from each member's durable ledger, also when Desktop is disconnected."""
    from gateway import hosted_room_driver as tasks
    room_id = binding.room_id
    room = service._room(room_id)
    all_tasks = tasks.list_tasks(service.db_path, room_id=room_id)
    for member in room['members']:
        if (member.get('target') or {}).get('kind', 'local') != 'local':
            continue
        candidates = [t for t in all_tasks if t['payload'].get('target_member_id', t['payload']['target_profile'])
                      == member['member_id'] and t['payload']['target_profile'] == member['profile']]
        if not candidates:
            continue
        latest = max(candidates, key=lambda task: task['payload']['source_event_seq'])
        from hermes_state_runtime import RuntimeStoreError
        try:
            _sweep_member(service, binding, member, latest)
        except (RuntimeStoreError, OSError, TimeoutError) as exc:
            # One unavailable profile must not prevent independent members or normal turns progressing.
            service.runtime._record_error(f"room {room_id} completion {member['member_id']}: {exc}")


def _sweep_member(service, binding, member, latest):
    room_id = binding.room_id
    rpc = service._resolve_member_transport(binding, latest)
    coords = {'profile': member['profile'], 'source': 'bot_room'}
    resolved = rpc.resolve_exact(**coords, title=f'Group: {room_id}')
    if resolved is None:
        return
    key = 'gateway.hosted.completion_cursor.v1:' + json.dumps(
        [room_id, member['member_id'], rpc.ref.profile_id, resolved['session_id']], separators=(',', ':'))
    with service.authority.db._read_ctx() as conn:
        stored = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    cursor = int(stored[0]) if stored else 0
    page = rpc.completions(**coords, session_id=resolved['session_id'], after_seq=cursor)
    selector = {'room_id': room_id, 'member_id': member['member_id'], 'profile': member['profile']}
    for report in page['reports']:
        try:
            publish_report(service, selector, report)
        except _ReportPending:
            return
    if page['cursor'] > cursor:
        service.authority.db._execute_write(lambda conn: conn.execute(
            'INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)', (key, str(page['cursor']))))
