"""Preconstructed FK metadata only; no runtime turns or result acceptance."""
import json


def ledger(conn, sid, kind):
    if kind.startswith('admission'):
        status = 'terminal' if kind.endswith('terminal') else 'queued'
        conn.execute('''INSERT INTO session_admissions(admission_id,request_id,principal_id,
            target_session_id,lineage_json,payload_json,payload_digest,intent,status,owner_epoch,outcome)
            VALUES(?,?,?,?,?,'{}',?,'queue',?,1,?)''',
            ('a-' + sid, 'r-' + sid, 'fixture', sid, json.dumps([sid]), 'a' * 64, status,
             'rejected' if status == 'terminal' else None))
    else:
        status = 'terminal' if kind.endswith('terminal') else 'registered'
        conn.execute('''INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,
            generation,status,adoption_digest) VALUES(?,?,'compute',1,0,?,?)''',
            ('w-' + sid, sid, status, 'b' * 64))


def snapshot(db):
    return {table: [dict(row) for row in db._read_all('SELECT * FROM ' + table)]
            for table in ('sessions', 'messages', 'session_admissions', 'worker_executions',
                          'worker_receipts', 'state_meta', 'gateway_routing')}


def ended(db, sid, *, parent=None, delegate=False):
    db.create_session(sid, source='tui', parent_session_id=parent,
                      model_config={'_delegate_from': parent} if delegate else None)
    db.end_session(sid, 'complete')
    db._execute_write(lambda conn: conn.execute(
        'UPDATE sessions SET started_at=1,ended_at=2,last_activity_at=1 WHERE id=?', (sid,)))
