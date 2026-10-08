"""Durable media retirement candidates survive transcript deletion and failed cleanup."""
import hashlib
import json

PREFIX = 'gateway.media.retired.v1.'


def retire_media(conn, payload):
    references = [*payload.get('attachments_v1', {}).get('media', []),
                  *payload.get('native_text_v1', {}).get('media', []),
                  *payload.get('api_turn_v1', {}).get('media', [])]
    for reference in references:
        key = PREFIX + hashlib.sha256(reference['path'].encode()).hexdigest()
        conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                     (key, json.dumps(reference)))


def collect_retired_media(db):
    from gateway.session_ingress_media import release_unheld_media
    with db._read_ctx() as conn:
        rows = conn.execute('SELECT key,value FROM state_meta WHERE key LIKE ?', (PREFIX + '%',)).fetchall()
    if rows:
        release_unheld_media(db, [json.loads(row['value']) for row in rows])
        from pathlib import Path
        gone = [(row['key'],) for row in rows if not Path(json.loads(row['value'])['path']).exists()]
        if gone:
            db._execute_write(lambda conn: conn.executemany('DELETE FROM state_meta WHERE key=?', gone))
