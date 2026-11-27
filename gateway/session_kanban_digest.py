"""Trusted Kanban wake batching and execution-time, read-only card snapshots.

Never infer a board/task from rendered prose. Old admissions without structured
references retain their text; new notices are re-read only after route authorization.
"""
import json
import sqlite3
from pathlib import Path

from gateway.session_persistence import authority_persistence
from persistence.authority.admission_order import is_kanban_input as is_kanban
from persistence.core.errors import AuthorityRefusal


def descriptor(payload):
    return payload.get('local_automation_v1') or payload.get('native_text_v1', {}).get('automation') or {}


def notice_metadata(metadata):
    if 'kanban_notices' not in metadata:
        return {}
    notices = metadata['kanban_notices']
    keys = {'db', 'board', 'task_id', 'platform', 'chat_id', 'thread_id'}
    if (not isinstance(notices, list) or not notices or any(
            not isinstance(n, dict) or set(n) != keys or any(not isinstance(v, str) for v in n.values())
            or not Path(n['db']).is_absolute() for n in notices)):
        raise AuthorityRefusal('invalid_params')
    return {'kanban_notices': [dict(n) for n in notices]}


def fold_event(authority, sid, event, identity):
    from gateway.session_authority import admission_row
    category = event.metadata.get('notification_category', 'result')
    # A live-state refresh replaces the digest text. Never fold a notice that
    # has no card reference (and therefore cannot be rendered by that refresh).
    structured = bool(event.metadata.get('kanban_notices'))
    rows = authority_persistence(authority).admissions.list(session_id=sid)
    waiting = [r for r in map(admission_row, rows)
               if r['status'] == 'queued' and is_kanban(r)
               and r['principal_id'] == 'automation:' + event.metadata['gateway_session_key']
               and descriptor(r['payload']).get('notification_category', 'result') == category
               and bool(descriptor(r['payload']).get('kanban_notices')) == structured]
    if not waiting:
        return []
    identities = {identity}
    notices = list(event.metadata.get('kanban_notices', []))
    texts = []
    for row in waiting:
        old = descriptor(row['payload'])
        identities.update(old.get('identities') or [row['request_id']])
        notices.extend(old.get('kanban_notices', []))
        texts.append(row['payload']['text'])
    # Texts have no wrapping header: incremental folding remains linear.
    event.text = '\n\n'.join([*texts, event.text])
    event.metadata['automation_identities'] = sorted(identities)
    if notices:
        event.metadata['kanban_notices'] = list({(n['db'], n['task_id'], n['platform'], n['chat_id'], n['thread_id']): n
                                               for n in notices}.values())
    return [r['admission_id'] for r in waiting]


def refresh_text(desc, historical):
    notices = desc.get('kanban_notices')
    if not notices:
        return historical
    texts = []
    for n in notices:
        try:
            # Explicit URI mode=ro: missing DBs are NOT created; no migrations.
            with sqlite3.connect(Path(n['db']).as_uri() + '?mode=ro', uri=True, timeout=1) as conn:
                conn.row_factory = sqlite3.Row
                sub = conn.execute('SELECT 1 FROM kanban_notify_subs WHERE task_id=? AND platform=? '
                                   "AND chat_id=? AND COALESCE(thread_id,'')=?",
                                   (n['task_id'], n['platform'], n['chat_id'], n['thread_id'])).fetchone()
                if sub is None:
                    continue
                task = conn.execute('SELECT status,title,assignee FROM tasks WHERE id=?', (n['task_id'],)).fetchone()
                if task is None or task['status'] == 'archived':
                    continue
                texts.append(f"[kanban] {n['board']}/{n['task_id']} — Current status: {task['status']}\n"
                             f"Title: {task['title']}\nAssignee: {task['assignee'] or '(none)'}\n"
                             'Historical events superseded by this execution-time snapshot.')
        except (OSError, sqlite3.Error, ValueError):
            texts.append('[kanban] Current state unavailable; historical notification withheld.')
    return '\n\n'.join(texts) or None
