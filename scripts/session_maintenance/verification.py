"""Fail-closed, read-only comparison of offline SQLite maintenance snapshots.

No SessionDB construction: it migrates data on open. Reports contain counts/hashes,
never conversation text or UID values. Only message_uid NULL -> fresh 32-hex and
explicit maintenance state_meta transitions may differ. Not a live DB audit tool.
"""
from collections import Counter
from contextlib import closing
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import sqlite3


class VerificationError(RuntimeError):
    pass


UID = re.compile(r"[0-9a-f]{32}\Z")
CURSOR = 'message_uid_backfill_id'
DONE = 'message_uid_backfill'
FTS_REMOVABLE = frozenset({
    'fts_optimize_available', 'fts_rebuild_high_water', 'fts_rebuild_progress',
    'fts_cjk_rebuild_high_water', 'fts_cjk_rebuild_progress', 'fts_cjk_stale',
    'fts_stale', 'fts_rebuild_deferral', 'fts_tool_full_content_high_water',
})
FTS_TABLES = frozenset(base + suffix
                      for base in ('messages_fts', 'messages_fts_trigram', 'messages_fts_cjk')
                      for suffix in ('', '_data', '_idx', '_docsize', '_config', '_content'))


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def encode(row):
    # Preserve SQL storage types (including BLOB vs text, integer vs float).
    return json.dumps([(type(v).__name__, v.hex() if isinstance(v, bytes) else v)
                       for v in row], ensure_ascii=True, separators=(',', ':'),
                      allow_nan=False).encode() + b'\n'


def open_snapshot(path):
    path = Path(path).resolve(strict=True)
    require(not any(Path(str(path) + s).exists() for s in ('-wal', '-shm', '-journal')),
            'Snapshot has SQLite sidecars; supply a standalone cold snapshot')
    c = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
    c.execute('PRAGMA query_only=ON')
    return c


def tables(c):
    return dict(c.execute("SELECT name,sql FROM sqlite_master WHERE type='table'"))


def schema(c, table):
    return c.execute('PRAGMA table_xinfo(' + quote(table) + ')').fetchall()


def table_hash(c, table):
    # Sort row digests, not SQL values: handles NULL, BLOB and duplicate rows.
    hashes = sorted(hashlib.sha256(encode(r)).digest()
                    for r in c.execute('SELECT * FROM ' + quote(table)))
    return {'rows': len(hashes), 'sha256': hashlib.sha256(b''.join(hashes)).hexdigest()}


def uid_inventory(c):
    counts = Counter(row[0] for row in c.execute('SELECT message_uid FROM messages'))
    duplicates = {uid: n for uid, n in counts.items() if uid is not None and n > 1}
    return counts, {
        'null_rows': counts.get(None, 0),
        'duplicate_uid_groups': len(duplicates),
        'rows_in_duplicate_uid_groups': sum(duplicates.values()),
        'duplicate_uid_excess_rows': sum(n - 1 for n in duplicates.values()),
        'non_null_rows': sum(n for uid, n in counts.items() if uid is not None),
    }


def verify_meta(a, b, changed_ids):
    before = dict(a.execute('SELECT key,value FROM state_meta'))
    after = dict(b.execute('SELECT key,value FROM state_meta'))
    changed = {k for k in before.keys() | after.keys()
               if (k in before, before.get(k)) != (k in after, after.get(k))}
    allowed = FTS_REMOVABLE | {CURSOR, DONE, 'fts_storage_version'}
    require(not changed - allowed, 'Unexpected state_meta change')
    for key in changed & FTS_REMOVABLE:
        require(key not in after, 'Maintenance marker was added or modified, not cleared')
    if 'fts_storage_version' in changed:
        require(after.get('fts_storage_version') == '3', 'Unexpected FTS version transition')

    def cursor_value(meta):
        value = meta.get(CURSOR, '0')
        require(isinstance(value, str) and re.fullmatch(r'0|[1-9][0-9]*', value) is not None,
                'Invalid UID backfill cursor')
        return int(value)

    low = cursor_value(before)
    high = a.execute('SELECT COALESCE(MAX(id),0) FROM messages').fetchone()[0]
    if DONE in before:
        require(before.get(DONE) == after.get(DONE) and CURSOR not in changed and not changed_ids,
                'UID backfill changed after completion')
    elif DONE in after:
        require(after[DONE] == '1' and CURSOR not in after, 'Invalid UID completion markers')
        upper = max(low, high)
        require(not b.execute('SELECT 1 FROM messages WHERE id > ? AND message_uid IS NULL LIMIT 1',
                              (low,)).fetchone(), 'Completion left unfilled rows in the walked range')
        require(all(low < i <= upper for i in changed_ids), 'UID changed outside completion range')
    else:
        upper = cursor_value(after)
        require(upper >= low and (upper - low) % 2000 == 0, 'Invalid UID cursor advancement')
        if CURSOR in changed:
            require(low < upper < high, 'Partial UID cursor outside valid range')
        require(all(low < i <= upper for i in changed_ids), 'UID changed outside recorded cursor range')
        if upper > low:
            require(not b.execute('SELECT 1 FROM messages WHERE id > ? AND id <= ? AND message_uid IS NULL LIMIT 1',
                                  (low, upper)).fetchone(), 'UID cursor skipped NULL rows')
    return sorted(changed)


def verify_snapshots(before_path, after_path):
    require(Path(before_path).resolve() != Path(after_path).resolve(), 'Distinct snapshots required')
    with closing(open_snapshot(before_path)) as a, closing(open_snapshot(after_path)) as b:
        ta, tb = tables(a), tables(b)
        protected_a, protected_b = set(ta) - FTS_TABLES, set(tb) - FTS_TABLES
        require(protected_a == protected_b, 'Protected table set changed')
        require({'sessions', 'messages', 'state_meta'} <= protected_a, 'Required tables missing')
        for table in sorted(protected_a):
            require(ta[table] == tb[table] and schema(a, table) == schema(b, table),
                    'Protected schema changed: ' + table)
        for table in ('sessions', 'messages'):
            pk = [r for r in schema(a, table) if r[5]]
            require(len(pk) == 1 and pk[0][1] == 'id', 'Unsupported row identity: ' + table)
        require(next(r for r in schema(a, 'messages') if r[1] == 'id')[2].upper() == 'INTEGER',
                'Legacy non-integer message IDs need a separately reviewed contract')
        columns = [r[1] for r in schema(a, 'messages')]
        require('message_uid' in columns, 'message_uid column missing')
        uid_index = columns.index('message_uid')
        id_index = columns.index('id')
        old_counts, old_inventory = uid_inventory(a)
        new_counts, new_inventory = uid_inventory(b)
        changed_ids = []
        digests = {key: hashlib.sha256() for key in ('before_full', 'after_full', 'before_protected', 'after_protected')}
        count = 0
        sentinel = object()
        rows_a = a.execute('SELECT * FROM messages ORDER BY id')
        rows_b = b.execute('SELECT * FROM messages ORDER BY id')
        for old, new in zip_longest(rows_a, rows_b, fillvalue=sentinel):
            require(old is not sentinel and new is not sentinel, 'Message rows added or removed')
            require(old[id_index] == new[id_index], 'Message row identity changed')
            old_protected = old[:uid_index] + old[uid_index + 1:]
            new_protected = new[:uid_index] + new[uid_index + 1:]
            require(encode(old_protected) == encode(new_protected), 'Protected message field changed')
            old_uid, new_uid = old[uid_index], new[uid_index]
            if encode((old_uid,)) != encode((new_uid,)):
                require(old_uid is None, 'Existing non-NULL UID modified')
                require(isinstance(new_uid, str) and UID.fullmatch(new_uid) is not None,
                        'Backfilled UID is not a lowercase 32-hex string')
                require(old_counts.get(new_uid, 0) == 0 and new_counts.get(new_uid) == 1,
                        'Backfilled UID collision')
                changed_ids.append(old[id_index])
            for key, row in [('before_full', old), ('after_full', new),
                             ('before_protected', old_protected), ('after_protected', new_protected)]:
                digests[key].update(encode(row))
            count += 1
        unchanged = {}
        for table in sorted(protected_a - {'messages', 'state_meta'}):
            first, second = table_hash(a, table), table_hash(b, table)
            require(first == second, 'Protected table rows changed: ' + table)
            unchanged[table] = first
        meta_changes = verify_meta(a, b, changed_ids)
        return {
            'contract_version': 1, 'status': 'passed',
            'messages': {'rows': count, **{k + '_sha256': v.hexdigest() for k, v in digests.items()}},
            'unchanged_tables': unchanged,
            'uid_backfill': {'rows': len(changed_ids), 'first_id': min(changed_ids, default=None),
                             'last_id': max(changed_ids, default=None),
                             'ids_sha256': hashlib.sha256(encode(changed_ids)).hexdigest()},
            'uid_inventory_before': old_inventory, 'uid_inventory_after': new_inventory,
            'state_meta_changed_keys': meta_changes,
            'limitations': ['FTS derived tables require separate layout, marker, integrity and search checks.',
                            'Immutable mode is only safe for cold standalone snapshots, never a live database.'],
        }
