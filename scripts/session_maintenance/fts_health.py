"""Read-only FTS completion gate, using installed DDL as the exact contract.

Expected objects are built only in an in-memory synthetic schema. Never run
FTS integrity-check/rebuild INSERT commands against the evidence or live store.
"""
from contextlib import closing
import sqlite3

from hermes_state_common import FTS_SQL, FTS_TRIGRAM_SQL
from hermes_state_fts import (FTS_CJK_TABLE_SQL, FTS_CJK_TRIGGER_SQL,
                              load_fts5_cjk_extension)
from verification import require


def verify_fts_health(c):
    names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    require('messages_fts' in names, 'Base FTS table is missing')
    markers = [r[0] for r in c.execute("SELECT key FROM state_meta WHERE key LIKE 'fts%'")
               if ('rebuild' in r[0] or 'teardown' in r[0]
                   or r[0] in ('fts_stale', 'fts_cjk_stale', 'fts_optimize_available'))]
    require(not markers, 'FTS incomplete/stale markers remain')
    require(not any(n.startswith('fts_v22_trash_') for n in names), 'FTS trash tables remain')
    families = ['messages_fts']
    ddl = FTS_SQL
    if 'messages_fts_trigram' in names:
        families.append('messages_fts_trigram')
        ddl += FTS_TRIGRAM_SQL
    if 'messages_fts_cjk' in names:
        families.append('messages_fts_cjk')
        ddl += FTS_CJK_TABLE_SQL + FTS_CJK_TRIGGER_SQL
    with closing(sqlite3.connect(':memory:')) as expected:
        load_fts5_cjk_extension(expected)
        expected.executescript('''
            CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,
                                  content TEXT,tool_name TEXT,tool_calls TEXT);
            CREATE TABLE sessions(id TEXT PRIMARY KEY,source TEXT,model_config TEXT);
            CREATE TABLE state_meta(key TEXT PRIMARY KEY,value TEXT);
        ''')
        expected.executescript(ddl)
        objects = []
        for family in families:
            objects.extend([family, family + '_src'] +
                           [family + '_' + kind for kind in ('insert', 'update', 'delete')])
        for name in objects:
            want = expected.execute('SELECT type,sql FROM sqlite_master WHERE name=?', (name,)).fetchone()
            got = c.execute('SELECT type,sql FROM sqlite_master WHERE name=?', (name,)).fetchone()
            require(want is not None and got == want, 'FTS object missing or differs from installed DDL: ' + name)
    return {'status': 'passed', 'families': families, 'validated_objects': objects,
            'pending_or_stale_markers': markers,
            'scope': 'Exact table/view/sync-trigger DDL and completion markers; not an exhaustive token audit'}
