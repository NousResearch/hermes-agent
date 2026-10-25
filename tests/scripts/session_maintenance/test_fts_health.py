from contextlib import closing
import sqlite3
import unittest

import offline_rebuild  # Loads the pinned installed source path, no database opens.
from hermes_state_common import FTS_SQL, FTS_TRIGRAM_SQL
from fts_health import verify_fts_health
from verification import VerificationError


class RealFtsTests(unittest.TestCase):
    def setUp(self):
        self.c = sqlite3.connect(':memory:')
        self.addCleanup(self.c.close)
        self.c.executescript('''
            CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,
                                  content TEXT,tool_name TEXT,tool_calls TEXT);
            CREATE TABLE sessions(id TEXT PRIMARY KEY,source TEXT,model_config TEXT);
            CREATE TABLE state_meta(key TEXT PRIMARY KEY,value TEXT);
            INSERT INTO sessions VALUES('synthetic','cli',NULL);
        ''' + FTS_SQL + FTS_TRIGRAM_SQL)
        self.c.execute("INSERT INTO messages VALUES(1,'synthetic','user','Hermes maintenance',NULL,NULL)")

    def matches(self, family, query):
        return self.c.execute('SELECT rowid FROM ' + family + ' WHERE ' + family + ' MATCH ?', (query,)).fetchall()

    def test_real_insert_update_delete_sync_and_health(self):
        result = verify_fts_health(self.c)
        self.assertEqual(result['families'], ['messages_fts', 'messages_fts_trigram'])
        for family in result['families']:
            self.assertEqual(self.matches(family, 'Hermes'), [(1,)])
        self.c.execute("UPDATE messages SET content='uniquenewtoken' WHERE id=1")
        for family in result['families']:
            self.assertEqual(self.matches(family, 'Hermes'), [])
            self.assertEqual(self.matches(family, 'uniquenewtoken'), [(1,)])
        self.c.execute('DELETE FROM messages WHERE id=1')
        for family in result['families']:
            self.assertEqual(self.matches(family, 'uniquenewtoken'), [])
        self.assertEqual(verify_fts_health(self.c)['status'], 'passed')

    def test_missing_each_sync_trigger_is_rejected(self):
        for family in ('messages_fts', 'messages_fts_trigram'):
            for kind in ('insert', 'update', 'delete'):
                with self.subTest(family=family, kind=kind):
                    self.c.execute('SAVEPOINT probe')
                    self.c.execute('DROP TRIGGER ' + family + '_' + kind)
                    with self.assertRaisesRegex(VerificationError, 'FTS object missing'):
                        verify_fts_health(self.c)
                    self.c.execute('ROLLBACK TO probe')
                    self.c.execute('RELEASE probe')

    def test_noop_trigger_with_correct_name_is_rejected(self):
        self.c.executescript('DROP TRIGGER messages_fts_insert; '
            'CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN SELECT 1; END;')
        with self.assertRaisesRegex(VerificationError, 'FTS object missing'):
            verify_fts_health(self.c)

    def test_changed_trigram_source_view_rejected(self):
        self.c.executescript('DROP VIEW messages_fts_trigram_src; '
                            'CREATE VIEW messages_fts_trigram_src AS SELECT * FROM messages;')
        with self.assertRaisesRegex(VerificationError, 'FTS object missing'):
            verify_fts_health(self.c)

    def test_stale_rebuild_teardown_and_available_flags_rejected(self):
        for key in ('fts_stale', 'fts_cjk_stale', 'fts_rebuild_progress', 'fts_cjk_rebuild_high_water',
                    'fts_teardown_fts_v22_trash_messages_fts_data_progress', 'fts_optimize_available'):
            with self.subTest(key=key):
                self.c.execute('DELETE FROM state_meta')
                self.c.execute('INSERT INTO state_meta VALUES(?,?)', (key, '1'))
                with self.assertRaisesRegex(VerificationError, 'markers remain'):
                    verify_fts_health(self.c)

    def test_missing_base_rejected(self):
        self.c.execute('DROP TABLE messages_fts')
        with self.assertRaisesRegex(VerificationError, 'Base FTS'):
            verify_fts_health(self.c)

    def test_leftover_trash_rejected(self):
        self.c.execute('CREATE TABLE fts_v22_trash_messages_fts_data(id INTEGER)')
        with self.assertRaisesRegex(VerificationError, 'trash'):
            verify_fts_health(self.c)


if __name__ == '__main__':
    unittest.main()
