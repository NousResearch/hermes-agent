from contextlib import closing
import shutil
import sqlite3
from pathlib import Path
import tempfile
import unittest

from verification import VerificationError, verify_snapshots


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.before, self.after = [Path(self.temp.name) / n for n in ('before.db', 'after.db')]
        with closing(sqlite3.connect(self.before)) as c, c:
            c.executescript('''
                CREATE TABLE sessions(id TEXT PRIMARY KEY, title TEXT, metadata BLOB);
                CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, content TEXT,
                  reasoning TEXT, metadata BLOB, message_uid TEXT);
                CREATE TABLE state_meta(key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE system_prompts(id INTEGER PRIMARY KEY, content TEXT);
                INSERT INTO sessions VALUES('synthetic','title',x'00');
                INSERT INTO messages VALUES(1,'synthetic','hello','reason',x'01',NULL);
                INSERT INTO messages VALUES(2,'synthetic','hello2','reason2',x'02',NULL);
                INSERT INTO messages VALUES(4001,'synthetic','later','reason',x'03','existing');
                INSERT INTO state_meta VALUES('message_uid_backfill_id','0');
                INSERT INTO state_meta VALUES('other','protected');
                INSERT INTO system_prompts VALUES(1,'protected prompt');
            ''')
        shutil.copyfile(self.before, self.after)

    def edit(self, sql, args=(), path=None):
        with closing(sqlite3.connect(path or self.after)) as c, c:
            c.execute(sql, args)

    def backfill(self):
        self.edit("UPDATE messages SET message_uid=printf('%032x',id) WHERE id <= 2000")
        self.edit("UPDATE state_meta SET value='2000' WHERE key='message_uid_backfill_id'")

    def verify(self):
        return verify_snapshots(self.before, self.after)

    def test_unchanged(self):
        self.assertEqual(self.verify()['uid_backfill']['rows'], 0)

    def test_successful_partial_backfill(self):
        self.backfill()
        result = self.verify()
        self.assertEqual(result['uid_backfill']['rows'], 2)
        self.assertNotEqual(result['messages']['before_full_sha256'], result['messages']['after_full_sha256'])
        self.assertEqual(result['messages']['before_protected_sha256'], result['messages']['after_protected_sha256'])

    def test_successful_completed_backfill(self):
        self.backfill()
        self.edit("DELETE FROM state_meta WHERE key='message_uid_backfill_id'")
        self.edit("INSERT INTO state_meta VALUES('message_uid_backfill','1')")
        self.assertEqual(self.verify()['uid_backfill']['rows'], 2)

    def test_collisions_with_new_or_existing_uid(self):
        for uid in ('f' * 32, 'existing'):
            with self.subTest(uid=uid):
                self.backfill()
                if uid == 'existing':
                    self.edit('UPDATE messages SET message_uid=? WHERE id=4001', ('f'*32,), self.before)
                    self.edit('UPDATE messages SET message_uid=? WHERE id=4001', ('f'*32,))
                self.edit('UPDATE messages SET message_uid=? WHERE id IN (1,2)', ('f'*32,))
                with self.assertRaisesRegex(VerificationError, 'collision'):
                    self.verify()

    def test_existing_uid_changes_rejected(self):
        for uid in (None, 'a'*32, ''):
            with self.subTest(uid=uid):
                self.edit('UPDATE messages SET message_uid=? WHERE id=4001', (uid,))
                with self.assertRaisesRegex(VerificationError, 'non-NULL'):
                    self.verify()

    def test_bad_backfill_format(self):
        for uid in ('', 'A'*32, 'x'*32, b'0'*32, '0'*31):
            with self.subTest(uid=uid):
                self.edit('UPDATE messages SET message_uid=? WHERE id=1', (uid,))
                with self.assertRaisesRegex(VerificationError, '32-hex'):
                    self.verify()

    def test_protected_message_fields(self):
        for column, value in [('content', 'mutation'), ('reasoning', 'mutation'),
                              ('metadata', b'changed'), ('session_id', 'other')]:
            with self.subTest(column=column):
                shutil.copyfile(self.before, self.after)
                self.backfill()
                self.edit('UPDATE messages SET ' + column + '=? WHERE id=1', (value,))
                with self.assertRaisesRegex(VerificationError, 'Protected message'):
                    self.verify()

    def test_session_and_system_prompt_changes(self):
        for sql in ["UPDATE sessions SET title='changed'", "UPDATE sessions SET metadata=x'ff'",
                    "UPDATE system_prompts SET content='changed'"]:
            with self.subTest(sql=sql):
                shutil.copyfile(self.before, self.after)
                self.edit(sql)
                with self.assertRaisesRegex(VerificationError, 'Protected table'):
                    self.verify()

    def test_unknown_meta_change(self):
        for sql in ["UPDATE state_meta SET value='changed' WHERE key='other'",
                    "INSERT INTO state_meta VALUES('fts_unknown','1')",
                    "DELETE FROM state_meta WHERE key='other'"]:
            with self.subTest(sql=sql):
                shutil.copyfile(self.before, self.after)
                self.edit(sql)
                with self.assertRaisesRegex(VerificationError, 'Unexpected state_meta'):
                    self.verify()

    def test_row_loss_addition_or_identity_mutation(self):
        for sql in ["DELETE FROM messages WHERE id=1", "UPDATE messages SET id=3 WHERE id=1",
                    "INSERT INTO messages SELECT 6001,session_id,content,reasoning,metadata,message_uid FROM messages WHERE id=4001"]:
            with self.subTest(sql=sql):
                shutil.copyfile(self.before, self.after)
                self.edit(sql)
                with self.assertRaises(VerificationError):
                    self.verify()

    def test_schema_change(self):
        self.edit('ALTER TABLE messages ADD COLUMN extra TEXT')
        with self.assertRaisesRegex(VerificationError, 'schema'):
            self.verify()

    def test_existing_duplicate_population_preserved(self):
        for path in (self.before, self.after):
            self.edit("INSERT INTO messages VALUES(4002,'synthetic','other','reason',x'01','existing')", path=path)
        self.backfill()
        r = self.verify()
        self.assertEqual(r['uid_inventory_before']['duplicate_uid_groups'], 1)
        self.assertEqual(r['uid_inventory_after']['duplicate_uid_groups'], 1)

    def test_missing_or_invalid_cursor_advancement(self):
        for cursor in ('0', '1000', '6000', '-1', 'bad'):
            with self.subTest(cursor=cursor):
                self.backfill()
                self.edit("UPDATE state_meta SET value=? WHERE key='message_uid_backfill_id'", (cursor,))
                with self.assertRaises(VerificationError):
                    self.verify()

    def test_cursor_skipped_null_row(self):
        self.backfill()
        self.edit('UPDATE messages SET message_uid=NULL WHERE id=2')
        with self.assertRaisesRegex(VerificationError, 'skipped NULL'):
            self.verify()

    def test_completed_marker_does_not_authorize_mutation(self):
        for path in (self.before, self.after):
            self.edit("INSERT INTO state_meta VALUES('message_uid_backfill','1')", path=path)
        self.backfill()
        with self.assertRaisesRegex(VerificationError, 'after completion'):
            self.verify()

    def test_cold_snapshot_required(self):
        Path(str(self.after) + '-wal').touch()
        with self.assertRaisesRegex(VerificationError, 'sidecars'):
            self.verify()

    def test_no_database_writes(self):
        a, b = self.before.read_bytes(), self.after.read_bytes()
        self.verify()
        self.assertEqual(a, self.before.read_bytes())
        self.assertEqual(b, self.after.read_bytes())
        self.assertEqual(sorted(p.name for p in Path(self.temp.name).iterdir()), ['after.db', 'before.db'])


if __name__ == '__main__':
    unittest.main()
