"""Exercise the actual installed UID migration on synthetic data, not the optimizer."""
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch


from hermes_state_schema import SessionSchemaMixin
import hermes_state_schema
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_verification
from verification import verify_snapshots


class MetadataWriter:
    def set_meta(self, key, value, *, cursor):
        cursor.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)', (key, value))


class UpstreamMigrationTests(unittest.TestCase):
    def test_actual_bounded_migration_satisfies_contract_and_preserves_existing_uids(self):
        fixture = test_verification.VerificationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with closing(sqlite3.connect(fixture.after, isolation_level=None)) as c:
            with patch.object(hermes_state_schema, '_MESSAGE_UID_BACKFILL_BUDGET_S', 0.0):
                SessionSchemaMixin._advance_message_uid_backfill(MetadataWriter(), c.cursor())
                self.assertEqual(c.execute("SELECT value FROM state_meta WHERE key='message_uid_backfill_id'").fetchone()[0], '2000')
                self.assertEqual(verify_snapshots(fixture.before, fixture.after)['uid_backfill']['rows'], 2)
                SessionSchemaMixin._advance_message_uid_backfill(MetadataWriter(), c.cursor())
                SessionSchemaMixin._advance_message_uid_backfill(MetadataWriter(), c.cursor())
            self.assertEqual(c.execute("SELECT value FROM state_meta WHERE key='message_uid_backfill'").fetchone()[0], '1')
            self.assertEqual(c.execute('SELECT message_uid FROM messages WHERE id=4001').fetchone()[0], 'existing')
            result = verify_snapshots(fixture.before, fixture.after)
            self.assertEqual(result['uid_backfill']['rows'], 2)
            self.assertIn('message_uid_backfill', result['state_meta_changed_keys'])


if __name__ == '__main__':
    unittest.main()
