from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import unittest

import verification
import test_verification
from bounded_verification import UIDBudget, verify_bounded
from service_runner import orchestrate

HERE = Path(__file__).resolve().parent


class BoundedTests(unittest.TestCase):
    def setUp(self):
        self.fx = test_verification.VerificationTests()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.budget = UIDBudget(0, 2000, 2, 2000, 2)

    def verify(self, budget=None):
        return verify_bounded(self.fx.before, self.fx.after, budget or self.budget)

    def test_expected_backfill(self):
        self.fx.backfill()
        result = self.verify()
        self.assertEqual(result['uid_backfill']['rows'], 2)
        self.assertEqual(result['messages']['before_protected_sha256'], result['messages']['after_protected_sha256'])
        self.assertNotEqual(result['messages']['before_full_sha256'], result['messages']['after_full_sha256'])

    def test_unchanged_under_zero_budget(self):
        self.assertEqual(self.verify(UIDBudget(0, 0, 0, 0, 0))['uid_backfill']['rows'], 0)

    def test_existing_duplicate_membership_preserved(self):
        for p in (self.fx.before, self.fx.after):
            self.fx.edit("INSERT INTO messages VALUES(4002,'synthetic','duplicate','r',x'00','existing')", path=p)
        self.fx.backfill()
        self.assertEqual(self.verify()['uid_inventory_after']['duplicate_uid_groups'], 1)

    def test_content_deletion_insertion_identity_and_existing_uid_mutations_block(self):
        import shutil
        attacks = [
            "UPDATE messages SET content='changed' WHERE id=1",
            "UPDATE messages SET reasoning='changed' WHERE id=1",
            "UPDATE messages SET metadata=x'ff' WHERE id=1",
            "UPDATE messages SET session_id='other' WHERE id=1",
            "DELETE FROM messages WHERE id=1",
            "UPDATE messages SET id=3 WHERE id=1",
            "INSERT INTO messages VALUES(6001,'synthetic','new','r',x'00',NULL)",
            "UPDATE messages SET message_uid='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' WHERE id=4001",
            "UPDATE messages SET message_uid=NULL WHERE id=4001",
            "UPDATE sessions SET title='changed'",
            "DELETE FROM sessions",
            "UPDATE system_prompts SET content='changed'",
        ]
        for sql in attacks:
            with self.subTest(sql=sql):
                shutil.copyfile(self.fx.before, self.fx.after)
                self.fx.backfill()
                self.fx.edit(sql)
                with self.assertRaises(verification.VerificationError):
                    self.verify()

    def test_fresh_to_fresh_collision(self):
        self.fx.backfill()
        self.fx.edit("UPDATE messages SET message_uid=printf('%032x',1) WHERE id=2")
        with self.assertRaisesRegex(verification.VerificationError, 'collision'):
            self.verify()

    def test_fresh_to_existing_collision_with_valid_uid(self):
        for p in (self.fx.before, self.fx.after):
            self.fx.edit('UPDATE messages SET message_uid=? WHERE id=4001', ('f'*32,), path=p)
        self.fx.backfill()
        self.fx.edit('UPDATE messages SET message_uid=? WHERE id=1', ('f'*32,))
        with self.assertRaisesRegex(verification.VerificationError, 'collision'):
            self.verify()

    def test_bounded_row_count(self):
        self.fx.backfill()
        with self.assertRaisesRegex(verification.VerificationError, 'row count exceeds'):
            self.verify(UIDBudget(0, 2000, 1))

    def test_self_declared_larger_cursor_not_authority(self):
        self.fx.backfill()
        self.fx.edit("UPDATE state_meta SET value='4000' WHERE key='message_uid_backfill_id'")
        # Existing candidate has no independent ceiling; this is a negative control.
        self.assertEqual(verification.verify_snapshots(self.fx.before, self.fx.after)['status'], 'passed')
        with self.assertRaisesRegex(verification.VerificationError, 'range exceeds'):
            self.verify()

    def test_completed_marker_not_budget_escape(self):
        self.fx.backfill()
        self.fx.edit("DELETE FROM state_meta WHERE key='message_uid_backfill_id'")
        self.fx.edit("INSERT INTO state_meta VALUES('message_uid_backfill','1')")
        self.assertEqual(verification.verify_snapshots(self.fx.before, self.fx.after)['status'], 'passed')
        with self.assertRaisesRegex(verification.VerificationError, 'range exceeds'):
            self.verify()

    def test_exact_expected_count(self):
        self.fx.backfill()
        with self.assertRaisesRegex(verification.VerificationError, 'Unexpected UID change count'):
            self.verify(UIDBudget(0, 2000, 3, 2000, 3))

    def test_changed_baseline_cursor(self):
        self.fx.backfill()
        with self.assertRaisesRegex(verification.VerificationError, 'baseline UID cursor'):
            self.verify(UIDBudget(2000, 4000, 2))

    def test_runtime_epoch_stays_protected(self):
        for p in (self.fx.before, self.fx.after):
            self.fx.edit('CREATE TABLE runtime_epoch(id INTEGER PRIMARY KEY, epoch INTEGER)', path=p)
            self.fx.edit('INSERT INTO runtime_epoch VALUES(1,4)', path=p)
        self.fx.backfill()
        self.fx.edit('UPDATE runtime_epoch SET epoch=5')
        with self.assertRaisesRegex(verification.VerificationError, 'runtime_epoch'):
            self.verify()

    def test_non_fts_index_change_blocks(self):
        self.fx.backfill()
        self.fx.edit('CREATE INDEX new_unapproved_index ON messages(content)')
        self.assertEqual(verification.verify_snapshots(self.fx.before, self.fx.after)['status'], 'passed')
        with self.assertRaisesRegex(verification.VerificationError, 'catalog'):
            self.verify()

    def test_bounded_verifier_has_no_snapshot_writes(self):
        self.fx.backfill()
        before = [p.read_bytes() for p in (self.fx.before, self.fx.after)]
        self.verify()
        self.assertEqual(before, [p.read_bytes() for p in (self.fx.before, self.fx.after)])
        self.assertEqual(sorted(p.name for p in self.fx.before.parent.iterdir()), ['after.db', 'before.db'])


class RestartTests(unittest.TestCase):
    def policy(self, active=True, fail=None, error=RuntimeError, nonzero=False, health=True, record_fail=None):
        calls, records = [], []
        def action(name, result=None):
            def call():
                calls.append(name)
                if fail == name:
                    raise error('synthetic fault')
                return result
            return call
        def record(event):
            records.append(event)
            if record_fail is not None and event.get('phase') == record_fail and event.get('status') != 'blocked':
                raise OSError('synthetic receipt persistence failure')
        with redirect_stderr(io.StringIO()):
            rc = orchestrate(initially_active=active, preflight=action('preflight'),
                stop=action('stop'), maintain=action('maintain', 1 if nonzero else 0),
                start=action('start'),                healthy=action('healthy', health), verified=lambda: True, record=record)
        return rc, calls, records

    def test_verified_active_restart_once(self):
        rc, calls, events = self.policy()
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ['preflight', 'stop', 'maintain', 'start', 'healthy'])
        self.assertEqual(events[-1]['status'], 'complete')

    def test_inactive_stays_inactive(self):
        rc, calls, _ = self.policy(active=False)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ['preflight', 'maintain'])

    def test_errors_and_interrupts_before_verification_never_restart(self):
        for failure in ('preflight', 'stop', 'maintain'):
            for error in (RuntimeError, KeyboardInterrupt, InterruptedError, SystemExit):
                with self.subTest(failure=failure, error=error.__name__):
                    rc, calls, events = self.policy(fail=failure, error=error)
                    self.assertEqual(rc, 1)
                    self.assertNotIn('start', calls)
                    self.assertFalse(events[-1]['maintenance_verified'])

    def test_nonzero_and_verified_record_failure_block_restart(self):
        for options in ({'nonzero': True}, {'record_fail': 'verified-offline'}):
            with self.subTest(options=options):
                rc, calls, _ = self.policy(**options)
                self.assertEqual(rc, 1)
                self.assertNotIn('start', calls)

    def test_restart_health_failure_keeps_offline_result_separate(self):
        for options in ({'fail': 'start'}, {'fail': 'healthy'}, {'health': False}):
            with self.subTest(options=options):
                rc, calls, events = self.policy(**options)
                self.assertEqual(rc, 1)
                self.assertEqual(calls.count('start'), 1)
                self.assertTrue(events[-1]['maintenance_verified'])
                self.assertEqual(events[-1]['status'], 'blocked')

    def test_none_and_bool_return_values_never_authorize_restart(self):
        for result in (None, False, True, '0'):
            starts = []
            with self.subTest(result=result), redirect_stderr(io.StringIO()):
                rc = orchestrate(initially_active=True, preflight=lambda: None,
                    stop=lambda: None, maintain=lambda: result, verified=lambda: True,
                    start=lambda: starts.append(True), healthy=lambda: True, record=lambda event: None)
            self.assertEqual(rc, 1)
            self.assertEqual(starts, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
