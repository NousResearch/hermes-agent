from contextlib import closing, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import offline_rebuild as worker
from service_runner import orchestrate
from bounded_verification import UIDBudget
import test_verification


class ServicePolicyTests(unittest.TestCase):
    def run_policy(self, active=True, fail=None, error=RuntimeError):
        calls, records = [], []
        def action(name, result=None):
            def call():
                calls.append(name)
                if fail == name:
                    raise error('synthetic fault')
                return result
            return call
        with redirect_stderr(io.StringIO()):
            rc = orchestrate(initially_active=active, preflight=action('preflight'),
                stop=action('stop'), maintain=action('maintain', 0), start=action('start'),
                healthy=action('healthy', True), verified=lambda: True, record=records.append)
        return rc, calls, records

    def test_success_restarts_only_previously_active_gateway(self):
        rc, calls, records = self.run_policy()
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ['preflight', 'stop', 'maintain', 'start', 'healthy'])
        self.assertEqual(records[-1]['status'], 'complete')

    def test_inactive_gateway_not_started(self):
        rc, calls, records = self.run_policy(active=False)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ['preflight', 'maintain'])
        self.assertEqual(records[-1]['gateway'], 'left-inactive')

    def test_preflight_stop_and_verification_failure_never_restart(self):
        for failed in ('preflight', 'stop', 'maintain'):
            with self.subTest(failed=failed):
                rc, calls, records = self.run_policy(fail=failed)
                self.assertEqual(rc, 1)
                self.assertNotIn('start', calls)
                self.assertEqual(records[-1]['status'], 'blocked')
                self.assertFalse(records[-1]['maintenance_verified'])

    def test_interrupt_and_sigterm_path_never_restart(self):
        for error in (KeyboardInterrupt, InterruptedError, SystemExit):
            with self.subTest(error=error):
                rc, calls, _records = self.run_policy(fail='maintain', error=error)
                self.assertEqual(rc, 1)
                self.assertNotIn('start', calls)

    def test_restart_failure_is_not_maintenance_failure_or_success(self):
        for failed in ('start', 'healthy'):
            with self.subTest(failed=failed):
                rc, calls, records = self.run_policy(fail=failed)
                self.assertEqual(rc, 1)
                self.assertEqual(calls.count('start'), 1)
                self.assertTrue(records[-1]['maintenance_verified'])
                self.assertEqual(records[-1]['status'], 'blocked')

    def test_nonzero_maintenance_return_fails_closed(self):
        calls, records = [], []
        with redirect_stderr(io.StringIO()):
            rc = orchestrate(initially_active=True, preflight=lambda: None,
                stop=lambda: None, maintain=lambda: 1, start=lambda: calls.append('start'),
                healthy=lambda: True, verified=lambda: True, record=records.append)
        self.assertEqual(rc, 1)
        self.assertEqual(calls, [])


class WrapperIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Reuse synthetic fixture setup, not its entire test suite.
        self.fixture = test_verification.VerificationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.home = Path(self.fixture.temp.name)
        self.db = self.fixture.after
        for name, value in [('HOME', self.home), ('ROOT', self.home), ('DB', self.db),
                            ('holders', lambda p: []), ('load_fts5_cjk_extension', lambda c: None),
                            ('fts_layout_current', lambda c: True), ('fts_layout_drift', lambda c: []),
                            ('verify_fts_health', lambda c: {'status': 'passed'}),
                            ('probes', lambda c: {'Hermes': [1], 'maintenance': [2]})]:
            context = patch.object(worker, name, value)
            context.start()
            self.addCleanup(context.stop)
        disk = patch.object(worker.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12))
        disk.start()
        self.addCleanup(disk.stop)
        self.effects = []

    def fake_command(self, args, log, timeout):
        self.effects.append(args[0])
        if args[0] != 'zstd':
            raise AssertionError('Only synthetic backup commands allowed')
        if '-o' in args:
            shutil.copyfile(Path(args[args.index('-o')-1]), Path(args[-1]))
        log.write_text('synthetic backup fixture, not a zstd execution\n')

    def fake_optimize(self, args, **kwargs):
        self.effects.append('synthetic-uid-backfill')
        self.fixture.backfill()
        return SimpleNamespace(returncode=0)

    def run_worker(self, change=None):
        def fake(args, **kwargs):
            if any(str(arg).endswith('quick_check.py') for arg in args):
                return SimpleNamespace(returncode=0, stdout='["ok"]')
            result = self.fake_optimize(args, **kwargs)
            if change:
                change()
            return result
        with patch.object(worker, 'command', self.fake_command), \
             patch.object(worker.subprocess, 'run', fake), redirect_stdout(io.StringIO()):
            return worker.main(execute_offline=True, budget=UIDBudget(0, 2000, 2, 2000, 2))

    def receipt(self):
        paths = list(self.home.glob('offline-*/receipt.json'))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text())

    def test_requires_explicit_arm_before_database_or_service_actions(self):
        with patch.object(worker, 'exclusive', side_effect=AssertionError('must not be called')):
            with self.assertRaisesRegex(RuntimeError, 'Execution disabled'):
                worker.main()
        self.assertFalse((self.home / '.lock').exists())

    def test_integrated_success_records_backfill_not_identical_full_rows(self):
        self.assertEqual(self.run_worker(), 0)
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'complete')
        self.assertFalse(receipt['full_rows_unchanged'])
        self.assertTrue(receipt['conversation_content_and_protected_fields_unchanged'])
        self.assertEqual(receipt['verification']['uid_backfill']['rows'], 2)
        self.assertTrue(Path(receipt['backup']).is_file())

    def test_integrated_metadata_mutation_blocks_and_preserves_backup(self):
        with self.assertRaisesRegex(RuntimeError, 'Protected message'):
            self.run_worker(lambda: self.fixture.edit("UPDATE messages SET metadata=x'ff' WHERE id=1"))
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'blocked')
        self.assertTrue(Path(receipt['backup']).is_file())

    def test_wal_snapshot_readers_do_not_create_sidecars(self):
        with closing(sqlite3.connect(self.db)) as c:
            self.assertEqual(c.execute('PRAGMA journal_mode=WAL').fetchone()[0], 'wal')
        self.assertEqual(self.run_worker(), 0)
        run_dir = Path(self.receipt()['directory'])
        self.assertFalse(list(run_dir.glob('state-*.db-wal')))
        self.assertFalse(list(run_dir.glob('state-*.db-shm')))

    def test_zero_exit_with_bad_layout_does_not_pass(self):
        with patch.object(worker, 'fts_layout_current', lambda c: False):
            with self.assertRaisesRegex(RuntimeError, 'Index layout'):
                self.run_worker()
        self.assertEqual(self.receipt()['status'], 'blocked')

    def test_both_search_probes_must_have_baseline_matches(self):
        with patch.object(worker, 'probes', lambda c: {'Hermes': [1], 'maintenance': []}):
            with self.assertRaisesRegex(RuntimeError, 'search IDs'):
                self.run_worker()
        self.assertEqual(self.receipt()['status'], 'blocked')

    def test_old_contract_reproduces_false_block_for_valid_backfill(self):
        before = worker.fingerprint(self.db)
        self.fixture.backfill()
        after = worker.fingerprint(self.db)
        self.assertNotEqual(before, after)  # Original wrapper's unconditional blocked branch.
        self.assertEqual(worker.verify_snapshots(self.fixture.before, self.db)['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
