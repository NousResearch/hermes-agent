"""Fail-closed operator safety, using only synthetic databases and fake services."""
from contextlib import redirect_stderr
from dataclasses import asdict
import io
import json
import sqlite3
import unittest
from unittest.mock import patch

import pytest
import offline_rebuild as worker
from bounded_verification import UIDBudget
from service_runner import orchestrate, ready_once, verify_receipt
import test_wrapper

pytestmark = pytest.mark.platforms('linux')


class FailurePathTests(unittest.TestCase):
    def setUp(self):
        self.fx = test_wrapper.WrapperIntegrationTests()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.budget = UIDBudget(0, 2000, 2, 2000, 2)

    def test_holders_refuse_before_any_connection_or_backup(self):
        with patch.object(worker, 'holders', return_value=[123]), patch.object(worker.sqlite3, 'connect') as connect:
            with self.assertRaisesRegex(RuntimeError, 'still open'):
                worker.main(execute_offline=True, budget=self.budget)
            connect.assert_not_called()
        self.assertEqual(self.fx.effects, [])
        self.assertFalse((self.fx.home / '.lock').exists())

    def test_invalid_budget_refuses_before_database_access(self):
        for budget in (None, UIDBudget(0, 2000, -1), UIDBudget(0, 2000, 2, 4000)):
            with self.subTest(budget=budget), patch.object(worker.sqlite3, 'connect') as connect:
                with self.assertRaises(RuntimeError):
                    worker.main(execute_offline=True, budget=budget)
                connect.assert_not_called()

    def test_holder_race_after_baseline_never_checkpoints(self):
        original = sqlite3.connect
        calls = []
        def track(path, *args, **kwargs):
            calls.append(str(path))
            return original(path, *args, **kwargs)
        with patch.object(worker, 'exclusive', side_effect=[None, None, RuntimeError('late holder')]), \
                patch.object(worker.sqlite3, 'connect', side_effect=track):
            with self.assertRaisesRegex(RuntimeError, 'late holder'):
                worker.main(execute_offline=True, budget=self.budget)
        self.assertTrue(calls)
        self.assertTrue(all('mode=ro' in path for path in calls))
        self.assertEqual(self.fx.receipt()['status'], 'blocked')

    def test_zero_exit_with_integrity_exception_stays_blocked(self):
        with patch.object(worker, 'check_integrity', side_effect=sqlite3.DatabaseError('synthetic integrity failure')):
            with self.assertRaisesRegex(sqlite3.DatabaseError, 'integrity failure'):
                self.fx.run_worker()
        self.assertEqual(self.fx.receipt()['status'], 'blocked')

    def test_persisted_postconditions_required_independently_of_exit_code(self):
        self.assertEqual(self.fx.run_worker(), 0)
        self.assertTrue(verify_receipt(self.fx.home, self.budget))
        path = next(self.fx.home.glob('offline-*/receipt.json'))
        good = json.loads(path.read_text())
        for key, bad in [('status', 'running'), ('quick_check', []), ('layout_current', False),
                         ('pending_markers', ['fts_stale']), ('backup_verified', False),
                         ('conversation_content_and_protected_fields_unchanged', False),
                         ('search_after', {}), ('fts_health', {'status': 'unknown'})]:
            with self.subTest(key=key):
                value = dict(good); value[key] = bad
                path.write_text(json.dumps(value))
                self.assertFalse(verify_receipt(self.fx.home, self.budget))
        path.write_text(json.dumps(good))
        self.assertFalse(verify_receipt(self.fx.home, UIDBudget(0, 4000, 2)))

    def test_missing_postconditions_never_restart_even_after_zero(self):
        for verified in (False, None):
            starts = []
            with redirect_stderr(io.StringIO()):
                rc = orchestrate(initially_active=True, preflight=lambda: None, stop=lambda: None,
                    maintain=lambda: 0, verified=lambda: verified, start=lambda: starts.append(True),
                    healthy=lambda: True, record=lambda event: None)
            self.assertEqual(rc, 1)
            self.assertEqual(starts, [])

    def test_unknown_initial_state_never_controls_service(self):
        calls = []
        with redirect_stderr(io.StringIO()):
            rc = orchestrate(initially_active=None, preflight=lambda: calls.append('preflight'),
                stop=lambda: calls.append('stop'), maintain=lambda: 0, verified=lambda: True,
                start=lambda: calls.append('start'), healthy=lambda: True, record=lambda event: None)
        self.assertEqual(rc, 1)
        self.assertEqual(calls, [])

    def test_restart_readiness_requires_current_matching_gateway(self):
        path = self.fx.home / 'gateway_state.json'
        stamp = '2026-01-01T00:00:01+00:00'
        good = {'pid': 123, 'start_time': 55, 'updated_at': stamp, 'gateway_state': 'running',
                'kind': 'hermes-gateway', 'served_profiles': [], 'session_store': {'status': 'ok'},
                'platforms': {'synthetic': {'state': 'connected', 'writer_pid': 123,
                                          'writer_start_time': 55, 'updated_at': stamp}}}
        def command(args): return '123' if 'show' in args else 'active'
        with patch('service_runner.command', side_effect=command), patch('service_runner.process_start', return_value=55):
            path.write_text(json.dumps(good))
            self.assertTrue(ready_once(self.fx.home, 'synthetic.service', 1, {'synthetic'}))
            for key, value in [('pid', 999), ('gateway_state', 'starting'),
                               ('session_store', {'status': 'error'}),
                               ('platforms', {'synthetic': {'state': 'error'}})]:
                with self.subTest(key=key):
                    bad = dict(good); bad[key] = value; path.write_text(json.dumps(bad))
                    self.assertFalse(ready_once(self.fx.home, 'synthetic.service', 1, {'synthetic'}))
            path.write_text(json.dumps(good))
            self.assertFalse(ready_once(self.fx.home, 'synthetic.service', 10**12, {'synthetic'}))
