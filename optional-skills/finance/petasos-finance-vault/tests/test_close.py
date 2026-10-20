"""Lane B tests: petasos.close — month freeze + archive (design §5, §6 C5;
R3.5/R3.6/R3.8, DG-10).

Freezing a month makes it immutable: closed_months.json is written once and
never adjusted (drift = gate failure + loud close error, fix the ledger not
the freeze). Close preconditions (R3.8): gates green AND no needs-confirm
rows open in the closing month. Archive = FULL ledger copy + Monthly Review
with frozen_total.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import laneb

REPO = Path(__file__).resolve().parent.parent
MONTHLY_CLOSE = REPO / 'scripts' / 'monthly_close.py'


def read(path):
    return Path(path).read_text(encoding='utf-8')


def closed_json(knobs):
    return json.loads(read(os.path.join(laneb.ledger_path(knobs),
                                        'closed_months.json')))


class TestClose(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.knobs = laneb.make_vault(self.root, rows=[
            laneb.row(id='2026-07-05.01', date='2026-07-05', amount=111000,
                      category='July spend'),
            laneb.row(id='2026-08-01.01', date='2026-08-01', amount=150000,
                      category='Food'),
            laneb.row(id='2026-08-02.01', date='2026-08-02', amount=250000,
                      category='Food'),
        ])
        self.L = laneb.ledger_path(self.knobs)

    def _close(self, month='2026-08'):
        from petasos.close import close_month
        return close_month(self.knobs, month)

    def test_close_freezes_expense_total_and_archives(self):
        self.assertEqual(self._close(), 0)
        self.assertEqual(closed_json(self.knobs), {'2026-08': 400000})
        arc = os.path.join(self.root, '4-Archives', 'Finance', '2026-08')
        self.assertTrue(os.path.isfile(os.path.join(
            arc, 'transactions-2026-08.jsonl')))

    def test_freeze_covers_expenses_only_income_reviewed_not_frozen(self):
        knobs = laneb.make_vault(self.root, rows=[
            laneb.row(id='2026-08-01.01', date='2026-08-01', amount=400000),
            laneb.row(id='2026-08-10.01', date='2026-08-10', dir='in',
                      kind='salary', amount=2000000, old_ref='mail:feed0001',
                      via='EMP-A'),
        ], ledger_dir='2-Areas/Finance/_Ledger')
        from petasos.close import close_month
        self.assertEqual(close_month(knobs, '2026-08'), 0)
        self.assertEqual(closed_json(knobs), {'2026-08': 400000})
        review = read(os.path.join(self.root, '4-Archives', 'Finance', '2026-08',
                                   'Monthly Review 2026-08.md'))
        self.assertIn('frozen_total: 400000', review)
        self.assertIn('2.000.000', review)  # income shown in the review

    def test_double_close_idempotent(self):
        """R3.6: second run verifies the frozen value + refreshes snapshot."""
        self.assertEqual(self._close(), 0)
        before = read(os.path.join(self.L, 'closed_months.json'))
        self.assertEqual(self._close(), 0)
        self.assertEqual(read(os.path.join(self.L, 'closed_months.json')), before)

    def test_drift_after_close_fails_loud_and_never_rewrites_freeze(self):
        """R3.5: fix the drift, never the freeze — closed_months.json is
        immutable once written."""
        self.assertEqual(self._close(), 0)
        with open(os.path.join(self.L, 'transactions.jsonl'), 'a',
                  encoding='utf-8') as f:
            f.write(json.dumps(laneb.row(id='2026-08-20.01', date='2026-08-20',
                                         amount=50000)) + '\n')
        from petasos.close import close_month
        self.assertNotEqual(close_month(self.knobs, '2026-08'), 0)
        self.assertEqual(closed_json(self.knobs), {'2026-08': 400000})  # untouched

    def test_close_aborts_with_open_needs_confirm(self):
        """R3.8: no freezing while confirmations are pending in the month."""
        with open(os.path.join(self.L, 'transactions.jsonl'), 'a',
                  encoding='utf-8') as f:
            f.write(json.dumps(laneb.row(
                id='2026-08-21.01', date='2026-08-21', amount=700000,
                status='needs-confirm', ask='source unknown')) + '\n')
        from petasos.close import close_month
        rc = close_month(self.knobs, '2026-08')
        self.assertNotEqual(rc, 0)
        self.assertEqual(closed_json(self.knobs), {})  # nothing frozen
        self.assertFalse(os.path.exists(
            os.path.join(self.root, '4-Archives', 'Finance', '2026-08')))

    def test_close_aborts_when_gates_red(self):
        with open(os.path.join(self.L, 'transactions.jsonl'), 'a',
                  encoding='utf-8') as f:
            f.write(json.dumps(laneb.row(id='2026-08-01.01',
                                         date='2026-08-01')) + '\n')  # dup id
        self.assertNotEqual(self._close(), 0)
        self.assertEqual(closed_json(self.knobs), {})
        self.assertFalse(os.path.exists(
            os.path.join(self.root, '4-Archives', 'Finance', '2026-08')))

    def test_vault_local_gen_views_is_gate_authority(self):
        """C6: a vault-local gen_views.py owns the close gate verdict —
        v1-tuned vaults are judged by their own gates, not the package's."""
        with open(os.path.join(self.L, 'gen_views.py'), 'w', encoding='utf-8') as f:
            f.write('print("ALL GATES PASS (vault authority)")\n')
        # dup-id row is red under package gates; the vault script says green
        with open(os.path.join(self.L, 'transactions.jsonl'), 'a',
                  encoding='utf-8') as f:
            f.write(json.dumps(laneb.row(id='2026-08-01.01',
                                         date='2026-08-01')) + '\n')
        self.assertEqual(self._close(), 0)

    def test_vault_local_gen_views_red_blocks_close(self):
        with open(os.path.join(self.L, 'gen_views.py'), 'w', encoding='utf-8') as f:
            f.write('print("✗ FAIL Chain: vault says red")\n'
                    'import sys as _s; _s.exit(1)\n')
        self.assertNotEqual(self._close(), 0)
        self.assertEqual(closed_json(self.knobs), {})

    def test_archive_holds_full_ledger_copy_not_just_the_month(self):
        """R3.6: the archive ledger copy is an intentional restore point."""
        self.assertEqual(self._close(), 0)
        snap = read(os.path.join(self.root, '4-Archives', 'Finance', '2026-08',
                                 'transactions-2026-08.jsonl'))
        full = read(os.path.join(self.L, 'transactions.jsonl'))
        self.assertEqual(snap, full)
        self.assertIn('2026-07-05.01', snap)  # other months included

    def test_monthly_review_content(self):
        self.assertEqual(self._close(), 0)
        review = read(os.path.join(self.root, '4-Archives', 'Finance', '2026-08',
                                   'Monthly Review 2026-08.md'))
        self.assertIn('# 📋 Monthly Review — 2026-08 (CLOSED)', review)
        self.assertIn('frozen_total: 400000', review)
        self.assertIn('| Income (0 rows) | 0 |', review)
        self.assertIn('| Tracked expenses (2 rows) | 400.000 |', review)
        self.assertIn('| Food | 400.000 | 100% |', review)

    def test_default_month_is_previous(self):
        from petasos.close import close_month
        knobs = laneb.make_vault(self.root, rows=[laneb.row()], ledger_dir='Fin/B')
        rc = close_month(knobs, None)
        self.assertEqual(rc, 0)
        prev = (date.today().replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
        self.assertIn(prev, closed_json(knobs))


class TestCloseCLI(unittest.TestCase):
    def test_cli_freezes_month(self):
        with tempfile.TemporaryDirectory() as tmp:
            laneb.make_vault(tmp, rows=[
                laneb.row(id='2026-08-01.01', date='2026-08-01', amount=300000)])
            env = dict(os.environ, PETASOS_VAULT=tmp)
            r = subprocess.run([sys.executable, str(MONTHLY_CLOSE), '2026-08'],
                               capture_output=True, text=True, env=env, cwd=REPO)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('FROZEN 2026-08', r.stdout)
            closed = json.loads(read(os.path.join(
                tmp, '2-Areas/Finance/_Ledger', 'closed_months.json')))
            self.assertEqual(closed, {'2026-08': 300000})

    def test_cli_rejects_malformed_month(self):
        with tempfile.TemporaryDirectory() as tmp:
            laneb.make_vault(tmp, rows=[laneb.row()])
            env = dict(os.environ, PETASOS_VAULT=tmp)
            r = subprocess.run([sys.executable, str(MONTHLY_CLOSE), 'august'],
                               capture_output=True, text=True, env=env, cwd=REPO)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('YYYY-MM', r.stderr)


if __name__ == '__main__':
    unittest.main()
