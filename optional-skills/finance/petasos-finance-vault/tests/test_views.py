"""Lane B tests: petasos.views — view regeneration + gate subprocess contract
(design §5/§6 C6, R3.1/R3.2/R3.4, S-4, DG-9/DG-15).

Views are GENERATED and gated: Dashboards.md / Salary History.md are written
beside the ledger dir, snapshot_totals are repaired+persisted (DG-9), and the
CLI exits 0 iff ALL GATES PASS — the property the pipeline (Lane A, M3 wiring)
depends on for "abort the write pipeline when any gate fails".
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import laneb

REPO = Path(__file__).resolve().parent.parent
GEN_VIEWS = REPO / 'scripts' / 'gen_views.py'


def read(path):
    return Path(path).read_text(encoding='utf-8')


class TestRegeneration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def test_writes_dashboard_and_salary_views_beside_ledger(self):
        rows = [
            laneb.row(id='2026-10-01.01', date='2026-10-01', amount=100000,
                      category='Food'),
            laneb.row(id='2026-10-05.01', date='2026-10-05', dir='in',
                      kind='salary', amount=700000, old_ref='mail:feed000a',
                      via='EMP-A'),
        ]
        knobs = laneb.make_vault(self.root, rows=rows)
        from petasos.views import regenerate
        rep = regenerate(knobs)
        self.assertTrue(rep.ok)
        dash = read(os.path.join(self.root, '2-Areas/Finance/Dashboards.md'))
        self.assertIn('# 📊 Dashboards', dash)
        self.assertIn('generator: gen_views.py', dash)
        self.assertIn('| **NET WORTH** | **11.400** |', dash)   # starter accounts
        self.assertIn('| 2026-10 | 700.000 | 100.000 | 600.000 |', dash)
        self.assertIn('Monthly flow (transfers excluded)', dash)
        sal = read(os.path.join(self.root, '2-Areas/Finance/Salary History.md'))
        self.assertIn('# 💼 Salary History (auto-generated)', sal)
        self.assertIn('| 2026 | EMP-A | 700.000 |', sal)

    def test_regeneration_idempotent_modulo_timestamp(self):
        """R3.1: unchanged ledger+balances → identical view content (modulo
        the generated timestamp); the ledger/state/closed files are untouched
        (C6: the view side NEVER writes them)."""
        rows = [laneb.row(id='2026-10-01.01', amount=100000, category='Food')]
        knobs = laneb.make_vault(self.root, rows=rows)
        L = laneb.ledger_path(knobs)
        from petasos.views import regenerate
        regenerate(knobs)
        first = read(os.path.join(L, '..', 'Dashboards.md'))
        protected = {name: read(os.path.join(L, name)) for name in
                     ('transactions.jsonl', 'balances.json', 'closed_months.json')}
        protected['state marker'] = 'absent'  # no state.json created
        regenerate(knobs)
        second = read(os.path.join(L, '..', 'Dashboards.md'))
        strip = lambda t: '\n'.join(l for l in t.splitlines()
                                    if not l.startswith('generated:'))
        self.assertEqual(strip(first), strip(second))
        for name in ('transactions.jsonl', 'closed_months.json'):
            self.assertEqual(protected[name], read(os.path.join(L, name)))
        self.assertNotIn('state.json', os.listdir(L))

    def test_hand_corrupted_snapshot_total_repaired_and_flagged(self):
        """R3.4/DG-9: repair persists on the drift run; next run is clean."""
        rows = [laneb.row(id='2026-10-01.01', amount=100000)]
        knobs = laneb.make_vault(self.root, rows=rows,
                                 snapshot_totals={'cash_total': 999})
        L = laneb.ledger_path(knobs)
        from petasos.views import regenerate
        rep = regenerate(knobs)  # drift run
        self.assertFalse(rep.ok)
        self.assertTrue(any('DRIFT' in l for l in rep.lines), rep.lines)
        bal = json.loads(read(os.path.join(L, 'balances.json')))
        self.assertEqual(bal['snapshot_totals']['cash_total'], 50)  # repaired
        rep2 = regenerate(knobs)  # clean run
        self.assertTrue(rep2.ok)

    def test_transfers_excluded_from_flow(self):
        rows = [
            laneb.row(id='2026-10-01.01', amount=100000, category='Food'),
            laneb.row(id='2026-10-02.01', date='2026-10-02', kind='transfer',
                      amount=999999, desc='internal move'),
        ]
        knobs = laneb.make_vault(self.root, rows=rows)
        from petasos.views import regenerate
        regenerate(knobs)
        dash = read(os.path.join(laneb.ledger_path(knobs), '..', 'Dashboards.md'))
        self.assertIn('| 2026-10 | 0 | 100.000 | -100.000 |', dash)
        self.assertNotIn('999.999', dash)

    def test_salary_view_only_written_with_salary_rows(self):
        knobs = laneb.make_vault(self.root, rows=[laneb.row(amount=50000)])
        from petasos.views import regenerate
        regenerate(knobs)
        self.assertFalse(os.path.exists(
            os.path.join(laneb.ledger_path(knobs), '..', 'Salary History.md')))

    def test_year_filtered_salary_totals_differ_from_raw_sum(self):
        """S-4: totals come from the generated year-filtered view, never an
        ad-hoc unfiltered sum."""
        rows = [
            laneb.row(id='2025-03-10.01', date='2025-03-10', dir='in',
                      kind='salary', amount=700000, old_ref='mail:feed0001',
                      via='EMP-A'),
            laneb.row(id='2025-09-10.01', date='2025-09-10', dir='in',
                      kind='salary', amount=500000, old_ref='mail:feed0002',
                      via='EMP-A'),
            laneb.row(id='2026-03-10.01', date='2026-03-10', dir='in',
                      kind='salary', amount=800000, old_ref='mail:feed0003',
                      via='EMP-B'),
        ]
        knobs = laneb.make_vault(self.root, rows=rows)
        from petasos.views import regenerate
        regenerate(knobs)
        sal = read(os.path.join(laneb.ledger_path(knobs), '..', 'Salary History.md'))
        self.assertIn('| 2025 | EMP-A | 1.200.000 |', sal)
        self.assertIn('| 2026 | EMP-B | 800.000 |', sal)
        self.assertNotIn('2.000.000', sal)  # raw sum must not appear unfiltered

    def test_needs_confirm_section_clears_after_owner_confirmation(self):
        """Card invariant #4 (R2.5 'never re-asked'): open confirmations are
        surfaced on the dashboard exactly until the owner classifies them;
        after the relabel-style confirmation they never reappear."""
        rows = [laneb.row(id='2026-10-02.01', date='2026-10-02',
                          status='needs-confirm', amount=700000,
                          ask='source unknown', desc='mystery credit',
                          kind='oneoff', dir='in')]
        knobs = laneb.make_vault(self.root, rows=rows)
        L = laneb.ledger_path(knobs)
        from petasos.views import regenerate
        regenerate(knobs)
        dash = read(os.path.join(L, '..', 'Dashboards.md'))
        self.assertIn('Open confirmations', dash)
        self.assertIn('mystery credit', dash)
        # owner confirms: relabel in place (C1 permitted fields only)
        path = os.path.join(L, 'transactions.jsonl')
        confirmed = json.loads(read(path))
        confirmed.update(status='posted', kind='salary', category='Salary',
                         src='user-confirm:2026-10-13')
        confirmed.pop('ask', None)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(confirmed) + '\n')
        regenerate(knobs)
        dash2 = read(os.path.join(L, '..', 'Dashboards.md'))
        self.assertNotIn('Open confirmations', dash2)
        self.assertNotIn('mystery credit', dash2)

    def test_custom_ledger_dir_views_land_in_its_parent(self):
        knobs = laneb.make_vault(self.root, rows=[laneb.row()],
                                 ledger_dir='Fin/Books')
        from petasos.views import regenerate
        regenerate(knobs)
        self.assertTrue(os.path.exists(os.path.join(self.root, 'Fin', 'Dashboards.md')))


class TestCLIContract(unittest.TestCase):
    """C6/R3.2: gen_views is a subprocess with a stdout contract — the
    executable form of 'after every ledger write ALL gates pass or the write
    pipeline aborts' (Lane A's pipeline wiring lands at M3 and MUST treat a
    nonzero exit as fatal)."""

    def _run(self, vault, ledger_dir='2-Areas/Finance/_Ledger'):
        env = dict(os.environ, PETASOS_VAULT=vault, PETASOS_LEDGER_DIR=ledger_dir)
        return subprocess.run([sys.executable, str(GEN_VIEWS)],
                              capture_output=True, text=True, env=env, cwd=REPO)

    def test_healthy_vault_exits_zero_all_gates_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            laneb.make_vault(tmp, rows=[laneb.row()])
            r = self._run(tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip().splitlines()[-1], 'ALL GATES PASS')

    def test_corrupted_ledger_exits_nonzero_naming_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [laneb.row(id='2026-10-01.01'), laneb.row(id='2026-10-01.01')]
            laneb.make_vault(tmp, rows=rows)
            r = self._run(tmp)
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual(r.stdout.strip().splitlines()[-1], 'GATES FAILED')
            self.assertIn('ID uniqueness', r.stdout)

    def test_missing_ledger_fails_loud_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            laneb.make_vault(tmp, rows=[])
            os.remove(os.path.join(laneb.ledger_path(laneb.knobs_for(tmp)),
                                   'transactions.jsonl'))
            r = self._run(tmp)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('transactions.jsonl', r.stderr)

    def test_missing_vault_knob_fails_loud(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ('PETASOS_VAULT', 'OBSIDIAN_VAULT_PATH')}
        r = subprocess.run([sys.executable, str(GEN_VIEWS)],
                           capture_output=True, text=True, env=env, cwd=REPO)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('PETASOS_VAULT', r.stderr)


if __name__ == '__main__':
    unittest.main()
