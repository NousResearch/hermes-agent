"""Lane B tests: petasos.gates — the gate evaluator (design §5, §6 C6, R3.3).

Gate set (R3.3 + DG-1 REC A, all resolved option A):
  (a) ID uniqueness
  (b) per closed month: ledger expense sum == frozen total   (R3.5/DG-10)
  (c) balances arithmetic: assets/debts/net_worth/liquid       (+ R3.4 drift)
  (d) chain: bal_after(n) = bal_after(n-1) -/+ amount(n) where evidence exists

stdout contract (C6/DG-15): one line per gate `✓ <name>` / `✗ FAIL <name> —
<detail>`; final line `ALL GATES PASS` or `GATES FAILED`.
"""
import copy
import unittest
from datetime import date
from decimal import Decimal  # noqa: F401  (amounts are ints; guard against float drift)

import laneb

from petasos.gates import GateReport, derive_totals, run_gates


def money(n):
    return f"{n:,.0f}".replace(',', '.')


class TestGateReportShape(unittest.TestCase):
    def test_report_is_dataclass_ok_and_lines(self):
        r = GateReport(ok=True, lines=['x'])
        self.assertTrue(r.ok)
        self.assertEqual(r.lines, ['x'])


class TestIDUniqueness(unittest.TestCase):
    def test_unique_ids_pass(self):
        rows = [laneb.row(id='2026-08-01.01'), laneb.row(id='2026-08-01.02')]
        rep = run_gates(rows, laneb.balances(), {})
        self.assertTrue(rep.ok)
        self.assertTrue(any(l.startswith('✓ ID uniqueness') for l in rep.lines))

    def test_duplicate_id_fails_naming_duplicates(self):
        rows = [laneb.row(id='2026-08-01.01'), laneb.row(id='2026-08-01.01'),
                laneb.row(id='2026-08-01.02')]
        rep = run_gates(rows, laneb.balances(), {})
        self.assertFalse(rep.ok)
        line = next(l for l in rep.lines if 'ID uniqueness' in l)
        self.assertTrue(line.startswith('✗ FAIL'))
        self.assertIn('DUPES', line)
        self.assertIn('2026-08-01.01', line)


class TestFrozenMonths(unittest.TestCase):
    def test_frozen_month_drift_fails_with_both_totals(self):
        rows = [laneb.row(id='2026-08-01.01', date='2026-08-01', amount=150000),
                laneb.row(id='2026-08-02.01', date='2026-08-02', amount=250000)]
        rep = run_gates(rows, laneb.balances(), {'2026-08': 350000})
        self.assertFalse(rep.ok)
        line = next(l for l in rep.lines if '2026-08' in l)
        self.assertTrue(line.startswith('✗ FAIL'))
        self.assertIn('350.000', line)   # frozen
        self.assertIn('400.000', line)   # live ledger sum

    def test_frozen_month_matching_sum_passes(self):
        rows = [laneb.row(id='2026-08-01.01', date='2026-08-01', amount=400000)]
        rep = run_gates(rows, laneb.balances(), {'2026-08': 400000})
        self.assertTrue(rep.ok)
        self.assertTrue(any(l.startswith('✓') and '2026-08' in l for l in rep.lines))

    def test_late_income_into_closed_month_does_not_fail_gate(self):
        """DG-10/R3.5: freeze covers EXPENSES only; late income may append."""
        rows = [laneb.row(id='2026-08-01.01', date='2026-08-01', amount=400000),
                laneb.row(id='2026-08-15.01', date='2026-08-15', dir='in',
                          kind='salary', amount=5000000, old_ref='mail:feed0001')]
        rep = run_gates(rows, laneb.balances(), {'2026-08': 400000})
        self.assertTrue(rep.ok)

    def test_expense_append_into_closed_month_fails_gate(self):
        """R3.5: an expense landing in a closed month drifts the frozen sum."""
        rows = [laneb.row(id='2026-08-01.01', date='2026-08-01', amount=400000),
                laneb.row(id='2026-08-20.01', date='2026-08-20', amount=50000)]
        rep = run_gates(rows, laneb.balances(), {'2026-08': 400000})
        self.assertFalse(rep.ok)


class TestBalancesArithmetic(unittest.TestCase):
    def test_derive_totals_all_kinds(self):
        totals = derive_totals(laneb.starter_accounts())
        self.assertEqual(totals, {
            'bank_total': 1500,        # checking + savings + savings* prefix
            'cash_total': 50,
            'deposits_total': 5000,
            'funds_total': 700,
            'property_total': 9000,
            'assets_total': 16250,
            'cc_outstanding_total': 250,
            'debts_total': 4850,       # mortgage 4000 + loan 600 + cc 250
            'net_worth': 11400,
            'liquid_total': 1550,
        })

    def test_unused_kinds_sum_to_zero(self):
        totals = derive_totals({})
        self.assertEqual(totals['assets_total'], 0)
        self.assertEqual(totals['net_worth'], 0)

    def test_hand_corrupted_snapshot_total_flags_drift(self):
        """R3.4: drift is FLAGGED in gate output (repair+persistence is views' job)."""
        accounts = {'CASH-DRAWER': {'kind': 'cash', 'balance': 50}}
        stored = {'cash_total': 999, 'net_worth': 0, 'assets_total': 0}
        rep = run_gates([], {'accounts': accounts, 'snapshot_totals': stored}, {})
        self.assertFalse(rep.ok)
        line = next(l for l in rep.lines if 'assets−debts' in l)
        self.assertTrue(line.startswith('✗ FAIL'))
        self.assertIn('DRIFT', line)
        self.assertIn('cash_total', line)

    def test_liquid_identity_gate_passes(self):
        rep = run_gates([], {'accounts': laneb.starter_accounts(),
                             'snapshot_totals': {}}, {})
        self.assertTrue(any(l.startswith('✓ Arithmetic: liquid') for l in rep.lines))


class TestChainGate(unittest.TestCase):
    """DG-1 REC A: ledger gate asserts per-account bal_after consistency
    where evidence exists (consecutive bal_after-bearing rows)."""

    def test_consistent_chain_passes(self):
        rows = [
            laneb.row(id='2026-10-01.01', date='2026-10-01', dir='out',
                      amount=100, bal_after=900),
            laneb.row(id='2026-10-01.02', date='2026-10-01', dir='out',
                      amount=50, bal_after=850),
            laneb.row(id='2026-10-02.01', date='2026-10-02', dir='in',
                      kind='salary', amount=200, bal_after=1050,
                      old_ref='mail:feed0002'),
        ]
        rep = run_gates(rows, laneb.balances(), {})
        self.assertTrue(any(l.startswith('✓ Chain') for l in rep.lines), rep.lines)
        self.assertTrue(rep.ok)

    def test_chain_break_fails_naming_the_break(self):
        rows = [
            laneb.row(id='2026-10-01.01', date='2026-10-01', dir='out',
                      amount=100, bal_after=900),
            laneb.row(id='2026-10-01.02', date='2026-10-01', dir='out',
                      amount=50, bal_after=700),  # expected 850
        ]
        rep = run_gates(rows, laneb.balances(), {})
        self.assertFalse(rep.ok)
        line = next(l for l in rep.lines if 'Chain' in l)
        self.assertTrue(line.startswith('✗ FAIL'))
        self.assertIn('2026-10-01.02', line)
        self.assertIn('850', line)

    def test_no_evidence_passes(self):
        rows = [laneb.row(id='2026-10-01.01'),  # no bal_after anywhere
                laneb.row(id='2026-10-01.02')]
        rep = run_gates(rows, laneb.balances(), {})
        self.assertTrue(any(l.startswith('✓ Chain') for l in rep.lines), rep.lines)
        self.assertTrue(rep.ok)

    def test_offsetting_pair_sharing_balance_passes(self):
        """F-1 (t_d2548b02): an offsetting ± pair sharing one balance is ONE
        batched alert (13/10 incident) — the pipeline's check_chain accepts
        it and appends both rows, so the ledger gate must accept them too."""
        rows = [
            laneb.row(id='2026-10-13.01', date='2026-10-13', dir='in',
                      kind='oneoff', amount=100000, bal_after=4800000,
                      old_ref='mail:pairin01'),
            laneb.row(id='2026-10-13.02', date='2026-10-13', dir='out',
                      amount=100000, bal_after=4800000,
                      old_ref='mail:pairout2'),
        ]
        rep = run_gates(rows, laneb.balances(), {})
        self.assertTrue(any(l.startswith('✓ Chain') for l in rep.lines), rep.lines)
        self.assertTrue(rep.ok, rep.lines)

    def test_chain_continues_from_the_pairs_shared_balance(self):
        """The pair is balance-neutral: the next row chains from the shared
        bal_after, mirroring pipeline.check_chain's semantics exactly."""
        rows = [
            laneb.row(id='2026-10-13.01', date='2026-10-13', dir='out',
                      amount=500, bal_after=1000),      # pair leg 1 (hold)
            laneb.row(id='2026-10-13.02', date='2026-10-13', dir='in',
                      kind='oneoff', amount=500, bal_after=1000,
                      old_ref='mail:pairleg2'),          # pair leg 2 (reversal)
            laneb.row(id='2026-10-13.03', date='2026-10-13', dir='out',
                      amount=100, bal_after=900,
                      old_ref='mail:pairnext3'),         # chains from 1000
        ]
        rep = run_gates(rows, laneb.balances(), {})
        self.assertTrue(any(l.startswith('✓ Chain') for l in rep.lines), rep.lines)
        self.assertTrue(rep.ok, rep.lines)


class TestTaxonomyAndContract(unittest.TestCase):
    def test_new_open_month_transaction_never_fails_gate(self):
        """R3.7: LIVE periods are info-only; asserting them would fail on every
        legitimate new transaction."""
        cur = date.today()
        rows = [laneb.row(id=f'{cur.isoformat()}.01', date=cur.isoformat(),
                          amount=123456)]
        rep = run_gates(rows, {'accounts': laneb.starter_accounts(),
                               'snapshot_totals': {}}, {})
        self.assertTrue(rep.ok, rep.lines)

    def test_stdout_contract_lines_and_final_verdict(self):
        ok_rep = run_gates([laneb.row()], {'accounts': {}, 'snapshot_totals': {}}, {})
        self.assertEqual(ok_rep.lines[-1], 'ALL GATES PASS')
        for l in ok_rep.lines[:-1]:
            if l.startswith('ℹ'):
                continue
            self.assertTrue(l.startswith('✓ ') or l.startswith('✗ FAIL '), l)

        bad = run_gates([laneb.row(), laneb.row()], {'accounts': {}, 'snapshot_totals': {}}, {})
        self.assertEqual(bad.lines[-1], 'GATES FAILED')
        self.assertFalse(bad.ok)

    def test_run_gates_is_pure_over_its_inputs(self):
        """§4: gate evaluator is pure — snapshot-repair mutation belongs to views."""
        rows = [laneb.row()]
        bal = {'accounts': {'CASH-DRAWER': {'kind': 'cash', 'balance': 50}},
               'snapshot_totals': {'cash_total': 999}}
        rows_before, bal_before = copy.deepcopy(rows), copy.deepcopy(bal)
        run_gates(rows, bal, {})
        self.assertEqual(rows, rows_before)
        self.assertEqual(bal, bal_before)  # NOT repaired in place


if __name__ == '__main__':
    unittest.main()
