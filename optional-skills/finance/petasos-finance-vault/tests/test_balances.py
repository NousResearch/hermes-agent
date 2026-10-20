"""balances.json sync semantics (C5, S-2) — alerts authoritative.

Latest alert by recv_utc wins; an earlier alert NEVER resurrects a balance;
derived totals recompute from components on every sync; hand-maintained
non-alert accounts are preserved. All figures synthetic (R5.1).
"""
import copy
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from petasos.balances import recompute_totals, sync_from_alerts  # noqa: E402
from petasos.knob import load_knobs  # noqa: E402


def knobs_for():
    return load_knobs({'PETASOS_VAULT': '/nv', 'PETASOS_ACCT_NUM': '1234567',
                       'PETASOS_ACCT_KEY': 'BANK-A'})


def accounts():
    return {
        'BANK-A': {'kind': 'checking', 'balance': 0, 'verified': 'initial'},
        'SAVE-A': {'kind': 'savings', 'balance': 2000},
        'CASH-DRAWER': {'kind': 'cash', 'balance': 500},
        'TERM-1': {'kind': 'deposit', 'balance': 9000},
        'FUND-1': {'kind': 'fund', 'balance': 700},
        'PROP-1': {'kind': 'property', 'balance': 50000},
        'MORT-1': {'kind': 'mortgage', 'balance': 20000},
        'LOAN-1': {'kind': 'loan', 'balance': 3000},
        'CC-1': {'kind': 'cc', 'limit': 10000, 'outstanding': 250},
    }


def balances_obj():
    return {'_readme': 'synthetic', 'accounts': accounts(), 'snapshot_totals': {}}


def alert(recv_utc, bal_after, avail=None, acct='1234567', kind='ACB'):
    d = {'acct': acct, 'bal_after': bal_after, 'recv_utc': recv_utc, 'kind': kind}
    if avail is not None:
        d['avail'] = avail
    return d


class SyncFromAlerts(unittest.TestCase):
    def setUp(self):
        self.knobs = knobs_for()

    def test_latest_alert_by_recv_utc_wins(self):
        b = sync_from_alerts(balances_obj(),
                             [alert('2026-10-13 10:00', 1000),
                              alert('2026-10-13 09:00', 9000)], self.knobs)
        self.assertEqual(b['accounts']['BANK-A']['balance'], 1000)

    def test_earlier_alert_never_resurrects(self):
        # Order in the list must not matter: 09:00 first, 10:00 second.
        b = sync_from_alerts(balances_obj(),
                             [alert('2026-10-13 09:00', 9000),
                              alert('2026-10-13 10:00', 1000)], self.knobs)
        self.assertEqual(b['accounts']['BANK-A']['balance'], 1000)

    def test_verified_stamped_with_latest_alert(self):
        b = sync_from_alerts(balances_obj(), [alert('2026-10-13 10:00', 1000)],
                             self.knobs)
        v = b['accounts']['BANK-A']['verified']
        self.assertIn('email', v)
        self.assertIn('2026-10-13 10:00', v)
        self.assertIn('auto-sync', v)

    def test_available_recorded_when_present(self):
        b = sync_from_alerts(balances_obj(),
                             [alert('2026-10-13 10:00', 1000, avail=900)],
                             self.knobs)
        self.assertEqual(b['accounts']['BANK-A']['available'], 900)

    def test_pending_hold_available_recorded(self):
        # S-3: avail < balance is a hold — balances record it (the row
        # decision belongs to the pipeline).
        b = sync_from_alerts(balances_obj(),
                             [alert('2026-10-13 10:00', 1000, avail=750)],
                             self.knobs)
        self.assertEqual(b['accounts']['BANK-A']['available'], 750)
        self.assertEqual(b['accounts']['BANK-A']['balance'], 1000)

    def test_other_accounts_alerts_ignored(self):
        b = sync_from_alerts(balances_obj(),
                             [alert('2026-10-13 10:00', 1000, acct='9999999')],
                             self.knobs)
        self.assertEqual(b['accounts']['BANK-A']['balance'], 0)

    def test_hand_maintained_accounts_preserved_byte_identical(self):
        base = balances_obj()
        snapshot = copy.deepcopy(base['accounts'])
        b = sync_from_alerts(base, [alert('2026-10-13 10:00', 1000)], self.knobs)
        for name, entry in b['accounts'].items():
            if name != 'BANK-A':
                self.assertEqual(entry, snapshot[name], f'{name} must be untouched')

    def test_no_alerts_leaves_accounts_unchanged(self):
        base = balances_obj()
        snapshot = copy.deepcopy(base['accounts'])
        b = sync_from_alerts(base, [], self.knobs)
        self.assertEqual(b['accounts'], snapshot)


class DerivedTotals(unittest.TestCase):
    """C5/C6 — snapshot_totals are DERIVED, recomputed from components."""

    def test_totals_recomputed_from_components_on_every_sync(self):
        b = sync_from_alerts(balances_obj(), [alert('2026-10-13 10:00', 1000)],
                             knobs_for())
        t = b['snapshot_totals']
        # bank = checking 1000 + savings 2000; cash 500; assets add
        # deposit 9000 + fund 700 + property 50000; debts = mortgage 20000
        # + loan 3000 + cc outstanding 250.
        self.assertEqual(t['bank_total'], 3000)
        self.assertEqual(t['cash_total'], 500)
        self.assertEqual(t['assets_total'], 3000 + 500 + 9000 + 700 + 50000)
        self.assertEqual(t['debt_total'], 20000 + 3000 + 250)
        self.assertEqual(t['net_worth'], t['assets_total'] - t['debt_total'])
        self.assertEqual(t['liquid_total'], t['bank_total'] + t['cash_total'])

    def test_drifted_totals_repaired_by_recompute(self):
        b = balances_obj()
        b['snapshot_totals']['bank_total'] = 999999        # hand-corrupted
        recompute_totals(b)
        self.assertEqual(b['snapshot_totals']['bank_total'], 2000)

    def test_savings_prefix_kind_counts_as_bank(self):
        b = balances_obj()
        b['accounts']['SAVE-JOINT'] = {'kind': 'savings-joint', 'balance': 400}
        recompute_totals(b)
        self.assertEqual(b['snapshot_totals']['bank_total'], 2400)

    def test_cc_counts_via_outstanding_not_balance(self):
        b = balances_obj()
        b['accounts']['CC-1']['balance'] = 999999          # cc ignores balance
        recompute_totals(b)
        self.assertEqual(b['snapshot_totals']['debt_total'], 20000 + 3000 + 250)


if __name__ == '__main__':
    unittest.main()
