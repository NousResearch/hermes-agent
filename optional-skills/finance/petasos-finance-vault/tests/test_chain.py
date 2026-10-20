"""Chain verification (R1.3, R1.8, G1/DG-1) — bal(n) = bal(n−1) ∓ amount(n)
per account, checked BEFORE writing anything; an offsetting ± pair sharing
one balance is ONE batched alert, not an unalerted transaction. Same-day
rows order by narrated time (R1.8). All figures synthetic (R5.1).
"""
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from petasos.pipeline import check_chain, narrated_time, order_txs  # noqa: E402


def tx(acct='1001', bal=None, amount=100, dir='OUT', dt=None, date=None, recv=''):
    d = {'acct': acct, 'amount': amount, 'dir': dir, 'kind': 'ACB'}
    if bal is not None:
        d['bal_after'] = bal
    if dt:
        d['datetime'] = dt
    if date:
        d['date'] = date
    if recv:
        d['recv_utc'] = recv
    return d


class NarratedTime(unittest.TestCase):
    def test_datetime_field_normalizes(self):
        self.assertEqual(narrated_time(tx(dt='13/10/26 10:25:55')), '2026-10-13 10:25:55')
        self.assertEqual(narrated_time(tx(dt='13/10/2026 10:25:55')), '2026-10-13 10:25:55')

    def test_date_only_falls_back_to_midnight(self):
        self.assertEqual(narrated_time(tx(date='13/10/2026')), '2026-10-13 00:00:00')

    def test_iso_date_passthrough(self):
        self.assertEqual(narrated_time(tx(date='2026-12-01')), '2026-12-01 00:00:00')


class OrderByNarratedTime(unittest.TestCase):
    """R1.8 — out-of-order same-day alerts yield narrated-time order."""

    def test_out_of_order_alerts_sorted_into_true_time_order(self):
        later = tx(bal=50, amount=50, dt='13/10/2026 10:05:00')
        earlier = tx(bal=100, amount=50, dt='13/10/2026 10:00:00')
        ordered = order_txs([later, earlier])
        self.assertEqual([t['bal_after'] for t in ordered], [100, 50])

    def test_sorted_chain_is_then_monotone(self):
        later = tx(bal=50, amount=50, dt='13/10/2026 10:05:00')
        earlier = tx(bal=100, amount=50, dt='13/10/2026 10:00:00')
        self.assertEqual(check_chain(order_txs([later, earlier])), [])


class CheckChain(unittest.TestCase):
    def test_clean_chain_passes(self):
        txs = [tx(bal=100, amount=50, dt='13/10/2026 09:00:00'),
               tx(bal=50, amount=50, dt='13/10/2026 09:05:00')]
        self.assertEqual(check_chain(txs), [])

    def test_credit_debit_chain_passes(self):
        txs = [tx(bal=100, amount=50, dt='13/10/2026 09:00:00'),
               tx(bal=150, amount=50, dir='IN', dt='13/10/2026 09:05:00')]
        self.assertEqual(check_chain(txs), [])

    def test_contradicting_sequence_reports_break(self):
        txs = [tx(bal=100, amount=50, dt='13/10/2026 09:00:00'),
               tx(bal=999, amount=50, dt='13/10/2026 09:05:00')]   # expected 50
        breaks = check_chain(txs)
        self.assertEqual(len(breaks), 1)
        self.assertIn('1001', breaks[0])                    # account named

    def test_offsetting_pair_sharing_balance_is_one_batched_alert(self):
        txs = [tx(bal=1000, amount=500, dt='13/10/2026 09:00:00'),   # hold
               tx(bal=1000, amount=500, dir='IN', dt='13/10/2026 09:01:00'),  # reversal
               tx(bal=900, amount=100, dt='13/10/2026 09:05:00')]    # chains from 1000
        self.assertEqual(check_chain(txs), [])

    def test_breaks_isolate_per_account(self):
        txs = [tx(acct='1001', bal=100, amount=50, dt='13/10/2026 09:00:00'),
               tx(acct='1001', bal=50, amount=50, dt='13/10/2026 09:05:00'),
               tx(acct='2002', bal=100, amount=50, dt='13/10/2026 09:00:00'),
               tx(acct='2002', bal=1, amount=50, dt='13/10/2026 09:05:00')]
        breaks = check_chain(txs)
        self.assertEqual(len(breaks), 1)
        self.assertIn('2002', breaks[0])

    def test_alerts_without_balance_are_ignored(self):
        txs = [tx(bal=None, amount=50), tx(bal=None, amount=60, dir='IN')]
        self.assertEqual(check_chain(txs), [])


if __name__ == '__main__':
    unittest.main()
