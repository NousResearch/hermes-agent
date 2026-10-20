"""F-1 seam regression (t_d2548b02): pipeline ↔ gates on a batched ± pair.

An offsetting ± alert pair sharing one balance is ONE batched alert (13/10
incident log), not an unalerted transaction. petasos.pipeline.check_chain
accepts the pair and appends both rows — the REAL gen_views gates must then
accept those same rows, or the pipeline exits 1 AFTER the append and the
vault stays red on every later run (R3.2 violation).

Neither lane's isolated suite can catch this: pipeline tests run a FAKE
gen_views; gate tests hand-build rows. This file runs the true seam —
pipeline.main() LIVE over a scratch vault with the BUNDLED real
scripts/gen_views.py (views=None → no vault-local shadow) — so the C6
subprocess contract is exercised end-to-end. Found by the t_688aa7e3
review (F-1); unit-level pair semantics live in test_gates.TestChainGate.
"""
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_pipeline import FakeTransport, Vault, run_main  # noqa: E402


def acb_pair_body(marker, sign, amt, bal, acct='1234567', content='PAY SYNTH X'):
    """ACB balance-alert shape with an explicit ± decimal amount (13/10
    incident form: 'Ghi có +100,000.00' / 'Ghi nợ -100,000.00')."""
    return (f'<html><body>tài khoản {acct} của quý khách<br>'
            f'Số dư mới: {bal:,} VND tính đến 13/10/2026<br>'
            f'Giao dịch mới nhất: Ghi {marker} {sign}{amt:,}.00 VND<br>'
            f'Nội dung giao dịch: {content}.</body></html>')


def pair_threads():
    """The batched-alert pair: +100.000 in, −100.000 out, one shared balance."""
    return [
        {'thread_id': 'pair0001',
         'subject': 'bao so du ACB',
         'timestamp': '2026-10-13T11:00:00Z',
         'body': acb_pair_body('có', '+', 100000, 4800000,
                               content='BATCHED LEG IN')},
        {'thread_id': 'pair0002',
         'subject': 'bao so du ACB',
         'timestamp': '2026-10-13T11:05:00Z',
         'body': acb_pair_body('nợ', '-', 100000, 4800000,
                               content='BATCHED LEG OUT')},
    ]


class BatchedAlertPairSeam(unittest.TestCase):
    def _run_pair_live(self, vault):
        return run_main(vault, FakeTransport(pair_threads()))

    def test_live_pair_passes_the_real_gates_end_to_end(self):
        v = Vault(views=None)              # no vault-local shadow → bundled
        code, out = self._run_pair_live(v)  # REAL scripts/gen_views.py (C6)
        rows = v.rows()
        # the pipeline side accepted the pair: both legs appended, shapes exact
        self.assertEqual(len(rows), 2, out)
        self.assertEqual(sorted(r['dir'] for r in rows), ['in', 'out'])
        self.assertEqual({r['amount'] for r in rows}, {100000})
        self.assertEqual({r['bal_after'] for r in rows}, {4800000})
        # the gates side must agree on the SAME rows (the F-1 seam)
        self.assertEqual(code, 0, out)
        self.assertIn('ALL GATES PASS', out)

    def test_vault_is_not_stuck_red_after_the_pair_append(self):
        """R3.2 — after the write, EVERY later gates run passes too; the
        pre-fix defect left the vault failing gates until hand-repair."""
        v = Vault(views=None)
        self._run_pair_live(v)             # appends the pair whatever gates say
        self.assertEqual(len(v.rows()), 2)
        r = subprocess.run([sys.executable,
                            os.path.join(REPO, 'scripts', 'gen_views.py')],
                           capture_output=True, text=True, env=v.env())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('ALL GATES PASS', r.stdout)


if __name__ == '__main__':
    unittest.main()
