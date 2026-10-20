"""Pipeline orchestration (INV-1, G2, G5, G6, G7, S-5, R4.5, R3.2, R5.3,
C2/C6/C8, DG-2/5/6/7/15). Transport/views/clock are injected; the fake
gen_views script exercises the C6 subprocess contract. All data synthetic.
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from petasos.pipeline import main, resolve_views_script  # noqa: E402
from petasos.digest import render_digest  # noqa: E402


# ---------------- scratch vault + fake transport ----------------

GOOD_VIEWS = "import sys\nprint('ALL GATES PASS')\n"
BAD_VIEWS = "import sys\nprint('GATES FAILED')\nprint('✗ FAIL dup-id — test', file=sys.stderr)\nsys.exit(1)\n"


def acb_body(bal, amt, acct='1234567', content='PAY SYNTH X'):
    return (f'<html><body>tài khoản {acct} của quý khách<br>'
            f'Số dư mới: {bal:,} VND tính đến 13/10/2026<br>'
            f'Giao dịch mới nhất: Ghi nợ {amt:,} VND<br>'
            f'Nội dung giao dịch: {content}.</body></html>')


class FakeTransport:
    """AgentMail-shaped fake: threads list + per-thread detail."""

    def __init__(self, threads):
        self.threads = threads      # [{'thread_id','subject','timestamp','body'}]
        self.calls = []

    def __call__(self, path):
        from urllib.parse import urlparse, parse_qs
        self.calls.append(path)
        if path.endswith('threads') or '/threads?' in path:
            return {'threads': [{'thread_id': t['thread_id'], 'subject': t['subject'],
                                 'timestamp': t['timestamp']} for t in self.threads]}
        tid = path.rstrip('/').split('/')[-1]
        for t in self.threads:
            if t['thread_id'] == tid:
                return {'messages': [{'subject': t['subject'],
                                      'extracted_html': t['body'],
                                      'timestamp': t['timestamp']}]}
        raise urllib.error.HTTPError(path, 404, 'nope', None, None)


def acb_thread(tid, bal, amt, ts='2026-10-13T09:00:00Z', subject='bao so du ACB',
               body=None):
    return {'thread_id': tid, 'subject': subject, 'timestamp': ts,
            'body': body if body is not None else acb_body(bal, amt)}


class Vault:
    """Scratch vault with ledger tree, rules copy, fake gen_views."""

    def __init__(self, ledger_dir='9-Test/_Ledger', views=GOOD_VIEWS, rows=None,
                 state=None, balances=None, with_banks=False):
        self.root = tempfile.mkdtemp(prefix='petasos-pipe-vault-')
        self.ledger_rel = ledger_dir
        self.ledger = os.path.join(self.root, ledger_dir)
        os.makedirs(self.ledger, exist_ok=True)
        with open(os.path.join(self.ledger, 'transactions.jsonl'), 'w', encoding='utf-8') as f:
            for r in (rows or []):
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        self.state_path = os.path.join(self.ledger, 'state.json')
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(state if state is not None else {'schema': 1, 'seen_threads': []}, f)
        with open(os.path.join(self.ledger, 'balances.json'), 'w', encoding='utf-8') as f:
            json.dump(balances if balances is not None else {
                '_readme': 'synthetic', 'snapshot_totals': {},
                'accounts': {'BANK-A': {'kind': 'checking', 'balance': 0},
                             'CASH': {'kind': 'cash', 'balance': 100}}}, f, indent=1)
        with open(os.path.join(self.ledger, 'rules.md'), 'w', encoding='utf-8') as f:
            f.write(open(os.path.join(REPO, 'templates', 'rules.md'),
                         encoding='utf-8').read())
        if views is not None:
            with open(os.path.join(self.ledger, 'gen_views.py'), 'w', encoding='utf-8') as f:
                f.write(views)
        if with_banks:
            with open(os.path.join(self.ledger, 'banks.yaml'), 'w', encoding='utf-8') as f:
                f.write(open(os.path.join(REPO, 'banks.yaml'), encoding='utf-8').read())

    def env(self, **extra):
        env = {'PETASOS_VAULT': self.root, 'PETASOS_LEDGER_DIR': self.ledger_rel,
               'PETASOS_ACCT_NUM': '1234567', 'PETASOS_ACCT_KEY': 'BANK-A',
               'AGENTMAIL_API_KEY': 'test-key', 'PETASOS_INBOX': 'test@inbox.test',
               'PETASOS_OWN_NAMES': 'ANN LEE'}
        env.update(extra)
        return env

    def snapshot(self):
        """(path, sha256, mtime_ns, ino) for every file under the vault."""
        snap = {}
        for dirpath, _dirs, files in os.walk(self.root):
            for name in sorted(files):
                p = os.path.join(dirpath, name)
                st = os.stat(p)
                with open(p, 'rb') as f:
                    snap[p] = (hashlib.sha256(f.read()).hexdigest(),
                               st.st_mtime_ns, st.st_ino)
        return snap

    def rows(self):
        p = os.path.join(self.ledger, 'transactions.jsonl')
        with open(p, encoding='utf-8') as f:
            return [json.loads(l) for l in f if l.strip()]

    def state(self):
        with open(self.state_path, encoding='utf-8') as f:
            return json.load(f)


def run_main(vault, transport, argv=(), **env_extra):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv), environ=vault.env(**env_extra), transport=transport)
    return code, out.getvalue() + err.getvalue()


# ---------------- tests ----------------

class DryWritesNothing(unittest.TestCase):
    """INV-1 — full-vault hash identical across --dry (state.json included)."""

    def test_dry_changes_no_file_by_hash_or_mtime(self):
        v = Vault(rows=[{'id': '2026-10-12.01', 'date': '2026-10-12', 'dir': 'out',
                         'kind': 'expense', 'old_ref': 'mail:oldthread', 'amount': 1,
                         'currency': 'VND', 'desc': 'x', 'status': 'posted',
                         'src': 'email:ACB'}])
        before = v.snapshot()
        digest_dir = os.path.join(tempfile.gettempdir(), 'petasos-pipe')
        digest_existed = os.path.exists(digest_dir)
        code, out = run_main(v, FakeTransport([acb_thread('aaaa0001', 2000, 500),
                                               acb_thread('aaaa0002', 1500, 500,
                                                          ts='2026-10-13T09:05:00Z')]),
                             argv=['--dry'])
        self.assertEqual(code, 0, out)
        self.assertEqual(v.snapshot(), before)          # hashes AND mtimes identical
        if not digest_existed:
            self.assertFalse(os.path.exists(digest_dir))  # DG-5: no scratch effects

    def test_dry_does_not_mark_seen(self):
        # The fetch-marks-seen trap: what a dry preview showed must still be
        # processed by the next live run.
        v = Vault()
        t = FakeTransport([acb_thread('bbbb0001', 2000, 500)])
        code, _ = run_main(v, t, argv=['--dry'])
        self.assertEqual(code, 0)
        code, out = run_main(v, t)                       # live, same threads
        self.assertEqual(code, 0, out)
        self.assertEqual(len(v.rows()), 1)               # dry consumed nothing


class LiveRun(unittest.TestCase):
    def test_full_run_appends_verifies_syncs_state_and_digest(self):
        v = Vault()
        code, out = run_main(v, FakeTransport([acb_thread('cccc0001', 2000, 500),
                                               acb_thread('cccc0002', 1500, 500,
                                                          ts='2026-10-13T09:05:00Z')]))
        self.assertEqual(code, 0, out)
        rows = v.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['id'], '2026-10-13.01')  # narrated-time order
        self.assertEqual(rows[0]['old_ref'], 'mail:cccc0001')   # DG-7 full id
        self.assertEqual(rows[0]['src'], 'email:ACB')
        state = v.state()
        self.assertIn('cccc0001', state['seen_threads'])
        self.assertIn('cccc0002', state['seen_threads'])
        self.assertTrue(state['last_run'])
        with open(os.path.join(v.ledger, 'balances.json'), encoding='utf-8') as f:
            bal = json.load(f)
        self.assertEqual(bal['accounts']['BANK-A']['balance'], 1500)  # latest alert
        self.assertIn('ALL GATES PASS', out)             # gate stdout passed through
        self.assertIn('2 new transactions', out)         # DG-15 cron string

    def test_run_with_no_fresh_threads_updates_last_run_only(self):
        v = Vault(state={'schema': 1, 'seen_threads': ['dddd0001']})
        code, out = run_main(v, FakeTransport([acb_thread('dddd0001', 2000, 500)]))
        self.assertEqual(code, 0, out)
        self.assertEqual(v.rows(), [])
        self.assertTrue(v.state()['last_run'])

    def test_state_loss_rerun_appends_zero_duplicates(self):
        # R1.9: delete state.json; the old_ref guard covers already-ledgered
        # threads.
        v = Vault()
        code, _ = run_main(v, FakeTransport([acb_thread('eeee0001', 2000, 500)]))
        self.assertEqual(code, 0)
        self.assertEqual(len(v.rows()), 1)
        os.unlink(v.state_path)                          # state lost
        code, out = run_main(v, FakeTransport([acb_thread('eeee0001', 2000, 500)]))
        self.assertEqual(code, 0, out)
        self.assertEqual(len(v.rows()), 1)               # zero duplicates

    def test_v1_era_8char_refs_still_guard(self):
        # DG-7 swap compat: guard prefix-matches v1-era refs.
        v = Vault(rows=[{'id': '2026-10-13.01', 'date': '2026-10-13', 'dir': 'out',
                         'kind': 'expense', 'old_ref': 'mail:ffff0001', 'amount': 1,
                         'currency': 'VND', 'desc': 'x', 'status': 'posted',
                         'src': 'email:ACB'}])
        code, out = run_main(v, FakeTransport(
            [acb_thread('ffff0001XYZW', 2000, 500)]))    # full id, v1-era prefix
        self.assertEqual(code, 0, out)
        self.assertEqual(len(v.rows()), 1)               # not re-appended


class WriteOrderAndSerialization(unittest.TestCase):
    """DG-6 — append → verify → balances → state → gates → digest; lockfile."""

    def test_balances_failure_leaves_ledger_appended_state_unwritten(self):
        import petasos.pipeline as pp
        v = Vault()
        orig = pp.sync_from_alerts

        def boom(balances, txs, knobs):
            raise RuntimeError('simulated crash between append and balances')

        pp.sync_from_alerts = boom
        try:
            code, out = run_main(v, FakeTransport([acb_thread('gggg0001', 2000, 500)]))
        finally:
            pp.sync_from_alerts = orig
        self.assertNotEqual(code, 0)
        self.assertEqual(len(v.rows()), 1)               # ledger IS authoritative
        self.assertNotIn('gggg0001', v.state()['seen_threads'])  # state not reached

    def test_gates_failure_after_append_is_fatal_nonzero(self):
        v = Vault(views=BAD_VIEWS)
        code, out = run_main(v, FakeTransport([acb_thread('hhhh0001', 2000, 500)]))
        self.assertNotEqual(code, 0)                     # R3.2 fatal
        self.assertEqual(len(v.rows()), 1)               # append already happened
        self.assertIn('GATES FAILED', out)

    def test_second_concurrent_run_exits_loudly(self):
        v = Vault()
        lock = os.path.join(v.ledger, 'petasos.lock')
        with open(lock, 'w', encoding='utf-8') as f:
            f.write('synthetic holder\n')
        before = v.snapshot()
        code, out = run_main(v, FakeTransport([acb_thread('iiii0001', 2000, 500)]))
        self.assertNotEqual(code, 0)                     # S-5 loud
        self.assertIn('lock', out.lower())
        self.assertEqual(v.snapshot(), before)           # touched NOTHING


class ChainBreakAppendsNothing(unittest.TestCase):
    def test_contradicting_balances_append_nothing_and_report(self):
        v = Vault()
        code, out = run_main(v, FakeTransport(
            [acb_thread('jjjj0001', 2000, 500),
             acb_thread('jjjj0002', 999, 500, ts='2026-10-13T09:05:00Z')]))  # expected 1500
        self.assertNotEqual(code, 0)                     # R1.3
        self.assertIn('chain break', out)
        self.assertEqual(v.rows(), [])                   # nothing appended
        self.assertNotIn('jjjj0001', v.state()['seen_threads'])  # no seen-marking


class UnparsedSurfacing(unittest.TestCase):
    """DG-2 — subject-keyword hit whose body fails requires: ASK + probe hint,
    no fabricated row, thread marked seen."""

    def test_subject_match_parse_miss_lands_in_digest_ask(self):
        v = Vault()
        t = acb_thread('kkkk0001', 0, 0, body='<html>Garbage with no fields</html>')
        code, out = run_main(v, FakeTransport([t]))
        self.assertEqual(code, 0, out)
        self.assertEqual(v.rows(), [])                   # no fabricated row
        self.assertIn('UNPARSED', out)
        self.assertIn('--probe', out)                    # probe hint
        self.assertIn('kkkk0001', v.state()['seen_threads'])   # asked once

    def test_digest_contains_needs_confirmation_block(self):
        rows = [{'id': '2026-10-13.01', 'date': '2026-10-13', 'dir': 'out',
                 'kind': 'oneoff', 'old_ref': 'mail:xyz', 'amount': 700000,
                 'currency': 'VND', 'desc': 'mystery', 'status': 'needs-confirm',
                 'src': 'email:ACB', 'ask': 'source unknown'}]
        text = render_digest(rows, [], '2026-10-13 07:00')
        self.assertIn('1 new transactions', text)
        self.assertIn('NEEDS CONFIRMATION', text)        # DG-15 cron strings
        self.assertIn('source unknown', text)


class FetchSkip(unittest.TestCase):
    """C8 — inbox invisible (401/403/404/URLError): skip notice, touch
    NOTHING, exit 0."""

    def _assert_skip(self, exc):
        v = Vault()
        before = v.snapshot()

        def transport(path):
            raise exc
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([], environ=v.env(), transport=transport)
        self.assertEqual(code, 0)
        self.assertIn('Skipping mail fetch', out.getvalue())
        self.assertEqual(v.snapshot(), before)

    def test_404_skips_touching_nothing(self):
        self._assert_skip(urllib.error.HTTPError('u', 404, 'Not Found', None, None))

    def test_401_skips_touching_nothing(self):
        self._assert_skip(urllib.error.HTTPError('u', 401, 'Unauthorized', None, None))

    def test_urlerror_skips_touching_nothing(self):
        self._assert_skip(urllib.error.URLError('gone'))


class LoudConfigFailures(unittest.TestCase):
    def test_missing_registry_never_fetches(self):
        # INV-2: registry resolution fails loud BEFORE any mail fetch.
        v = Vault(with_banks=True)
        env = v.env(PETASOS_BANKS='/nonexistent/banks.yaml')
        calls = []

        def transport(path):
            calls.append(path)
            return {}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([], environ=env, transport=transport)
        self.assertNotEqual(code, 0)
        self.assertEqual(calls, [])                      # never fetched

    def test_missing_credentials_are_loud_not_silent(self):
        v = Vault()
        env = v.env()
        env.pop('AGENTMAIL_API_KEY')
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([], environ=env, transport=FakeTransport([]))
        self.assertNotEqual(code, 0)
        self.assertIn('AGENTMAIL_API_KEY', out.getvalue() + err.getvalue())


class ViewsResolution(unittest.TestCase):
    """C6 — vault-local gen_views.py shadows the skill's; loud if neither."""

    def test_vault_copy_wins(self):
        from petasos.knob import load_knobs
        v = Vault()
        knobs = load_knobs(v.env())
        self.assertEqual(os.path.abspath(resolve_views_script(knobs)),
                         os.path.abspath(os.path.join(v.ledger, 'gen_views.py')))

    def test_falls_back_to_bundled_then_loud(self):
        from petasos.knob import load_knobs
        knobs = load_knobs({'PETASOS_VAULT': '/nonexistent-vault'})
        self.assertTrue(os.path.samefile(resolve_views_script(knobs),
                                         os.path.join(REPO, 'scripts', 'gen_views.py')))
        with tempfile.TemporaryDirectory(prefix='petasos-noviews-') as d:
            self.assertIsNone(resolve_views_script(knobs, bundled_root=d))


class EnvOnlyScratchRun(unittest.TestCase):
    """R5.3 — every knob via env runs the full pipeline with zero source
    edits (custom currency rides through to the rows)."""

    def test_custom_ledger_dir_and_currency(self):
        v = Vault(ledger_dir='7-Other/Deep/_L')
        code, out = run_main(v, FakeTransport([acb_thread('llll0001', 2000, 500)]),
                             PETASOS_CURRENCY='USD')
        self.assertEqual(code, 0, out)
        rows = v.rows()
        self.assertEqual(rows[0]['currency'], 'USD')


if __name__ == '__main__':
    unittest.main()
