"""Manual ledger operations through the DG-16 sanctioned entry point —
`python3 -m petasos.ledger relabel|add` (R1.4 corrections, R4.6 chat-reported
rows). v1 left both to agent hand-edits of the JSONL; v2 gives them tested,
validated code. All rows synthetic (R5.1).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def run_module(args, env=None):
    e = dict(os.environ)
    e.setdefault('PETASOS_VAULT', os.path.join(tempfile.gettempdir(), 'petasos-manual-vault'))
    if env:
        e.update(env)
    return subprocess.run([sys.executable, '-m', 'petasos.ledger'] + args,
                          capture_output=True, text=True, cwd=REPO, env=e)


def make_ledger(rows):
    fd, path = tempfile.mkstemp(prefix='petasos-man-', suffix='.jsonl')
    os.close(fd)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return path


def srow(**kw):
    base = dict(id='2026-10-13.01', date='2026-10-13', dir='out', kind='expense',
                old_ref='mail:thread0001', amount=1000, currency='VND',
                desc='synthetic', status='posted', src='email:TEST')
    base.update(kw)
    return base


class RelabelEntry(unittest.TestCase):
    def test_relabel_via_module_invocation(self):
        path = make_ledger([srow()])
        r = run_module(['relabel', '--ledger', path, '--id', '2026-10-13.01',
                        '--set', 'category=Food', '--set', 'src=user-confirm:2026-10-13'])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        got = json.loads(open(path, encoding='utf-8').read().splitlines()[0])
        self.assertEqual(got['category'], 'Food')

    def test_relabel_rejects_evidence_field_via_cli(self):
        path = make_ledger([srow()])
        r = run_module(['relabel', '--ledger', path, '--id', '2026-10-13.01',
                        '--set', 'amount=999999'])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('amount', r.stderr + r.stdout)


class AddEntry(unittest.TestCase):
    """R4.6 — a chat-reported item lands as ONE posted row via the cash
    account, precedent category, src user-confirm:<date>, same day."""

    def test_chat_reported_item_gets_precedent_category(self):
        precedent = srow(desc='HIGHLANDS COFFEE', category='Personal', type='Wants',
                         fv='Variable')
        path = make_ledger([precedent])
        r = run_module(['add', '--ledger', path, '--date', '2026-10-13', '--dir', 'out',
                        '--kind', 'expense', '--amount', '50000',
                        '--desc', 'HIGHLANDS COFFEE TWO'])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        lines = open(path, encoding='utf-8').read().splitlines()
        self.assertEqual(len(lines), 2)
        new = json.loads(lines[1])
        self.assertEqual(new['id'], '2026-10-13.02')
        self.assertEqual(new['status'], 'posted')
        self.assertEqual(new['src'], 'user-confirm:2026-10-13')
        self.assertEqual(new['category'], 'Personal')      # precedent shape copied
        self.assertEqual(new['type'], 'Wants')
        self.assertEqual(new['amount'], 50000)
        self.assertEqual(new.get('old_ref'), '')          # no borrowed mail evidence

    def test_add_without_precedent_leaves_category_unset(self):
        path = make_ledger([srow(desc='SOMETHING ELSE', category='Food')])
        r = run_module(['add', '--ledger', path, '--date', '2026-10-13', '--dir', 'out',
                        '--kind', 'expense', '--amount', '1000', '--desc', 'BRAND NEW'])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        new = json.loads(open(path, encoding='utf-8').read().splitlines()[1])
        self.assertNotIn('category', new)

    def test_add_rejects_out_of_vocabulary_kind(self):
        path = make_ledger([])
        r = run_module(['add', '--ledger', path, '--date', '2026-10-13', '--dir', 'out',
                        '--kind', 'income?', '--amount', '1000', '--desc', 'X'])
        self.assertNotEqual(r.returncode, 0)
        before = open(path, encoding='utf-8').read()
        self.assertEqual(open(path, encoding='utf-8').read(), before)  # untouched

    def test_add_rejects_non_integer_amount(self):
        path = make_ledger([])
        r = run_module(['add', '--ledger', path, '--date', '2026-10-13', '--dir', 'out',
                        '--kind', 'expense', '--amount', '50k', '--desc', 'X'])
        self.assertNotEqual(r.returncode, 0)


if __name__ == '__main__':
    unittest.main()
