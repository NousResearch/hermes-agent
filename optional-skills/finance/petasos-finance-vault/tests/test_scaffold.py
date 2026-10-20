"""Lane B tests: petasos.scaffold — vault scaffolder (design §5, §4; C2/C3/C4
shadowing seeds; R5.1 privacy).

Creates the ledger tree with starter data the engine SHADOWS (vault-side
rules.md + banks.yaml override the skill-bundled copies — user extends
without touching skill code). Idempotent: existing files are never touched.
Scaffold output contains zero private deployment details.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import laneb

REPO = Path(__file__).resolve().parent.parent
INIT_VAULT = REPO / 'scripts' / 'init_vault.py'

SEEDED = ('transactions.jsonl', 'state.json', 'closed_months.json',
          'balances.json', 'rules.md', 'banks.yaml')

# R5.1 privacy sweep: the deny-list is DERIVED from the frozen fixture corpus
# (fixtures/expected.yaml acct/beneficiary/card/src values — the sealed P-1..P-4
# classes) so this file itself carries no private literals; an IPv4 pattern
# sweep covers P-3 structurally.
def _corpus_private_literals():
    import yaml
    with open(REPO / 'fixtures' / 'expected.yaml', encoding='utf-8') as f:
        expected = yaml.safe_load(f)
    lits = set()

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ('acct', 'beneficiary', 'card', 'src') \
                        and isinstance(v, str) and len(v) >= 5:
                    lits.add(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(expected)
    return sorted(lits)


IPV4_RE = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b')


class TestScaffold(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.vault = os.path.join(self._tmp.name, 'vault')
        os.makedirs(self.vault)

    def _scaffold(self, vault_path=None):
        from petasos.scaffold import scaffold
        knobs = laneb.knobs_for(self.vault)
        return scaffold(vault_path if vault_path is not None else self.vault, knobs)

    def seeded_paths(self, ledger_dir=None):
        L = os.path.join(self.vault, ledger_dir or laneb.DEFAULT_LEDGER_DIR)
        return {name: os.path.join(L, name) for name in SEEDED}

    def test_empty_dir_creates_the_full_set(self):
        self.assertEqual(self._scaffold(), 0)
        for name, path in self.seeded_paths().items():
            self.assertTrue(os.path.isfile(path), name)
        self.assertEqual(Path(self.seeded_paths()['transactions.jsonl']).read_text(), '')
        self.assertEqual(json.loads(Path(self.seeded_paths()['closed_months.json']).read_text()), {})
        # C2-shaped starter state: schema 1, empty seen list, no last_run yet.
        state = json.loads(Path(self.seeded_paths()['state.json']).read_text())
        self.assertEqual(state, {'schema': 1, 'seen_threads': []})
        # starter balances: one synthetic placeholder per account kind (C5)
        bal = json.loads(Path(self.seeded_paths()['balances.json']).read_text())
        kinds = sorted(a['kind'] for a in bal['accounts'].values())
        self.assertEqual(kinds, ['cash', 'cc', 'checking', 'deposit', 'fund',
                                 'loan', 'mortgage', 'property', 'savings'])
        self.assertEqual(bal['snapshot_totals'], {})

    def test_seeded_rules_and_banks_are_the_bundled_copies(self):
        """C3/C4 shadowing seeds: vault-side copies equal the skill-bundled
        files byte-for-byte, so the user can extend them freely."""
        self.assertEqual(self._scaffold(), 0)
        self.assertEqual(Path(self.seeded_paths()['rules.md']).read_bytes(),
                         (REPO / 'templates' / 'rules.md').read_bytes())
        self.assertEqual(Path(self.seeded_paths()['banks.yaml']).read_bytes(),
                         (REPO / 'banks.yaml').read_bytes())

    def test_rerun_touches_nothing(self):
        """Idempotent: user edits survive a re-run byte-for-byte."""
        self.assertEqual(self._scaffold(), 0)
        paths = self.seeded_paths()
        with open(paths['transactions.jsonl'], 'a', encoding='utf-8') as f:
            f.write('{"id":"2026-10-13.01"}\n')
        with open(paths['rules.md'], 'a', encoding='utf-8') as f:
            f.write('\n<!-- my own rule notes -->\n')
        before = {n: Path(p).read_bytes() for n, p in paths.items()}
        self.assertEqual(self._scaffold(), 0)
        after = {n: Path(p).read_bytes() for n, p in paths.items()}
        self.assertEqual(before, after)

    def test_output_contains_zero_private_details(self):
        """R5.1: no private deployment details in scaffolded files."""
        self.assertEqual(self._scaffold(), 0)
        deny = _corpus_private_literals()
        self.assertTrue(deny, 'corpus deny-list must not be empty')
        for name, path in self.seeded_paths().items():
            text = Path(path).read_text(encoding='utf-8')
            for lit in deny:
                self.assertNotIn(lit, text, f'{name} leaked {lit!r}')
            self.assertIsNone(IPV4_RE.search(text), f'{name} leaked an IP')

    def test_missing_vault_dir_fails_loud(self):
        rc = self._scaffold(vault_path=os.path.join(self._tmp.name, 'nope'))
        self.assertNotEqual(rc, 0)

    def test_no_vault_source_fails_loud(self):
        from petasos.scaffold import scaffold
        rc = scaffold('', laneb.knobs_for(''))
        self.assertNotEqual(rc, 0)

    def test_custom_ledger_dir(self):
        from petasos.scaffold import scaffold
        knobs = laneb.knobs_for(self.vault, 'Fin/Books')
        self.assertEqual(scaffold(self.vault, knobs), 0)
        for name, path in self.seeded_paths('Fin/Books').items():
            self.assertTrue(os.path.isfile(path), name)


class TestScaffoldCLI(unittest.TestCase):
    def test_cli_scaffolds_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = os.path.join(tmp, 'v')
            os.makedirs(vault)
            r = subprocess.run([sys.executable, str(INIT_VAULT), vault],
                               capture_output=True, text=True, cwd=REPO)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('scaffold', r.stdout.lower())
            self.assertTrue(os.path.isfile(os.path.join(
                vault, '2-Areas/Finance/_Ledger', 'balances.json')))
            # env-only invocation (PETASOS_VAULT, no arg) also works
            r2 = subprocess.run([sys.executable, str(INIT_VAULT)],
                                capture_output=True, text=True,
                                env=dict(os.environ, PETASOS_VAULT=vault), cwd=REPO)
            self.assertEqual(r2.returncode, 0, r2.stderr)
            self.assertIn('kept', r2.stdout)

    def test_cli_without_vault_fails_nonzero(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ('PETASOS_VAULT', 'OBSIDIAN_VAULT_PATH')}
        r = subprocess.run([sys.executable, str(INIT_VAULT)],
                           capture_output=True, text=True, env=env, cwd=REPO)
        self.assertNotEqual(r.returncode, 0)


if __name__ == '__main__':
    unittest.main()
