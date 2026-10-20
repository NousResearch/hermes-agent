"""Fixture parity — the frozen v1.1.0 corpus must parse identically under v2.

Design §8 row 1 / brief §4 + INV-3/INV-4/INV-5:
- the sealed fixtures parse to the exact expected dicts (subset match) via
  the v2 package API (petasos.knob → registry → parse);
- any regex regression flips the `--test` exit code;
- `--probe` prints one MATCH/MISS line per field regex per format and exits
  0 iff some format fully parses;
- optional pipes + 2/4-digit years parse identical (INV-4);
- per-bank decimal separator as data with the ×100 guard (INV-5).

Expected values are READ from the sealed fixtures/expected.yaml — no private
literals live in this file (R5.1). CLI tests pin v1-parity behavior that the
v2 rewrite of scripts/bank_adapters.py must preserve.
"""
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FIXTURES = os.path.join(REPO, 'fixtures')


def base_env(**extra):
    """os.environ minus every petasos knob, plus caller overrides."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PETASOS_', 'OBSIDIAN_'))}
    env.pop('AGENTMAIL_API_KEY', None)
    env.update(extra)
    return env


def run_cli(args, env):
    return subprocess.run([sys.executable, os.path.join(REPO, 'scripts', 'bank_adapters.py')] + args,
                          capture_output=True, text=True, env=env, cwd=REPO)


class FixtureParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from petasos.knob import load_knobs
        from petasos.registry import resolve_registry
        with open(os.path.join(FIXTURES, 'expected.yaml'), encoding='utf-8') as f:
            cls.expected = yaml.safe_load(f)
        # empty scratch vault → resolution must land on the bundled registry
        cls.scratch = tempfile.mkdtemp(prefix='petasos-parity-')
        knobs = load_knobs({'PETASOS_VAULT': cls.scratch})
        cls.registry = resolve_registry(knobs)

    def parse(self, fname):
        from petasos.parse import parse_any, read_email
        with open(os.path.join(FIXTURES, fname), encoding='utf-8') as f:
            raw = f.read()
        subj, body = read_email(raw)
        return parse_any(self.registry, subj, body)

    def test_registry_resolves_to_bundled_examples(self):
        self.assertEqual(self.registry.source, 'bundled')
        self.assertTrue(os.path.samefile(self.registry.path, os.path.join(REPO, 'banks.yaml')))

    def test_every_fixture_parses_to_expected_subset(self):
        fails = []
        for fname, spec in sorted(self.expected.items()):
            got = self.parse(fname)
            for k, v in spec.get('expect', {}).items():
                if got.get(k) != v:
                    fails.append(f'{fname}: {k}={got.get(k)!r} != {v!r}')
        self.assertEqual(fails, [], 'fixture parity drift:\n' + '\n'.join(fails))

    def test_pipes_and_no_pipes_parse_identical(self):
        # INV-4: tag-stripping yields spaces, so separators must stay optional.
        from petasos.parse import parse_any
        subj = 'Thong bao chuyển tiền ngoài BIDV'
        plain = ('Transaction time: 11/10/2026 17:57:04 Reference number: REF12345 '
                 'Transaction amount: 3,500,000 VND Beneficiary name: JOHN Q PUBLIC '
                 'Số tài khoản: 123456789 Debit account: 9988776655')
        piped = plain.replace(': ', ': | ')
        a = parse_any(self.registry, subj, plain)
        b = parse_any(self.registry, subj, piped)
        self.assertTrue(a and b, f'both variants must parse: plain={a} piped={b}')
        self.assertEqual(a, b)

    def test_two_and_four_digit_years_normalize_to_same_iso(self):
        from petasos.parse import normalize_date
        self.assertEqual(normalize_date('13/10/26'), '2026-10-13')
        self.assertEqual(normalize_date('13/10/2026'), '2026-10-13')
        self.assertEqual(normalize_date('2026-12-01'), '2026-12-01')  # ISO passthrough

    def test_decimal_separator_flip_is_exactly_x100(self):
        from petasos.parse import to_int
        right = to_int('9,876,543.00', '.')
        wrong = to_int('9,876,543.00', ',')
        self.assertEqual(right, 9876543)
        self.assertEqual(wrong, right * 100)  # INV-5: wrong declaration = ×100 bug


class AdapterCliContract(unittest.TestCase):
    """CLI parity pins (v1 behavior the v2 script must preserve) + DG-8 warning."""

    def test_fixture_corpus_green_exits_zero(self):
        r = run_cli(['--test'], base_env())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('7/7 fixtures pass', r.stdout)

    def test_probe_known_email_exits_zero_with_match_lines(self):
        r = run_cli(['--probe', os.path.join(FIXTURES, 'acb__balance-alert__atm.txt')], base_env())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('MATCH', r.stdout)
        self.assertIn('MISS', r.stdout)      # per-field visibility: misses shown too
        self.assertIn('PARSED:', r.stdout)
        self.assertIn('RESULT:', r.stdout)

    def test_probe_unknown_email_exits_nonzero(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write('Subject: hello there\nnothing banky inside\n')
            path = f.name
        try:
            r = run_cli(['--probe', path], base_env())
        finally:
            os.unlink(path)
        self.assertNotEqual(r.returncode, 0)

    def test_regex_regression_flips_test_exit(self):
        # A registry whose regex never fires must fail --test (INV-3: a dead
        # regex is visible through these tools).
        with open(os.path.join(REPO, 'banks.yaml'), encoding='utf-8') as f:
            text = f.read()
        broken = text.replace('Số dư mới', 'Số dư CŨ KHÔNG TỒN TẠI')
        with tempfile.TemporaryDirectory(prefix='petasos-reg-') as d:
            reg = os.path.join(d, 'banks.yaml')
            with open(reg, 'w', encoding='utf-8') as f:
                f.write(broken)
            r = run_cli(['--test'], base_env(PETASOS_BANKS=reg))
        self.assertNotEqual(r.returncode, 0, r.stdout)

    def test_banks_names_active_file_and_warns_on_overlap(self):
        # R2.2: --banks names the file it used; DG-8: overlapping subject_any
        # across banks → load-time warning naming BOTH banks (+ first-wins).
        with open(os.path.join(REPO, 'banks.yaml'), encoding='utf-8') as f:
            text = f.read()
        overlapped = text + ("\nTESTBANK:\n  name: Test Bank\n  amount:\n"
                             "    decimal_separator: \".\"\n  formats:\n"
                             "    - name: balance alert dupe\n      subject_any: [\"bao so du\"]\n"
                             "      kind: TESTB\n      fields:\n"
                             "        - 'Số dư mới[^:]*:\\\\s*(?P<bal_after>[\\\\d,.]+)'\n"
                             "      requires: [bal_after]\n      amounts: [bal_after]\n")
        with tempfile.TemporaryDirectory(prefix='petasos-reg-') as d:
            reg = os.path.join(d, 'banks.yaml')
            with open(reg, 'w', encoding='utf-8') as f:
                f.write(overlapped)
            r = run_cli(['--banks'], base_env(PETASOS_BANKS=reg))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(reg, r.stdout)                       # names the file it used
        warn_lines = [ln for ln in r.stdout.splitlines()
                      if 'overlap' in ln.lower() and 'ACB' in ln and 'TESTBANK' in ln]
        self.assertTrue(warn_lines, 'DG-8 overlap warning naming both banks missing:\n' + r.stdout)


if __name__ == '__main__':
    unittest.main()
