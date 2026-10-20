"""Shadowing + loud failures (R2.2, INV-2, INV-6, R5.4).

User data SHADOWS the skill install: $PETASOS_BANKS → vault copy → bundled
examples; every CLI names the file it used; load failures are LOUD and happen
BEFORE any mail fetch. All test data is synthetic (R5.1).
"""
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from petasos.knob import load_knobs  # noqa: E402
from petasos.registry import RegistryError, fixtures_dir, resolve_registry  # noqa: E402

BUNDLED = os.path.join(REPO, 'banks.yaml')


def scratch_vault_with_registry(tmp, banks_text=None):
    ledger = os.path.join(tmp, '2-Areas', 'Finance', '_Ledger')
    os.makedirs(ledger, exist_ok=True)
    if banks_text is None:
        with open(BUNDLED, encoding='utf-8') as f:
            banks_text = f.read()
    path = os.path.join(ledger, 'banks.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(banks_text)
    return path


class ShadowOrder(unittest.TestCase):
    """INV-6 — env > vault > bundled, each named by `source`."""

    def test_vault_copy_wins_over_bundled(self):
        with tempfile.TemporaryDirectory(prefix='petasos-sh-') as tmp:
            path = scratch_vault_with_registry(tmp)
            knobs = load_knobs({'PETASOS_VAULT': tmp})
            reg = resolve_registry(knobs)
            self.assertEqual(reg.source, 'vault')
            self.assertEqual(os.path.abspath(reg.path), os.path.abspath(path))

    def test_env_overrides_vault_copy(self):
        with tempfile.TemporaryDirectory(prefix='petasos-sh-') as tmp:
            scratch_vault_with_registry(tmp)
            with tempfile.TemporaryDirectory(prefix='petasos-sh-env-') as envdir:
                env_path = os.path.join(envdir, 'banks.yaml')
                with open(BUNDLED, encoding='utf-8') as f:
                    text = f.read()
                with open(env_path, 'w', encoding='utf-8') as f:
                    f.write(text.replace('kind: ACB', 'kind: ACB_ENV'))
                knobs = load_knobs({'PETASOS_VAULT': tmp, 'PETASOS_BANKS': env_path})
                reg = resolve_registry(knobs)
                self.assertEqual(reg.source, 'env')
                self.assertEqual(os.path.abspath(reg.path), os.path.abspath(env_path))
                self.assertEqual(reg.banks['ACB']['formats'][0]['kind'], 'ACB_ENV')

    def test_no_vault_copy_falls_back_to_bundled(self):
        with tempfile.TemporaryDirectory(prefix='petasos-sh-') as tmp:
            os.makedirs(os.path.join(tmp, 'empty-ledger'), exist_ok=True)
            knobs = load_knobs({'PETASOS_VAULT': tmp, 'PETASOS_LEDGER_DIR': 'empty-ledger'})
            reg = resolve_registry(knobs)
            self.assertEqual(reg.source, 'bundled')
            self.assertTrue(os.path.samefile(reg.path, BUNDLED))


class FixturesResolution(unittest.TestCase):
    """R2.2 — fixtures resolve beside the ACTIVE registry first, then bundled."""

    def test_fixtures_beside_active_registry_win(self):
        with tempfile.TemporaryDirectory(prefix='petasos-fx-') as tmp:
            reg_path = scratch_vault_with_registry(tmp)
            fx = os.path.join(os.path.dirname(reg_path), 'fixtures')
            os.makedirs(fx, exist_ok=True)
            reg = resolve_registry(load_knobs({'PETASOS_VAULT': tmp}))
            self.assertEqual(os.path.abspath(fixtures_dir(reg)), os.path.abspath(fx))

    def test_bundled_fixtures_when_none_beside_registry(self):
        reg = resolve_registry(load_knobs({'PETASOS_VAULT': '/nonexistent-vault'}))
        self.assertTrue(os.path.samefile(fixtures_dir(reg), os.path.join(REPO, 'fixtures')))


class LoudFailures(unittest.TestCase):
    """INV-2 — missing registry / missing PyYAML fail LOUDLY, never silently."""

    def test_missing_registry_names_every_searched_path(self):
        with tempfile.TemporaryDirectory(prefix='petasos-none-') as bundled_root:
            knobs = load_knobs({'PETASOS_VAULT': '/nonexistent-vault'})
            with self.assertRaises(RegistryError) as cm:
                resolve_registry(knobs, bundled_root=bundled_root)
            msg = str(cm.exception)
            self.assertIn('banks.yaml not found', msg)
            self.assertIn(os.path.join('/nonexistent-vault', '2-Areas', 'Finance',
                                       '_Ledger', 'banks.yaml'), msg)
            self.assertIn(os.path.join(bundled_root, 'banks.yaml'), msg)

    def test_explicit_env_path_missing_is_loud_not_fallthrough(self):
        # An explicitly named registry that does not exist must not silently
        # fall back to the vault/bundled copies (v2 repair of a v1 surprise).
        knobs = load_knobs({'PETASOS_VAULT': '/nonexistent-vault',
                            'PETASOS_BANKS': '/nonexistent-vault/banks.yaml'})
        with self.assertRaises(RegistryError) as cm:
            resolve_registry(knobs)
        self.assertIn('$PETASOS_BANKS', str(cm.exception))

    def test_missing_engine_definition_fails_loudly(self):
        # Task item 5 / G3-adjacent: a format lacking its engine definition
        # (kind/subject_any/fields) is a registry error, not a silent skip.
        bad = ('ACB:\n  name: x\n  amount:\n    decimal_separator: "."\n'
               '  formats:\n    - name: balance alert\n      subject_any: ["bao so du"]\n')
        with tempfile.TemporaryDirectory(prefix='petasos-bad-') as tmp:
            path = scratch_vault_with_registry(tmp, bad)
            knobs = load_knobs({'PETASOS_VAULT': tmp})
            with self.assertRaises(RegistryError) as cm:
                resolve_registry(knobs)
            msg = str(cm.exception)
            self.assertIn('ACB', msg)
            self.assertIn('balance alert', msg)
            self.assertIn('kind', msg)

    def test_uncompilable_regex_fails_loudly(self):
        bad = ('ACB:\n  name: x\n  amount:\n    decimal_separator: "."\n'
               '  formats:\n    - name: balance alert\n      subject_any: ["bao so du"]\n'
               '      kind: ACB\n      fields:\n        - "(?P<broken[unclosed>"\n'
               '      requires: []\n')
        with tempfile.TemporaryDirectory(prefix='petasos-badrx-') as tmp:
            scratch_vault_with_registry(tmp, bad)
            knobs = load_knobs({'PETASOS_VAULT': tmp})
            with self.assertRaises(RegistryError) as cm:
                resolve_registry(knobs)
            self.assertIn('regex', str(cm.exception).lower())


class CliLoudFailures(unittest.TestCase):
    """The CLI surface of INV-2 — nonzero exit, message on stderr/stdout."""

    @staticmethod
    def run_cli(args, env):
        return subprocess.run(
            [sys.executable, os.path.join(REPO, 'scripts', 'bank_adapters.py')] + args,
            capture_output=True, text=True, env=env, cwd=REPO)

    def test_missing_registry_exits_nonzero_naming_paths(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('PETASOS_', 'OBSIDIAN_'))}
        env['PETASOS_BANKS'] = '/nonexistent-vault/banks.yaml'
        r = self.run_cli(['--banks'], env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('$PETASOS_BANKS', r.stdout + r.stderr)

    def test_missing_pyyaml_exits_nonzero_with_install_hint(self):
        # Simulate an absent PyYAML: a PYTHONPATH dir whose yaml.py raises
        # ImportError on import (the exact failure mode of a missing module).
        with tempfile.TemporaryDirectory(prefix='petasos-noyaml-') as d:
            with open(os.path.join(d, 'yaml.py'), 'w', encoding='utf-8') as f:
                f.write('raise ImportError("no yaml here")\n')
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(('PETASOS_', 'OBSIDIAN_'))}
            env['PYTHONPATH'] = d + os.pathsep + env.get('PYTHONPATH', '')
            r = self.run_cli(['--banks'], env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('pip install pyyaml', r.stdout + r.stderr)


if __name__ == '__main__':
    unittest.main()
