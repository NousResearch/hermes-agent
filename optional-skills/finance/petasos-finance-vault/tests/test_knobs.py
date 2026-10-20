"""Knobs — every user-specific is env/config, never a source edit (R5.3, C7).

§8 maps 'scratch-vault full run with zero source edits' here as well; that
end-to-end assertion lives in tests/test_pipeline.py (it needs the pipeline);
this file pins the resolver itself.
"""
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from petasos.knob import load_knobs  # noqa: E402


class Defaults(unittest.TestCase):
    def test_empty_environment_yields_defaults(self):
        k = load_knobs({'PETASOS_VAULT': '/tmp/vault-x'})
        self.assertEqual(k.vault, '/tmp/vault-x')
        self.assertEqual(k.ledger_dir, os.path.join('/tmp/vault-x', '2-Areas/Finance/_Ledger'))
        self.assertEqual(k.currency, 'VND')
        self.assertIsNone(k.threshold)          # env unset → rules.md decides (C7)
        self.assertEqual(k.acct_num, '')
        self.assertEqual(k.acct_key, '')
        self.assertEqual(k.own_names, [])
        self.assertIsNone(k.banks_path)
        self.assertIsNone(k.rules_path)
        self.assertEqual(k.api_base, 'https://api.agentmail.to/v0')
        self.assertEqual(k.inbox, '')


class Resolution(unittest.TestCase):
    def test_petasos_vault_shadows_obsidian_vault_path(self):
        k = load_knobs({'PETASOS_VAULT': '/v1', 'OBSIDIAN_VAULT_PATH': '/v2'})
        self.assertEqual(k.vault, '/v1')

    def test_obsidian_vault_path_used_when_petasos_vault_unset(self):
        k = load_knobs({'OBSIDIAN_VAULT_PATH': '/v2'})
        self.assertEqual(k.vault, '/v2')

    def test_ledger_dir_knob(self):
        k = load_knobs({'PETASOS_VAULT': '/v', 'PETASOS_LEDGER_DIR': 'Fin/Ledger'})
        self.assertEqual(k.ledger_dir, os.path.join('/v', 'Fin/Ledger'))

    def test_absolute_ledger_dir_passes_through(self):
        k = load_knobs({'PETASOS_VAULT': '/v', 'PETASOS_LEDGER_DIR': '/elsewhere/led'})
        self.assertEqual(k.ledger_dir, '/elsewhere/led')


class EveryKnobIsEnvDriven(unittest.TestCase):
    def test_all_env_keys_map_to_knobs(self):
        k = load_knobs({
            'PETASOS_VAULT': '/scratch-vault',
            'PETASOS_LEDGER_DIR': '9-Test/_L',
            'PETASOS_CURRENCY': 'USD',
            'PETASOS_THRESHOLD': '750000',
            'PETASOS_ACCT_NUM': '1234567',
            'PETASOS_ACCT_KEY': 'BANK-X',
            'PETASOS_OWN_NAMES': 'Ann Lee, bob doe',
            'PETASOS_BANKS': '/tmp/mybanks.yaml',
            'PETASOS_RULES': '/tmp/myrules.md',
            'PETASOS_API_BASE': 'https://mail.example.test/v0',
            'PETASOS_INBOX': 'test@example.test',
        })
        self.assertEqual(k.vault, '/scratch-vault')
        self.assertEqual(k.ledger_dir, os.path.join('/scratch-vault', '9-Test/_L'))
        self.assertEqual(k.currency, 'USD')
        self.assertEqual(k.threshold, 750000)
        self.assertEqual(k.acct_num, '1234567')
        self.assertEqual(k.acct_key, 'BANK-X')
        self.assertEqual(k.own_names, ['ANN LEE', 'BOB DOE'])
        self.assertEqual(k.banks_path, '/tmp/mybanks.yaml')
        self.assertEqual(k.rules_path, '/tmp/myrules.md')
        self.assertEqual(k.api_base, 'https://mail.example.test/v0')
        self.assertEqual(k.inbox, 'test@example.test')

    def test_unparsable_threshold_falls_back_to_none(self):
        # v1 parity: `toint(env) or <fallback>` — a bad value never crashes
        # the run; the rules.md/default chain decides (C7 precedence).
        k = load_knobs({'PETASOS_VAULT': '/v', 'PETASOS_THRESHOLD': 'not-a-number'})
        self.assertIsNone(k.threshold)


if __name__ == '__main__':
    unittest.main()
