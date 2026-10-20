"""rules.md — section-aware parser (R2.4, S-6, G4/DG-4, DG-3, DG-14).

The classification brain is one human-editable file; the parser binds tables
to their OWNING section (v1 was section-blind: §3a rows leaked into the
auto-post list), validates shape, and exits loudly naming section + line on a
malformed table. Resolution chain: $PETASOS_RULES → vault copy → bundled
template (v1 crashed without a vault copy — DG-14 repairs).
"""
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from petasos.knob import load_knobs  # noqa: E402
from petasos.rules import Rules, RulesError, load_rules  # noqa: E402

BUNDLED = os.path.join(REPO, 'templates', 'rules.md')

STARTER = open(BUNDLED, encoding='utf-8').read() if os.path.exists(BUNDLED) else ''


def write_rules(directory, text, name='rules.md'):
    path = os.path.join(directory, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def rules_for(text=None, ledger_rel='9-Ledger'):
    """Rules from `text` placed as the vault copy; else the bundled template."""
    if text is None:
        return load_rules(load_knobs({'PETASOS_VAULT': '/nonexistent-vault'}))
    with tempfile.TemporaryDirectory(prefix='petasos-rls-') as tmp:
        ledger = os.path.join(tmp, ledger_rel)
        os.makedirs(ledger, exist_ok=True)
        write_rules(ledger, text)
        knobs = load_knobs({'PETASOS_VAULT': tmp, 'PETASOS_LEDGER_DIR': ledger_rel})
        return load_rules(knobs)


class SectionAwareParsing(unittest.TestCase):
    def test_bundled_template_loads_with_expected_sections(self):
        r = rules_for()
        self.assertIsInstance(r, Rules)
        self.assertEqual(r.source, 'bundled')
        self.assertTrue(any(p == 'IB LUNCH' for p, *_ in r.auto_post),
                        'starter §1 patterns must parse')
        self.assertTrue(any(p == 'ELECTRICITY|EVN' for p, *_ in r.known_route),
                        'starter §3a patterns must parse')
        self.assertIn('ACB', r.kind_route, 'starter §5 kind routing must parse')

    def test_section_3a_rows_do_not_leak_into_auto_post(self):
        # G4: v1's section-blind regex put §3a rows in the auto-post list.
        r = rules_for()
        auto_patterns = [p for p, *_ in r.auto_post]
        self.assertNotIn('ELECTRICITY|EVN', auto_patterns)

    def test_new_section_1_row_classifies_without_code_change(self):
        text = STARTER.replace('| `GRAB|BE-` | Transport | Needs | Variable | ride-hailing |',
                               '| `GRAB|BE-` | Transport | Needs | Variable | ride-hailing |\n'
                               '| `NEWPATTERN` | TestCat | TestType | TestFV | added as data |')
        r = rules_for(text)
        self.assertIn(('NEWPATTERN', 'TestCat', 'TestType', 'TestFV'), r.auto_post)

    def test_kind_routing_table_parses(self):
        # DG-3: kind → (tier, category, type, fv) replaces v1 code branches.
        text = STARTER.replace(
            '| `ACB` | balance-alert | Personal | Needs | Variable |',
            '| `ACB` | balance-alert | Personal | Needs | Variable |\n'
            '| `TESTK` | plumbing |  |  |  |')
        r = rules_for(text)
        self.assertEqual(r.kind_route.get('TESTK'), ('plumbing', '', '', ''))

    def test_aggregate_notes_collected(self):
        r = rules_for()
        joined = '\n'.join(r.aggregate_notes)
        self.assertIn('Credit-card statements', joined)

    def test_threshold_default_parsed_from_backtick_value(self):
        text = STARTER + '\nDefault ceiling: `threshold_auto_post`: **750,000** for review\n'
        r = rules_for(text)
        self.assertEqual(r.threshold_default, 750000)

    def test_threshold_absent_is_none(self):
        r = rules_for()
        self.assertIsNone(r.threshold_default)


class MalformedTablesAreLoud(unittest.TestCase):
    """S-6 — a malformed table is DETECTED, never silently swallowed."""

    def test_uncompilable_pattern_fails_loud_at_load(self):
        """§1/§3a patterns must compile — fail naming the line, not crash
        classify on the first real transaction (live finding 14/10)."""
        text = STARTER.replace(
            '|---|---|---|---|---|\n',
            '|---|---|---|---|---|\n| `+opens-with-repeat` | X | Y | Z | |\n', 1)
        with self.assertRaises(RulesError) as cm:
            rules_for(text)
        self.assertIn('does not compile', str(cm.exception))

    def test_wrong_column_count_names_section_and_line(self):
        text = STARTER.replace('| `IB LUNCH` | Food | Needs | Variable | workplace lunch |',
                               '| `IB LUNCH` | Food |')  # truncated row
        with self.assertRaises(RulesError) as cm:
            rules_for(text)
        msg = str(cm.exception)
        self.assertIn('1', msg)            # section identified
        self.assertIn('line', msg.lower())

    def test_unparseable_row_names_section_and_line(self):
        text = STARTER.replace('| `TUITION` | Education | Needs | Variable |',
                               '| TUITION no backticks | Education | Needs | Variable |')
        with self.assertRaises(RulesError) as cm:
            rules_for(text)
        msg = str(cm.exception)
        self.assertIn('3a', msg)
        self.assertIn('line', msg.lower())


class ResolutionChain(unittest.TestCase):
    """DG-14 — $PETASOS_RULES → vault copy → bundled template, file named."""

    def test_vault_copy_wins_and_is_named(self):
        with tempfile.TemporaryDirectory(prefix='petasos-rls-') as tmp:
            ledger = os.path.join(tmp, '9-Ledger')
            os.makedirs(ledger, exist_ok=True)
            path = write_rules(ledger, STARTER + '\n<!-- vault copy -->\n')
            r = load_rules(load_knobs({'PETASOS_VAULT': tmp, 'PETASOS_LEDGER_DIR': '9-Ledger'}))
            self.assertEqual(r.source, 'vault')
            self.assertEqual(os.path.abspath(r.path), os.path.abspath(path))

    def test_env_overrides_vault_copy(self):
        with tempfile.TemporaryDirectory(prefix='petasos-rls-') as tmp:
            ledger = os.path.join(tmp, '9-Ledger')
            os.makedirs(ledger, exist_ok=True)
            write_rules(ledger, STARTER)
            env_path = write_rules(tmp, STARTER.replace('Food', 'FoodEnv'), name='env-rules.md')
            r = load_rules(load_knobs({'PETASOS_VAULT': tmp, 'PETASOS_LEDGER_DIR': '9-Ledger',
                                       'PETASOS_RULES': env_path}))
            self.assertEqual(r.source, 'env')
            self.assertTrue(any(c == 'FoodEnv' for _, c, _, _ in r.auto_post))

    def test_missing_vault_copy_falls_back_to_bundled_not_crash(self):
        # v1 required the vault copy and crashed; DG-14 REC repairs this.
        with tempfile.TemporaryDirectory(prefix='petasos-rls-') as tmp:
            os.makedirs(os.path.join(tmp, '9-Ledger'), exist_ok=True)
            r = load_rules(load_knobs({'PETASOS_VAULT': tmp, 'PETASOS_LEDGER_DIR': '9-Ledger'}))
            self.assertEqual(r.source, 'bundled')

    def test_explicit_env_path_missing_is_loud(self):
        knobs = load_knobs({'PETASOS_VAULT': '/nonexistent-vault',
                            'PETASOS_RULES': '/nonexistent-vault/rules.md'})
        with self.assertRaises(RulesError):
            load_rules(knobs)


if __name__ == '__main__':
    unittest.main()
