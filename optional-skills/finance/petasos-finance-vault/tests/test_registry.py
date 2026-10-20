"""Registry + generic engine semantics (R2.1, R2.3, INV-4, INV-5, G8/DG-8).

The engine walks banks.yaml only — adding/fixing a bank is a DATA edit that
must never touch engine code. All test data is synthetic (R5.1).
"""
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from petasos.knob import load_knobs  # noqa: E402
from petasos.parse import parse_any, read_email, to_int  # noqa: E402
from petasos.registry import (Registry, resolve_registry, subject_keywords,  # noqa: E402
                              validate_registry)

BUNDLED_YAML = os.path.join(REPO, 'banks.yaml')


def bundled_registry():
    knobs = load_knobs({'PETASOS_VAULT': '/nonexistent-scratch-vault'})
    return resolve_registry(knobs)


class ToInt(unittest.TestCase):
    """INV-5 — per-bank decimal separator is data; strip the declared tail
    FIRST (×100 guard), then thousands separators."""

    def test_dot_decimal_with_comma_thousands(self):
        self.assertEqual(to_int('13,703,097.00', '.'), 13703097)

    def test_dot_thousands_no_tail(self):
        self.assertEqual(to_int('1.234.567', '.'), 1234567)

    def test_comma_decimal_with_dot_thousands(self):
        self.assertEqual(to_int('1.234.567,89', ','), 1234567)

    def test_plain_digits(self):
        self.assertEqual(to_int('3500000', '.'), 3500000)

    def test_sign_preserved(self):
        self.assertEqual(to_int('-1,234', '.'), -1234)

    def test_empty_and_none_are_none(self):
        self.assertIsNone(to_int('', '.'))
        self.assertIsNone(to_int(None, '.'))


class ReadEmail(unittest.TestCase):
    """C3 email convention: first line `Subject:`; HTML stripped iff <> present;
    entities unescaped; whitespace collapsed."""

    def test_subject_line_extracted(self):
        subj, body = read_email('Subject: Bao so du ACB\nGiao dich\nmoi\n')
        self.assertEqual(subj, 'Bao so du ACB')
        self.assertEqual(body, 'Giao dich moi')

    def test_html_stripped_to_spaces_and_entities_unescaped(self):
        raw = 'Subject: x\n<table><tr><td>Amount:&nbsp;100&amp;50</td></tr></table>'
        _, body = read_email(raw)
        self.assertEqual(body, 'Amount: 100&50')

    def test_plain_text_kept_verbatim_modulo_whitespace(self):
        _, body = read_email('Subject: x\nA>B stays  because  no tags')
        self.assertEqual(body, 'A>B stays because no tags')


class EngineSemantics(unittest.TestCase):
    """R2.3 — first format (registry order) whose subject matches AND whose
    requires all capture wins; direction BEFORE amount conversion; defaults
    fill; marker popped."""

    def setUp(self):
        self.reg = bundled_registry()

    def parse(self, subject, body, reg=None):
        return parse_any(reg or self.reg, subject, body)

    def test_subject_match_but_requires_fail_falls_through(self):
        # ACB subject, body that matches no ACB requires → no match at all.
        got = self.parse('Bao so du tu ACB', 'nothing useful here')
        self.assertEqual(got, {})

    def test_new_bank_as_pure_yaml_parses_without_code_change(self):
        reg = bundled_registry()
        reg.banks['TESTBANK'] = {
            'name': 'Test Bank',
            'amount': {'decimal_separator': '.'},
            'formats': [{
                'name': 'spend alert', 'subject_any': ['TESTBANK SPEND'],
                'kind': 'TB',
                'fields': [r'Amount: (?P<amount>[\d,]+) \w+ at (?P<merchant>.+?) on'],
                'requires': ['amount'], 'amounts': ['amount'],
                'defaults': {'merchant': '?'},
            }],
        }
        got = self.parse('TESTBANK SPEND 123',
                         'Amount: 2,500 VND at TEST MERCHANT on Monday', reg)
        self.assertEqual(got['kind'], 'TB')
        self.assertEqual(got['amount'], 2500)

    def test_data_only_change_alters_only_that_field(self):
        # R2.3: rename ACB's kind in the registry → only `kind` differs in
        # the parse of the same email.
        with open(os.path.join(REPO, 'fixtures', 'acb__balance-alert__voucher.txt'),
                  encoding='utf-8') as f:
            subj, body = read_email(f.read())
        base = self.parse(subj, body)
        reg2 = bundled_registry()
        reg2.banks['ACB']['formats'][0]['kind'] = 'ACB2'
        changed = parse_any(reg2, subj, body)
        self.assertEqual(changed['kind'], 'ACB2')
        self.assertEqual({k: v for k, v in changed.items() if k != 'kind'},
                         {k: v for k, v in base.items() if k != 'kind'})

    def test_registry_order_decides_between_matching_banks(self):
        reg = bundled_registry()
        reg.banks['AAA_DUP'] = {
            'name': 'first in order',
            'amount': {'decimal_separator': '.'},
            'formats': [{
                'name': 'dupe', 'subject_any': ['bao so du'], 'kind': 'DUPFIRST',
                'fields': [r'Số dư mới[^:]*:\s*(?P<bal_after>[\d,.]+)'],
                'requires': ['bal_after'], 'amounts': ['bal_after'],
            }],
        }
        # ACB comes first in the bundled registry → ACB wins despite the dup.
        with open(os.path.join(REPO, 'fixtures', 'acb__balance-alert__voucher.txt'),
                  encoding='utf-8') as f:
            subj, body = read_email(f.read())
        got = self.parse(subj, body, reg)
        self.assertEqual(got['kind'], 'ACB')

    def test_first_regex_capturing_a_group_wins(self):
        fmt = {
            'name': 'x', 'subject_any': ['SUBJ X'], 'kind': 'X',
            'fields': [r'val (?P<v>\d+)', r'val (?P<v>\d+)\s+extra'],
            'requires': [], 'defaults': {},
        }
        bank = {'amount': {'decimal_separator': '.'}}
        from petasos.parse import parse_format
        got = parse_format(fmt, bank, 'SUBJ X', 'val 7 extra')
        self.assertEqual(got['v'], '7')

    def test_direction_before_amount_conversion_sign_of(self):
        # sign_of reads the raw '+1,234' string before to_int strips the sign.
        fmt = {
            'name': 'x', 'subject_any': ['SUBJ S'], 'kind': 'S',
            'fields': [r'change (?P<amount>[+\-.,\d]+) VND'],
            'requires': ['amount'], 'amounts': ['amount'],
            'direction': {'sign_of': 'amount'},
        }
        bank = {'amount': {'decimal_separator': '.'}}
        from petasos.parse import parse_format
        credit = parse_format(fmt, bank, 'SUBJ S', 'change +1,234 VND')
        debit = parse_format(fmt, bank, 'SUBJ S', 'change -1,234 VND')
        self.assertEqual((credit['dir'], credit['amount']), ('IN', 1234))
        self.assertEqual((debit['dir'], debit['amount']), ('OUT', -1234))

    def test_marker_direction_and_marker_group_popped(self):
        # S-1: 'Ghi có' = IN, 'Ghi nợ' = OUT; the marker group never leaks.
        fmt = {
            'name': 'x', 'subject_any': ['SUBJ M'], 'kind': 'M',
            'fields': [r'GD: (?P<dir_marker>Ghi (?:nợ|có)) (?P<amount>[\d,.]+)'],
            'requires': ['amount'], 'amounts': ['amount'],
            'direction': {'marker': {'group': 'dir_marker',
                                     'out_contains': ['nợ'], 'in_contains': ['có']}},
        }
        bank = {'amount': {'decimal_separator': '.'}}
        from petasos.parse import parse_format
        credit = parse_format(fmt, bank, 'SUBJ M', 'GD: Ghi có 2,000.00')
        debit = parse_format(fmt, bank, 'SUBJ M', 'GD: Ghi nợ 2,000.00')
        self.assertEqual((credit['dir'], credit['amount']), ('IN', 2000))
        self.assertEqual((debit['dir'], debit['amount']), ('OUT', 2000))
        self.assertNotIn('dir_marker', credit)

    def test_defaults_fill_missing_optional_fields(self):
        # ACB without tài khoản/account digits → defaults fill '?' / ''.
        # (Subject matching is case-sensitive v1 parity: keyword is lowercase.)
        body = ('Số dư mới: 8,000,000.00 VND tính đến 13/10/2026 '
                'Giao dịch mới nhất: Ghi nợ 1,000,000.00 VND Nội dung giao dịch: PAY X.')
        got = self.parse('bao so du', body)
        self.assertEqual(got.get('acct'), '?')
        self.assertEqual(got.get('content'), 'PAY X')

    def test_required_field_satisfied_by_default_still_matches(self):
        # v1 parity: requires are checked AFTER defaults are applied.
        fmt = {
            'name': 'x', 'subject_any': ['SUBJ D'], 'kind': 'D',
            'fields': [r'amount (?P<amount>[\d,.]+)'],
            'requires': ['ref'], 'amounts': ['amount'],
            'defaults': {'ref': '?'},
        }
        bank = {'amount': {'decimal_separator': '.'}}
        from petasos.parse import parse_format
        self.assertIsNotNone(parse_format(fmt, bank, 'SUBJ D', 'amount 1,000'))

    def test_bankless_registry_returns_empty_not_crash(self):
        # R5.4: deleting all example banks leaves a working engine that
        # reports no-match instead of crashing.
        empty = Registry(path='/synthetic/none.yaml', banks={}, source='bundled')
        self.assertEqual(parse_any(empty, 'anything', 'anything'), {})
        self.assertEqual(subject_keywords(empty), [])


class ValidateRegistry(unittest.TestCase):
    """DG-8/G8 — overlap detection at load time (warning, first-wins documented)."""

    def test_no_overlap_yields_no_warnings(self):
        reg = bundled_registry()
        self.assertEqual(validate_registry(reg.banks), [])

    def test_cross_bank_overlap_warns_naming_both(self):
        reg = bundled_registry()
        reg.banks['TESTBANK'] = {
            'name': 'Test Bank', 'amount': {'decimal_separator': '.'},
            'formats': [{'name': 'dupe', 'subject_any': ['bao so du'], 'kind': 'TB',
                         'fields': [r'x (?P<a>\d+)'], 'requires': ['a']}],
        }
        warns = validate_registry(reg.banks)
        joined = '\n'.join(warns)
        self.assertIn('overlap', joined.lower())
        self.assertIn('ACB', joined)
        self.assertIn('TESTBANK', joined)


if __name__ == '__main__':
    unittest.main()
