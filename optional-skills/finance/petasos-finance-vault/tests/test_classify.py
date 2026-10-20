"""Classify — precedence ladder + autonomy tiers (R4.1–R4.6, S-1, S-3, G3/DG-3).

Precedence (§7, v1-observed order preserved): §6 confirmed → §3a known
merchant → §1 auto-post patterns → §5 kind routing (data-driven, replaces v1
code branches) → own-name guard → unknown ≥ threshold needs-confirm. Mystery
credits hold kind `oneoff` (C1 vocabulary repair of v1's `income?`). All test
data synthetic (R5.1).
"""
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from petasos.classify import Decision, classify, effective_threshold, is_pending_hold  # noqa: E402
from petasos.knob import load_knobs  # noqa: E402
from petasos.rules import load_rules  # noqa: E402

STARTER = open(os.path.join(REPO, 'templates', 'rules.md'), encoding='utf-8').read()


def rules_with(text=None):
    if text is None:
        return load_rules(load_knobs({'PETASOS_VAULT': '/nonexistent-vault'}))
    with tempfile.TemporaryDirectory(prefix='petasos-cls-') as tmp:
        ledger = os.path.join(tmp, '9-Ledger')
        os.makedirs(ledger, exist_ok=True)
        with open(os.path.join(ledger, 'rules.md'), 'w', encoding='utf-8') as f:
            f.write(text)
        return load_rules(load_knobs({'PETASOS_VAULT': tmp, 'PETASOS_LEDGER_DIR': '9-Ledger'}))


def knobs_with(**env):
    base = {'PETASOS_VAULT': '/nonexistent-vault', 'PETASOS_OWN_NAMES': 'ANN LEE'}
    base.update(env)
    return load_knobs(base)


def tx(kind='ACB', dir='OUT', amount=50000, **extra):
    d = {'kind': kind, 'dir': dir, 'amount': amount}
    d.update(extra)
    return d


class PrecedenceLadder(unittest.TestCase):
    """R4.2 — §3a posts no ask; §1 posts with ask at/above threshold; unknown
    above-threshold lands needs-confirm."""

    def setUp(self):
        self.rules = rules_with()
        self.knobs = knobs_with(PETASOS_THRESHOLD='500000')

    def test_known_merchant_above_threshold_posts_without_ask(self):
        d = classify(tx(amount=900000, content='EVN ELECTRICITY BILL'), self.rules, self.knobs)
        self.assertEqual(d.status, 'posted')
        self.assertIsNone(d.ask)
        self.assertEqual(d.category, 'Utilities')

    def test_auto_post_pattern_above_threshold_posts_with_ask(self):
        d = classify(tx(amount=900000, content='HIGHLANDS COFFEE ONE'), self.rules, self.knobs)
        self.assertEqual(d.status, 'posted')
        self.assertEqual(d.ask, 'above-thresh-but-pattern')
        self.assertEqual(d.category, 'Personal')

    def test_auto_post_pattern_below_threshold_posts_silently(self):
        d = classify(tx(amount=10000, content='IB LUNCH PLACE'), self.rules, self.knobs)
        self.assertEqual(d.status, 'posted')
        self.assertIsNone(d.ask)
        self.assertEqual(d.category, 'Food')

    def test_unknown_above_threshold_needs_confirm(self):
        d = classify(tx(amount=900000, content='MYSTERY SHOP'), self.rules, self.knobs)
        self.assertEqual(d.status, 'needs-confirm')
        self.assertEqual(d.kind, 'expense')       # vocabulary-valid holding kind
        self.assertIsNone(d.category)


class KindRoutingTiers(unittest.TestCase):
    """§5 data-driven tiers replace v1's hardcoded kind branches (G3/DG-3)."""

    def setUp(self):
        self.rules = rules_with()
        self.knobs = knobs_with(PETASOS_THRESHOLD='500000')

    def test_cc_bill_payment_is_one_plumbing_row_never_expense(self):
        # R4.1 double-count guard: BIDV credit-card payment receipt.
        d = classify(tx(kind='BIDV_CCPAY', amount=3000000), self.rules, self.knobs)
        self.assertEqual(d.kind, 'transfer')
        self.assertEqual(d.status, 'posted')
        self.assertIsNone(d.ask)

    def test_hsbc_cc_always_expense_even_above_threshold(self):
        d = classify(tx(kind='HSBC_CC', amount=9000000, merchant='BIG STORE'),
                     self.rules, self.knobs)
        self.assertEqual((d.kind, d.status), ('expense', 'posted'))
        self.assertEqual(d.category, 'Personal')

    def test_bidv_card_below_threshold_expense_above_needs_confirm(self):
        below = classify(tx(kind='BIDV_CARD', amount=10000, merchant='GRAB X'),
                         self.rules, self.knobs)
        above = classify(tx(kind='BIDV_CARD', amount=700000, merchant='MYSTERY'),
                         self.rules, self.knobs)
        self.assertEqual((below.kind, below.status), ('expense', 'posted'))
        self.assertEqual(above.status, 'needs-confirm')

    def test_balance_alert_out_below_threshold_expense_with_row_category(self):
        d = classify(tx(kind='ACB', dir='OUT', amount=100000, content='PAY MERCHANT X'),
                     self.rules, self.knobs)
        self.assertEqual((d.kind, d.status), ('expense', 'posted'))
        self.assertEqual((d.category, d.type, d.fv), ('Personal', 'Needs', 'Variable'))

    def test_transfer_receipt_own_name_needs_confirm_destination_unknown(self):
        d = classify(tx(kind='BIDV_XFER', amount=3500000, beneficiary='ANN LEE'),
                     self.rules, self.knobs)
        self.assertEqual((d.kind, d.status), ('transfer', 'needs-confirm'))
        self.assertEqual(d.ask, 'own-name transfer — destination unknown')

    def test_transfer_receipt_external_payee_needs_confirm(self):
        d = classify(tx(kind='BIDV_XFER', amount=3500000, beneficiary='SOMEONE ELSE'),
                     self.rules, self.knobs)
        self.assertEqual((d.kind, d.status), ('transfer', 'needs-confirm'))

    def test_unknown_kind_without_route_needs_confirm_vocabulary_valid(self):
        d = classify(tx(kind='NEWBANK', amount=100000), self.rules, self.knobs)
        self.assertEqual(d.status, 'needs-confirm')
        self.assertIn(d.kind, ('expense', 'transfer', 'salary', 'rental', 'oneoff'))

    def test_desc_spans_content_merchant_beneficiary(self):
        # v1 parity: narration = content + merchant + beneficiary joined.
        d = classify(tx(amount=10000, content='', merchant='', beneficiary='IB LUNCH'),
                     self.rules, self.knobs)
        self.assertEqual(d.category, 'Food')


class OwnNameGuard(unittest.TestCase):
    """R4.3 — own-name transfers are needs-confirm, NEVER expenses."""

    def setUp(self):
        self.rules = rules_with()
        self.knobs = knobs_with(PETASOS_THRESHOLD='500000')

    def test_own_name_out_never_an_expense(self):
        d = classify(tx(amount=10000, content='CHUYEN ANN LEE 123'), self.rules, self.knobs)
        self.assertEqual(d.kind, 'transfer')
        self.assertEqual(d.status, 'needs-confirm')
        self.assertEqual(d.ask, 'own-name transfer — destination unknown')

    def test_own_name_guard_beats_below_threshold_expense(self):
        # Even a small own-name OUT must not post as an expense.
        d = classify(tx(amount=5000, content='TO ANN LEE'), self.rules, self.knobs)
        self.assertNotEqual(d.kind, 'expense')


class MysteryCredits(unittest.TestCase):
    """R4.1 — IN ≥ threshold → needs-confirm 'source unknown', kind oneoff
    (C1 repair: v1 wrote the out-of-vocabulary kind `income?`)."""

    def setUp(self):
        self.rules = rules_with()
        self.knobs = knobs_with(PETASOS_THRESHOLD='500000')

    def test_signed_credit_parses_in_then_needs_confirm_as_oneoff(self):
        # S-1 end-to-end: a '+' credit alert parses dir IN and classifies to
        # the mystery-credit tier.
        from petasos.parse import parse_format
        fmt = {'name': 'x', 'subject_any': ['SUBJ S'], 'kind': 'S',
               'fields': [r'change (?P<amount>[+\-.,\d]+) VND'],
               'requires': ['amount'], 'amounts': ['amount'],
               'direction': {'sign_of': 'amount'}}
        got = parse_format(fmt, {'amount': {'decimal_separator': '.'}},
                           'SUBJ S', 'change +700,000 VND')
        self.assertEqual(got['dir'], 'IN')
        d = classify(got, self.rules, self.knobs)
        self.assertEqual(d.status, 'needs-confirm')
        self.assertEqual(d.ask, 'source unknown')

    def test_acb_credit_above_threshold_oneoff_needs_confirm(self):
        d = classify(tx(dir='IN', amount=700000, content='UNKNOWN SOURCE'),
                     self.rules, self.knobs)
        self.assertEqual((d.kind, d.status, d.ask),
                         ('oneoff', 'needs-confirm', 'source unknown'))

    def test_acb_credit_below_threshold_oneoff_posted(self):
        d = classify(tx(dir='IN', amount=100000, content='SMALL REFUND'),
                     self.rules, self.knobs)
        self.assertEqual((d.kind, d.status), ('oneoff', 'posted'))

    def test_marker_credit_parses_in(self):
        # S-1 marker half: 'Ghi có' = IN.
        from petasos.parse import parse_format
        fmt = {'name': 'x', 'subject_any': ['SUBJ M'], 'kind': 'M',
               'fields': [r'GD: (?P<dir_marker>Ghi (?:nợ|có)) (?P<amount>[\d,.]+)'],
               'requires': ['amount'], 'amounts': ['amount'],
               'direction': {'marker': {'group': 'dir_marker',
                                        'out_contains': ['nợ'], 'in_contains': ['có']}}}
        got = parse_format(fmt, {'amount': {'decimal_separator': '.'}},
                           'SUBJ M', 'GD: Ghi có 100,000.00')
        self.assertEqual(got['dir'], 'IN')


class ThresholdKnob(unittest.TestCase):
    """R4.4 — env moves the boundary; env → rules.md → 500000."""

    def test_env_threshold_moves_boundary(self):
        rules = rules_with()
        low = knobs_with(PETASOS_THRESHOLD='1000')
        d = classify(tx(amount=10000, content='IB LUNCH'), rules, low)
        self.assertEqual(d.ask, 'above-thresh-but-pattern')   # 10k ≥ 1k now

    def test_rules_threshold_used_when_env_unset(self):
        text = STARTER + '\nCeiling: `threshold_auto_post`: **1000**\n'
        rules = rules_with(text)
        knobs = knobs_with()  # no env threshold
        self.assertEqual(effective_threshold(knobs, rules), 1000)
        d = classify(tx(amount=10000, content='IB LUNCH'), rules, knobs)
        self.assertEqual(d.ask, 'above-thresh-but-pattern')

    def test_default_500000_when_neither_set(self):
        rules = rules_with()
        self.assertEqual(effective_threshold(knobs_with(), rules), 500000)


class ConfirmedByUser(unittest.TestCase):
    """R2.5 — a §6 item is auto-routed, never re-asked."""

    def test_confirmed_item_posts_without_ask(self):
        text = STARTER.replace(
            '| Date | Item | Meaning |\n|---|---|---|\n',
            '| Date | Item | Meaning |\n|---|---|---|\n'
            '| 2026-10-01 | SALARY COMPANY Z | confirmed salary |\n')
        rules = rules_with(text)
        knobs = knobs_with(PETASOS_THRESHOLD='500000')
        d = classify(tx(dir='IN', amount=9000000, content='SALARY COMPANY Z OCT'),
                     rules, knobs)
        self.assertEqual(d.status, 'posted')
        self.assertIsNone(d.ask)

    def test_confirmed_prose_item_with_regex_metachars_is_safe(self):
        """§6 prose items ('+900,000 Zalo (…)') must not crash classify —
        literal fallback (live finding 14/10: re.error 'nothing to repeat')."""
        text = STARTER.replace(
            '| Date | Item | Meaning |\n|---|---|---|\n',
            '| Date | Item | Meaning |\n|---|---|---|\n'
            '| 2026-09-26 | +900,000 Zalo (TRAN VAN EXAMPLEA) | domain-fee income |\n')
        rules = rules_with(text)
        knobs = knobs_with(PETASOS_THRESHOLD='500000')
        d = classify(tx(dir='IN', amount=900000,
                        content='+900,000 Zalo (TRAN VAN EXAMPLEA)'),
                     rules, knobs)
        self.assertNotEqual(d.ask, 'source unknown')  # §6 matched, not unknown


class PendingHold(unittest.TestCase):
    """S-3 — available < balance on the same alert is a pending hold: record,
    NO ledger row. The row-builder path never sees hold alerts."""

    def test_avail_below_balance_is_hold(self):
        self.assertTrue(is_pending_hold({'bal_after': 1000, 'avail': 900}))

    def test_avail_equal_balance_is_not_hold(self):
        self.assertFalse(is_pending_hold({'bal_after': 1000, 'avail': 1000}))

    def test_missing_either_field_is_not_hold(self):
        self.assertFalse(is_pending_hold({'bal_after': 1000}))
        self.assertFalse(is_pending_hold({'avail': 900}))
        self.assertFalse(is_pending_hold({}))


if __name__ == '__main__':
    unittest.main()
