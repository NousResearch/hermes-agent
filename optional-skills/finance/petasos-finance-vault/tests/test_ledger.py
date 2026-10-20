"""Ledger — append-only JSONL discipline, IDs, validity, in-place relabel
(R1.1, R1.2, R1.4–R1.7; C1/DG-11 schema frozen v1-verbatim; DG-7 full
thread-id refs). All rows synthetic (R5.1).
"""
import json
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from petasos.classify import Decision  # noqa: E402
from petasos.knob import load_knobs  # noqa: E402
from petasos.ledger import (LedgerError, append_rows, build_row, next_id,  # noqa: E402
                            relabel, validate_ledger, verify_append)


def row(**kw):
    base = dict(id='2026-10-13.01', date='2026-10-13', dir='out', kind='expense',
                old_ref='mail:thread0001', amount=1000, currency='VND',
                desc='synthetic', status='posted', src='email:TEST')
    base.update(kw)
    return base


def make_ledger(rows):
    fd, path = tempfile.mkstemp(prefix='petasos-led-', suffix='.jsonl')
    os.close(fd)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return path


class NextId(unittest.TestCase):
    """R1.5 — max suffix + 1, never rows-count + 1; batch feedback."""

    def test_gap_suffix_gets_max_plus_one(self):
        rows = [row(id='2026-10-13.01'), row(id='2026-10-13.03')]
        self.assertEqual(next_id(rows, '2026-10-13', []), '2026-10-13.04')

    def test_batch_feedback_keeps_batch_ids_unique(self):
        rows = [row(id='2026-10-13.01')]
        first = next_id(rows, '2026-10-13', [])
        second = next_id(rows, '2026-10-13', [first])
        self.assertEqual((first, second), ('2026-10-13.02', '2026-10-13.03'))

    def test_other_dates_do_not_interfere(self):
        rows = [row(id='2026-10-12.09', date='2026-10-12')]
        self.assertEqual(next_id(rows, '2026-10-13', []), '2026-10-13.01')


class BuildRow(unittest.TestCase):
    """C1 shape + R1.2 provenance + DG-7 full thread id + date normalization."""

    def setUp(self):
        self.knobs = load_knobs({'PETASOS_VAULT': '/nv', 'PETASOS_CURRENCY': 'VND'})

    def test_full_shape_from_tx_and_decision(self):
        tx = {'kind': 'BIDV_CARD', 'dir': 'OUT', 'amount': 49487,
              'datetime': '13/10/26 10:25:55', 'merchant': 'GRAB X',
              'src_thread': 'abcdef1234567890', 'bal_after': 123}
        d = Decision('expense', 'posted', 'Personal', 'Wants', 'Variable', None)
        r = build_row(tx, d, self.knobs)
        self.assertEqual(r['id'], '')  # caller allocates via next_id
        self.assertEqual(r['date'], '2026-10-13')       # 2-digit year normalized
        self.assertEqual(r['dir'], 'out')               # email IN/OUT → in/out
        self.assertEqual(r['old_ref'], 'mail:abcdef1234567890')  # DG-7: FULL id
        self.assertEqual(r['src'], 'email:BIDV_CARD')
        self.assertEqual(r['amount'], 49487)
        self.assertEqual(r['currency'], 'VND')
        self.assertEqual(r['desc'], 'GRAB X')
        self.assertEqual(r['bal_after'], 123)
        self.assertEqual(r['category'], 'Personal')

    def test_iso_date_field_wins_when_no_datetime(self):
        tx = {'kind': 'HSBC_ACC', 'dir': 'IN', 'amount': 1, 'date': '2026-12-01',
              'src_thread': 't' * 12}
        r = build_row(tx, Decision('oneoff', 'posted'), self.knobs)
        self.assertEqual(r['date'], '2026-12-01')

    def test_desc_truncated_to_90(self):
        tx = {'kind': 'X', 'dir': 'OUT', 'amount': 1, 'content': 'A' * 200,
              'src_thread': 't'}
        r = build_row(tx, Decision('expense', 'posted'), self.knobs)
        self.assertLessEqual(len(r['desc']), 90)

    def test_optional_fields_omitted_when_none(self):
        tx = {'kind': 'X', 'dir': 'OUT', 'amount': 1, 'src_thread': 't'}
        r = build_row(tx, Decision('expense', 'needs-confirm'), self.knobs)
        for key in ('category', 'type', 'fv', 'ask', 'bal_after'):
            self.assertNotIn(key, r)


class ValidateLedger(unittest.TestCase):
    """R1.6/R1.2 — closed status vocabulary, resolvable src, unique ids."""

    def test_valid_rows_pass(self):
        self.assertEqual(validate_ledger([row(), row(id='2026-10-13.02')]), [])

    def test_status_outside_vocabulary_fails(self):
        problems = validate_ledger([row(status='income?')])
        self.assertTrue(problems and 'status' in problems[0])

    def test_src_must_resolve_to_a_provenance_form(self):
        for bad in ('weird', 'email:', 'user-confirm:someday', 'import:'):
            problems = validate_ledger([row(src=bad)])
            self.assertTrue(problems, f'src={bad!r} must be rejected')

    def test_v1_compat_provenance_forms_accepted(self):
        """Live history carries v1 evidence labels — the write gate must
        accept them, not refuse the whole ledger (live finding 14/10)."""
        for good in ('chat:telegram-2026-10-14', 'bidv-screenshot:2026-10-01',
                     'user-count:2026-10-07', 'msb-screenshot:2026-09-29',
                     'email:ACB', 'user-confirm:2026-10-13', 'import:BankAccounts-IT'):
            self.assertEqual(validate_ledger([row(src=good)]), [],
                             f'src={good!r} must be accepted')

    def test_duplicate_ids_fail(self):
        problems = validate_ledger([row(), row()])
        self.assertTrue(problems and 'duplicate' in problems[0])


class AppendDiscipline(unittest.TestCase):
    """R1.1/R1.7 — append-only byte identity, exact count growth, re-read."""

    def test_append_preserves_prefix_and_grows_exactly(self):
        path = make_ledger([row()])
        before = open(path, 'rb').read()
        new = [row(id='2026-10-13.02'), row(id='2026-10-13.03')]
        append_rows(path, new)
        after = open(path, 'rb').read()
        self.assertTrue(after.startswith(before))          # R1.1 byte-identical prefix
        self.assertEqual(after[len(before):].count(b'\n'), 2)  # trailing newline each
        verify_append(path, 1, new)                        # R1.7 must not raise

    def test_verify_append_raises_on_count_mismatch(self):
        path = make_ledger([row()])
        with self.assertRaises(LedgerError):
            verify_append(path, 1, [row(id='2026-10-13.02'), row(id='2026-10-13.03')])

    def test_invalid_row_rejected_before_touching_file(self):
        path = make_ledger([row()])
        before = open(path, 'rb').read()
        with self.assertRaises(LedgerError):
            append_rows(path, [row(id='2026-10-13.02', status='bogus')])
        self.assertEqual(open(path, 'rb').read(), before)  # untouched

    def test_duplicate_of_existing_id_rejected(self):
        path = make_ledger([row()])
        with self.assertRaises(LedgerError):
            append_rows(path, [row()])                     # same id again


class RelabelInPlace(unittest.TestCase):
    """R1.4 — corrections relabel in place; ONLY the sanctioned field set."""

    def test_category_only_correction_touches_one_row(self):
        other = row(id='2026-10-13.02', desc='other row stays')
        path = make_ledger([row(), other])
        before_other = json.dumps(other, ensure_ascii=False)
        relabel(path, '2026-10-13.01',
                {'category': 'Food', 'src': 'user-confirm:2026-10-13'})
        lines = open(path, encoding='utf-8').read().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1], before_other)           # byte-identical sibling
        first = json.loads(lines[0])
        self.assertEqual(first['category'], 'Food')
        self.assertEqual(first['src'], 'user-confirm:2026-10-13')
        self.assertEqual(first['old_ref'], 'mail:thread0001')  # evidence kept

    def test_evidence_fields_are_never_relabelable(self):
        for field in ('id', 'date', 'dir', 'amount', 'currency', 'old_ref', 'bal_after'):
            path = make_ledger([row()])
            with self.assertRaises(LedgerError, msg=f'{field} must be rejected'):
                relabel(path, '2026-10-13.01', {field: 'x'})

    def test_status_only_promotable_to_posted(self):
        path = make_ledger([row(status='needs-confirm')])
        relabel(path, '2026-10-13.01', {'status': 'posted'})
        self.assertEqual(json.loads(open(path, encoding='utf-8').read())['status'],
                         'posted')
        path = make_ledger([row(status='needs-confirm')])
        with self.assertRaises(LedgerError):
            relabel(path, '2026-10-13.01', {'status': 'candidate'})

    def test_kind_relabel_only_from_needs_confirm(self):
        path = make_ledger([row(status='needs-confirm', kind='oneoff')])
        relabel(path, '2026-10-13.01', {'kind': 'salary'})
        self.assertEqual(json.loads(open(path, encoding='utf-8').read())['kind'], 'salary')
        path = make_ledger([row(status='posted')])
        with self.assertRaises(LedgerError):
            relabel(path, '2026-10-13.01', {'kind': 'salary'})

    def test_unknown_row_id_is_loud(self):
        path = make_ledger([row()])
        with self.assertRaises(LedgerError):
            relabel(path, '2099-01-01.01', {'category': 'Food'})


if __name__ == '__main__':
    unittest.main()
