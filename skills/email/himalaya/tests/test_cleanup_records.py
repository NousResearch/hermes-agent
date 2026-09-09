import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cleanup_records import decision_gate, replay, summary, append_event, locked


def message(mid='original', folder='inbox-id'):
    return {'id': mid, 'parentFolderId': folder, 'receivedDateTime': '2026-02-01T12:00:00Z',
            'internetMessageId': '<synthetic@example.invalid>', 'subject': 'Message',
            'isRead': False, 'isDraft': False, 'importance': 'normal', 'flag': {'flagStatus': 'notFlagged'}, 'categories': []}


def assessment(kinds=None):
    return {'content_kinds': kinds or ['promotion'], 'fraud_status': 'not_suspected',
            'body_reviewed': True, 'protection_fields_verified': True, 'remove_authorized': True,
            'content_only_removable': True, 'reason': 'Individually checked as an ordinary promotion'}


def event(eid, kind, **values):
    return dict(event_id=eid, type=kind, record_id='stable-record-1', at='2026-09-09T01:00:00Z', **values)


def planned_events():
    return [event('r', 'register', account='hotmail', message=message()),
            event('d', 'decide', proposed='REMOVE', assessment=assessment()),
            event('p', 'plan_move', operation_id='move-1', action='cleanup', decision_revision=1,
                  source_id='original', source_folder='inbox-id', destination_folder='cleanup-id',
                  membership_evidence='local/preflight.json', authorized=True)]


class DecisionTests(unittest.TestCase):
    def test_protected_categories_never_inherit_sender_remove(self):
        for kind in ('invoice', 'payment', 'purchase', 'refund', 'security', 'trial_conversion', 'order_update'):
            with self.subTest(kind=kind):
                self.assertEqual(decision_gate(message(), assessment(['promotion', kind]), 'REMOVE')[0], 'KEEP')

    def test_missing_fields_are_unknown_not_unprotected(self):
        for field in ('isDraft', 'importance', 'flag', 'categories'):
            msg = message()
            del msg[field]
            self.assertEqual(decision_gate(msg, assessment(), 'REMOVE')[0], 'REVIEW')

    def test_flagged_categorized_and_draft_candidates_require_review(self):
        for patch in ({'flag': {'flagStatus': 'flagged'}}, {'categories': ['Personal']},
                      {'importance': 'high'}, {'isDraft': True}):
            msg = dict(message(), **patch)
            self.assertEqual(decision_gate(msg, assessment(), 'REMOVE')[0], 'REVIEW')

    def test_auth_pass_does_not_override_phishing(self):
        data = dict(assessment(['invoice']), fraud_status='suspected', dkim='pass', dmarc='pass')
        self.assertEqual(decision_gate(message(), data, 'REMOVE')[0], 'REVIEW')

    def test_no_body_or_authorization_blocks_remove(self):
        for field in ('body_reviewed', 'remove_authorized', 'content_only_removable', 'protection_fields_verified'):
            data = dict(assessment(), **{field: False})
            self.assertEqual(decision_gate(message(), data, 'REMOVE')[0], 'REVIEW')

    def test_reviewed_ordinary_promotion_passes(self):
        self.assertEqual(decision_gate(message(), assessment(), 'REMOVE'), ('REMOVE', []))


class JournalTests(unittest.TestCase):
    def test_move_rescue_changes_ids_and_reconciles_counts(self):
        events = planned_events()
        events += [event('s', 'submitted', operation_id='move-1'),
                   event('c', 'confirmed', operation_id='move-1', source_absent=True, match_unique=True,
                         destination_message=message('after-move', 'cleanup-id'), evidence='local/verification.json'),
                   event('d2', 'decide', proposed='KEEP', assessment=assessment(['invoice'])),
                   event('p2', 'plan_move', operation_id='rescue-1', action='rescue', decision_revision=2,
                         source_id='after-move', source_folder='cleanup-id', destination_folder='inbox-id',
                         membership_evidence='local/rescue-preflight.json', authorized=True),
                   event('c2', 'confirmed', operation_id='rescue-1', source_absent=True, match_unique=True,
                         destination_message=message('after-rescue', 'inbox-id'), evidence='local/rescue-verification.json')]
        records, _ = replay(events)
        self.assertEqual(records['stable-record-1']['message']['id'], 'after-rescue')
        totals = summary(events)
        self.assertEqual(totals['decisions'], {'KEEP': 1})
        self.assertEqual((totals['confirmed_cleanup_moves'], totals['confirmed_rescue_moves']), (1, 1))
        self.assertEqual(totals['current_folder_counts'], {'inbox-id': 1})

    def test_old_id_or_wrong_source_cannot_be_planned(self):
        for field, value in (('source_id', 'stale'), ('source_folder', 'wrong-folder')):
            events = planned_events()
            events[-1][field] = value
            with self.assertRaises(ValueError):
                replay(events)

    def test_unknown_outcome_blocks_retry_and_revision(self):
        events = planned_events()+[event('u', 'unknown', operation_id='move-1')]
        self.assertEqual(summary(events)['pending_or_unknown_operations'], 1)
        with self.assertRaises(ValueError):
            replay(events+[event('d2', 'decide', proposed='KEEP', assessment={})])
        with self.assertRaises(ValueError):
            replay(events+[event('s', 'submitted', operation_id='move-1')])

    def test_confirmation_requires_unique_destination_and_source_absence(self):
        for patch in ({'match_unique': False}, {'source_absent': False}, {'evidence': ''}):
            confirmation = event('c', 'confirmed', operation_id='move-1', source_absent=True,
                                 match_unique=True, evidence='file.json', destination_message=message('new', 'cleanup-id'))
            confirmation.update(patch)
            with self.assertRaises(ValueError):
                replay(planned_events()+[confirmation])

    def test_changed_identity_cannot_confirm(self):
        wrong = message('new', 'cleanup-id')
        wrong['subject'] = 'Different message'
        with self.assertRaises(ValueError):
            replay(planned_events()+[event('c', 'confirmed', operation_id='move-1', source_absent=True,
                                          match_unique=True, evidence='file.json', destination_message=wrong)])

    def test_changed_or_missing_read_state_cannot_confirm(self):
        for new_state in (True, None):
            wrong = message('new', 'cleanup-id')
            wrong['isRead'] = new_state
            with self.assertRaises(ValueError):
                replay(planned_events()+[event('c', 'confirmed', operation_id='move-1', source_absent=True,
                                              match_unique=True, evidence='file.json', destination_message=wrong)])

    def test_duplicate_event_is_idempotent_but_conflict_rejected(self):
        events = planned_events()
        self.assertEqual(summary(events), summary(events+[events[-1]]))
        conflict = dict(events[-1], destination_folder='another-folder')
        with self.assertRaises(ValueError):
            replay(events+[conflict])

    def test_atomic_append_and_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'journal.json'
            first = planned_events()[0]
            append_event(path, first)
            append_event(path, first)
            self.assertEqual(len(json.loads(path.read_text())), 1)
            lock = path.with_name('journal.json.lock')
            with locked(lock):
                with self.assertRaises(FileExistsError):
                    append_event(path, planned_events()[1])
            self.assertFalse(lock.exists())
            self.assertEqual(len(json.loads(path.read_text())), 1)


if __name__ == '__main__':
    unittest.main()
