import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from graph_scan import Scope, collect, timestamp, category_check
from cleanup_records import decision_gate, append_event, replay


class CategoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((Path(__file__).parent / 'fixtures/graph-categories.json').read_text())
        self.scope = Scope('hotmail', 'synthetic-inbox')
        self.start = timestamp('2026-02-01T00:00:00Z')
        self.end = timestamp('2026-02-02T00:00:00Z')
        self.assessment = dict(content_kinds=['promotion'], fraud_status='not_suspected',
                               body_reviewed=True, protection_fields_verified=True,
                               remove_authorized=True, content_only_removable=True,
                               reason='Synthetic body reviewed under the authorized cleanup policy')
        self.scan = self.scan_rows(self.payload['messages'])
        self.msg = self.scan['messages'][0]
        self.evidence = self.scan['metadata_evidence'][self.msg['id']]

    def scan_rows(self, rows, version='himalaya v2.1.0 +msgraph'):
        return collect(self.scope, self.start, self.end, lambda *_: {'messages': rows}, cli_version=version)

    def gate(self, msg=None, evidence=None, account='hotmail', assessment=None):
        return decision_gate(self.msg if msg is None else msg,
                             self.assessment if assessment is None else assessment, 'REMOVE',
                             metadata_evidence=self.evidence if evidence is None else evidence, account=account)

    def events(self, evidence=True):
        events = [dict(event_id='register', type='register', record_id='stable',
                       at='2026-02-02T00:00:00Z', account='hotmail', message=self.msg),
                  dict(event_id='decide', type='decide', record_id='stable',
                       at='2026-02-02T00:00:01Z', proposed='REMOVE', assessment=self.assessment)]
        if evidence:
            events[1]['metadata_evidence'] = self.evidence
        return events

    def plan(self):
        return dict(event_id='plan', type='plan_move', record_id='stable', at='2026-02-02T00:00:02Z',
                    operation_id='pilot', action='cleanup', decision_revision=1,
                    source_id=self.msg['id'], source_folder=self.msg['parentFolderId'],
                    destination_folder='synthetic-recovery', membership_evidence='local/preflight.json', authorized=True)

    def test_selected_cli_omission_passes_without_changing_snapshot(self):
        before = copy.deepcopy(self.payload)
        self.assertEqual(self.gate(), ('REMOVE', []))
        self.assertEqual(self.payload, before)
        self.assertNotIn('categories', self.msg)
        self.assertEqual(category_check(self.msg, self.evidence, 'hotmail'),
                         {'state': 'empty', 'basis': 'selected_cli_contract', 'wire_shape': 'unknown'})

    def test_absent_without_provenance_stays_review(self):
        self.assertEqual(self.gate(evidence={})[0], 'REVIEW')
        scan = self.scan_rows([self.msg], version=None)
        self.assertEqual(scan['metadata_evidence'], {})

    def test_populated_null_and_wrong_types_never_pass(self):
        for value in (['Personal'], None, '', {}, False, [None]):
            with self.subTest(value=value):
                msg = dict(self.msg, categories=value)
                scan = self.scan_rows([msg])
                self.assertEqual(self.gate(msg, scan['metadata_evidence'][msg['id']])[0], 'REVIEW')

    def test_explicit_empty_remains_backwards_compatible(self):
        self.assertEqual(self.gate(dict(self.msg, categories=[]), {})[0], 'REMOVE')

    def test_missing_other_protection_fields_still_block(self):
        for field in ('isDraft', 'importance', 'flag'):
            msg = copy.deepcopy(self.msg)
            del msg[field]
            scan = self.scan_rows([msg])
            self.assertEqual(self.gate(msg, scan['metadata_evidence'][msg['id']])[0], 'REVIEW')

    def test_body_policy_fraud_and_authorization_still_apply(self):
        for patch in ({'body_reviewed': False}, {'remove_authorized': False},
                      {'protection_fields_verified': False}, {'content_only_removable': False},
                      {'fraud_status': 'suspected'}, {'content_kinds': ['promotion', 'invoice']}):
            self.assertNotEqual(self.gate(assessment=dict(self.assessment, **patch))[0], 'REMOVE')

    def test_cross_account_and_changed_snapshot_rejected(self):
        for account in ('gmail', '', None):
            self.assertEqual(self.gate(account=account)[0], 'REVIEW')
        for patch in ({'id': 'another'}, {'parentFolderId': 'other-folder'}, {'isRead': False},
                      {'subject': 'changed'}):
            self.assertEqual(self.gate(dict(self.msg, **patch))[0], 'REVIEW')

    def test_unsupported_version_source_and_bad_evidence_rejected(self):
        for patch in ({'cli_version': 'himalaya v2.2.0 +msgraph'}, {'cli_version': 'himalaya v2.1.0-dev'},
                      {'source': 'shared-mime-read'}, {'source': 'native-get-trace'},
                      {'schema_version': 99}, {'snapshot_sha256': 'wrong'}, {'account': 'other'},
                      {'observed_at': None}, {'scope': None}):
            self.assertEqual(self.gate(evidence=dict(self.evidence, **patch))[0], 'REVIEW')

    def test_unselected_field_or_changed_command_rejected(self):
        for mode in ('unselected', 'get', 'different-account', 'extra-select'):
            evidence = copy.deepcopy(self.evidence)
            args = evidence['argv']
            if mode == 'unselected':
                args[args.index('--select')+1] = 'id,parentFolderId'
            elif mode == 'get':
                args[:] = ['himalaya', '--account', 'hotmail', '--json', 'msgraph', 'message', 'get', self.msg['id']]
            elif mode == 'different-account':
                args[args.index('--account')+1] = 'other'
            else:
                args.extend(['--select', 'id'])
            self.assertEqual(self.gate(evidence=evidence)[0], 'REVIEW')

    def test_unsupported_collector_version_fails_before_fetch(self):
        with self.assertRaises(ValueError):
            collect(self.scope, self.start, self.end, lambda *_: self.fail('must not fetch'),
                    cli_version='himalaya v2.2.0 +msgraph')

    def test_nonterminal_rows_do_not_receive_evidence(self):
        result = collect(self.scope, self.start, self.end,
                         lambda *_: {'messages': [self.msg], 'next_page': 'unfollowed'},
                         max_requests=1, cli_version='himalaya v2.1.0')
        self.assertFalse(result['complete'])
        self.assertEqual(result['metadata_evidence'], {})

    def test_journal_preserves_interpretation_and_replays_legacy_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'journal.json'
            for event in self.events():
                append_event(path, event)
            saved = json.loads(path.read_text())
            # Old plans remain readable, but cannot be newly appended/executed.
            records, operations = replay(saved+[self.plan()])
            record = records['stable']
            self.assertEqual(record['decision'], 'REMOVE')
            self.assertEqual(record['category_check']['basis'], 'selected_cli_contract')
            self.assertEqual(record['metadata_evidence'], self.evidence)
            self.assertNotIn('categories', record['message'])
            self.assertEqual(operations['pilot']['state'], 'planned')

    def test_review_can_be_persisted_but_cannot_plan_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'journal.json'
            for event in self.events(evidence=False):
                append_event(path, event)
            saved = json.loads(path.read_text())
            records, operations = replay(saved)
            self.assertEqual(records['stable']['decision'], 'REVIEW')
            self.assertEqual(operations, {})
            with self.assertRaisesRegex(ValueError, 'passing REMOVE'):
                append_event(path, self.plan())
            self.assertEqual(json.loads(path.read_text()), saved)


if __name__ == '__main__':
    unittest.main()
