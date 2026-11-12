import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from operation_support import json_digest, save_new, run_recorded
from review_support import decode_shared, load_review, validate
from task_support import task_lock, atomic_json, attempt_path
from scan_support import ScanScope, list_argv
from gmail_scan import collect
from backend_operations import target_from_capture, target_get_argv, verify_target
import gmail_cleanup as gmail
from support24 import review_fixture
from test_operation_support import write_capture


class SharedReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.r, self.a = review_fixture(self.root, {'id': 'synthetic-message'},
                                       text='Sale. '*100+'Your existing subscription renews tomorrow.')

    def test_complete_chunks_cover_tail(self):
        r = load_review(self.r)
        self.assertIn('renews tomorrow', r['chunks'][-1]['text'])
        self.assertEqual(''.join(c['text'] for c in r['chunks']),
                         'Sale. '*100+'Your existing subscription renews tomorrow.')

    def test_partial_review_rejected(self):
        a = json.loads(self.a.read_bytes())
        a['chunk_reviews'] = a['chunk_reviews'][:1]
        with self.assertRaises(ValueError): validate(load_review(self.r), a)

    def test_reordered_review_rejected(self):
        a = json.loads(self.a.read_bytes())
        a['chunk_reviews'].reverse()
        with self.assertRaises(ValueError): validate(load_review(self.r), a)

    def test_protected_mixed_message_rejected(self):
        a = json.loads(self.a.read_bytes())
        a['protected_content'] = True
        with self.assertRaises(ValueError): validate(load_review(self.r), a)

    def test_provisional_or_fraud_rejected(self):
        for key, value in [('provisional', True), ('fraud_status', 'suspected'), ('attachment_dependent', True)]:
            a = json.loads(self.a.read_bytes())
            a[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate(load_review(self.r), a)

    def test_source_change_invalidates_review(self):
        (self.root/'body.json').write_text('{}')
        with self.assertRaises(ValueError): load_review(self.r)

    def test_plain_text_angle_brackets_preserved(self):
        p = {'parts': [{'body': {'Text': 'Reference <ORDER-123>'}}],
             'text_body': [0], 'html_body': [], 'attachments': []}
        self.assertEqual(decode_shared(p)['blocks'][0]['text'], 'Reference <ORDER-123>')

    def test_selected_alternatives_and_attachment_exclusion(self):
        p = {'parts': [{'body': {'Text': 'Boilerplate'}}, {'body': {'Html': '<p>Payment due</p>'}},
                       {'body': {'Text': 'ATTACHMENT'}}],
             'text_body': [0, 2], 'html_body': [1], 'attachments': [2]}
        r = decode_shared(p)
        self.assertEqual(len(r['blocks']), 2)
        self.assertIn('Payment due', r['blocks'][1]['text'])
        self.assertNotIn('ATTACHMENT', str(r['blocks']))

    def test_encoding_warning_not_silent(self):
        p = {'parts': [{'body': {'Text': 'sale'}, 'is_encoding_problem': True}],
             'text_body': [0], 'html_body': [], 'attachments': []}
        self.assertTrue(decode_shared(p)['warnings'])

    def test_missing_or_invalid_indices_rejected(self):
        for p in [{'parts': []}, {'parts': [], 'text_body': [True], 'html_body': [], 'attachments': []}]:
            with self.assertRaises(ValueError): decode_shared(p)


class DurableScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.scope = ScanScope('work', labels=('INBOX',))
        self.checkpoint = self.root/'scan.json'
        self.pages, self.argv = [], []

    def runner(self, argv, cwd, directory, **kw):
        self.argv.append(argv)
        payload = self.pages.pop(0)
        if isinstance(payload, Exception): raise payload
        return write_capture(directory, argv, cwd, payload)

    def scan(self, budget=100, stopped=lambda: False, scope=None):
        return collect(scope or self.scope, self.checkpoint, self.root/'captures', self.root,
                       sys.executable, stopped, budget, self.runner)

    def test_resume_budget_and_same_page_duplicates(self):
        self.pages = [{'ids': [{'id': 'a'}, {'id': 'a'}], 'next_page': 'next'},
                      {'ids': [{'id': 'a'}, {'id': 'b'}]}]
        self.assertEqual(self.scan(1)['reason'], 'page_budget')
        state = self.scan(1)
        self.assertEqual(state['ids'], ['a', 'b'])
        self.assertTrue(state['complete'])
        self.assertIn('next', self.argv[-1])
        self.assertEqual(len(set(state['attempts'])), 2)

    def test_scope_change_rejected(self):
        self.pages = [{'ids': [], 'next_page': 'n'}]
        self.scan(1)
        with self.assertRaises(ValueError): self.scan(scope=ScanScope('other', labels=('INBOX',)))

    def test_repeat_across_resume_rejected(self):
        self.pages = [{'ids': [], 'next_page': 'n'}, {'ids': [], 'next_page': 'n'}]
        self.scan(1)
        self.scan(1)
        self.assertEqual(self.scan()['reason'], 'repeated_cursor')
        self.assertEqual(len(self.argv), 2)

    def test_error_retains_cursor_unique_new_attempt(self):
        self.pages = [{'ids': [{'id': 'a'}], 'next_page': 'n'}, ValueError('failure')]
        self.assertEqual(self.scan()['reason'], 'error')
        self.pages = [{'ids': [{'id': 'b'}]}]
        state = self.scan()
        self.assertEqual(state['ids'], ['a', 'b'])
        self.assertEqual(len(set(state['attempts'])), 3)

    def test_stop_during_page_commits_then_stops(self):
        self.pages = [{'ids': [{'id': 'a'}], 'next_page': 'n'}]
        state = self.scan(stopped=lambda: len(self.argv) > 0)
        self.assertEqual(state['ids'], ['a'])
        self.assertEqual(state['reason'], 'stopped')
        self.assertEqual(len(self.argv), 1)

    def test_explicit_executable(self):
        self.assertEqual(list_argv(self.scope, executable=sys.executable)[0], sys.executable)


class StateTests(unittest.TestCase):
    def test_live_lock_and_reacquire(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'lock'
            with task_lock(path):
                with self.assertRaises(ValueError):
                    with task_lock(path): pass
            self.assertTrue(path.exists())
            with task_lock(path): pass

    def test_atomic_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state'
            atomic_json(path, {'revision': 1})
            with patch('task_support.os.replace', side_effect=OSError('test')):
                with self.assertRaises(OSError): atomic_json(path, {'revision': 2})
            self.assertEqual(json.loads(path.read_bytes()), {'revision': 1})

    def test_attempt_names_do_not_reuse_counters(self):
        self.assertNotEqual(attempt_path('/tmp', 'body'), attempt_path('/tmp', 'body'))

    def test_failed_target_initialization_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_new(root/'labels.json', {'labels': [{'id': 'Label_demo', 'name': 'Recovery'}]})
            with self.assertRaises(ValueError):
                verify_target(root/'labels.json', 'Recovery', 'work', 'gmail', root/'target',
                              root, executable='certainly-not-a-real-program-123')
            self.assertTrue((root/'target/request.json').exists())
            self.assertTrue((root/'target/failure.json').exists())


class GmailOperationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.runtime = {'executable': sys.executable, 'cwd': str(self.root),
                        'cli_version': 'himalaya v2.1.0 +gmail', 'config': ''}
        self.label = {'id': 'Label_demo', 'name': 'Recovery', 'type': 'user'}
        self.target_response = {'labels': [self.label]}
        write_capture(self.root/'target-get', target_get_argv(self.runtime, 'work', 'gmail', self.label['id']),
                      self.root, self.target_response)
        target = target_from_capture(self.runtime, 'work', 'gmail', self.label, self.root/'target-get')
        self.target = self.root/'target.json'
        save_new(self.target, target)
        self.message = {'id': 'synthetic-mail', 'label-ids': ['INBOX', 'UNREAD', 'CATEGORY_PROMOTIONS'],
                        'headers': [{'name': 'Subject', 'value': 'A sale'}], 'internal-date': '123'}
        self.current = copy.deepcopy(self.message)
        write_capture(self.root/'source', gmail.get_argv(self.runtime, 'work', self.message['id']),
                      self.root, self.message)
        self.review, self.assessment = review_fixture(self.root, self.message)
        self.journal = self.root/'journal.json'
        self.calls = []

    def plan(self, authorized=True):
        return gmail.plan(self.journal, 'record', 'operation', self.target, self.root/'source',
                          self.review, self.assessment, authorized)

    def runner(self, argv, cwd, directory, before_launch=lambda *_: None, **kw):
        self.calls.append(argv)
        if 'modify' in argv:
            payload = {'message': 'synthetic modified'}
            self.current['label-ids'] = ['UNREAD', 'CATEGORY_PROMOTIONS', 'Label_demo']
        elif 'labels' in argv:
            payload = self.target_response
        else:
            payload = copy.deepcopy(self.current)
        return write_capture(directory, argv, cwd, payload, before_launch=before_launch)

    def execute(self, stopped=lambda: False):
        return gmail.execute(self.journal, 'work', 'operation', self.root/'captures', stopped, self.runner)

    def reconcile(self):
        return gmail.reconcile(self.journal, 'work', 'operation', self.root/'captures', lambda: False, self.runner)

    def test_plan_execute_unknown_then_confirm(self):
        proposal = self.plan()
        self.assertEqual(proposal['argv'][-1], self.message['id'])
        self.assertEqual(self.execute()['status'], 'unknown')
        self.assertEqual(self.reconcile()['status'], 'confirmed')
        self.assertIn('UNREAD', self.current['label-ids'])

    def test_no_automatic_retry(self):
        self.plan()
        self.execute()
        with self.assertRaises(ValueError): self.execute()

    def test_cancel_only_before_submission(self):
        self.plan()
        self.assertEqual(gmail.cancel(self.journal, 'work', 'operation', 'Source changed')['status'], 'cancelled')
        with self.assertRaises(ValueError): self.execute()

    def test_unknown_cannot_be_cancelled(self):
        self.plan()
        self.execute()
        with self.assertRaises(ValueError): gmail.cancel(self.journal, 'work', 'operation', 'Try again')

    def test_target_count_change_is_not_identity_change(self):
        self.plan()
        self.target_response['labels'][0]['messagesTotal'] = 10
        self.assertEqual(self.execute()['status'], 'unknown')

    def test_target_identity_change_blocks_modify(self):
        self.plan()
        self.target_response['labels'][0]['name'] = 'Renamed'
        with self.assertRaises(ValueError): self.execute()
        self.assertFalse(any('modify' in a for a in self.calls))

    def test_changed_protection_blocks_mutation(self):
        self.plan()
        self.current['label-ids'].append('STARRED')
        with self.assertRaises(ValueError): self.execute()
        self.assertFalse(any('modify' in a for a in self.calls))

    def test_unread_change_remains_unknown(self):
        self.plan()
        self.execute()
        self.current['label-ids'].remove('UNREAD')
        self.assertEqual(self.reconcile()['status'], 'unknown')

    def test_unrelated_label_change_remains_unknown(self):
        self.plan()
        self.execute()
        self.current['label-ids'].remove('CATEGORY_PROMOTIONS')
        self.assertEqual(self.reconcile()['status'], 'unknown')

    def test_no_effect_verified(self):
        self.plan()
        self.execute()
        self.current = copy.deepcopy(self.message)
        self.assertEqual(self.reconcile()['status'], 'failed_no_effect_verified')

    def test_stop_prevents_any_call(self):
        self.plan()
        self.assertEqual(self.execute(lambda: True)['status'], 'stopped_before_preflight')
        self.assertEqual(self.calls, [])

    def test_plan_without_authorization_rejected(self):
        with self.assertRaises(ValueError): self.plan(False)

    def test_tampered_id_even_with_new_hash_rejected(self):
        self.plan()
        state = json.loads(self.journal.read_bytes())
        op = state['operations']['operation']
        op['plan']['argv'][-1] = 'another-id'
        op['plan_sha256'] = json_digest(op['plan'])
        atomic_json(self.journal, state)
        with self.assertRaises(ValueError): self.execute()

    def test_changed_review_rejected(self):
        self.plan()
        self.assessment.write_text('{}')
        with self.assertRaises(ValueError): self.execute()

    def test_duplicate_record_for_same_id_rejected(self):
        self.plan()
        with self.assertRaises(ValueError):
            gmail.plan(self.journal, 'other', 'other-op', self.target, self.root/'source',
                       self.review, self.assessment, True)


if __name__ == '__main__':
    unittest.main()
