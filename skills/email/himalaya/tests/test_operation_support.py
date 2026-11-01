import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from operation_support import run_recorded, load_capture, successful_json, digest, save_new, unique_record
from backend_operations import (target_get_argv, target_from_capture, validate_target,
                                graph_move_argv, validate_graph_plan)
from cleanup_records import append_event, replay, decision_gate
from graph_scan import collect, Scope, timestamp
from graph_move import prepare_plan, execute_plan, cancel_plan
from support24 import review_fixture


def write_capture(directory, argv, cwd, payload, code=0, before_launch=lambda *_: None):
    """Synthetic execution boundary. No Himalaya process or mailbox."""
    directory = Path(directory)
    directory.mkdir()
    invocation = dict(schema_version=1, argv=argv, cwd=str(cwd), shell=False,
                      timeout_seconds=30, recorded_at='2026-09-09T00:00:00Z')
    save_new(directory/'invocation.json', invocation)
    before_launch(invocation, directory/'invocation.json')
    raw = json.dumps(payload).encode()
    (directory/'stdout.bin').write_bytes(raw)
    (directory/'stderr.bin').write_bytes(b'')
    save_new(directory/'result.json', dict(status='completed', returncode=code,
             invocation_sha256=digest((directory/'invocation.json').read_bytes()),
             stdout_sha256=digest(raw), stderr_sha256=digest(b'')))
    return str(directory)


class ProcessRecordTests(unittest.TestCase):
    def test_records_actual_argv_cwd_outputs_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            value = 'opaque A+/= $(not-a-command) "quotes"'
            args = [sys.executable, '-c', 'import json,sys; print(json.dumps(sys.argv[1:])); sys.stderr.write("detail"); sys.exit(7)', value]
            run_recorded(args, root, root/'attempt')
            invocation, result, stdout = load_capture(root/'attempt')
            self.assertEqual(invocation['argv'][1:], args[1:])
            self.assertTrue(Path(invocation['argv'][0]).is_absolute())
            self.assertEqual(invocation['cwd'], str(root))
            self.assertFalse(invocation['shell'])
            self.assertEqual(json.loads(stdout), [value])
            self.assertEqual(result['returncode'], 7)
            self.assertEqual((root/'attempt/stderr.bin').read_bytes(), b'detail')
            with self.assertRaises(ValueError): successful_json(root/'attempt')

    def test_capture_exists_before_process_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            def before(invocation, path):
                self.assertEqual(json.loads(path.read_bytes()), invocation)
                self.assertFalse((root/'effect').exists())
            run_recorded([sys.executable, '-c', 'from pathlib import Path; Path("effect").write_text("ok")'],
                         root, root/'attempt', before_launch=before)
            self.assertTrue((root/'effect').exists())

    def test_capture_directory_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            args = [sys.executable, '-c', 'print("{}")']
            run_recorded(args, root, root/'attempt')
            with self.assertRaises(FileExistsError): run_recorded(args, root, root/'attempt')

    def test_stop_prevents_launch_and_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run_recorded([sys.executable, '-c', 'raise Exception()'], root, root/'attempt',
                         stopped=lambda: True, before_launch=lambda *_: self.fail('must not submit'))
            self.assertEqual(load_capture(root/'attempt')[1]['status'], 'stopped_before_launch')

    def test_timeout_retains_output_and_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run_recorded([sys.executable, '-c', 'import time; print("started",flush=True); time.sleep(10)'],
                         root, root/'attempt', timeout=.15)
            self.assertEqual(load_capture(root/'attempt')[1]['status'], 'timeout')

    def test_tampered_capture_and_error_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run_recorded([sys.executable, '-c', 'print(\'{"error":"failure"}\')'], root, root/'attempt')
            with self.assertRaises(ValueError): successful_json(root/'attempt')
            (root/'attempt/stdout.bin').write_bytes(b'{}')
            with self.assertRaises(ValueError): load_capture(root/'attempt')


class VerifiedTargetPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.runtime = dict(executable=str(self.root/'himalaya'), cwd=str(self.root),
                            cli_version='himalaya v2.1.0 +msgraph', config='')
        self.folder = {'id': 'synthetic-VAAlk-destination', 'displayName': 'Recovery', 'totalItemCount': 3}
        self.capture = self.root/'folder-get'
        write_capture(self.capture, target_get_argv(self.runtime, 'work', 'msgraph', self.folder['id']),
                      self.root, {'folders': [self.folder]})
        self.target = target_from_capture(self.runtime, 'work', 'msgraph', self.folder, self.capture)
        self.target_path = self.root/'target.json'
        save_new(self.target_path, self.target)
        self.msg = json.loads((Path(__file__).parent/'fixtures/graph-categories.json').read_text())['messages'][0]
        scan = collect(Scope('work', self.msg['parentFolderId']), timestamp('2026-02-01T00:00:00Z'),
                       timestamp('2026-02-02T00:00:00Z'), lambda *_: {'messages': [self.msg]},
                       cli_version=self.runtime['cli_version'])
        self.preflight = self.root/'preflight.json'
        save_new(self.preflight, scan)
        self.journal = self.root/'journal.json'
        self.review_path, self.assessment_path = review_fixture(self.root, self.msg, 'msgraph')
        self.assessment = dict(content_kinds=['promotion'], fraud_status='not_suspected', body_reviewed=True,
                               protection_fields_verified=True, content_only_removable=True,
                               remove_authorized=True, reason='Synthetic reviewed sale')
        append_event(self.journal, dict(event_id='r', record_id='record', type='register', at='2026-09-09T00:00:00Z',
                                       account='work', backend='msgraph', message=self.msg))
        append_event(self.journal, dict(event_id='d', record_id='record', type='decide', at='2026-09-09T00:00:01Z',
                                       proposed='REMOVE', assessment=self.assessment,
                                       metadata_evidence=scan['metadata_evidence'][self.msg['id']]))

    def plan(self):
        return prepare_plan(self.journal, 'record', self.target_path, self.preflight, 'op', True,
                            review_path=self.review_path, assessment_path=self.assessment_path)

    def test_plan_loads_ids_from_verified_records(self):
        plan = self.plan()
        self.assertEqual(plan['destination_folder'], self.folder['id'])
        self.assertEqual(plan['argv'][-2:], [self.msg['id'], self.folder['id']])
        records, ops = replay(json.loads(self.journal.read_bytes()))
        self.assertEqual(ops['op']['plan_schema'], 2)
        self.assertEqual(validate_graph_plan(plan, records['record'], True), plan['argv'])

    def test_cancel_unsubmitted_graph_plan_retains_history(self):
        self.plan()
        cancel_plan(self.journal, 'op', 'New review required')
        records, ops = replay(json.loads(self.journal.read_bytes()))
        self.assertEqual(ops['op']['state'], 'cancelled')
        self.assertIsNone(records['record']['pending_operation'])

    def test_missing_review_blocks_new_graph_plan(self):
        with self.assertRaises(ValueError):
            prepare_plan(self.journal, 'record', self.target_path, self.preflight, 'op', True)

    def test_extra_character_in_plan_id_or_argv_is_rejected(self):
        plan = self.plan()
        records, _ = replay(json.loads(self.journal.read_bytes()))
        for change in ('id', 'argv', 'both'):
            bad = copy.deepcopy(plan)
            wrong = self.folder['id'].replace('VAAlk', 'VAAAlk')
            if change in ('id', 'both'): bad['destination_folder'] = wrong
            if change in ('argv', 'both'): bad['argv'][-1] = wrong
            with self.assertRaises(ValueError): validate_graph_plan(bad, records['record'])

    def test_wrong_get_id_or_name_is_rejected(self):
        for change in ({'id': 'wrong'}, {'displayName': 'Other'}):
            with self.assertRaises(ValueError):
                target_from_capture(self.runtime, 'work', 'msgraph', dict(self.folder, **change), self.capture)

    def test_duplicate_target_name_is_rejected(self):
        with self.assertRaises(ValueError):
            unique_record({'folders': [self.folder, dict(self.folder, id='another')]}, 'folders', 'displayName', 'Recovery')

    def test_target_account_backend_or_capture_mismatch_is_rejected(self):
        for account, backend in [('other', 'msgraph'), ('work', 'gmail')]:
            with self.assertRaises(ValueError): validate_target(self.target, account, backend)
        (self.capture/'stdout.bin').write_bytes(b'{}')
        with self.assertRaises(ValueError): self.plan()

    def test_gmail_uses_same_verifier_with_distinct_adapter(self):
        label = {'id': 'Label_synthetic', 'name': 'Recovery', 'type': 'user'}
        args = target_get_argv(self.runtime, 'work', 'gmail', label['id'])
        self.assertEqual(args[-5:], ['gmail', 'labels', 'get', '--', label['id']])
        cap = self.root/'label-get'
        write_capture(cap, args, self.root, {'labels': [label]})
        evidence = target_from_capture(self.runtime, 'work', 'gmail', label, cap)
        self.assertEqual(validate_target(evidence, 'work', 'gmail', True), label['id'])
        with self.assertRaises(ValueError): validate_target(evidence, 'work', 'msgraph')
        with self.assertRaises(ValueError): decision_gate(self.msg, self.assessment, 'REMOVE', backend='gmail')

    def test_missing_authorization_and_changed_preflight_block_plan(self):
        with self.assertRaises(ValueError):
            prepare_plan(self.journal, 'record', self.target_path, self.preflight, 'op', False)
        scan = json.loads(self.preflight.read_bytes())
        scan['messages'][0]['isRead'] = False
        self.preflight.write_text(json.dumps(scan))
        with self.assertRaises(ValueError): self.plan()

    def test_old_journal_readable_but_new_legacy_plan_rejected(self):
        events = json.loads(self.journal.read_bytes())
        old = dict(event_id='old', record_id='record', type='plan_move', at='2026-09-09T00:00:00Z',
                   operation_id='old', action='cleanup', decision_revision=1, source_id=self.msg['id'],
                   source_folder=self.msg['parentFolderId'], destination_folder=self.folder['id'],
                   membership_evidence='old-file.json', authorized=True)
        self.assertEqual(replay(events+[old])[1]['old']['plan_schema'], 1)
        with self.assertRaises(ValueError): append_event(self.journal, old)

    def test_execution_records_one_attempt_then_blocks_retry(self):
        plan = self.plan()
        seen = []
        def boundary(args, cwd, directory, timeout, stopped, before_launch):
            seen.append(copy.deepcopy(args))
            return write_capture(directory, args, cwd, {'error': 'simulated rejection'}, 1, before_launch)
        with patch('graph_move.run_recorded', side_effect=boundary):
            result = execute_plan(self.journal, 'op', self.root/'move')
            self.assertEqual(result['returncode'], 1)
            with self.assertRaises(ValueError): execute_plan(self.journal, 'op', self.root/'retry')
        self.assertEqual(seen, [plan['argv']])
        _, ops = replay(json.loads(self.journal.read_bytes()))
        self.assertEqual(ops['op']['state'], 'unknown')
        self.assertEqual(ops['op']['execution']['argv'], plan['argv'])

    def test_legacy_pending_plan_cannot_execute(self):
        plan = self.plan()
        events = json.loads(self.journal.read_bytes())
        del events[-1]['plan_schema']
        self.journal.write_text(json.dumps(events))
        with patch('graph_move.run_recorded', side_effect=AssertionError('must not launch')):
            with self.assertRaises(ValueError): execute_plan(self.journal, 'op', self.root/'move')

    def test_exit_zero_stays_unknown_until_verified(self):
        self.plan()
        def boundary(args, cwd, directory, timeout, stopped, before_launch):
            return write_capture(directory, args, cwd, {'message': 'Synthetic success'}, 0, before_launch)
        with patch('graph_move.run_recorded', side_effect=boundary):
            result = execute_plan(self.journal, 'op', self.root/'move')
        self.assertEqual(result['returncode'], 0)
        _, operations = replay(json.loads(self.journal.read_bytes()))
        self.assertEqual(operations['op']['state'], 'unknown')

    def test_changed_destination_capture_blocks_execution(self):
        self.plan()
        (self.capture/'stdout.bin').write_bytes(b'{}')
        with patch('graph_move.run_recorded', side_effect=AssertionError('must not launch')):
            with self.assertRaises(ValueError): execute_plan(self.journal, 'op', self.root/'move')


if __name__ == '__main__':
    unittest.main()
