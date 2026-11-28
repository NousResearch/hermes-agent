import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from graph_scan import Scope, Runner, collect, list_argv, read_argv, timestamp, atomic_write


def row(mid='A+/=opaque', received='2026-01-01T12:00:00Z'):
    return {'id': mid, 'receivedDateTime': received, 'parentFolderId': 'folder-id'}


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.scope = Scope('hotmail', 'inbox', top=2)
        self.start, self.end = timestamp('2026-01-01T00:00:00Z'), timestamp('2026-01-03T00:00:00Z')

    def test_split_uses_terminal_children_not_parent_subset(self):
        calls = []
        fixture = json.loads((Path(__file__).parent / 'fixtures/graph-pages.json').read_text())
        def fetch(scope, lower, upper):
            calls.append((scope, lower, upper))
            if lower == self.start and upper == self.end:
                return fixture['parent']
            return fixture['left'] if lower == self.start else fixture['right']
        result = collect(self.scope, self.start, self.end, fetch)
        self.assertTrue(result['complete'])
        self.assertEqual((result['requests'], result['unique_messages']), (3, 2))
        self.assertTrue(all(scope is self.scope for scope, _, _ in calls))
        self.assertEqual(calls[1][2], calls[2][1])

    def test_short_page_with_cursor_is_not_terminal(self):
        result = collect(self.scope, self.start, self.end,
                         lambda *_: {'messages': [], 'next_page': 'next'}, max_requests=1)
        self.assertFalse(result['complete'])
        self.assertEqual(result['reason'], 'request_budget')
        self.assertEqual(len(result['unresolved_intervals']), 2)

    def test_full_page_without_cursor_is_terminal(self):
        result = collect(self.scope, self.start, self.end,
                         lambda *_: {'messages': [row('a'), row('b')]})
        self.assertTrue(result['complete'])
        self.assertEqual(result['unique_messages'], 2)

    def test_one_second_dense_range_stays_incomplete(self):
        result = collect(self.scope, self.start, timestamp('2026-01-01T00:00:01Z'),
                         lambda *_: {'messages': [], 'next_page': 'unfollowable'})
        self.assertEqual(result['reason'], 'dense_interval_requires_continuation')
        self.assertFalse(result['complete'])

    def test_bad_responses_never_become_zero_success(self):
        for payload in ({'error': 'Invalid filter', 'messages': []}, {}, {'value': []},
                        {'messages': [{'id': 'x'}]}, {'messages': [], 'next_page': 5},
                        {'messages': [row(received='2025-12-31T23:59:59Z')]},
                        {'messages': [row(received='2026-01-03T00:00:00Z')]}):
            with self.subTest(payload=payload):
                result = collect(self.scope, self.start, self.end, lambda *_: payload)
                self.assertEqual(result['reason'], 'error')
                self.assertFalse(result['complete'])

    def test_stop_during_fetch_prevents_split_and_next_call(self):
        stop = [False]
        calls = []
        def fetch(*_):
            calls.append(1)
            stop[0] = True
            return {'messages': [row()], 'next_page': 'next'}
        result = collect(self.scope, self.start, self.end, fetch, lambda: stop[0])
        self.assertEqual((len(calls), result['reason'], result['complete']), (1, 'stopped', False))

    def test_stop_before_fetch_makes_no_call(self):
        result = collect(self.scope, self.start, self.end, lambda *_: self.fail(), lambda: True)
        self.assertEqual(result['requests'], 0)

    def test_partial_success_survives_later_failure_and_checkpoint(self):
        snapshots = []
        calls = [0]
        def fetch(*_):
            calls[0] += 1
            if calls[0] == 1:
                return {'messages': [], 'next_page': 'next'}
            if calls[0] == 2:
                return {'messages': [row()]}
            raise RuntimeError('Simulated failure')
        result = collect(self.scope, self.start, self.end, fetch,
                         checkpoint=lambda r: snapshots.append(json.loads(json.dumps(r))))
        self.assertEqual((result['unique_messages'], result['reason']), (1, 'error'))
        self.assertFalse(snapshots[-1]['complete'])
        self.assertEqual(len(result['terminal_intervals']), 1)

    def test_argv_preserves_odata_paths_and_ids(self):
        scope = Scope('hotmail', 'AA+/=Folder', config='C:/My Profile/config.toml', extra_filter="subject eq 'Bob''s invoice'")
        args = list_argv(scope, self.start, self.end)
        self.assertEqual(args[args.index('--folder')+1], 'AA+/=Folder')
        self.assertIn("subject eq 'Bob''s invoice'", args[args.index('--filter')+1])
        self.assertNotIn('--skip', args)
        rid = 'A+/=id with spaces $(do-not-run)'
        raw_args = read_argv(scope, rid)
        self.assertEqual(raw_args[-1], rid)
        self.assertNotIn('--seen', raw_args)
        self.assertNotIn('--json', raw_args)

    def test_runner_preserves_argv_without_shell_execution(self):
        tricky = 'C:/Path With Spaces/AA+/= $(echo no) "quoted"'
        result = Runner().run([sys.executable, '-c', 'import json,sys; print(json.dumps(sys.argv[1:]))', tricky])
        self.assertEqual(result, [tricky])

    def test_runner_exit_and_json_errors(self):
        for script in ('raise SystemExit(3)', 'print("not-json")'):
            with self.assertRaises(RuntimeError):
                Runner().run([sys.executable, '-c', script])

    def test_runner_timeout_is_bounded(self):
        with self.assertRaises(RuntimeError):
            Runner(timeout=.05).run([sys.executable, '-c', 'import time; time.sleep(10)'])

    def test_runner_stop_prevents_launch(self):
        with self.assertRaises(RuntimeError):
            Runner(stopped=lambda: True).run(['definitely-not-an-executable'])

    def test_atomic_checkpoint_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.json'
            atomic_write(path, {'complete': False})
            atomic_write(path, {'complete': True})
            self.assertTrue(json.loads(path.read_text())['complete'])
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_invalid_interval_and_naive_timestamp(self):
        with self.assertRaises(ValueError):
            timestamp('2026-01-01')
        with self.assertRaises(ValueError):
            collect(self.scope, self.end, self.start, None)


if __name__ == '__main__':
    unittest.main()
