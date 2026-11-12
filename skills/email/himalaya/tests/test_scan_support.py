import json
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scan_support import (ScanScope, collect_pages, epoch_bounds,
                          in_internal_date_window, list_argv, month_windows, parse_page)


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.scope = ScanScope("work", ("INBOX",), "subject:invoice")
        self.pages = json.loads((Path(__file__).parent / "fixtures/gmail-pages.json").read_text())["pages"]

    def run_pages(self, pages, **kwargs):
        calls = []
        def fetch(scope, token):
            calls.append((scope, token))
            value = pages[len(calls) - 1]
            if isinstance(value, Exception):
                raise value
            return value
        return collect_pages(self.scope, fetch, lambda: False, **kwargs), calls

    def test_cursor_exhaustion_deduplicates_and_follows_empty_page(self):
        result, calls = self.run_pages(self.pages)
        self.assertEqual(result.ids, ["a", "b", "c"])
        self.assertTrue(result.exhausted)
        self.assertEqual(result.reason, "exhausted")
        self.assertEqual([token for _, token in calls], [None, "cursor-one", "cursor-two"])
        self.assertTrue(all(scope is self.scope for scope, _ in calls))

    def test_page_budget_is_incomplete(self):
        result, calls = self.run_pages(self.pages, max_pages=1)
        self.assertEqual((result.reason, result.exhausted, len(calls)), ("page_budget", False, 1))

    def test_requested_limit_preserves_unreturned_page_ids(self):
        result, calls = self.run_pages(self.pages, limit=1)
        self.assertEqual(result.ids, ["a"])
        self.assertEqual(result.pending_ids, ["b"])
        self.assertEqual(result.next_page, "cursor-one")
        self.assertEqual((result.reason, result.exhausted, len(calls)), ("requested_limit", False, 1))

    def test_error_keeps_partial_evidence_without_retry(self):
        result, calls = self.run_pages([self.pages[0], RuntimeError("request failed")])
        self.assertEqual(result.ids, ["a", "b"])
        self.assertEqual(result.reason, "error")
        self.assertFalse(result.exhausted)
        self.assertEqual(len(calls), 2)

    def test_repeated_cursor_stops_before_another_request(self):
        result, calls = self.run_pages([self.pages[0], {"ids": [], "next_page": "cursor-one"}])
        self.assertEqual((result.reason, result.exhausted, len(calls)), ("repeated_cursor", False, 2))

    def test_stop_before_first_call(self):
        def fetch(*args):
            self.fail("No fetch permitted")
        result = collect_pages(self.scope, fetch, lambda: True)
        self.assertEqual((result.reason, result.pages), ("stopped", 0))

    def test_stop_during_call_does_not_schedule_next_page(self):
        stopped, calls = [False], []
        def fetch(*args):
            calls.append(args)
            stopped[0] = True
            return self.pages[0]
        result = collect_pages(self.scope, fetch, lambda: stopped[0])
        self.assertEqual((result.reason, result.pages, len(calls)), ("stopped", 1, 1))
        self.assertEqual(result.ids, ["a", "b"])

    def test_invalid_payload_is_not_empty_success(self):
        for payload in [[], {}, {"messages": []}, {"ids": [], "nextPageToken": "x"},
                        {"ids": [], "next_page": 7}, {"ids": [{"id": 42}]},
                        {"ids": [{"id": "a"}, {}]}, {"ids": [], "next_page": ""}]:
            with self.subTest(payload=payload):
                result, _ = self.run_pages([payload])
                self.assertEqual((result.reason, result.ids, result.exhausted), ("error", [], False))

    def test_terminal_empty_and_null_pages(self):
        for payload in [{"ids": []}, {"ids": [], "next_page": None}]:
            result, _ = self.run_pages([payload])
            self.assertEqual((result.reason, result.ids, result.exhausted), ("exhausted", [], True))

    def test_scope_and_untrusted_query_are_single_argv_values(self):
        scope = ScanScope("work", ("INBOX", "Label_1"), 'from:a OR subject:$(echo x)', "a b.toml")
        first, second = list_argv(scope), list_argv(scope, "token $(echo x)")
        self.assertEqual(second[:-2], first)
        self.assertEqual(second[-2:], ["--page-token", "token $(echo x)"])
        self.assertEqual(first.count("--label"), 2)
        self.assertEqual(first[first.index("--query") + 1], scope.query)
        self.assertNotIn("--include-spam-trash", first)

    def test_account_scopes_do_not_deduplicate_each_other(self):
        fetch = lambda scope, token: {"ids": [{"id": "same"}]}
        results = [collect_pages(ScanScope(account, ("INBOX",)), fetch, lambda: False)
                   for account in ("work", "personal")]
        self.assertEqual([r.ids for r in results], [["same"], ["same"]])

    def test_invalid_budgets_and_scope(self):
        for budget in (0, -1, True):
            with self.assertRaises(ValueError):
                collect_pages(self.scope, None, lambda: False, max_pages=budget)
        with self.assertRaises(ValueError):
            ScanScope("", ("INBOX",))
        with self.assertRaises(ValueError):
            ScanScope("work", ["INBOX"])


class CalendarTests(unittest.TestCase):
    def test_backward_clipped_months(self):
        actual = month_windows(date(2026, 6, 15), date(2026, 9, 1))
        self.assertEqual(actual, [(date(2026, 8, 1), date(2026, 9, 1)),
                                  (date(2026, 7, 1), date(2026, 8, 1)),
                                  (date(2026, 6, 15), date(2026, 7, 1))])

    def test_leap_year_and_year_rollover(self):
        self.assertEqual((month_windows(date(2024, 2, 1), date(2024, 3, 1))[0][1]
                          - date(2024, 2, 1)).days, 29)
        windows = month_windows(date(2025, 12, 20), date(2026, 1, 5))
        self.assertEqual(windows[0], (date(2026, 1, 1), date(2026, 1, 5)))
        self.assertEqual(windows[1], (date(2025, 12, 20), date(2026, 1, 1)))

    def test_windows_cover_each_day_once(self):
        start, end = date(2023, 12, 13), date(2025, 3, 18)
        windows = month_windows(start, end)
        self.assertEqual(sum((b-a).days for a,b in windows), (end-start).days)
        self.assertTrue(all(a < b for a,b in windows))
        self.assertTrue(all(windows[i][0] == windows[i+1][1] for i in range(len(windows)-1)))

    def test_reversed_and_equal_bounds_fail(self):
        for end in (date(2026, 1, 1), date(2025, 1, 1)):
            with self.assertRaises(ValueError):
                month_windows(date(2026, 1, 1), end)

    def test_epoch_half_open_boundary_membership(self):
        low, high = epoch_bounds(date(2024, 2, 1), date(2024, 3, 1), "UTC")
        self.assertEqual(high-low, 29*86400)
        for value, expected in [(low*1000-1, False), (low*1000, True),
                                (high*1000-1, True), (high*1000, False)]:
            self.assertEqual(in_internal_date_window(str(value), low, high), expected)
        with self.assertRaises(ValueError):
            in_internal_date_window(None, low, high)

    def test_timezone_offset_is_explicit(self):
        start, end = date(2026, 3, 1), date(2026, 4, 1)
        utc = epoch_bounds(start, end, "UTC")
        la = epoch_bounds(start, end, "America/Los_Angeles")
        self.assertEqual(la[0]-utc[0], 8*3600)
        self.assertEqual(la[1]-utc[1], 7*3600)


if __name__ == "__main__":
    unittest.main()
