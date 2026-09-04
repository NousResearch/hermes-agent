"""Regression tests for dashboard sidebar scan coalescing."""

import inspect
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from hermes_cli.web_routers import profiles


class SidebarCacheTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(profiles, "_SIDEBAR_CACHE_TTL_SECONDS", 5.0)
        patcher.start()
        self.addCleanup(patcher.stop)
        profiles._sidebar_profile_cache_clear()
        self.addCleanup(profiles._sidebar_profile_cache_clear)

    def test_profile_cache_uses_db_and_wal_fingerprint_and_defensive_copies(self):
        with tempfile.TemporaryDirectory() as root:
            db_path = Path(root) / "state.db"
            wal_path = Path(f"{db_path}-wal")
            db_path.write_bytes(b"db-v1")
            wal_path.write_bytes(b"wal-v1")
            first_fingerprint = profiles._sidebar_db_fingerprint(db_path)
            first_key = (str(db_path), first_fingerprint, False, 0, (), 50, 100, ())
            payload = {"recents": None, "cron": [{"id": "one"}], "messaging": []}

            profiles._sidebar_profile_cache_put(first_key, payload)
            cached = profiles._sidebar_profile_cache_get(first_key)
            cached["cron"][0]["id"] = "mutated"
            self.assertEqual(
                profiles._sidebar_profile_cache_get(first_key)["cron"][0]["id"],
                "one",
            )

            wal_path.write_bytes(b"wal-v2-is-different")
            second_fingerprint = profiles._sidebar_db_fingerprint(db_path)
            second_key = (str(db_path), second_fingerprint, False, 0, (), 50, 100, ())
            self.assertNotEqual(first_fingerprint, second_fingerprint)
            self.assertIsNone(profiles._sidebar_profile_cache_get(second_key))

            profiles._sidebar_profile_cache_put(second_key, payload)
            self.assertIsNone(profiles._sidebar_profile_cache_get(first_key))

    def test_profile_cache_is_lru_bounded(self):
        with mock.patch.object(profiles, "_SIDEBAR_PROFILE_CACHE_MAX_ENTRIES", 2):
            for index in range(3):
                key = (f"/db/{index}", (index, None), False, 0, (), 50, 100, ())
                profiles._sidebar_profile_cache_put(key, {"index": index})
            self.assertEqual(len(profiles._SIDEBAR_PROFILE_CACHE), 2)

    def test_applies_defaults_and_returns_defensive_copies(self):
        calls = 0

        @profiles._sidebar_singleflight_cache
        def scan(profile="all", limit=20):
            nonlocal calls
            calls += 1
            return {"profile": profile, "rows": [{"limit": limit}]}

        first = scan()
        first["rows"][0]["limit"] = 999
        second = scan(profile="all", limit=20)

        self.assertEqual(calls, 1)
        self.assertEqual(second, {"profile": "all", "rows": [{"limit": 20}]})

    def test_coalesces_concurrent_identical_scans(self):
        workers = 12
        entered = threading.Event()
        release = threading.Event()
        calls = 0
        calls_lock = threading.Lock()

        @profiles._sidebar_singleflight_cache
        def scan(profile="all"):
            nonlocal calls
            with calls_lock:
                calls += 1
            entered.set()
            self.assertTrue(release.wait(timeout=2))
            return {"profile": profile, "rows": []}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(scan, "default") for _ in range(workers)]
            self.assertTrue(entered.wait(timeout=1))
            time.sleep(0.05)
            release.set()
            results = [future.result(timeout=2) for future in futures]

        self.assertEqual(calls, 1)
        self.assertEqual(results, [{"profile": "default", "rows": []}] * workers)

    def test_expires(self):
        clock = iter((100.0, 100.0, 100.0, 106.0, 106.0, 106.0))
        calls = 0

        @profiles._sidebar_singleflight_cache
        def scan():
            nonlocal calls
            calls += 1
            return {"generation": calls}

        with mock.patch.object(profiles.time, "monotonic", side_effect=clock):
            self.assertEqual(scan(), {"generation": 1})
            self.assertEqual(scan(), {"generation": 2})
        self.assertEqual(calls, 2)

    def test_does_not_cache_failures(self):
        calls = 0

        @profiles._sidebar_singleflight_cache
        def scan():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient")
            return {"ok": True}

        with self.assertRaisesRegex(RuntimeError, "transient"):
            scan()
        self.assertEqual(scan(), {"ok": True})
        self.assertEqual(scan(), {"ok": True})
        self.assertEqual(calls, 2)

    def test_does_not_cache_payloads_that_carry_profile_errors(self):
        # A 200 with a non-empty errors[] is how a failed profile scan is
        # reported. Caching it for the TTL keeps the empty recents page in
        # front of a store that has already recovered.
        calls = 0

        @profiles._sidebar_singleflight_cache
        def scan():
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "errors": [{"profile": "default", "error": "disk I/O error"}],
                    "recents": {"sessions": []},
                }
            return {"errors": [], "recents": {"sessions": [{"id": "yesterday"}]}}

        first = scan()
        second = scan()

        self.assertEqual(first["errors"][0]["error"], "disk I/O error")
        self.assertEqual(second["recents"]["sessions"], [{"id": "yesterday"}])
        self.assertEqual(calls, 2)

    def test_can_be_disabled(self):
        calls = 0

        @profiles._sidebar_singleflight_cache
        def scan():
            nonlocal calls
            calls += 1
            return calls

        with mock.patch.object(profiles, "_SIDEBAR_CACHE_TTL_SECONDS", 0.0):
            self.assertEqual((scan(), scan()), (1, 2))

    def test_preserves_fastapi_signature(self):
        def scan(profile: str = "all", limit: int = 20):
            return profile, limit

        wrapped = profiles._sidebar_singleflight_cache(scan)

        self.assertEqual(inspect.signature(wrapped), inspect.signature(scan))


if __name__ == "__main__":
    unittest.main()
