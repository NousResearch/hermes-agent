# -*- coding: utf-8 -*-
"""Tests for agent.key_router."""
import time, tempfile, unittest
from pathlib import Path
from unittest.mock import MagicMock


def _setup_isolation():
    import agent.key_router as kr
    tmp = tempfile.mkdtemp()
    kr.STATE_DIR = Path(tmp)
    kr.STATE_FILE = Path(tmp) / "key_pool.json"
    kr._router = None
    return tmp


class TestKeyEntry(unittest.TestCase):
    def setUp(self):
        _setup_isolation()

    def _make(self, **kw):
        from agent.key_router import KeyEntry
        d = {"api_key": "ak_test_key_12345678"}
        d.update(kw)
        return KeyEntry(**d)

    def test_short_key(self):
        self.assertEqual(self._make(api_key="ak_abcdefgh12345678").short_key, "ak_abcde…5678")

    def test_short_key_short(self):
        self.assertEqual(self._make(api_key="short").short_key, "short")

    def test_available_when_ok(self):
        self.assertTrue(self._make().is_available)

    def test_unavailable_when_exhausted(self):
        e = self._make()
        e.mark_exhausted(error_code=429, retry_after=3600)
        self.assertFalse(e.is_available)

    def test_available_after_cooldown(self):
        import agent.key_router as kr
        orig = kr.COOLDOWN_MIN
        kr.COOLDOWN_MIN = 0
        try:
            e = self._make()
            e.mark_exhausted(error_code=429, retry_after=0.1)
            time.sleep(0.15)
            self.assertTrue(e.is_available)
        finally:
            kr.COOLDOWN_MIN = orig

    def test_mark_ok_resets(self):
        e = self._make()
        e.mark_exhausted(error_code=429)
        e.mark_ok()
        self.assertEqual(e.status, "ok")

    def test_to_dict_roundtrip(self):
        from agent.key_router import KeyEntry
        e = self._make(label="primary", priority=0)
        e.mark_exhausted(error_code=402)
        e2 = KeyEntry.from_dict(e.to_dict())
        self.assertEqual(e2.api_key, e.api_key)
        self.assertEqual(e2.status, "exhausted")


class TestSelection(unittest.TestCase):
    def setUp(self):
        _setup_isolation()

    def _router(self, strategy="fill_first"):
        from agent.key_router import KeyRouter
        r = KeyRouter("https://api.example.com/v1", strategy=strategy)
        r.add_key("ak_primary", label="primary", priority=0)
        r.add_key("ak_backup", label="backup", priority=1)
        return r

    def test_fill_first(self):
        self.assertEqual(self._router().select_key().api_key, "ak_primary")

    def test_round_robin(self):
        from agent.key_router import KeyRouter
        r = KeyRouter("https://api.example.com/v1", strategy="round_robin")
        r.add_key("ak_a", label="a", priority=0)
        r.add_key("ak_b", label="b", priority=0)
        self.assertEqual({r.select_key().api_key for _ in range(10)}, {"ak_a", "ak_b"})

    def test_least_used(self):
        from agent.key_router import KeyRouter
        r = KeyRouter("https://api.example.com/v1", strategy="least_used")
        r.add_key("ak_a", label="a", priority=0)
        r.add_key("ak_b", label="b", priority=0)
        r._keys[0].request_count = 3
        self.assertEqual(r.select_key().api_key, "ak_b")

    def test_exhausted_skipped(self):
        r = self._router()
        r._keys[0].mark_exhausted(error_code=429)
        self.assertEqual(r.select_key().api_key, "ak_backup")

    def test_all_exhausted(self):
        r = self._router()
        for k in r._keys:
            k.mark_exhausted(error_code=429, retry_after=9999)
        self.assertIsNone(r.select_key())


class TestErrors(unittest.TestCase):
    def setUp(self):
        _setup_isolation()

    def _router(self):
        from agent.key_router import KeyRouter
        r = KeyRouter("https://api.example.com/v1")
        r.add_key("ak_primary", label="primary", priority=0)
        r.add_key("ak_backup", label="backup", priority=1)
        return r

    def test_mark_exhausted(self):
        r = self._router()
        r.mark_exhausted("ak_primary", error_code=429, error_message="rate limited")
        self.assertEqual(r._keys[0].status, "exhausted")

    def test_mark_ok(self):
        r = self._router()
        r.mark_exhausted("ak_primary", error_code=429)
        r.mark_ok("ak_primary")
        self.assertEqual(r._keys[0].status, "ok")

    def test_is_exhaustion_429(self):
        from agent.key_router import KeyRouter
        exc = Exception("rate limited"); exc.status_code = 429
        self.assertTrue(KeyRouter._is_exhaustion(exc))

    def test_is_exhaustion_quota(self):
        from agent.key_router import KeyRouter
        self.assertTrue(KeyRouter._is_exhaustion(Exception("quota exceeded")))

    def test_not_exhaustion(self):
        from agent.key_router import KeyRouter
        self.assertFalse(KeyRouter._is_exhaustion(Exception("bad request")))

    def test_extract_retry_after(self):
        from agent.key_router import KeyRouter
        self.assertEqual(KeyRouter._extract_retry_after(Exception("retry after 120 seconds")), 120.0)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        _setup_isolation()

    def test_save_and_load(self):
        from agent.key_router import KeyRouter
        r = KeyRouter("https://api.example.com/v1")
        r.add_key("ak_primary", label="primary", priority=0)
        r.add_key("ak_backup", label="backup", priority=1)
        r.mark_exhausted("ak_primary", error_code=429)
        r.save_state()

        r2 = KeyRouter("https://api.example.com/v1")
        r2.add_key("ak_primary", label="primary", priority=0)
        r2.add_key("ak_backup", label="backup", priority=1)
        r2._load_state()
        self.assertEqual(r2._keys[0].status, "exhausted")

    def test_different_url_ignored(self):
        from agent.key_router import KeyRouter
        r = KeyRouter("https://api.example.com/v1")
        r.add_key("ak_primary", label="primary", priority=0)
        r.save_state()
        r2 = KeyRouter("https://other.example.com/v1")
        r2.add_key("ak_primary", label="primary", priority=0)
        r2._load_state()
        self.assertEqual(r2._keys[0].status, "ok")


class TestFailover(unittest.TestCase):
    def setUp(self):
        _setup_isolation()

    def _router(self):
        from agent.key_router import KeyRouter
        r = KeyRouter("https://api.example.com/v1")
        r.add_key("ak_primary", label="primary", priority=0)
        r.add_key("ak_backup", label="backup", priority=1)
        return r

    def test_success_first_key(self):
        from agent.key_router import call_with_failover
        fn = MagicMock(return_value="ok")
        self.assertEqual(call_with_failover(self._router(), fn), "ok")
        self.assertEqual(fn.call_count, 1)

    def test_failover_on_429(self):
        from agent.key_router import call_with_failover
        exc = Exception("rate limited"); exc.status_code = 429
        fn = MagicMock(side_effect=[exc, "ok"])
        self.assertEqual(call_with_failover(self._router(), fn), "ok")
        self.assertEqual(fn.call_count, 2)

    def test_all_exhausted_raises(self):
        from agent.key_router import call_with_failover
        exc = Exception("quota exceeded"); exc.status_code = 402
        with self.assertRaises(Exception):
            call_with_failover(self._router(), MagicMock(side_effect=exc))

    def test_non_exhaustion_raises_immediately(self):
        from agent.key_router import call_with_failover
        exc = Exception("bad request"); exc.status_code = 400
        fn = MagicMock(side_effect=exc)
        with self.assertRaises(Exception):
            call_with_failover(self._router(), fn)
        self.assertEqual(fn.call_count, 1)

if __name__ == "__main__":
    unittest.main()
