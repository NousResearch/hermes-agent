"""Tests for command_token_source cache TTL handling.

Proves the cache never exceeds the advertised token TTL (the max(ttl-60, 5.0)
bug forced a 5s cache even for a 1s TTL).
"""
import time
import unittest
from unittest import mock

from agent.command_token_source import CommandTokenSource, _TOKEN_REFRESH_LEEWAY_SECONDS


class TestCacheNeverExceedsTTL(unittest.TestCase):
    def _make_source(self, ttl):
        """Create a source whose _mint returns a token with the given TTL."""
        source = CommandTokenSource("echo fake", "test")
        with mock.patch(
            "agent.command_token_source._mint",
            return_value=("fake-token", float(ttl)),
        ):
            source()
        return source

    def test_cache_never_exceeds_advertised_ttl(self):
        """The cache must be <= max(0, ttl - leeway), never forced to 5s."""
        for ttl in (1.0, 5.0, 10.0, 30.0, 60.0, 120.0):
            source = self._make_source(ttl)
            cache_duration = source._expires_at - time.monotonic()
            max_allowed = max(0.0, ttl - _TOKEN_REFRESH_LEEWAY_SECONDS)
            self.assertLessEqual(
                cache_duration,
                max_allowed,
                f"TTL={ttl}: cache {cache_duration:.2f}s exceeds max allowed {max_allowed:.2f}s",
            )

    def test_short_ttl_not_forced_to_5s(self):
        """A 1s TTL must NOT be cached for 5s (the bug)."""
        source = self._make_source(1.0)
        cache_duration = source._expires_at - time.monotonic()
        self.assertLess(cache_duration, 5.0, f"Cache {cache_duration:.2f}s: bug forces 5s!")


if __name__ == "__main__":
    unittest.main(verbosity=2)

