#!/usr/bin/env python3
"""Unit tests that do not need a live browser or personal data."""
import unittest

from browser_inspect import norm_url


class NormUrlTests(unittest.TestCase):
    def test_strips_query_and_slash(self):
        self.assertEqual(
            norm_url("https://example.com/group/1/?sorting=new"),
            "https://example.com/group/1",
        )

    def test_same_page_is_equal(self):
        a = "https://example.com/page"
        b = "https://example.com/page/"
        self.assertEqual(norm_url(a), norm_url(b))

    def test_different_paths_differ(self):
        self.assertNotEqual(
            norm_url("https://example.com/a"),
            norm_url("https://example.com/b"),
        )


if __name__ == "__main__":
    unittest.main()
