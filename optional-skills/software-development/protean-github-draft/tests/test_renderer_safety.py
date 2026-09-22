#!/usr/bin/env python3
"""Regression tests for the local review artifact trust boundary."""
from __future__ import annotations

import sys
import unittest
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import render_draft_html as renderer  # noqa: E402
import draft_pipeline  # noqa: E402


class RendererSafetyTests(unittest.TestCase):
    def test_documented_markdown_and_fences_remain_faithful(self):
        draft = "# Heading\n\nA **normal** [link](https://example.com).\n\n~~~python\nprint('ok')\n~~~\n"
        page = renderer.build_page(
            draft=draft,
            repo="owner/repository",
            tab="pull request draft",
            title="A normal title",
            badge="DRAFT — NOT POSTED",
            src_lines=6,
            owners="reviewer",
            state="awaiting human review",
        )
        self.assertIn("Heading", page)
        self.assertIn("normal", page)
        self.assertIn('href="https://example.com"', page)
        self.assertIn("print", page)
        self.assertEqual(draft_pipeline.fence_bodies(draft), ["print('ok')"])

    def test_hostile_markdown_is_inert(self):
        draft = (
            '<script>alert(1)</script>\n'
            '<img src="javascript:alert(2)" onerror="owned()">\n'
            '[run](javascript:alert(3))\n'
            '<div onclick="owned()">text</div>\n'
        )
        page = renderer.build_page(
            draft=draft,
            repo='owner" onmouseover="owned()',
            tab="pull request draft",
            title='title</title><script>owned()</script>',
            badge='badge"><script>owned()</script>',
            src_lines=4,
            owners='owner</span><script>owned()</script>',
            state='state" onclick="owned()',
        )
        self.assertNotIn("<script>owned()</script>", page)
        self.assertNotIn("<img", page)
        self.assertNotRegex(page.lower(), r'<[^>]+(?:href|src)=["\']javascript:')
        self.assertNotRegex(page.lower(), r'<[^>]+\bonerror=')
        self.assertNotRegex(page.lower(), r'<[^>]+\bonclick=')
        self.assertNotIn("owner</span>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("&lt;/span&gt;", page)


if __name__ == "__main__":
    unittest.main()
