#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Tests for recon_kit.py."""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recon_kit import add_months, main, parse_date, slugify

REPORT = """---
title: 'market research: widgets'
---

# Report

The market is growing fast [1] and pricing clusters at $20 [2].
An uncited assertion sits here [4].

```
[9] inside a fence never counts
```

## Source appendix

| [n] | Supports | Publisher | Pub date | Accessed | Confidence |
| --- | --- | --- | --- | --- | --- |
| [1] | market growth | [Gartner](https://example.com/g) | 2026-01 | 2026-07-01 | high |
| [2] | pricing | [Acme](javascript:alert(1)) | 2026-05 | 2026-07-01 | medium |
| [3] | never cited | https://example.com/x | 2025-11 | 2026-07-01 | low |
"""

MEMLOG = """---
topic: widgets
updated: 2026-07-22T10:00
---

- (decision) plan approved
- (source) round 1 batch: 4 sources
- (claim) ref=[1] status=verified class=size/growth pub=2026-01 — market growing 12% CAGR
- (claim) ref=[2] status=unverified class=pricing pub=2026-05 — pricing clusters at $20
- (claim) ref=[2] status=verified class=pricing pub=2026-05 — confirmed by second source
- (claim) status=unverified class=behavior pub=2025-03 — users churn at day 8
- (event) dimension 1 complete
"""


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, json.loads(buf.getvalue())


class CitationsTest(unittest.TestCase):
    def test_cross_check(self):
        report = Path(__file__).parent / "_report.md"
        report.write_text(REPORT, encoding="utf-8")
        try:
            code, result = run(["citations", str(report)])
        finally:
            report.unlink()
        self.assertEqual(result["dangling_markers"], [4])
        self.assertEqual(result["orphaned_rows"], [3])
        self.assertNotIn(9, result["markers"])  # fenced content ignored
        self.assertEqual(code, 1)


class TallyTest(unittest.TestCase):
    def test_last_status_wins_per_ref(self):
        log = Path(__file__).parent / "_memlog.md"
        log.write_text(MEMLOG, encoding="utf-8")
        try:
            code, result = run(["tally", str(log)])
        finally:
            log.unlink()
        self.assertEqual(result["by_type"]["claim"], 4)
        self.assertEqual(result["claims"], {"unverified": 1, "verified": 2})
        self.assertEqual(result["claims_total"], 3)  # ref=[2] counted once
        self.assertEqual(code, 0)


class StalenessTest(unittest.TestCase):
    def test_dates(self):
        self.assertEqual(parse_date("2026-01"), date(2026, 1, 1))
        self.assertEqual(add_months(date(2026, 1, 31), 1), date(2026, 2, 28))

    def test_windows(self):
        claims = json.dumps([
            {"claim": "sizing", "class": "size/growth", "pub_date": "2024-06"},
            {"claim": "pricing", "class": "pricing", "pub_date": "2026-06"},
            {"claim": "odd", "class": "unmapped", "pub_date": "2026-06"},
        ])
        f = Path(__file__).parent / "_claims.json"
        f.write_text(claims, encoding="utf-8")
        try:
            code, result = run([
                "staleness", str(f),
                "--windows", '{"size/growth": 18, "pricing": 3}',
                "--today", "2026-07-22",
            ])
        finally:
            f.unlink()
        self.assertEqual(result["stale_count"], 1)  # sizing recheck 2025-12 < today
        self.assertEqual(result["earliest_recheck"], "2025-12-01")
        self.assertEqual(result["no_window_classes"], ["unmapped"])
        self.assertEqual(code, 1)


class SlugTest(unittest.TestCase):
    def test_deterministic_folder(self):
        self.assertEqual(slugify("Créme Brûlée: AI Tools!"), "creme-brulee-ai-tools")
        code, result = run(["slug", "SMB Accounting SaaS", "--type", "market",
                            "--date", "2026-07-22"])
        self.assertEqual(result["folder"], "market-smb-accounting-saas-2026-07-22")
        self.assertEqual(code, 0)


class EscapeSourcesTest(unittest.TestCase):
    def test_escaping_and_url_validation(self):
        report = Path(__file__).parent / "_report.md"
        report.write_text(REPORT, encoding="utf-8")
        try:
            code, result = run(["escape-sources", str(report)])
        finally:
            report.unlink()
        self.assertEqual(result["rows"], 3)
        self.assertTrue(any(u.startswith("javascript:") for u in result["invalid_urls"]))
        self.assertNotIn("javascript:", result["html"])  # never linked
        self.assertIn('href="https://example.com/g"', result["html"])
        self.assertIn('id="src-1"', result["html"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
