"""Acceptance: prefetch budget (default 8000 tokens) and truncation order.

Truncation drops in priority order — system-pin first, then provenance
completeness, recency, salience — and always appends a marker so the model
knows context was omitted.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load  # noqa: E402

budget_mod = load("budget")


def entry(ref, content, pinned=False, created=1_700_000_000.0, salience=0.5,
          provenance=True):
    return {
        "ref": ref,
        "content": content,
        "salience": salience,
        "pinned": pinned,
        "provenance": ({
            "source_session": "s", "author": "Ada", "created_at": created,
            "origin_repo": "default", "trace_id": "t",
        } if provenance else {}),
    }


class BudgetTest(unittest.TestCase):
    def test_default_budget_is_8000(self):
        self.assertEqual(budget_mod.DEFAULT_BUDGET_TOKENS, 8000)

    def test_everything_fits_means_no_marker(self):
        entries = [entry("cog-s-%d" % i, "short fact %d" % i) for i in range(3)]
        text, kept, dropped = budget_mod.apply_budget(entries)
        self.assertEqual(kept, 3)
        self.assertEqual(dropped, 0)
        self.assertNotIn("omitted", text)

    def test_truncation_order_pin_then_provenance_then_recency_then_salience(self):
        big = "x" * 400  # ~100 tokens each; budget fits ~2 plus marker
        old_rich = entry("cog-old", big + "old", created=1_700_000_000.0, salience=0.9)
        new_rich = entry("cog-new", big + "new", created=1_700_000_100.0, salience=0.1)
        pinned = entry("cog-pin", big + "pin", pinned=True,
                       created=1_700_000_000.0, salience=0.0)
        no_prov = entry("cog-noprov", big + "noprovenance",
                        created=1_700_000_200.0, salience=1.0, provenance=False)
        text, kept, dropped = budget_mod.apply_budget(
            [no_prov, old_rich, new_rich, pinned], budget_tokens=250)
        # Pinned survives despite lowest salience; provenance-less drops first.
        self.assertIn("cog-pin", text)
        self.assertNotIn("cog-noprov", text)
        # Recency beats salience among equals: newer (low salience) beats older.
        self.assertIn("cog-new", text)
        self.assertNotIn("cog-old", text)
        self.assertGreaterEqual(dropped, 1)
        # Marker names the drop count and the budget.
        marker = budget_mod.truncation_marker(dropped, 250)
        self.assertIn(marker, text)
        self.assertIn("250", marker)

    def test_degenerate_budget_keeps_top_entry_and_says_so(self):
        entries = [entry("cog-s-%d" % i, "fact %d" % i) for i in range(5)]
        text, kept, dropped = budget_mod.apply_budget(entries, budget_tokens=1)
        self.assertEqual(kept, 1)
        self.assertEqual(dropped, 4)
        self.assertIn("omitted", text)


if __name__ == "__main__":
    unittest.main()
