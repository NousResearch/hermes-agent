"""T3 PR97786 — regression coverage for cron/suggestions.py provenance fail-closed.

Covers contracts:
  C9 — provenance lookup failure fails closed.
  C15 — suggestion/background paths obey retained authority.

Each of the 4 mutation entry points (add_suggestion, accept_suggestion,
dismiss_suggestion, clear_resolved) must refuse mutations when the retained
self-improvement Decision is deny. The fixture swaps in a deny Decision and
verifies that each entry point raises PermissionError.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from agent.self_improvement_policy import (
    Decision,
    deny as decision_deny,
)
from agent.self_improvement_decision_context import self_improvement_decision_scope


class TestSuggestionsProvenanceFailClosed(unittest.TestCase):
    """C9/C15 — every mutation entry point must refuse when Decision is deny."""

    def setUp(self) -> None:
        # Use a private temp home so tests don't pollute the real ~/.hermes.
        self._tmp_home = tempfile.TemporaryDirectory()
        self._saved_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self._tmp_home.name
        from cron import suggestions as cron_suggestions
        cron_suggestions.SUGGESTIONS_FILE = None  # let _current_suggestions_file resolve from home

    def tearDown(self) -> None:
        if self._saved_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._saved_home
        self._tmp_home.cleanup()

    def _deny_decision(self) -> Decision:
        return decision_deny("test_deny", "test")

    def test_add_suggestion_refuses_when_denied(self) -> None:
        from cron.suggestions import add_suggestion
        with self_improvement_decision_scope(self._deny_decision()):
            with self.assertRaises(PermissionError):
                add_suggestion(
                    title="t", description="d", source="catalog",
                    job_spec={"name": "x"}, dedup_key="k",
                )

    def test_accept_suggestion_refuses_when_denied(self) -> None:
        from cron.suggestions import add_suggestion, accept_suggestion
        # First add while ALLOWED so we have a pending record to try to accept.
        with self_improvement_decision_scope(decision_allow_for_test()):
            rec = add_suggestion(
                title="t", description="d", source="catalog",
                job_spec={"name": "x"}, dedup_key="k2",
            )
            self.assertIsNotNone(rec)
        # Now switch to DENY and try to accept.
        with self_improvement_decision_scope(self._deny_decision()):
            with self.assertRaises(PermissionError):
                accept_suggestion(rec["id"])  # type: ignore[index]

    def test_dismiss_suggestion_refuses_when_denied(self) -> None:
        from cron.suggestions import add_suggestion, dismiss_suggestion
        with self_improvement_decision_scope(decision_allow_for_test()):
            rec = add_suggestion(
                title="t", description="d", source="catalog",
                job_spec={"name": "x"}, dedup_key="k3",
            )
            self.assertIsNotNone(rec)
        with self_improvement_decision_scope(self._deny_decision()):
            with self.assertRaises(PermissionError):
                dismiss_suggestion(rec["id"])  # type: ignore[index]

    def test_clear_resolved_refuses_when_denied(self) -> None:
        from cron.suggestions import clear_resolved
        with self_improvement_decision_scope(self._deny_decision()):
            with self.assertRaises(PermissionError):
                clear_resolved()

    def test_add_suggestion_allowed_when_decision_allows(self) -> None:
        """Sanity: when the Decision is allow, mutations proceed."""
        from cron.suggestions import add_suggestion
        with self_improvement_decision_scope(decision_allow_for_test()):
            rec = add_suggestion(
                title="t", description="d", source="catalog",
                job_spec={"name": "x"}, dedup_key="k4",
            )
            self.assertIsNotNone(rec)
            self.assertEqual(rec["status"], "pending")


def decision_allow_for_test() -> Decision:
    from agent.self_improvement_policy import allow as decision_allow
    return decision_allow("test_allow", "test")


if __name__ == "__main__":
    unittest.main()
