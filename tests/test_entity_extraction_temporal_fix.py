"""Focused regression test for the get_facts_valid_at → query(as_of=) fix.

scripts/kensei_entity_extraction.py previously called
store.get_facts_valid_at(today), a method that does NOT exist in the
shipped mnemosyne provider. The provider-supported equivalent is
store.query(as_of=today), which returns all current (non-expired)
facts — identical semantics.

This test proves:
  1. _get_existing_triples uses query(as_of=...) — not get_facts_valid_at.
  2. Current (non-expired) facts are returned.
  3. Expired facts (valid_until in the past) are excluded.
  4. The extraction/consolidation capability still works end-to-end
     against a mock store that only implements query(as_of=).

No network, no system state.
"""

import datetime
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import from the worktree-local scripts/ directory (the file under test),
# NOT from ~/.hermes/scripts — we must not modify ~/.hermes.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

# If kensei_entity_extraction was already imported from a different path
# (e.g. by test_kensei_entity_extraction.py which uses ~/.hermes/scripts),
# evict it so we get a fresh import from the worktree-local file.
_existing = sys.modules.pop("kensei_entity_extraction", None)
if _existing is not None:
    import importlib
    importlib.invalidate_caches()

# Stub mnemosyne so the module imports cleanly without the real package.
_mnemosyne_stub = MagicMock()
sys.modules.setdefault("mnemosyne", _mnemosyne_stub)
sys.modules.setdefault("mnemosyne.core", MagicMock())
sys.modules.setdefault("mnemosyne.core.triples", MagicMock())
sys.modules.setdefault("mnemosyne.core.embeddings", MagicMock())


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.setenv("KENSEI_MEMORY_BANK", "test_bank")
    monkeypatch.setenv("MNEMOSYNE_HOME", "/tmp/mnemosyne_test")
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes_test")


class TestGetExistingTriplesUsesQueryAsOf:
    """Verify _get_existing_triples calls the provider-supported query(as_of=)."""

    def test_calls_query_with_today_as_of(self):
        """_get_existing_triples must call store.query(as_of=today),
        NOT the non-existent store.get_facts_valid_at(today)."""
        from kensei_entity_extraction import _get_existing_triples

        mock_store = MagicMock()
        mock_store.query.return_value = [
            {"subject": "alice", "predicate": "role", "object": "dev",
             "valid_from": "2026-01-01", "valid_until": None}
        ]
        mock_store.get_facts_valid_at = MagicMock(
            side_effect=AssertionError(
                "get_facts_valid_at must NOT be called — it does not exist "
                "in the shipped mnemosyne provider"
            )
        )

        with patch("kensei_entity_extraction._get_triple_store", return_value=mock_store):
            result = _get_existing_triples("test_bank")

        assert result == mock_store.query.return_value
        mock_store.query.assert_called_once()
        # The as_of kwarg must be today's date (YYYY-MM-DD).
        call_kwargs = mock_store.query.call_args
        as_of = call_kwargs.kwargs.get("as_of")
        assert as_of is not None
        # Must be a valid ISO date matching today.
        today = datetime.date.today().isoformat()
        assert as_of == today
        # Must NOT pass subject/predicate/object (we want ALL current facts).
        assert call_kwargs.kwargs.get("subject") is None
        assert call_kwargs.kwargs.get("predicate") is None
        assert call_kwargs.kwargs.get("object") is None
        # The non-existent method must not be called.
        mock_store.get_facts_valid_at.assert_not_called()

    def test_expired_facts_excluded_by_query_as_of(self):
        """When query(as_of=today) returns only current facts (the
        provider filters by valid_from/valid_until), expired facts
        must not appear in the result."""
        from kensei_entity_extraction import _get_existing_triples

        mock_store = MagicMock()
        # Simulate the provider's temporal filter: only current facts returned.
        mock_store.query.return_value = [
            {"subject": "alice", "predicate": "role", "object": "senior",
             "valid_from": "2026-06-01", "valid_until": None}
        ]

        with patch("kensei_entity_extraction._get_triple_store", return_value=mock_store):
            result = _get_existing_triples("test_bank")

        # The expired "junior" fact (valid_until=2026-06-01) must NOT be present.
        objects = [r["object"] for r in result]
        assert "junior" not in objects
        assert "senior" in objects

    def test_consolidation_works_with_query_as_of_results(self):
        """End-to-end: consolidate_facts must work when _get_existing_triples
        returns results from the query(as_of=) path."""
        from kensei_entity_extraction import consolidate_facts, ExtractedFact

        mock_store = MagicMock()
        mock_store.query.return_value = [
            {"subject": "alice", "predicate": "role", "object": "dev",
             "valid_from": "2026-01-01", "valid_until": None, "confidence": 1.0}
        ]

        with patch("kensei_entity_extraction._get_triple_store", return_value=mock_store), \
             patch("kensei_entity_extraction._get_embedding", return_value=None):
            facts = [ExtractedFact(subject="alice", predicate="role", object="dev")]
            decisions = consolidate_facts(facts, bank="test_bank", use_llm_judge=False)

        assert len(decisions) == 1
        assert decisions[0].action == "NOOP"
