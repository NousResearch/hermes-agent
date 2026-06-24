"""Tests for hermes_storage.registry (phase feedback-AGE-240-B).

Covers:
  - ReasonQuery dataclass construction and validation.
  - MemoryRegistry.reason delegates correctly to FactRetriever.reason.
  - Edge cases: empty entities, limit=0, category filtering.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call

from hermes_storage.registry import MemoryRegistry, ReasonQuery


# ---------------------------------------------------------------------------
# ReasonQuery dataclass tests
# ---------------------------------------------------------------------------


class TestReasonQuery:
    def test_minimal_construction(self):
        q = ReasonQuery(entities=["alice"])
        assert q.entities == ["alice"]
        assert q.category is None
        assert q.limit == 10

    def test_full_construction(self):
        q = ReasonQuery(entities=["alice", "bob"], category="people", limit=5)
        assert q.entities == ["alice", "bob"]
        assert q.category == "people"
        assert q.limit == 5

    def test_empty_entities_raises(self):
        with pytest.raises(ValueError, match="entities"):
            ReasonQuery(entities=[])

    def test_zero_limit_raises(self):
        with pytest.raises(ValueError, match="limit"):
            ReasonQuery(entities=["alice"], limit=0)

    def test_negative_limit_raises(self):
        with pytest.raises(ValueError, match="limit"):
            ReasonQuery(entities=["alice"], limit=-1)

    def test_limit_one_is_valid(self):
        q = ReasonQuery(entities=["x"], limit=1)
        assert q.limit == 1

    def test_multiple_entities(self):
        entities = ["peppi", "backend", "auth"]
        q = ReasonQuery(entities=entities)
        assert q.entities == entities

    def test_category_none_by_default(self):
        q = ReasonQuery(entities=["e"])
        assert q.category is None

    def test_category_string(self):
        q = ReasonQuery(entities=["e"], category="tech")
        assert q.category == "tech"


# ---------------------------------------------------------------------------
# MemoryRegistry.reason delegation tests
# ---------------------------------------------------------------------------


class TestMemoryRegistryReason:
    def _make_registry(self, return_value=None):
        """Return a (registry, mock_retriever) pair."""
        mock_retriever = MagicMock()
        mock_retriever.reason.return_value = return_value or []
        registry = MemoryRegistry(retriever=mock_retriever)
        return registry, mock_retriever

    def test_delegates_entities(self):
        registry, mock_retriever = self._make_registry()
        query = ReasonQuery(entities=["alice", "bob"])
        registry.reason(query)
        mock_retriever.reason.assert_called_once_with(
            entities=["alice", "bob"],
            category=None,
            limit=10,
        )

    def test_delegates_category(self):
        registry, mock_retriever = self._make_registry()
        query = ReasonQuery(entities=["alice"], category="people")
        registry.reason(query)
        mock_retriever.reason.assert_called_once_with(
            entities=["alice"],
            category="people",
            limit=10,
        )

    def test_delegates_limit(self):
        registry, mock_retriever = self._make_registry()
        query = ReasonQuery(entities=["alice"], limit=3)
        registry.reason(query)
        mock_retriever.reason.assert_called_once_with(
            entities=["alice"],
            category=None,
            limit=3,
        )

    def test_returns_retriever_result(self):
        expected = [
            {"fact_id": 1, "content": "Alice works on backend.", "score": 0.9},
            {"fact_id": 2, "content": "Backend uses Python.", "score": 0.7},
        ]
        registry, _ = self._make_registry(return_value=expected)
        result = registry.reason(ReasonQuery(entities=["alice", "backend"]))
        assert result == expected

    def test_returns_empty_list_when_no_facts(self):
        registry, _ = self._make_registry(return_value=[])
        result = registry.reason(ReasonQuery(entities=["unknown_entity"]))
        assert result == []

    def test_single_entity_query(self):
        registry, mock_retriever = self._make_registry(return_value=[])
        query = ReasonQuery(entities=["solo"])
        registry.reason(query)
        mock_retriever.reason.assert_called_once_with(
            entities=["solo"],
            category=None,
            limit=10,
        )

    def test_multiple_calls_are_independent(self):
        registry, mock_retriever = self._make_registry(return_value=[])
        q1 = ReasonQuery(entities=["a"], category="cat1", limit=5)
        q2 = ReasonQuery(entities=["b", "c"], category=None, limit=20)
        registry.reason(q1)
        registry.reason(q2)
        assert mock_retriever.reason.call_count == 2
        mock_retriever.reason.assert_any_call(entities=["a"], category="cat1", limit=5)
        mock_retriever.reason.assert_any_call(entities=["b", "c"], category=None, limit=20)

    def test_retriever_stored_as_attribute(self):
        mock_retriever = MagicMock()
        registry = MemoryRegistry(retriever=mock_retriever)
        assert registry._retriever is mock_retriever

    def test_reason_propagates_retriever_exception(self):
        mock_retriever = MagicMock()
        mock_retriever.reason.side_effect = RuntimeError("db error")
        registry = MemoryRegistry(retriever=mock_retriever)
        with pytest.raises(RuntimeError, match="db error"):
            registry.reason(ReasonQuery(entities=["x"]))

    def test_all_fields_forwarded_together(self):
        registry, mock_retriever = self._make_registry(return_value=[])
        query = ReasonQuery(entities=["peppi", "backend"], category="work", limit=7)
        registry.reason(query)
        mock_retriever.reason.assert_called_once_with(
            entities=["peppi", "backend"],
            category="work",
            limit=7,
        )
